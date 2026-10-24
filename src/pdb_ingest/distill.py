"""Stage 2 baseline: multi-head simplex + entropy distillation on frozen text embeddings (torch imported lazily)."""
import json
import math
from .battery import ENNEAGRAM_TYPES,MBTI_TYPES
from .text import AP_TYPES

CORE_SYSTEMS=('MBTI','ENNEAGRAM','AP')
LAMBDA=.1

def class_lists(systems=CORE_SYSTEMS):
    lists={'MBTI':list(MBTI_TYPES),'ENNEAGRAM':list(ENNEAGRAM_TYPES),'AP':sorted(AP_TYPES)}
    return {s:lists[s] for s in systems}

def load_targets(conn,systems=CORE_SYSTEMS):
    """profile_num -> {system: (simplex aligned to class_lists, entropy bits, N)}; only training-eligible heads.
    Returns (targets, dropped) where dropped counts vote mass on labels outside the class list, per system."""
    classes=class_lists(systems); targets={}; dropped={s:0 for s in systems}
    rows=conn.execute('''SELECT v.profile_num,v.system_name,v.vote_counts_json,v.total_system_votes FROM vote_distributions v
                         JOIN profiles p USING(profile_num) WHERE p.complete AND v.training_eligible''').fetchall()
    for n,system,counts,total in rows:
        if system not in classes: continue
        counts=json.loads(counts); index={c:i for i,c in enumerate(classes[system])}
        vec=[0.0]*len(index)
        for label,v in counts.items():
            if label in index: vec[index[label]]+=v
            else: dropped[system]+=v
        kept=sum(vec)
        if kept<10: continue
        p=[v/kept for v in vec]
        targets.setdefault(n,{})[system]=(p,-sum(x*math.log2(x) for x in p if x),int(kept))
    return targets,dropped

def tensors(ids,X,targets,classes,torch):
    """Stack features and per-system targets for profile ids; mask=1 only where the head is eligible."""
    x=torch.tensor([X[n] for n in ids],dtype=torch.float32); out={}
    for s,cl in classes.items():
        P=torch.zeros(len(ids),len(cl)); H=torch.zeros(len(ids)); N=torch.zeros(len(ids)); M=torch.zeros(len(ids))
        for i,n in enumerate(ids):
            t=targets[n].get(s)
            if t: P[i]=torch.tensor(t[0]); H[i]=t[1]; N[i]=t[2]; M[i]=1
        out[s]=(P,H,N,M)
    return x,out

def head_model(dim,classes,hidden=256,dropout=.2):
    import torch
    class HeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk=torch.nn.Sequential(torch.nn.Linear(dim,hidden),torch.nn.GELU(),torch.nn.Dropout(dropout))
            self.dist=torch.nn.ModuleDict({s:torch.nn.Linear(hidden,len(c)) for s,c in classes.items()})
            self.ent=torch.nn.ModuleDict({s:torch.nn.Linear(hidden,1) for s in classes})
        def forward(self,x):
            h=self.trunk(x)
            return {s:(torch.log_softmax(self.dist[s](h),-1),torch.nn.functional.softplus(self.ent[s](h)).squeeze(-1)) for s in self.dist}
    return HeadModel()

def loss(outputs,target,lam=LAMBDA):
    """Spec 5.1: sum_s 1[N>=10] log(1+N) KL(p||p_hat) + lam sum_s 1[N>=10] (H-H_hat)^2, averaged over profiles."""
    import torch
    total=0
    for s,(logq,hhat) in outputs.items():
        P,H,N,M=target[s]
        kl=(torch.xlogy(P,P)-P*logq).sum(-1)
        total=total+(M*torch.log1p(N)*kl).sum()+lam*(M*(H-hhat)**2).sum()
    return total/len(next(iter(outputs.values()))[1])

def kl_rows(P,Q):
    return [sum(p*math.log(p/q) for p,q in zip(pr,qr) if p>0) for pr,qr in zip(P,Q)]

def metrics(P,Q,H,Hhat):
    from .audits import spearman
    argmax=lambda r:max(range(len(r)),key=r.__getitem__)
    return {'n':len(P),'kl':sum(kl_rows(P,Q))/len(P),'top1':sum(argmax(p)==argmax(q) for p,q in zip(P,Q))/len(P),
            'entropy_mae':sum(abs(a-b) for a,b in zip(H,Hhat))/len(P),'entropy_spearman':spearman(H,Hhat) if len(P)>2 else float('nan')}

def cross_validate(X,targets,classes=None,*,folds=5,seed=0,epochs=300,patience=20,lr=1e-3,weight_decay=1e-2,hidden=256,device='cpu'):
    """Out-of-fold predictions for the head model and a train-fold-mean baseline; returns per-system metrics."""
    import torch
    from sklearn.model_selection import KFold
    classes=classes or class_lists()
    ids=sorted(n for n in targets if n in X)
    preds={s:{'P':[],'Q':[],'Qb':[],'H':[],'Hhat':[],'Hb':[]} for s in classes}
    for train_idx,test_idx in KFold(folds,shuffle=True,random_state=seed).split(ids):
        torch.manual_seed(seed)
        train=[ids[i] for i in train_idx]; test=[ids[i] for i in test_idx]
        cut=max(1,len(train)//7); val,fit=train[:cut],train[cut:]
        xf,tf=tensors(fit,X,targets,classes,torch)
        mu,sd=xf.mean(0),xf.std(0)+1e-6
        norm=lambda ids_:((tensors(ids_,X,targets,classes,torch)[0]-mu)/sd).to(device)
        to=lambda t:{s:tuple(v.to(device) for v in vals) for s,vals in t.items()}
        xfit,tfit=norm(fit),to(tf); xval,tval=norm(val),to(tensors(val,X,targets,classes,torch)[1])
        model=head_model(xf.shape[1],classes,hidden).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
        best=(float('inf'),None); stale=0
        for _ in range(epochs):
            model.train(); opt.zero_grad(); loss(model(xfit),tfit).backward(); opt.step()
            model.eval()
            with torch.no_grad(): v=loss(model(xval),tval).item()
            if v<best[0]-1e-6: best=(v,{k:t.clone() for k,t in model.state_dict().items()}); stale=0
            else:
                stale+=1
                if stale>=patience: break
        model.load_state_dict(best[1]); model.eval()
        with torch.no_grad(): out=model(norm(test))
        for s in classes:
            rows=[(i,targets[n][s]) for i,n in enumerate(test) if s in targets[n]]
            fit_rows=[targets[n][s] for n in train if s in targets[n]]
            if not rows or not fit_rows: continue
            k=len(classes[s]); base=[sum(r[0][j] for r in fit_rows)/len(fit_rows)+1e-3 for j in range(k)]
            z=sum(base); base=[b/z for b in base]; hb=sum(r[1] for r in fit_rows)/len(fit_rows)
            q=out[s][0].exp().cpu().tolist(); hh=out[s][1].cpu().tolist()
            for i,(p,h,_) in rows:
                preds[s]['P'].append(p); preds[s]['Q'].append(q[i]); preds[s]['Qb'].append(base)
                preds[s]['H'].append(h); preds[s]['Hhat'].append(hh[i]); preds[s]['Hb'].append(hb)
    result={}
    for s,d in preds.items():
        if not d['P']: result[s]={'n':0}; continue
        result[s]={'model':metrics(d['P'],d['Q'],d['H'],d['Hhat']),'baseline':metrics(d['P'],d['Qb'],d['H'],d['Hb'])}
    return result

VIEWS={'main':{'scrub':True,'mask_prompts':True},'unmasked':{'scrub':True,'mask_prompts':False},'raw':{'scrub':False,'mask_prompts':True}}

def run_distill(store,view='main',*,views_fn=None,encoder=None,out_root='data/stage2',folds=5,epochs=300,seed=0):
    """Embed every profile with targets under one encoder view, cross-validate, and write <out_root>/<view>/report.json."""
    import hashlib
    from pathlib import Path
    from . import embed
    if views_fn is None:
        from .encoder_text import encoder_view as views_fn
    tokenize,encode,stats=encoder or embed.hf_encoder()
    targets,dropped=load_targets(store.connection)
    X={}
    for n in sorted(targets):
        views=views_fn(store,n,**VIEWS[view])
        sha=hashlib.sha256(json.dumps(views,ensure_ascii=False).encode('utf-8')).hexdigest()
        vec=embed.cached_vector(Path(out_root)/'emb'/view,n,sha,lambda:embed.profile_vector(embed.embed_segments(views,tokenize,encode)))
        if vec is not None: X[n]=vec
    report={'view':view,'options':VIEWS[view],'targets':len(targets),'embedded':len(X),'dropped_votes_outside_classes':dropped,
            'encoder':{k:v for k,v in stats.items()},'folds':folds,'seed':seed,
            'metrics':cross_validate(X,targets,folds=folds,seed=seed,epochs=epochs)}
    path=Path(out_root)/view/'report.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=1),encoding='utf-8')
    return report
