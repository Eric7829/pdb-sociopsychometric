"""Fixing overconfident encoder heads on pasted text: regularisation vs training on excerpts. Database read-only.

  python tools/short_text_experiment.py --db data/pdb-top.duckdb [--samples top,top2]

For crowd letters, agent (Form M v3b) letters and agent Big Five, compares the panel's TF-IDF + encoder heads trained
  M0: as now (encoder logistic C=1 / RidgeCV on full profiles),
  M1: stronger regularisation of the encoder half (C grid / larger ridge alphas), full profiles,
  M2: full profiles plus up to 4 evenly spaced 250-word windows per profile as extra training rows (same label),
in 5-fold CV over profiles (a profile's windows never cross folds). Scored on held-out full profiles and on held-out
250-word excerpts (the first 250 words of pinned comments and Google Docs, as in tools/type_ablation.py): balanced
accuracy and log loss for letters, Spearman r for Big Five. Encoder vectors of excerpts and windows are cached in
data/stage2/excerpt_vectors_<model>.npz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from type_ablation import load
from pdb_ingest import embed,panel
from pdb_ingest.panel import LETTERS,l2

WINDOW=250; WINDOWS=4

def windows(doc):
    w=doc.split(); n=len(w)
    if n<=WINDOW: return []
    starts=np.linspace(0,n-WINDOW,min(WINDOWS,n//WINDOW)).astype(int)   # spread out, barely overlapping
    return [' '.join(w[s:s+WINDOW]) for s in sorted(set(starts))]

def vectors(texts,enc,path):
    """Encoder vectors for texts, cached by text hash."""
    cache=dict(np.load(path)) if path.exists() else {}; out=[]; new=0
    for t in texts:
        k=hashlib.sha256(t.encode('utf-8')).hexdigest()[:20]
        if k not in cache: cache[k]=np.array(embed.profile_vector(embed.embed_views([(t,[(0,len(t))])],enc)),dtype='float32'); new+=1
        out.append(cache[k])
    if new: np.savez(path,**cache)
    return l2(np,np.array(out))

def main():
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--samples',default='top,top2')
    p.add_argument('--output',type=Path,default=Path('data/stage2/short_text_experiment.json'))
    p.add_argument('--cs',default='',help='Only test these encoder-head C values (comma-separated), letters and the crowd 16-type head; skips M2 and Big Five')
    args=p.parse_args()
    data=load(args.db,tuple(args.samples.split(','))); ids=sorted(data)
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); vpath=Path('data/stage2')/f'excerpt_vectors_{embed.model_slug(enc[2]["model"])}.npz'
    D=[data[n]['doc'] for n in ids]; F=l2(np,[data[n]['vec'] for n in ids]); C=[data[n]['crop'] for n in ids]
    W=[windows(d) for d in D]; owner=np.array([i for i,ws in enumerate(W) for _ in ws]); Wt=[w for ws in W for w in ws]
    Ec=vectors(C,enc,vpath); Ew=vectors(Wt,enc,vpath) if Wt else np.zeros((0,F.shape[1]))
    print(json.dumps({'profiles':len(ids),'windows':len(Wt)}),flush=True)
    v3b={int(f.stem):json.loads(f.read_text(encoding='utf-8'))['scored']['type'] for f in panel.AGENT_FORMM.glob('*.json')}
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True)
    cv=list(KFold(5,shuffle=True,random_state=0).split(ids)); res={'profiles':len(ids),'windows':len(Wt)}
    def logloss(p,y): p=np.clip(p,1e-6,1-1e-6); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    def bal(p,y): return float(np.mean([np.mean((p>=.5)[y==v]==v) for v in (True,False) if (y==v).any()]))
    variants={'M0 C=1':(1,False),'M1 C=.1':(.1,False),'M1 C=.03':(.03,False),'M1 C=.01':(.01,False),
              'M2 C=1':(1,True),'M2 C=.1':(.1,True),'M2 C=.03':(.03,True)}
    if args.cs: variants={f'M1 C={c}':(float(c),False) for c in args.cs.split(',')}
    # Letters
    for target in ('crowd','agent'):
        lab={n:(data[n]['crowd'] if target=='crowd' else v3b.get(n)) for n in ids}; keep=np.array([lab[n] is not None for n in ids])
        res[target]={}
        for name,(Cg,aug) in variants.items():
            out={'full_bal':[],'crop_bal':[],'full_ll':[],'crop_ll':[],'crop_enc_extreme':[]}
            for i,(a,_) in enumerate(LETTERS):
                y=np.array([lab[n] is not None and lab[n][i]==a for n in ids]); pf=np.full(len(ids),np.nan); pc=pf.copy(); pe=pf.copy()
                for tr,te in cv:
                    tr=tr[keep[tr]]; te=te[keep[te]]
                    wtr=np.where(np.isin(owner,tr))[0] if aug else np.array([],int)
                    Dtr=[D[k] for k in tr]+[Wt[k] for k in wtr]; ytr=np.r_[y[tr],y[owner[wtr]]]; Etr=np.r_[F[tr],Ew[wtr]] if aug else F[tr]
                    t=tf().fit(Dtr); mt=LogisticRegression(C=4,max_iter=3000,class_weight='balanced').fit(t.transform(Dtr),ytr)
                    s=StandardScaler().fit(Etr); me=LogisticRegression(C=Cg,max_iter=3000,class_weight='balanced').fit(s.transform(Etr),ytr)
                    pf[te]=(mt.predict_proba(t.transform([D[k] for k in te]))[:,1]+me.predict_proba(s.transform(F[te]))[:,1])/2
                    ec=me.predict_proba(s.transform(Ec[te]))[:,1]; pe[te]=ec
                    pc[te]=(mt.predict_proba(t.transform([C[k] for k in te]))[:,1]+ec)/2
                m=keep
                out['full_bal'].append(bal(pf[m],y[m])); out['crop_bal'].append(bal(pc[m],y[m]))
                out['full_ll'].append(logloss(pf[m],y[m])); out['crop_ll'].append(logloss(pc[m],y[m]))
                out['crop_enc_extreme'].append(float(np.mean((pe[m]<.05)|(pe[m]>.95))))
            res[target][name]={k:round(float(np.mean(v)),4) for k,v in out.items()}|{'crop_bal_by_letter':[round(x,3) for x in out['crop_bal']]}
            print(json.dumps({'target':target,'variant':name,**res[target][name]}),flush=True)
    if args.cs:   # crowd 16-type head: TF-IDF multinomial (C=4) averaged with the encoder multinomial at each C
        from pdb_ingest.panel import MBTI
        keep=np.array([data[n]['crowd'] is not None for n in ids]); y=np.array([data[n]['crowd'] or '' for n in ids]); res['crowd_type']={}
        for c in [.3]+[float(x) for x in args.cs.split(',')]:
            Pf=np.zeros((len(ids),16)); Pc=Pf.copy()
            for tr,te in cv:
                tr=tr[keep[tr]]; te=te[keep[te]]
                t=tf().fit([D[k] for k in tr]); mt=LogisticRegression(C=4,max_iter=3000).fit(t.transform([D[k] for k in tr]),y[tr])
                s=StandardScaler().fit(F[tr]); me=LogisticRegression(C=c,max_iter=3000).fit(s.transform(F[tr]),y[tr]); cols=[MBTI.index(k) for k in mt.classes_]
                for P,Dx,Ex in ((Pf,[D[k] for k in te],F[te]),(Pc,[C[k] for k in te],Ec[te])):
                    P[np.ix_(te,cols)]=(mt.predict_proba(t.transform(Dx))+me.predict_proba(s.transform(Ex)))/2
            idx=np.array([MBTI.index(k) if k else 0 for k in y])
            sc=lambda P:{'exact':round(float(np.mean(P[keep].argmax(1)==idx[keep])),4),
                         'top3':round(float(np.mean([i in np.argsort(-p)[:3] for p,i in zip(P[keep],idx[keep])])),4),
                         'log_loss':round(float(-np.mean(np.log(np.clip(P[keep][np.arange(keep.sum()),idx[keep]],1e-6,1)))),4)}
            res['crowd_type'][f'C={c}']={'full':sc(Pf),'crop':sc(Pc)}; print(json.dumps({'target':'crowd_type','C':c,**res['crowd_type'][f'C={c}']}),flush=True)
        args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); return
    # Big Five (agent IPIP)
    R=np.array([data[n]['ipip'] is not None for n in ids]); res['big5']={}
    for name,(alpha,aug) in {'M0 RidgeCV':(None,False),'M1 alpha=1e3':(1e3,False),'M1 alpha=1e4':(1e4,False),'M2 RidgeCV':(None,True),'M2 alpha=1e3':(1e3,True)}.items():
        from sklearn.linear_model import RidgeCV
        full=[];crop=[]
        for k in 'OCEAN':
            y=np.array([data[n]['ipip'][k] if data[n]['ipip'] else np.nan for n in ids]); pf=np.full(len(ids),np.nan); pc=pf.copy()
            for tr,te in cv:
                tr=tr[R[tr]]; te=te[R[te]]; wtr=np.where(np.isin(owner,tr))[0] if aug else np.array([],int)
                Dtr=[D[i] for i in tr]+[Wt[i] for i in wtr]; ytr=np.r_[y[tr],y[owner[wtr]]]; Etr=np.r_[F[tr],Ew[wtr]] if aug else F[tr]
                t=tf().fit(Dtr); mt=RidgeCV(alphas=np.logspace(-1,4,12)).fit(t.transform(Dtr),ytr)
                s=StandardScaler().fit(Etr); me=(RidgeCV(alphas=np.logspace(-1,4,12)) if alpha is None else Ridge(alpha=alpha)).fit(s.transform(Etr),ytr)
                pf[te]=(mt.predict(t.transform([D[i] for i in te]))+me.predict(s.transform(F[te])))/2
                pc[te]=(mt.predict(t.transform([C[i] for i in te]))+me.predict(s.transform(Ec[te])))/2
            full.append(spearmanr(pf[R],y[R])[0]); crop.append(spearmanr(pc[R],y[R])[0])
        res['big5'][name]={'full':round(float(np.mean(full)),4),'crop':round(float(np.mean(crop)),4),'crop_by_domain':[round(float(x),3) for x in crop]}
        print(json.dumps({'target':'big5','variant':name,**res['big5'][name]}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
