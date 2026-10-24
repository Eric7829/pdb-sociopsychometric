"""Jev as a teacher for the crowd's modal MBTI type (database read-only).

  python tools/distill_jev.py label    --db data/pdb-top.duckdb   # Jev 16-type probabilities for every eligible profile (cached)
  python tools/distill_jev.py evaluate --db data/pdb-top.duckdb   # 5-fold CV: Jev, ours, tuned ensemble, distilled students

Honesty rules: our models' predictions are out-of-fold; the ensemble weight for each fold is chosen only on the OTHER folds;
students are trained on soft targets (alpha * Jev + (1-alpha) * crowd one-hot) and scored against crowd labels on held-out
folds, so Jev's labels for a test profile never enter its prediction.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_llm_classifier import SYSTEM,tool
from compare_type_models import evaluate,letters_to_types,pool,TYPES
from run_stage2 import readonly_store
from pdb_ingest import panel
from pdb_ingest.battery import MBTI_TYPES
from pdb_ingest.llm import OpenRouter,ToolCallError,load_api_key

MODEL='typesafe/jev-router'
OUT=Path('data/stage2/llm')/MODEL.replace('/','--')

def load(db):
    import hashlib
    from pdb_ingest.encoder_text import encoder_view
    store=readonly_store(db); rows=panel.load_rows(store.connection)
    cache=Path('data/stage2/emb/intfloat--multilingual-e5-small/main'); docs={}; vecs={}
    for n,r in rows.items():
        if 'mbti_counts' not in r: continue
        v=encoder_view(store,n); sha=hashlib.sha256(json.dumps(v,ensure_ascii=False).encode('utf-8')).hexdigest()
        f=cache/f'{n}_{sha[:16]}.npy'
        if f.exists(): docs[n]=' '.join(t[a:b] for t,rs in v for a,b in rs); vecs[n]=np.load(f)
    ids=sorted(docs)
    return ids,docs,vecs,np.array([panel.modal(rows[n]['mbti_counts']) for n in ids])

async def label(ids,docs,max_words=30000,concurrency=4):
    import httpx
    OUT.mkdir(parents=True,exist_ok=True); gate=asyncio.Semaphore(concurrency); stats={'cached':0,'labelled':0,'failed':0,'cost':0.0}
    async with httpx.AsyncClient() as http:
        llm=OpenRouter(http,load_api_key(),MODEL,provider=None)
        async def one(n):
            f=OUT/f'{n}.json'
            if f.exists(): stats['cached']+=1; return
            msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':'<text>\n'+' '.join(docs[n].split()[:max_words])+'\n</text>'}]
            info={}
            for attempt in range(3):
                try:
                    async with gate: r=await llm.call_tool(msgs,tool(),temperature=0,seed=1+attempt)
                except ToolCallError as error:
                    info=error.args[1] if len(error.args)>1 else {}; stats['cost']+=((info or {}).get('usage') or {}).get('cost',0) or 0
                    continue
                probs=r['arguments']['probabilities']; stats['cost']+=(r['usage'] or {}).get('cost',0) or 0; stats['labelled']+=1
                f.write_text(json.dumps({'type':r['arguments']['type'],'probabilities':probs,'usage':r['usage'],'provider':r['provider']}),encoding='utf-8')
                print(json.dumps({'event':'labelled','profile_num':n,'type':r['arguments']['type']}),flush=True)
                return
            stats['failed']+=1   # left uncached so a later run retries it
            print(json.dumps({'event':'failed','profile_num':n,'content':str((info or {}).get('content'))[:200]}),flush=True)
        await asyncio.gather(*(one(n) for n in ids))
    return stats

def jev_matrix(ids):
    P=np.zeros((len(ids),16)); ok=np.zeros(len(ids),bool)
    for i,n in enumerate(ids):
        f=OUT/f'{n}.json'
        if not f.exists(): continue
        probs=json.loads(f.read_text(encoding='utf-8'))['probabilities']
        P[i]=[max(float(probs.get(t,0)),0) for t in MBTI_TYPES]; s=P[i].sum()
        if s>0: P[i]/=s; ok[i]=True
    return np.clip(P,1e-4,1)/np.clip(P,1e-4,1).sum(1,keepdims=True),ok

def soft_fit(model,X,targets):
    """Multinomial fit on soft targets: one row per (profile, type) weighted by that type's target probability."""
    rows=[(i,j,w) for i,t in enumerate(targets) for j,w in enumerate(t) if w>1e-3]
    idx=[i for i,_,_ in rows]; Xs=[X[i] for i in idx] if isinstance(X,list) else X[idx]
    est=model.steps[-1][0]
    return model.fit(Xs,TYPES[[j for _,j,_ in rows]],**{f'{est}__sample_weight':np.array([w for _,_,w in rows])})

def evaluate_all(ids,docs,vecs,y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    J,ok=jev_matrix(ids); keep=np.where(ok)[0]
    ids=[ids[i] for i in keep]; y=y[keep]; J=J[keep]
    D=[docs[n] for n in ids]; E=np.array([vecs[n] for n in ids]); onehot=np.eye(16)[[MBTI_TYPES.index(t) for t in y]]
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True)
    folds=list(KFold(5,shuffle=True,random_state=0).split(ids)); alphas=(0.0,0.5,1.0)
    oof={k:np.zeros((len(ids),16)) for k in ['letters','type']+[f'student_a{a}' for a in alphas]}
    for train,test in folds:
        P=np.zeros((len(test),4))
        for i,(a,_) in enumerate(panel.LETTERS):
            yl=np.array([t[i]==a for t in y])
            mt=make_pipeline(tf(),LogisticRegression(C=4,max_iter=3000)).fit([D[k] for k in train],yl[train])
            me=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000)).fit(E[train],yl[train])
            P[:,i]=(mt.predict_proba([D[k] for k in test])[:,1]+me.predict_proba(E[test])[:,1])/2
        oof['letters'][test]=letters_to_types(P)
        for name,target in [('type',onehot)]+[(f'student_a{a}',a*J+(1-a)*onehot) for a in alphas]:
            if name=='student_a0.0': oof[name][test]=oof['type'][test]; continue   # alpha 0 is the crowd-only type model
            T=np.zeros((len(test),16))
            for model,X in ((make_pipeline(tf(),LogisticRegression(C=4,max_iter=3000)),D),(make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=3000)),E)):
                Xtr=[X[k] for k in train] if X is D else X[train]; Xte=[X[k] for k in test] if X is D else X[test]
                soft_fit(model,Xtr,target[train]); pr=model.predict_proba(Xte); cols=[MBTI_TYPES.index(c) for c in model.classes_]
                T[:,cols]+=pr/2
            oof[name][test]=(T+1e-4)/(T+1e-4).sum(1,keepdims=True)
    ours=pool(oof['letters'],oof['type'],0.4)
    # Ensemble with Jev: per fold, weight chosen on the other folds only.
    grid=np.round(np.arange(0,1.01,.1),1); ens=np.zeros_like(ours); chosen=[]
    for train,test in folds:
        w=min(grid,key=lambda w:evaluate(pool(J[train],ours[train],w),y[train])['log_loss']); chosen.append(float(w))
        ens[test]=pool(J[test],ours[test],w)
    res={'profiles':len(ids),'commonest_baseline':float(np.mean(y==max(set(y),key=list(y).count))),
         'jev':evaluate(J,y),'ours_letters':evaluate(oof['letters'],y),'ours_type':evaluate(oof['type'],y),
         'ours_blend':evaluate(ours,y),'ensemble_jev+ours':{'w_jev_per_fold':chosen,**evaluate(ens,y)}}
    for a in alphas[1:]:
        s=oof[f'student_a{a}']; res[f'student_alpha{a}']=evaluate(s,y)
        res[f'student_alpha{a}+letters']=evaluate(pool(oof['letters'],s,0.4),y)
        res[f'student_alpha{a}_agrees_with_jev_top1']=float(np.mean(s.argmax(1)==J.argmax(1)))
    return res

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd',choices=('label','evaluate')); p.add_argument('--db',type=Path,required=True)
    args=p.parse_args(); ids,docs,vecs,y=load(args.db)
    if args.cmd=='label': print(json.dumps(asyncio.run(label(ids,docs))))
    else:
        res=evaluate_all(ids,docs,vecs,y); Path('data/stage2/jev_distill.json').write_text(json.dumps(res,indent=1),encoding='utf-8')
        print(json.dumps(res,indent=1))

if __name__=='__main__': main()
