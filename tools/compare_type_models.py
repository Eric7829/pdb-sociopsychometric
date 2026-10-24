"""Letter-product vs direct 16-type vs blended models for the crowd's modal MBTI type (5-fold CV, database read-only).

  python tools/compare_type_models.py --db data/pdb-top.duckdb

"Letters" multiplies four binary letter probabilities (assumes letters are independent). "Type" is a 16-class multinomial
that learns type base rates and letter interactions directly. "Blend" is a log-linear pool p ~ letters^w * type^(1-w); w is
picked on the out-of-fold log loss (one parameter, so slightly optimistic).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from pdb_ingest import panel
from pdb_ingest.battery import MBTI_TYPES
from pdb_ingest.encoder_text import encoder_view

TYPES=np.array(MBTI_TYPES)

def letters_to_types(P):
    """P: (n,4) probability of each pair's first letter -> (n,16) type probabilities under independence."""
    out=np.ones((len(P),16))
    for j,t in enumerate(TYPES):
        for i,(a,_) in enumerate(panel.LETTERS): out[:,j]*=np.where(t[i]==a,P[:,i],1-P[:,i])
    return out

def pool(A,B,w):
    """Log-linear pool of two (n,16) probability arrays, renormalised."""
    L=w*np.log(np.clip(A,1e-9,1))+(1-w)*np.log(np.clip(B,1e-9,1)); L-=L.max(1,keepdims=True); E=np.exp(L)
    return E/E.sum(1,keepdims=True)

def evaluate(P,y):
    idx=np.array([MBTI_TYPES.index(t) for t in y]); top=P.argmax(1)
    return {'exact':float(np.mean(top==idx)),'top3':float(np.mean([i in np.argsort(-p)[:3] for p,i in zip(P,idx)])),
            'letters':float(np.mean([sum(a==b for a,b in zip(TYPES[k],t))/4 for k,t in zip(top,y)])),
            'log_loss':float(-np.mean(np.log(np.clip(P[np.arange(len(y)),idx],1e-9,1)))),'p_true':float(np.mean(P[np.arange(len(y)),idx]))}

def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); args=p.parse_args()
    store=readonly_store(args.db); rows=panel.load_rows(store.connection)
    cache=Path('data/stage2/emb/intfloat--multilingual-e5-small/main'); docs={}; vecs={}
    for n,r in rows.items():
        if 'mbti_counts' not in r: continue
        v=encoder_view(store,n); sha=hashlib.sha256(json.dumps(v,ensure_ascii=False).encode('utf-8')).hexdigest()
        f=cache/f'{n}_{sha[:16]}.npy'
        if f.exists(): docs[n]=' '.join(t[a:b] for t,rs in v for a,b in rs); vecs[n]=np.load(f)
    ids=sorted(docs); y=np.array([panel.modal(rows[n]['mbti_counts']) for n in ids])
    D=np.array([docs[n] for n in ids],dtype=object); E=np.array([vecs[n] for n in ids])
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True)
    oof={k:np.zeros((len(ids),16)) for k in ('letters_balanced','letters','type')}
    for train,test in KFold(5,shuffle=True,random_state=0).split(ids):
        for name,balanced in (('letters_balanced','balanced'),('letters',None)):
            P=np.zeros((len(test),4))
            for i,(a,_) in enumerate(panel.LETTERS):
                yl=np.array([t[i]==a for t in y])
                mt=make_pipeline(tf(),LogisticRegression(C=4,max_iter=3000,class_weight=balanced)).fit(list(D[train]),yl[train])
                me=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,class_weight=balanced)).fit(E[train],yl[train])
                P[:,i]=(mt.predict_proba(list(D[test]))[:,1]+me.predict_proba(E[test])[:,1])/2
            oof[name][test]=letters_to_types(P)
        # Direct 16-class models; classes absent from a training fold get probability 0 before averaging.
        T=np.zeros((len(test),16))
        for model,X in ((make_pipeline(tf(),LogisticRegression(C=4,max_iter=3000)),D),(make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=3000)),E)):
            Xtr,Xte=(list(X[train]),list(X[test])) if X is D else (X[train],X[test])
            model.fit(Xtr,y[train]); pr=model.predict_proba(Xte); cols=[MBTI_TYPES.index(c) for c in model.classes_]
            T[:,cols]+=pr/2
        oof['type'][test]=(T+1e-4)/(T+1e-4).sum(1,keepdims=True)
    res={'profiles':len(ids),'type_counts':{t:int((y==t).sum()) for t in MBTI_TYPES},
         'commonest_baseline':float(np.mean(y==max(set(y),key=list(y).count)))}
    for k,P in oof.items(): res[k]=evaluate(P,y)
    grid=[round(w,1) for w in np.arange(0,1.01,.1)]
    blends={w:evaluate(pool(oof['letters'],oof['type'],w),y) for w in grid}
    best=min(grid,key=lambda w:blends[w]['log_loss']); res['blend']={'w_letters':best,**blends[best]}
    res['blend_grid']={str(w):{k:round(v,3) for k,v in blends[w].items()} for w in grid}
    Path('data/stage2/type_models.json').write_text(json.dumps(res,indent=1),encoding='utf-8')
    print(json.dumps({k:v for k,v in res.items() if k not in ('blend_grid','type_counts')},indent=1))

if __name__=='__main__': main()
