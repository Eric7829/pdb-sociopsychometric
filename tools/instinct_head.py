"""Dominant instinct (sp/so/sx) from text, learned within each Enneagram core. Database read-only; no API calls.

  python tools/instinct_head.py --db data/pdb-top.duckdb

Instinct changes qualitatively by core (the countertypes: SX6 is the least agreeable 6, SX7 the most agreeable 7), so
one global head would average opposite directions away. Text features: TF-IDF reduced to 300 SVD components plus the
profile's gte-modernbert vector, standardized. Heads (3-class logistic regression, given the crowd's core):
  global:      one head for all cores,
  per-core:    a separate head per core, no sharing,
  pooled lam:  feature augmentation [x, lam * x * onehot(core)] (Daume 2007): the effective weights are
               w_global + lam * d_core, and the L2 penalty shrinks each core's deviation d_core towards zero, so a core
               keeps its own direction only where its data supports it. Larger lam = less pooling.
Scored by 5-fold CV (3 repeats, stratified by subtype) on the crowd's dominant instinct, and out of fold on the
profile creator's own instinct vote. Baselines: chance (33%) and each core's majority instinct (from training folds).
Reported overall, per core, and as recall on the usual countertypes (SX1 SP2 SP3 SP4 SX5 SX6 SO7 SO8 SO9).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from type_ablation import load
from subtype_model import COUNTER,self_labels
from so5_big5 import first_instinct
from run_stage2 import readonly_store
from pdb_ingest.panel import l2,modal

CORES='123456789'; INST=['sp','so','sx']

def features(tr,te,docs,V):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    tf=TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True); svd=TruncatedSVD(300,random_state=0)
    A=svd.fit_transform(tf.fit_transform([docs[i] for i in tr])); B=svd.transform(tf.transform([docs[i] for i in te]))
    s=StandardScaler().fit(np.hstack([A,V[tr]]))
    return s.transform(np.hstack([A,V[tr]])),s.transform(np.hstack([B,V[te]]))

def augment(X,core,lam):
    if lam is None: return X
    blocks=[X]+[lam*X*(core==c)[:,None] for c in CORES]
    return np.hstack(blocks)

def fit_predict(kind,Xtr,ytr,ctr,Xte,cte,C):
    from sklearn.linear_model import LogisticRegression
    P=np.zeros((len(Xte),3))
    if kind=='per-core':
        for c in CORES:
            m=ctr==c; t=cte==c
            if not t.any(): continue
            if len(set(ytr[m]))<2: P[t,INST.index(Counter(ytr[m]).most_common(1)[0][0])]=1; continue
            lr=LogisticRegression(C=C,max_iter=3000).fit(Xtr[m],ytr[m]); P[np.ix_(t,[INST.index(k) for k in lr.classes_])]=lr.predict_proba(Xte[t])
        return P
    lam=None if kind=='global' else float(kind.split('=')[1])
    lr=LogisticRegression(C=C,max_iter=5000).fit(augment(Xtr,ctr,lam),ytr)
    P[:,[INST.index(k) for k in lr.classes_]]=lr.predict_proba(augment(Xte,cte,lam)); return P

def main():
    from sklearn.model_selection import StratifiedKFold
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--C',type=float,default=.02)
    p.add_argument('--output',type=Path,default=Path('data/stage2/instinct_head.json'))
    args=p.parse_args(); store=readonly_store(args.db); c=store.connection
    data=load(args.db,('top','top2','es')); ennea={}; inst={}
    for n,system,counts,eligible in c.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible FROM vote_distributions').fetchall():
        if n in data and eligible and counts:
            if system=='ENNEAGRAM': ennea[n]=modal(json.loads(counts))
            if system=='INSTINCT': inst[n]=first_instinct(json.loads(counts))
    sv=self_labels(c)
    ids=sorted(n for n in data if ennea.get(n) and ennea[n][0] in CORES and inst.get(n) in INST)
    docs=[data[n]['doc'] for n in ids]; V=l2(np,[data[n]['vec'] for n in ids])
    core=np.array([ennea[n][0] for n in ids]); y=np.array([inst[n] for n in ids]); sub=np.array([a+b for a,b in zip(core,y)])
    selfi=np.array([(sv.get(n,(None,None))[1] or '') for n in ids])
    kinds=['global','per-core','pooled lam=0.5','pooled lam=1','pooled lam=2']
    res={'profiles':len(ids),'self_labelled':int((selfi!='').sum()),'C':args.C,'cells':dict(sorted(Counter(sub).items()))}
    acc={k:{'crowd':[],'balanced':[],'self':[],'counter':[],'per_core':{cc:[] for cc in CORES}} for k in kinds+['majority per core']}
    for r in range(3):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(ids,sub):
            Xtr,Xte=features(tr,te,docs,V)
            preds={k:np.array(INST)[fit_predict(k,Xtr,y[tr],core[tr],Xte,core[te],args.C).argmax(1)] for k in kinds}
            maj={cc:Counter(y[tr][core[tr]==cc]).most_common(1)[0][0] for cc in CORES if (core[tr]==cc).any()}
            preds['majority per core']=np.array([maj.get(cc,'sp') for cc in core[te]])
            for k,pr in preds.items():
                yt=y[te]; a=acc[k]; a['crowd'].append(np.mean(pr==yt)); a['balanced'].append(np.mean([np.mean(pr[yt==v]==v) for v in INST if (yt==v).any()]))
                m=selfi[te]!=''; a['self'].append(np.mean(pr[m]==selfi[te][m]))
                ct=np.isin(sub[te],COUNTER); a['counter'].append(np.mean(pr[ct]==yt[ct]))
                for cc in CORES:
                    mm=core[te]==cc
                    if mm.any(): a['per_core'][cc].append(np.mean(pr[mm]==yt[mm]))
    for k,a in acc.items():
        res[k]={'crowd':round(float(np.mean(a['crowd'])),3),'balanced':round(float(np.mean(a['balanced'])),3),'self':round(float(np.mean(a['self'])),3),
                'countertypes':round(float(np.mean(a['counter'])),3),'per_core':{cc:round(float(np.mean(v)),2) for cc,v in a['per_core'].items() if v}}
        print(json.dumps({k:res[k]}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
