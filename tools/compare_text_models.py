"""Compare text representations on the same profiles and folds (database opened read-only).

  python tools/compare_text_models.py --db data/pdb-top.duckdb [--view main]

Targets: the agent's IPIP domain raw scores (prompt stage3-v1; Spearman r of out-of-fold predictions) and each letter of the
crowd's modal MBTI type (balanced accuracy). Features: TF-IDF over pooled text, every encoder cached under data/stage2/emb,
and stacks that average a pair's z-scored out-of-fold predictions (or class probabilities), so nothing is fitted on test folds.
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
from pdb_ingest import distill
from pdb_ingest.encoder_text import encoder_view

LETTERS=[('E','I'),('S','N'),('T','F'),('J','P')]

def expand_matryoshka(encoders,dims):
    """Add '<encoder>@k' variants (leading k dims, re-normalised) for Matryoshka-trained encoders."""
    from pdb_ingest import embed
    out=dict(encoders)
    for name,vecs in encoders.items():
        if name.replace('--','/',1) not in embed.MATRYOSHKA_DIMS: continue
        full=len(next(iter(vecs.values())))
        for k in dims:
            if k<full: out[f'{name}@{k}']={n:np.asarray(embed.matryoshka(v,k)) for n,v in vecs.items()}
    return out

def zscore(x):
    x=np.asarray(x,dtype=float); sd=x.std()
    return (x-x.mean())/sd if sd else x*0

def stack(predictions):
    """Average of z-scored out-of-fold predictions; no parameters are fitted."""
    return np.mean([zscore(p) for p in predictions],axis=0)

def load(store,view,samples=('top',)):
    c=store.connection; ipip={}
    for n,d in c.execute('''SELECT r.profile_num,e.ipip_domains_json FROM agent_runs r JOIN psychometric_evaluations e ON e.run_id=r.run_id
                            LEFT JOIN crawl_queue q ON q.profile_num=r.profile_num WHERE r.complete AND r.prompt_version='stage3-v1'
                            AND coalesce(q.sample,'top') IN ({}) ORDER BY r.seed'''.format(','.join('?'*len(samples))),list(samples)).fetchall():
        ipip.setdefault(n,{k:v['raw'] for k,v in json.loads(d).items()})
    targets,_=distill.load_targets(c,samples=samples); classes=distill.class_lists()['MBTI']
    modal={n:classes[int(np.argmax(t['MBTI'][0]))] for n,t in targets.items() if 'MBTI' in t}
    docs={}; shas={}
    for n in sorted(set(ipip)|set(modal)):
        v=encoder_view(store,n,**distill.VIEWS[view])
        docs[n]=' '.join(t[a:b] for t,r in v for a,b in r)
        shas[n]=hashlib.sha256(json.dumps(v,ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
    encoders={}
    for d in sorted(Path('data/stage2/emb').glob(f'*/{view}')):
        vecs={n:np.load(f).astype(float) for n in shas if (f:=d/f'{n}_{shas[n]}.npy').exists()}
        if vecs: encoders[d.parent.name]=vecs
    return ipip,modal,docs,encoders

def main():
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression,RidgeCV
    from sklearn.model_selection import KFold,cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--view',default='main',choices=sorted(distill.VIEWS))
    p.add_argument('--output',type=Path,default=Path('data/stage2/compare.json'))
    p.add_argument('--samples',default='top',help='Comma-separated crawl samples, e.g. top,top2')
    p.add_argument('--matryoshka',default='',help='Comma-separated leading-dimension sizes to also test for MRL encoders, e.g. 64,128,256')
    args=p.parse_args()
    ipip,modal,docs,encoders=load(readonly_store(args.db),args.view,tuple(args.samples.split(',')))
    # Encoders cached for only part of these profiles (another sample, or an older scrubber) would shrink the shared set; drop them.
    need=set(ipip)|set(modal); encoders={k:v for k,v in encoders.items() if len(need&set(v))>=.9*len(need)}
    if args.matryoshka: encoders=expand_matryoshka(encoders,[int(x) for x in args.matryoshka.split(',')])
    cv=KFold(5,shuffle=True,random_state=0); alphas=np.logspace(-1,4,12); result={'view':args.view,'encoders':list(encoders)}
    def features(ids):
        f={'tfidf':(TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True).fit_transform([docs[n] for n in ids]),False)}
        for name,vecs in encoders.items(): f[name]=(np.array([vecs[n] for n in ids]),True)
        return f
    ids=sorted(n for n in ipip if all(n in v for v in encoders.values())); f=features(ids); result['ipip_profiles']=len(ids)
    result['ipip_spearman']={}
    for k in 'OCEAN':
        y=np.array([ipip[n][k] for n in ids]); preds={}
        for name,(X,dense) in f.items():
            model=make_pipeline(StandardScaler(),RidgeCV(alphas=alphas)) if dense else RidgeCV(alphas=alphas)
            preds[name]=cross_val_predict(model,X,y,cv=cv)
        for name in encoders: preds[f'tfidf+{name}']=stack([preds['tfidf'],preds[name]])
        result['ipip_spearman'][k]={name:round(float(spearmanr(pr,y)[0]),3) for name,pr in preds.items()}
    ids=sorted(n for n in modal if all(n in v for v in encoders.values())); f=features(ids); result['letter_profiles']=len(ids)
    result['letter_balanced_accuracy']={}
    for i,(a,b) in enumerate(LETTERS):
        y=np.array([modal[n][i]==a for n in ids]); probs={}
        for name,(X,dense) in f.items():
            clf=LogisticRegression(C=1 if dense else 4,max_iter=3000,class_weight='balanced')
            probs[name]=cross_val_predict(make_pipeline(StandardScaler(),clf) if dense else clf,X,y,cv=cv,method='predict_proba')[:,1]
        for name in encoders: probs[f'tfidf+{name}']=(probs['tfidf']+probs[name])/2
        bal=lambda pr:float(np.mean([np.mean((pr>=.5)[y==v]==v) for v in (True,False)]))
        result['letter_balanced_accuracy'][f'{a}/{b}']={name:round(bal(pr),3) for name,pr in probs.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=1),encoding='utf-8')
    print(json.dumps(result,indent=1))

if __name__=='__main__': main()
