"""Memorisation check: does Jev's accuracy on the crowd's modal MBTI type fall on profiles created after its release?

  python tools/jev_recency.py --top-db data/pdb-top.duckdb --recent-db data/pdb-recent.duckdb [--no-llm]

Control: our letter+type blend is trained on Top profiles only and applied to the recent ones (it cannot have memorised
them), and both models are also scored on Top profiles with vote counts matched to the recent sample (fewer votes means a
noisier crowd label for everyone). A memorisation signal is Jev's drop clearly exceeding ours.
"""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_llm_classifier import SYSTEM,tool
from compare_type_models import letters_to_types,pool,TYPES
from run_stage2 import readonly_store
from pdb_ingest import embed,panel
from pdb_ingest.battery import MBTI_TYPES
from pdb_ingest.encoder_text import encoder_view
from pdb_ingest.llm import OpenRouter,ToolCallError,load_api_key

MODEL='typesafe/jev-router'

def wilson(k,n,z=1.96):
    if not n: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(c-h,3),round(c+h,3))

def acc(hits):
    hits=list(hits); k=int(sum(hits)); return {'n':len(hits),'exact':round(k/len(hits),3) if hits else None,'ci95':wilson(k,len(hits))}

def corpus(db,encoder,cache):
    """profile -> (doc, e5 vector, crowd modal type, MBTI votes) for profiles with an eligible crowd MBTI distribution."""
    store=readonly_store(db); rows=panel.load_rows(store.connection); out={}

    votes=dict(store.connection.execute("SELECT profile_num,total_system_votes FROM vote_distributions WHERE system_name='MBTI'").fetchall())
    for n,r in rows.items():
        if 'mbti_counts' not in r: continue
        v=encoder_view(store,n); sha=hashlib.sha256(json.dumps(v,ensure_ascii=False).encode('utf-8')).hexdigest()
        vec=embed.cached_vector(cache,n,sha,lambda:embed.profile_vector(embed.embed_views(v,encoder)))
        doc=' '.join(t[a:b] for t,rs in v for a,b in rs)
        if vec is not None and doc.strip(): out[n]=(doc,np.asarray(vec),panel.modal(r['mbti_counts']),int(votes.get(n) or 0))
    return out

async def jev_labels(data,out_dir,max_words=30000):
    import httpx
    out_dir.mkdir(parents=True,exist_ok=True); gate=asyncio.Semaphore(4); preds={}; cost=[0.0]
    async with httpx.AsyncClient() as http:
        llm=OpenRouter(http,load_api_key(),MODEL,provider=None)
        async def one(n):
            f=out_dir/f'{n}.json'
            if not f.exists():
                msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':'<text>\n'+' '.join(data[n][0].split()[:max_words])+'\n</text>'}]
                for attempt in range(3):
                    try:
                        async with gate: r=await llm.call_tool(msgs,tool(),temperature=0,seed=1+attempt)
                    except ToolCallError: continue
                    cost[0]+=(r['usage'] or {}).get('cost',0) or 0
                    f.write_text(json.dumps({'type':r['arguments']['type'],'probabilities':r['arguments']['probabilities'],'usage':r['usage']}),encoding='utf-8')
                    break
                else: return
            probs=json.loads(f.read_text(encoding='utf-8'))['probabilities']; preds[n]=max(probs,key=lambda t:float(probs.get(t,0)))
        await asyncio.gather(*(one(n) for n in data))
    return preds,cost[0]

def train_ours(top):
    """Letter + type blend (as in compare_type_models) fitted on every Top profile; returns predict(docs, vectors) -> types."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ids=sorted(top); D=[top[n][0] for n in ids]; E=np.array([top[n][1] for n in ids]); y=np.array([top[n][2] for n in ids])
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True)
    letters=[]
    for i,(a,_) in enumerate(panel.LETTERS):
        yl=np.array([t[i]==a for t in y])
        if len(set(yl))<2: letters.append(float(yl[0])); continue   # one class in this training set: constant probability
        letters.append((make_pipeline(tf(),LogisticRegression(C=4,max_iter=3000)).fit(D,yl),make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000)).fit(E,yl)))
    types=(make_pipeline(tf(),LogisticRegression(C=4,max_iter=3000)).fit(D,y),make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=3000)).fit(E,y))
    def predict(docs,vecs):
        P=np.column_stack([np.full(len(docs),m) if isinstance(m,float) else (m[0].predict_proba(docs)[:,1]+m[1].predict_proba(vecs)[:,1])/2 for m in letters])
        T=np.zeros((len(docs),16))
        for m,X in zip(types,(docs,vecs)):
            T[:,[MBTI_TYPES.index(c) for c in m.classes_]]+=m.predict_proba(X)/2
        return TYPES[pool(letters_to_types(P),(T+1e-4)/(T+1e-4).sum(1,keepdims=True),.4).argmax(1)]
    return predict

def ours_oof_top(top,folds=5):
    """Out-of-fold predictions on Top, so our Top accuracy is not measured on training data."""
    from sklearn.model_selection import KFold
    ids=sorted(top); pred={}
    for train,test in KFold(folds,shuffle=True,random_state=0).split(ids):
        f=train_ours({ids[i]:top[ids[i]] for i in train})
        out=f([top[ids[i]][0] for i in test],np.array([top[ids[i]][1] for i in test]))
        pred.update({ids[i]:p for i,p in zip(test,out)})
    return pred

def main():
    from scipy.stats import fisher_exact
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--top-db',type=Path,required=True); p.add_argument('--recent-db',type=Path,required=True)
    p.add_argument('--no-llm',action='store_true',help='Skip Jev calls (smoke test); uses cached labels only')
    p.add_argument('--output',type=Path,default=Path('data/stage2/jev_recency.json'))
    args=p.parse_args()
    encoder=embed.hf_encoder(); cache=Path('data/stage2/emb')/embed.model_slug(encoder[2]['model'])/'main'
    top=corpus(args.top_db,encoder,cache); recent={n:v for n,v in corpus(args.recent_db,encoder,cache).items() if n not in top}
    top_jev={}
    for n in top:
        f=Path('data/stage2/llm')/MODEL.replace('/','--')/f'{n}.json'
        if f.exists():
            probs=json.loads(f.read_text(encoding='utf-8'))['probabilities']; top_jev[n]=max(probs,key=lambda t:float(probs.get(t,0)))
    recent_dir=Path('data/stage2/llm-recent')/MODEL.replace('/','--')
    if args.no_llm:
        recent_jev={}
        for n in recent:
            f=recent_dir/f'{n}.json'
            if f.exists(): probs=json.loads(f.read_text(encoding='utf-8'))['probabilities']; recent_jev[n]=max(probs,key=lambda t:float(probs.get(t,0)))
        cost=0.0
    else: recent_jev,cost=asyncio.run(jev_labels(recent,recent_dir))
    ours_top=ours_oof_top(top); predict=train_ours(top)
    rids=sorted(recent); ours_recent=dict(zip(rids,predict([recent[n][0] for n in rids],np.array([recent[n][1] for n in rids])))) if rids else {}
    # Vote-matched Top subset: Top profiles inside the recent sample's 10th-90th percentile vote range.
    rv=sorted(recent[n][3] for n in recent)
    lo,hi=(rv[int(.1*(len(rv)-1))],rv[int(.9*(len(rv)-1))]) if rv else (0,0)
    matched=[n for n in top if lo<=top[n][3]<=hi]
    both=lambda a,b,ids:[a[n]==b[n][2] for n in ids if n in a]
    res={'recent_profiles':len(recent),'top_profiles':len(top),'recent_votes_p10_p90':[lo,hi],'vote_matched_top':len(matched),
         'jev_cost_usd':round(cost,4),
         'jev':{'top_all':acc(both(top_jev,top,top)),'top_vote_matched':acc(both(top_jev,top,matched)),'recent':acc(both(recent_jev,recent,recent))},
         'ours':{'top_all':acc(both(ours_top,top,top)),'top_vote_matched':acc(both(ours_top,top,matched)),'recent':acc(both(ours_recent,recent,recent))}}
    for who in ('jev','ours'):
        a,b=res[who]['top_vote_matched'],res[who]['recent']
        if a['n'] and b['n']:
            ka,kb=round(a['exact']*a['n']),round(b['exact']*b['n'])
            res[who]['drop_matched_to_recent']=round(a['exact']-b['exact'],3)
            res[who]['fisher_p']=float(fisher_exact([[ka,a['n']-ka],[kb,b['n']-kb]])[1])
    if 'drop_matched_to_recent' in res['jev'] and 'drop_matched_to_recent' in res['ours']:
        res['memorisation_signal']=round(res['jev']['drop_matched_to_recent']-res['ours']['drop_matched_to_recent'],3)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')
    print(json.dumps(res,indent=1))

if __name__=='__main__': main()
