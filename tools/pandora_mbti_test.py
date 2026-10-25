"""The panel's MBTI cards on PANDORA Reddit users' self-reported types. Read-only; writes aggregates only.

  python tools/pandora_mbti_test.py --db data/pdb-top.duckdb [--limit 30]

Uses the cached panel models (samples top + top2; trains and caches them if missing). Text comes from
data/external/pandora-full/comments_g23.parquet with every comment that names a type (tools/pandora_subreddits.TYPE_TALK)
dropped; Assessor.assess then applies the panel's own scrub. Slices, one document per user with a 4-letter type:
  main:  vent/advice/mental-health/personal-question subreddits (grp 3), comments shuffled, up to 1000 words, >= 150,
  short: introduction subreddits (grp 2), up to 300 words, >= 50,
  short_no_introvert: the same without r/introvert ("I'm an introvert" nearly states E/I).
Per card (crowd, jev, agent): letter balanced accuracy and AUC, 16-type exact and top-3 (headline and prior-neutral),
against chance (balanced accuracy .5; exact = the label base rate's sum of squares under the card's own type mix).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pandora_subreddits import ROOT,TYPE_TALK
from run_stage2 import readonly_store
from pdb_ingest import embed,panel
from pdb_ingest.panel import LETTERS,MBTI

def documents(cap,floor,grp,exclude=()):
    import duckdb
    c=duckdb.connect(); C=str(ROOT/'comments_g23.parquet').replace('\\','/'); A=str(ROOT/'author_profiles.csv').replace('\\','/')
    ex=','.join("'"+s.lower()+"'" for s in exclude) or "''"
    rows=c.execute(f"""select s.author, upper(p.mbti), s.body from read_parquet('{C}') s join read_csv_auto('{A}') p using(author)
        where s.grp={grp} and lower(s.subreddit) not in ({ex}) and length(p.mbti)=4 and p.mbti not ilike '%x%'
        and not regexp_matches(lower(s.body),'{TYPE_TALK}') and s.body not in ('[deleted]','[removed]')""").fetchall()
    by={}
    for a,t,b in rows: by.setdefault((a,t),[]).append(b)
    docs=[]
    for (a,t),bs in sorted(by.items()):
        random.Random(a).shuffle(bs); out=[]; w=0
        for b in bs:
            out.append(b); w+=len(b.split())
            if w>=cap: break
        if w>=floor: docs.append((t,' '.join(' '.join(out).split()[:cap])))
    return docs

def score(results,truth):
    from sklearn.metrics import roc_auc_score
    out={}
    for card in ('crowd','jev','agent'):
        R=[r['mbti'][card] for r in results if r['mbti'].get(card)]
        if not R: continue
        L=np.array([[x['letters'][a+b] for a,b in LETTERS] for x in R]); y=np.array([[t[i]==a for i,(a,_) in enumerate(LETTERS)] for t in truth])
        o={'letters':{}}
        for i,(a,b) in enumerate(LETTERS):
            q=L[:,i]>=.5; o['letters'][a+b]={'balanced_accuracy':round(float(np.mean([np.mean(q[y[:,i]==v]==v) for v in (True,False)])),3),
                                             'auc':round(float(roc_auc_score(y[:,i],L[:,i])),3),'share_'+a:round(float(y[:,i].mean()),3)}
        o['letters_mean_bacc']=round(float(np.mean([v['balanced_accuracy'] for v in o['letters'].values()])),3)
        for k,get in (('headline',lambda x:x),('neutral',lambda x:x.get('neutral'))):
            got=[get(x) for x in R]
            if all(got):
                pred=[g['type'] for g in got]; top=[[z['type'] for z in g['top']] for g in got]; base=Counter(truth); pm=Counter(pred)
                o[k]={'exact':round(float(np.mean([p==t for p,t in zip(pred,truth)])),3),'top3':round(float(np.mean([t in tp for tp,t in zip(top,truth)])),3),
                      'chance_exact':round(sum(base[t]/len(truth)*pm[t]/len(pred) for t in MBTI),3),'predicted_mix':dict(pm.most_common(5))}
        out[card]=o
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--limit',type=int,default=None)
    p.add_argument('--output',type=Path,default=Path('data/stage2/pandora_mbti_test.json'))
    args=p.parse_args(); samples=('top','top2'); store=readonly_store(args.db); encoder=embed.hf_encoder()
    a=panel.cached_assessor(store,encoder,args.db,samples,Path('data/stage2/emb')/embed.model_slug(encoder[2]['model'])/'main')
    res={'models':a.loaded_from or 'trained'}
    for name,cap,floor,grp,ex in (('main',1000,150,3,()),('short',300,50,2,()),('short_no_introvert',300,50,2,('introvert',))):
        docs=documents(cap,floor,grp,ex)
        if args.limit: docs=random.Random(0).sample(docs,min(args.limit,len(docs)))
        out=[];truth=[]
        for t,d in docs:
            r=a.assess(d)
            if 'error' not in r: out.append(r); truth.append(t)
        res[name]={'users':len(truth),'mean_words':round(float(np.mean([r['words'] for r in out])),1),'types':dict(Counter(truth).most_common()),**score(out,truth)}
        print(json.dumps({'slice':name,**{k:v for k,v in res[name].items() if k!='types'}}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
