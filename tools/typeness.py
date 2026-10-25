"""How type-like are MBTI (16) and the Enneagram (9 cores, 27 subtypes) when Big Five is the ground? Read-only.

  python tools/typeness.py --db data/pdb-top.duckdb

Same profiles for every system: crowd MBTI type, Enneagram core and dominant instinct, and the Stage 3 agent's 30 IPIP
facets (probit of percentiles). Also repeated on profiles whose text never mentions typology terms.
Per system, on 30 facets and on the 5 domains:
  info:  share of the label's entropy that Big Five explains, 1 - CV cross-entropy / H(label), from a multinomial
         logistic regression (5-fold, 5 repeats). Comparable across different numbers of classes.
  lift:  CV balanced accuracy / chance.
  eta2:  between-type share of total Big Five variance (trace ratio): how much of the space the labels carve up.
  silhouette: mean silhouette of the labels in facet space; > 0 means types form separated clumps, ~0 means they
         overlap like slices of one continuous cloud.
  null:  the same statistics with labels shuffled (keeps class sizes), so each number has a floor.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from subtype_model import FACETS,probit
from so5_big5 import first_instinct
from run_stage2 import readonly_store
from pdb_ingest import panel
from pdb_ingest.battery import corpus_text
from pdb_ingest.panel import modal

TALK=re.compile(r'\b[ie][ns][tf][jp]\b|mbti|ennea|\b[1-9]w[1-9]\b|\btype [1-9]\b|tritype|\b(sp|so|sx)[ /-]?(sp|so|sx|[1-9])\b|socionics',re.I)

def stats(X,y,seed=0,reps=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score,log_loss,silhouette_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    classes,cnt=np.unique(y,return_counts=True); p=cnt/cnt.sum(); H=-(p*np.log(p)).sum()
    ce=[];ba=[]
    for r in range(reps):
        P=np.zeros((len(y),len(classes))); pred=np.empty(len(y),dtype=object)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed+r).split(X,y):
            s=StandardScaler().fit(X[tr]); m=LogisticRegression(C=.05,max_iter=5000).fit(s.transform(X[tr]),y[tr])
            P[te][:, :]=0; Q=np.zeros((len(te),len(classes))); Q[:,np.searchsorted(classes,m.classes_)]=m.predict_proba(s.transform(X[te])); P[te]=Q
            mb=LogisticRegression(C=.05,max_iter=5000,class_weight='balanced').fit(s.transform(X[tr]),y[tr]); pred[te]=mb.predict(s.transform(X[te]))
        ce.append(log_loss(y,P,labels=classes)); ba.append(balanced_accuracy_score(y,pred))
    Z=(X-X.mean(0))/X.std(0); tot=(Z**2).sum(); bw=sum((y==c).sum()*(Z[y==c].mean(0)**2).sum() for c in classes)
    return {'classes':len(classes),'entropy_bits':round(float(H/np.log(2)),2),'info':round(float(1-np.mean(ce)/H),3),
            'lift':round(float(np.mean(ba)*len(classes)),2),'eta2':round(float(bw/tot),3),'silhouette':round(float(silhouette_score(Z,y)),3)}

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/typeness.json'))
    args=p.parse_args(); store=readonly_store(args.db); c=store.connection; rows=panel.load_rows(c,('top','top2','es')); inst={}; fac={}
    for n,system,counts,eligible in c.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible FROM vote_distributions').fetchall():
        if system=='INSTINCT' and eligible and counts: inst[n]=first_instinct(json.loads(counts))
    for n,f in c.execute("SELECT r.profile_num,e.ipip_facets_json FROM agent_runs r JOIN psychometric_evaluations e ON e.run_id=r.run_id WHERE r.complete AND r.prompt_version='stage3-v1' ORDER BY r.seed").fetchall():
        if n not in fac and f: fac[n]={k:v['percentile'] for k,v in json.loads(f).items()}
    P=[]
    for n,r in rows.items():
        e=modal(r.get('ennea_counts') or {}); t=modal(r['mbti_counts']) if 'mbti_counts' in r else None
        if e and e[0] in '123456789' and t and inst.get(n) in ('sp','so','sx') and n in fac:
            P.append((n,t,e[0],e[0]+inst[n],probit([fac[n][k] for k in FACETS]),bool(TALK.search(corpus_text(store.load_snapshot(n).segments,'raw')))))
    res={}
    for subset,keep in (('all',lambda q:True),('no_typology_talk',lambda q:not q[5])):
        S=[q for q in P if keep(q)]; X=np.array([q[4] for q in S]); D=np.column_stack([X[:,i*6:(i+1)*6].mean(1) for i in range(5)])
        out={'n':len(S)}
        for name,j in (('MBTI 16',1),('Enneagram 9',2),('Enneagram 27',3)):
            y=np.array([q[j] for q in S]); rng=np.random.default_rng(0)
            out[name]={'facets':stats(X,y),'domains':stats(D,y),'null_facets':stats(X,rng.permutation(y),reps=2)}
            print(json.dumps({'subset':subset,'system':name,**out[name]}),flush=True)
        res[subset]=out
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
