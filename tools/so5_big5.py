"""Do social 5s (crowd Enneagram core 5 with social instinct first) have a distinct Big Five profile? Database read-only.

  python tools/so5_big5.py --db data/pdb-top.duckdb

Groups use crowd votes only (training-eligible ENNEAGRAM and INSTINCT systems): the dominant instinct is the one with the
most first-position votes over stackings (so/sx) and subtype codes (SO5). Two Big Five sources:
  agent: the DeepSeek agent's IPIP-NEO-120 percentiles (prompt A),
  crowd: PDB's SLOAN votes as letter shares (S=E, L=N, O=C, A=A, I=O), profiles with >= 10 SLOAN votes.
For each source: group means; Cohen's d and Mann-Whitney p (Holm over 5 domains) for SO5 vs SP5+SX5 and SO5 vs non-5
social-firsts; and whether the five domains separate SO5 from other 5s at all (logistic regression, repeated
stratified 5-fold AUC, with a 1000-shuffle permutation p). Only aggregates are printed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from pdb_ingest.panel import DOMAINS,load_rows,modal

SLOAN={'E':(0,'S'),'N':(1,'L'),'C':(2,'O'),'A':(3,'A'),'O':(4,'I')}

def first_instinct(counts):
    c=Counter()
    for k,v in counts.items(): c[k[:2].lower()]+=v
    c={k:v for k,v in c.items() if k in ('sp','so','sx')}
    return max(c,key=c.get) if c and sum(c.values()) else None

def holm(ps):
    order=np.argsort(ps); out=np.empty(len(ps)); run=0
    for r,i in enumerate(order): run=max(run,min(1,(len(ps)-r)*ps[i])); out[i]=run
    return out

def cohen_d(a,b): return (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))

def separability(X,y,reps=20,perms=1000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    def auc(y,rs):
        scores=[]
        for r in range(rs):
            p=np.zeros(len(y))
            for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(X,y):
                s=StandardScaler().fit(X[tr]); p[te]=LogisticRegression(C=1).fit(s.transform(X[tr]),y[tr]).predict_proba(s.transform(X[te]))[:,1]
            scores.append(roc_auc_score(y,p))
        return float(np.mean(scores))
    real=auc(y,reps); rng=np.random.default_rng(0); null=[auc(rng.permutation(y),1) for _ in range(perms)]
    return round(real,3),round((1+sum(n>=real for n in null))/(1+perms),4)

def main():
    from scipy.stats import mannwhitneyu
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/so5_big5.json'))
    args=p.parse_args(); conn=readonly_store(args.db).connection
    rows=load_rows(conn,('top','top2','es')); inst={}; sloan={}
    for n,system,counts,eligible,total in conn.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible,total_system_votes FROM vote_distributions').fetchall():
        if n not in rows or not counts: continue
        c=json.loads(counts)
        if system=='INSTINCT' and eligible: inst[n]=first_instinct(c)
        if system=='BIG5_SLOAN' and (total or 0)>=10:
            t=sum(c.values()); sloan[n]={k:sum(v for code,v in c.items() if len(code)==5 and code[i]==ch)/t for k,(i,ch) in SLOAN.items()}
    group={}
    for n,r in rows.items():
        core=modal(r.get('ennea_counts') or {}); i=inst.get(n)
        if core and i: group[n]=(core[0],i)
    res={'n_groups':{f'{c}{i}':v for (c,i),v in sorted(Counter(group.values()).items())}}
    for source,score in (('agent',{n:r['pct'] for n,r in rows.items() if 'pct' in r}),('crowd',sloan)):
        ids=[n for n in group if n in score]; X=np.array([[score[n][k] for k in DOMAINS] for n in ids],dtype=float)
        core=np.array([group[n][0] for n in ids]); ins=np.array([group[n][1] for n in ids])
        so5=(core=='5')&(ins=='so'); other5=(core=='5')&(ins!='so'); so_non5=(core!='5')&(ins=='so')
        out={'n':{'SO5':int(so5.sum()),'SP5+SX5':int(other5.sum()),'SO non-5':int(so_non5.sum()),'all':len(ids)},
             'means':{g:dict(zip(DOMAINS,np.round(X[m].mean(0),3).tolist())) for g,m in (('SO5',so5),('SP5',(core=='5')&(ins=='sp')),('SX5',(core=='5')&(ins=='sx')),('SO non-5',so_non5),('all',np.ones(len(ids),bool)))}}
        for name,ref in (('SO5 vs SP5+SX5',other5),('SO5 vs SO non-5',so_non5)):
            ps=[mannwhitneyu(X[so5,j],X[ref,j]).pvalue for j in range(5)]
            out[name]={k:{'d':round(float(cohen_d(X[so5,j],X[ref,j])),2),'p_holm':round(float(q),4)} for j,(k,q) in enumerate(zip(DOMAINS,holm(np.array(ps))))}
        m=(core=='5'); auc,pp=separability(X[m],so5[m]); out['separability_within_5s']={'auc':auc,'perm_p':pp}
        res[source]=out; print(json.dumps({'source':source,**out}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
