"""27 Enneagram subtypes (core x dominant instinct) from 30 IPIP-NEO facet percentiles. Database read-only.

  python tools/subtype_model.py --db data/pdb-top.duckdb [--people data/personal/ipip300.json]

Training data: PDB profiles with a crowd core and dominant instinct (tools/ennea_report) and the Stage 3 agent's facet
percentiles. Percentiles become normal scores (probit, clipped to 0.5-99.5), so the 0th and 100th percentiles of a
self-report stay finite.
Model: Gaussian classes with one shared covariance (Ledoit-Wolf shrinkage, from residuals around each subtype's
mean). Each subtype keeps its own mean, partially pooled towards core mean + the pooled instinct offset:
  mu_s = w * cell mean + (1 - w) * (core mean + instinct offset),  w = n_s / (n_s + k).
k = 0 is a free mean per subtype, k = infinity is purely additive (instinct shifts every core the same way). k is
chosen by cross-validated log loss. Baselines: independent core and instinct classifiers multiplied together.
Reported: 5-fold CV (10 repeats) top-1 / top-3 over 27 subtypes, core top-1, log loss, and top-3 recall of the usual
countertypes (SX1 SP2 SP3 SP4 SX5 SX6 SO7 SO8 SO9). With --people, each person's five likeliest subtypes under PDB
frequencies and under a flat prior, core and instinct totals, and the facets that most separate their top subtype
from the runner-up. Nothing about a person is written to the repo.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ennea_report import CORES,INST
from run_stage2 import readonly_store
from so5_big5 import first_instinct
from pdb_ingest import panel
from pdb_ingest.panel import modal

FACETS=[d+str(i) for d in 'NEOAC' for i in range(1,7)]
NAMES={'N1':'anxiety','N2':'anger','N3':'depression','N4':'self-consciousness','N5':'immoderation','N6':'vulnerability',
       'E1':'friendliness','E2':'gregariousness','E3':'assertiveness','E4':'activity','E5':'excitement-seeking','E6':'cheerfulness',
       'O1':'imagination','O2':'artistic interests','O3':'emotionality','O4':'adventurousness','O5':'intellect','O6':'liberalism',
       'A1':'trust','A2':'morality','A3':'altruism','A4':'cooperation','A5':'modesty','A6':'sympathy',
       'C1':'self-efficacy','C2':'orderliness','C3':'dutifulness','C4':'achievement-striving','C5':'self-discipline','C6':'cautiousness'}
SUBTYPES=[c+i for c in CORES for i in INST]
COUNTER=['1sx','2sp','3sp','4sp','5sx','6sx','7so','8so','9so']

def probit(p):
    from scipy.stats import norm
    return norm.ppf(np.clip(np.asarray(p,float),.5,99.5)/100)

def load(db):
    store=readonly_store(db); c=store.connection; rows=panel.load_rows(c,('top','top2','es')); inst={}; fac={}
    for n,system,counts,eligible in c.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible FROM vote_distributions').fetchall():
        if system=='INSTINCT' and eligible and counts: inst[n]=first_instinct(json.loads(counts))
    for n,f in c.execute("SELECT r.profile_num,e.ipip_facets_json FROM agent_runs r JOIN psychometric_evaluations e ON e.run_id=r.run_id WHERE r.complete AND r.prompt_version='stage3-v1' ORDER BY r.seed").fetchall():
        if n not in fac and f: fac[n]={k:v['percentile'] for k,v in json.loads(f).items()}
    X=[];y=[]
    for n,r in rows.items():
        e=modal(r.get('ennea_counts') or {})
        if e and e[0] in CORES and inst.get(n) in INST and n in fac: X.append(probit([fac[n][k] for k in FACETS])); y.append(e[0]+inst[n])
    return np.array(X),np.array(y)

class Model:
    def __init__(self,k): self.k=k
    def fit(self,X,y):
        from sklearn.covariance import LedoitWolf
        core=np.array([s[0] for s in y]); ins=np.array([s[1:] for s in y]); g=X.mean(0)
        cm={c:X[core==c].mean(0) if (core==c).any() else g for c in CORES}
        R=X-np.array([cm[c] for c in core]); off={i:R[ins==i].mean(0) if (ins==i).any() else 0*g for i in INST}
        self.mu={}; self.n={}
        for s in SUBTYPES:
            m=y==s; n=int(m.sum()); add=cm[s[0]]+off[s[1:]]
            w=0 if n==0 else 1 if self.k==0 else n/(n+self.k); self.mu[s]=w*(X[m].mean(0) if n else add)+(1-w)*add; self.n[s]=n
        res=X-np.array([self.mu[s] for s in y]); self.P=np.linalg.inv(LedoitWolf().fit(res).covariance_)
        tot=sum(self.n.values()); self.prior={s:(self.n[s]+1)/(tot+27) for s in SUBTYPES}
        return self
    def logp(self,X,flat=False):
        L=np.array([[-.5*(x-self.mu[s])@self.P@(x-self.mu[s])+(0 if flat else np.log(self.prior[s])) for s in SUBTYPES] for x in X])
        L-=L.max(1,keepdims=True); P=np.exp(L); return P/P.sum(1,keepdims=True)

class Independent:
    """Baseline: P(core | x) * P(instinct | x), each from a shared-covariance Gaussian on its own labels."""
    def fit(self,X,y):
        self.c=Model(np.inf).fit(X,np.array([s[0]+'sp' for s in y])); self.i=Model(np.inf).fit(X,np.array(['5'+s[1:] for s in y])); return self
    def logp(self,X,flat=False):
        Pc=self.c.logp(X,flat); Pi=self.i.logp(X,flat)
        pc=np.array([[Pc[r,SUBTYPES.index(c+'sp')] for c in CORES] for r in range(len(X))]); pc/=pc.sum(1,keepdims=True)
        pi=np.array([[Pi[r,SUBTYPES.index('5'+i)] for i in INST] for r in range(len(X))]); pi/=pi.sum(1,keepdims=True)
        return np.array([[pc[r,CORES.index(s[0])]*pi[r,INST.index(s[1:])] for s in SUBTYPES] for r in range(len(X))])

def evaluate(make,X,y,reps=10):
    from sklearn.model_selection import StratifiedKFold
    idx=np.array([SUBTYPES.index(s) for s in y]); out={'top1':[],'top3':[],'core':[],'log_loss':[],'countertype_top3':[]}
    strat=np.array([s[0] for s in y])
    for r in range(reps):
        P=np.zeros((len(y),27))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(X,strat): P[te]=make().fit(X[tr],y[tr]).logp(X[te])
        top=np.argsort(-P,1); out['top1'].append(np.mean(top[:,0]==idx)); out['top3'].append(np.mean([i in t[:3] for t,i in zip(top,idx)]))
        out['core'].append(np.mean([SUBTYPES[t[0]][0]==s[0] for t,s in zip(top,y)])); out['log_loss'].append(-np.mean(np.log(np.clip(P[np.arange(len(y)),idx],1e-12,1))))
        ct=np.isin(y,COUNTER); out['countertype_top3'].append(np.mean([i in t[:3] for t,i in zip(top[ct],idx[ct])]))
    return {k:round(float(np.mean(v)),4) for k,v in out.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--people',type=Path)
    p.add_argument('--output',type=Path,default=Path('data/stage2/subtype_model.json'))
    args=p.parse_args(); X,y=load(args.db)
    res={'profiles':len(y),'chance':{'top1':round(1/27,4),'top3':round(3/27,4),'core':round(1/9,4)},'models':{}}
    for k in (0,2,5,10,20,50,np.inf): res['models'][f'pooled k={k}']=evaluate(lambda:Model(k),X,y); print(json.dumps({f'k={k}':res['models'][f'pooled k={k}']}),flush=True)
    res['models']['independent core x instinct']=evaluate(Independent,X,y); print(json.dumps({'independent':res['models']['independent core x instinct']}),flush=True)
    best=min((m for m in res['models'] if m.startswith('pooled')),key=lambda m:res['models'][m]['log_loss']); res['chosen']=best
    k=float(best.split('=')[1]); args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps({'chosen':best}))
    if not args.people: return
    m=Model(k).fit(X,y); people=json.loads(args.people.read_text(encoding='utf-8')); out={}
    for name,f in people.items():
        if not isinstance(f,dict): continue
        x=probit([f[c] for c in FACETS])[None]; o={}
        for prior,flat in (('pdb_prior',False),('flat_prior',True)):
            P=m.logp(x,flat)[0]; top=np.argsort(-P)
            o[prior]={'top5':[(SUBTYPES[i],round(float(P[i]),3)) for i in top[:5]],
                      'core':sorted(((c,round(float(sum(P[SUBTYPES.index(c+i)] for i in INST)),3)) for c in CORES),key=lambda z:-z[1])[:4],
                      'instinct':{i:round(float(sum(P[SUBTYPES.index(c+i)] for c in CORES)),3) for i in INST}}
        a,b=[SUBTYPES[i] for i in np.argsort(-m.logp(x,True)[0])[:2]]; v=m.P@(m.mu[a]-m.mu[b]); con=(x[0]-(m.mu[a]+m.mu[b])/2)*v
        o['why']={'top':a,'runner_up':b,'facets_for_top':[(NAMES[FACETS[j]],round(float(con[j]),2)) for j in np.argsort(-con)[:5]],
                  'facets_for_runner_up':[(NAMES[FACETS[j]],round(float(con[j]),2)) for j in np.argsort(con)[:3]]}
        dist=sorted(((s,float(np.sqrt((x[0]-m.mu[s])@m.P@(x[0]-m.mu[s])))) for s in SUBTYPES),key=lambda z:z[1]); o['nearest_distance']=round(dist[0][1],2)
        o['typical_distance']=round(float(np.median([np.sqrt((X[j]-m.mu[y[j]])@m.P@(X[j]-m.mu[y[j]])) for j in range(len(y))])),2)
        out[name]=o
    Path('data/personal/subtype_guess.json').write_text(json.dumps(out,indent=1),encoding='utf-8'); print(json.dumps(out,indent=1))

if __name__=='__main__': main()
