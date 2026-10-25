"""27 Enneagram subtypes (core x dominant instinct) from 30 IPIP-NEO facet percentiles. Database read-only.

  python tools/subtype_model.py --db data/pdb-top.duckdb [--people data/personal/ipip300.json]

Training data: PDB profiles with a crowd core and dominant instinct (tools/ennea_report) and the Stage 3 agent's facet
percentiles, turned into normal scores (probit, clipped to 0.5-99.5).
Model: one class per subtype with a shared covariance (Ledoit-Wolf, from residuals around each subtype's mean). Each
subtype's mean is partially pooled, mu_s = w * cell mean + (1 - w) * base, w = n_s / (n_s + k), where base is
  pool='global': core mean + an instinct offset pooled over all cores (instinct shifts every core alike),
  pool='core':   core mean only (instinct effects are learned within each core).
Options, each judged by cross-validation:
  cap:    clip facet scores to +-cap so a block of extreme facets can't dominate,
  shape:  subtract each profile's own mean over the 30 facets (match the pattern, not the overall level),
  dist:   'gauss' or 't' (multivariate Student-t, 5 degrees of freedom: heavier tails, softer on outliers),
  theory: add lam * (coded direction) * (within-core facet SD) to each subtype's mean, from tools/subtype_theory.json
          (the PDB wiki's subtype descriptions, coded as expected facet directions within each core),
  off-map: when a person is further from every subtype than 95% of training profiles are from their own, the
          log-likelihoods are scaled down by (d95 / d)^2, flattening the probabilities, and the result is flagged.
Scored by 5-fold CV (10 repeats) against the crowd's subtype and, out of fold, against the profile creator's own
Enneagram and instinct self-votes. With --people: each person's shortlist under the chosen configuration, with the
facets that separate the top candidates and the wiki's description labels for each.
"""
import argparse
from collections import Counter
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
THEORY=json.loads((Path(__file__).with_name('subtype_theory.json')).read_text(encoding='utf-8'))
NAMES=THEORY['facets']
SUBTYPES=[c+i for c in CORES for i in INST]
COUNTER=['1sx','2sp','3sp','4sp','5sx','6sx','7so','8so','9so']
NU=5

def probit(p):
    from scipy.stats import norm
    return norm.ppf(np.clip(np.asarray(p,float),.5,99.5)/100)

def self_labels(c):
    """profile -> (core, instinct) from the creator's own confirmed badges (either may be None)."""
    out={}
    for system in ('ENNEAGRAM','INSTINCT'):
        for n,ev in c.execute(f"SELECT profile_num,evidence_json FROM vote_distributions WHERE system_name='{system}' AND self_vote_status='SUBTRACTED'").fetchall():
            labels=[e['label'] for e in json.loads(ev or '[]') if e.get('confirmed') and e.get('label')]
            if not labels: continue
            lab=Counter(labels).most_common(1)[0][0]; core,ins=out.get(n,(None,None))
            if system=='ENNEAGRAM' and lab[0] in CORES: core=lab[0]
            if system=='INSTINCT' and lab[:2].lower() in INST: ins=lab[:2].lower()
            out[n]=(core,ins)
    return out

def load(db):
    store=readonly_store(db); c=store.connection; rows=panel.load_rows(c,('top','top2','es')); inst={}; fac={}
    for n,system,counts,eligible in c.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible FROM vote_distributions').fetchall():
        if system=='INSTINCT' and eligible and counts: inst[n]=first_instinct(json.loads(counts))
    for n,f in c.execute("SELECT r.profile_num,e.ipip_facets_json FROM agent_runs r JOIN psychometric_evaluations e ON e.run_id=r.run_id WHERE r.complete AND r.prompt_version='stage3-v1' ORDER BY r.seed").fetchall():
        if n not in fac and f: fac[n]={k:v['percentile'] for k,v in json.loads(f).items()}
    sv=self_labels(c); X=[];y=[];s=[]
    for n,r in rows.items():
        e=modal(r.get('ennea_counts') or {})
        if e and e[0] in CORES and inst.get(n) in INST and n in fac:
            X.append(probit([fac[n][k] for k in FACETS])); y.append(e[0]+inst[n]); s.append(sv.get(n,(None,None)))
    return np.array(X),np.array(y),s

class Model:
    def __init__(self,k=50,pool='global',cap=None,shape=False,dist='gauss',lam=0.,temper=True):
        self.k=k; self.pool=pool; self.cap=cap; self.shape=shape; self.dist=dist; self.lam=lam; self.temper=temper
    def prep(self,X):
        X=np.array(X,float)
        if self.cap: X=np.clip(X,-self.cap,self.cap)
        if self.shape: X=X-X.mean(1,keepdims=True)
        return X
    def fit(self,X,y):
        from sklearn.covariance import LedoitWolf
        X=self.prep(X); core=np.array([s[0] for s in y]); ins=np.array([s[1:] for s in y]); g=X.mean(0)
        cm={c:X[core==c].mean(0) if (core==c).any() else g for c in CORES}
        csd={c:X[core==c].std(0) if (core==c).sum()>2 else X.std(0) for c in CORES}
        R=X-np.array([cm[c] for c in core]); off={i:(R[ins==i].mean(0) if (ins==i).any() else 0*g) if self.pool=='global' else 0*g for i in INST}
        self.mu={}; self.n={}
        for s in SUBTYPES:
            m=y==s; n=int(m.sum()); base=cm[s[0]]+off[s[1:]]
            w=0 if n==0 else 1 if self.k==0 else n/(n+self.k); mu=w*(X[m].mean(0) if n else base)+(1-w)*base
            if self.lam: mu=mu+self.lam*np.array([THEORY['subtypes'][s]['expect'].get(f,0) for f in FACETS])/2*csd[s[0]]
            self.mu[s]=mu; self.n[s]=n
        res=X-np.array([self.mu[s] for s in y]); self.P=np.linalg.inv(LedoitWolf().fit(res).covariance_)
        tot=sum(self.n.values()); self.prior={s:(self.n[s]+1)/(tot+27) for s in SUBTYPES}
        d=np.sqrt(np.einsum('ij,jk,ik->i',res,self.P,res)); self.d95=float(np.quantile(d,.95)); self.d50=float(np.median(d))
        return self
    def dists(self,X):
        X=self.prep(X); M=np.array([self.mu[s] for s in SUBTYPES]); D=X[:,None,:]-M[None]
        return np.sqrt(np.einsum('nsj,jk,nsk->ns',D,self.P,D))
    def proba(self,X,flat=False):
        D=self.dists(X); p=len(FACETS)
        L=-.5*D**2 if self.dist=='gauss' else -(NU+p)/2*np.log1p(D**2/NU)
        if self.temper: L=L*np.minimum(1,(self.d95/D.min(1,keepdims=True))**2)
        if not flat: L=L+np.log([self.prior[s] for s in SUBTYPES])
        L-=L.max(1,keepdims=True); P=np.exp(L); return P/P.sum(1,keepdims=True)

def evaluate(make,X,y,selfl,reps=10):
    from sklearn.model_selection import StratifiedKFold
    idx=np.array([SUBTYPES.index(s) for s in y]); acc={k:[] for k in ('top1','top3','core','log_loss','countertype_top3','self_core','self_inst','self_sub_top3')}
    sc=np.array([c for c,_ in selfl],dtype=object); si=np.array([i for _,i in selfl],dtype=object)
    for r in range(reps):
        P=np.zeros((len(y),27))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(X,[s[0] for s in y]): P[te]=make().fit(X[tr],y[tr]).proba(X[te])
        top=np.argsort(-P,1); core=np.array([[sum(P[j,SUBTYPES.index(c+i)] for i in INST) for c in CORES] for j in range(len(y))])
        ins=np.array([[sum(P[j,SUBTYPES.index(c+i)] for c in CORES) for i in INST] for j in range(len(y))])
        acc['top1'].append(np.mean(top[:,0]==idx)); acc['top3'].append(np.mean([i in t[:3] for t,i in zip(top,idx)]))
        acc['core'].append(np.mean(np.array(list(CORES))[core.argmax(1)]==np.array([s[0] for s in y])))
        acc['log_loss'].append(-np.mean(np.log(np.clip(P[np.arange(len(y)),idx],1e-12,1))))
        ct=np.isin(y,COUNTER); acc['countertype_top3'].append(np.mean([i in t[:3] for t,i in zip(top[ct],idx[ct])]))
        m=sc!=None; acc['self_core'].append(np.mean(np.array(list(CORES))[core[m].argmax(1)]==sc[m]))
        m2=si!=None; acc['self_inst'].append(np.mean(np.array(INST)[ins[m2].argmax(1)]==si[m2]))
        m3=m&m2; acc['self_sub_top3'].append(np.mean([SUBTYPES.index(c+i) in t[:3] for t,c,i in zip(top[m3],sc[m3],si[m3])]))
    return {k:round(float(np.mean(v)),4) for k,v in acc.items()}

def explain(m,x,a,b):
    """Facets that push towards subtype a over b (contributions to the log-likelihood ratio)."""
    x=m.prep(x[None])[0]; v=m.P@(m.mu[a]-m.mu[b]); con=(x-(m.mu[a]+m.mu[b])/2)*v; order=np.argsort(-con)
    return {'for_'+a:[NAMES[FACETS[j]] for j in order[:4] if con[j]>0],'for_'+b:[NAMES[FACETS[j]] for j in order[::-1][:3] if con[j]<0]}

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--people',type=Path)
    p.add_argument('--output',type=Path,default=Path('data/stage2/subtype_model.json'))
    args=p.parse_args(); X,y,selfl=load(args.db)
    res={'profiles':len(y),'self_labelled':{'core':sum(c is not None for c,_ in selfl),'instinct':sum(i is not None for _,i in selfl)},
         'chance':{'top1':round(1/27,4),'top3':round(3/27,4),'core':round(1/9,4),'instinct':round(1/3,4)},'models':{}}
    configs={'v1: gauss, global pool, k=50':dict(k=50,temper=False),
             'cap 2':dict(k=50,cap=2.,temper=False),'shape':dict(k=50,shape=True,temper=False),'student-t':dict(k=50,dist='t',temper=False),
             'per-core pool k=20':dict(k=20,pool='core',temper=False),'per-core pool k=50':dict(k=50,pool='core',temper=False),
             'theory lam=.5':dict(k=50,lam=.5,temper=False),'theory lam=1':dict(k=50,lam=1.,temper=False),'theory lam=2':dict(k=50,lam=2.,temper=False)}
    def run(name,cfg):
        res['models'][name]={**cfg,**evaluate(lambda:Model(**cfg),X,y,selfl)}; print(json.dumps({name:res['models'][name]}),flush=True)
    for name,cfg in configs.items(): run(name,cfg)
    base=res['models']['v1: gauss, global pool, k=50']
    better=lambda n:res['models'][n]['log_loss']<base['log_loss']-.002
    combo=dict(k=50,temper=True)
    for n,keys in (('cap 2',{'cap':2.}),('shape',{'shape':True}),('student-t',{'dist':'t'})):
        if better(n): combo.update(keys)
    pc=min(('per-core pool k=20','per-core pool k=50'),key=lambda n:res['models'][n]['log_loss'])
    if better(pc): combo.update(pool='core',k=res['models'][pc]['k'])
    lams=[n for n in configs if n.startswith('theory')]; bl=min(lams,key=lambda n:res['models'][n]['log_loss'])
    if better(bl): combo['lam']=res['models'][bl]['lam']
    run('v2: combined',combo); res['chosen']=combo
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')
    if not args.people: return
    m=Model(**combo).fit(X,y); people=json.loads(args.people.read_text(encoding='utf-8')); out={}
    for name,f in people.items():
        if not isinstance(f,dict): continue
        x=probit([f[c] for c in FACETS]); d=m.dists(x[None])[0]; o={'off_map':bool(d.min()>m.d95),'nearest_distance':round(float(d.min()),2),
                                                                     'typical_distance':round(m.d50,2),'distance_95':round(m.d95,2)}
        for prior,flat in (('pdb_prior',False),('flat_prior',True)):
            P=m.proba(x[None],flat)[0]; top=np.argsort(-P)
            o[prior]={'shortlist':[(SUBTYPES[i],round(float(P[i]),3),THEORY['subtypes'][SUBTYPES[i]]['name']) for i in top[:4]],
                      'core':sorted(((c,round(float(sum(P[SUBTYPES.index(c+i)] for i in INST)),3)) for c in CORES),key=lambda z:-z[1])[:4],
                      'instinct':{i:round(float(sum(P[SUBTYPES.index(c+i)] for c in CORES)),3) for i in INST}}
        sl=[s for s,_,_ in o['flat_prior']['shortlist']]; o['why']=[explain(m,x,sl[0],b) for b in sl[1:3]]
        # theory fit: correlation between the person's within-core deviation and each shortlisted subtype's coded directions
        o['theory_fit']={}
        for s in sl:
            e=np.array([THEORY['subtypes'][s]['expect'].get(f,0) for f in FACETS]); dev=m.prep(x[None])[0]-np.mean([m.mu[s[0]+i] for i in INST],0)
            o['theory_fit'][s]=round(float(np.corrcoef(dev[e!=0],e[e!=0])[0,1]),2) if (e!=0).sum()>2 else None
        out[name]=o
    Path('data/personal/subtype_guess.json').write_text(json.dumps(out,indent=1),encoding='utf-8'); print(json.dumps(out,indent=1))

if __name__=='__main__': main()
