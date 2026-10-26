"""Recalibrate MBTI Form M scoring for LLM respondents, following the Form M manual's method. Read-only.

  python tools/formm_recalibrate.py --db data/pdb-top.duckdb

The manual (ch. 7) scores Form M with IRT item parameters and Bayes EAP estimates of theta, then places each scale's
division point where it maximizes agreement with best-fit type (the type people choose after a feedback session);
exact ties go to I, N, F or P. Here the respondents are the Form M v3b agents (scrubbed text) and best-fit type is the
profile creator's own PDB self-vote. Variants per dichotomy, 5-fold CV (5 repeats) over self-voted profiles:
  official:        formm.score as shipped (human 2PL parameters, MLE, cut at 0 plus the S/T/J clarity adjustments),
  official, cut 0: the same theta without the adjustments,
  human 2PL + EAP + cut: human parameters, EAP theta (N(0,1) prior), division point fitted on the training folds,
  LLM 2PL + EAP + cut: 2PL refitted to the agents' answers (marginal ML, EM with Gauss-Hermite-style quadrature,
                   all training-fold agents), EAP theta, division point fitted on the training folds,
  item logistic:   L2 logistic regression from the items straight to the self-vote letter (a data-driven key; the
                   ceiling for this item set).
Scored by agreement with the self-vote letter (and balanced agreement), exact four-letter agreement, and agreement
with the crowd's modal type. Also reports each fitted division point and the share of the letter it produces.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from type_ablation import self_votes
from pdb_ingest import formm,panel
from pdb_ingest.panel import MBTI

DICH=list(formm.DICHOTOMIES); Q=np.linspace(-4,4,41); W=np.exp(-Q**2/2); W/=W.sum()

def responses(folder):
    qs=formm.load_questions(); params=formm.load_parameters(); out={}
    for f in Path(folder).glob('*.json'):
        a=json.loads(f.read_text(encoding='utf-8'))['answers']
        out[int(f.stem)]={n:qs[n-1]['options'][a[str(n)]]['scoreKey'] for n in params if a.get(str(n)) in ('A','B')}
    return out

def items_of(d):
    p=formm.load_parameters(); return [n for n in sorted(p) if p[n]['dichotomy']==d]

def eap(U,a,b):
    """U: (people, items) of 0/1/nan. Posterior mean of theta under N(0,1)."""
    P=1/(1+np.exp(-a[None,:]*(Q[:,None]-b[None,:])))                    # (quad, items)
    L=np.zeros((len(U),len(Q)))
    for j in range(U.shape[1]):
        u=U[:,j]; m=~np.isnan(u)
        L[m]+=np.where(u[m,None]==1,np.log(P[None,:,j]),np.log(1-P[None,:,j]))
    L+=np.log(W); L-=L.max(1,keepdims=True); post=np.exp(L); post/=post.sum(1,keepdims=True)
    return post@Q

def fit_2pl(U,iters=40):
    """Marginal ML 2PL by EM over the quadrature grid; slope-intercept form with a weak prior on the slope."""
    from scipy.optimize import minimize
    k=U.shape[1]; a=np.ones(k); d=np.zeros(k)
    for _ in range(iters):
        P=1/(1+np.exp(-(a[None,:]*Q[:,None]+d[None,:])))
        L=np.zeros((len(U),len(Q)))
        for j in range(k):
            u=U[:,j]; m=~np.isnan(u)
            L[m]+=np.where(u[m,None]==1,np.log(P[None,:,j]),np.log(1-P[None,:,j]))
        L+=np.log(W); L-=L.max(1,keepdims=True); post=np.exp(L); post/=post.sum(1,keepdims=True)
        for j in range(k):
            u=U[:,j]; m=~np.isnan(u); n_q=post[m].sum(0); r_q=(post[m]*u[m,None]).sum(0)
            def nll(x):
                p=np.clip(1/(1+np.exp(-(x[0]*Q+x[1]))),1e-9,1-1e-9)
                return -(r_q*np.log(p)+(n_q-r_q)*np.log(1-p)).sum()+.5*(np.log(max(x[0],1e-3)))**2*4
            r=minimize(nll,[a[j],d[j]],bounds=[(.05,6),(-12,12)],method='L-BFGS-B'); a[j],d[j]=r.x
    return a,-d/a

def best_cut(t,y):
    """Division point maximizing agreement with y (1 = first pole); ties at the cut go to the second pole, as Form M."""
    c=np.unique(np.r_[t,0]); acc=[np.mean((t>x)==y) for x in c]; i=int(np.argmax(acc))
    j=i+1 if i+1<len(c) else i; return float((c[i]+c[j])/2)

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/formm_recalibrate.json'))
    args=p.parse_args(); store=readonly_store(args.db); c=store.connection
    R=responses(panel.AGENT_FORMM); sv=self_votes(c); rows=panel.load_rows(c,('top','top2','es'))
    crowd={n:panel.modal(r['mbti_counts']) for n,r in rows.items() if 'mbti_counts' in r}
    ids=sorted(n for n in R if n in sv); allids=sorted(R)
    params=formm.load_parameters(); res={'agents':len(allids),'self_voted':len(ids)}
    U={d:np.array([[R[n].get(i,np.nan) for i in items_of(d)] for n in allids],float) for d in DICH}
    idx={n:k for k,n in enumerate(allids)}; sel=np.array([idx[n] for n in ids])
    Y={d:np.array([sv[n][k]==formm.DICHOTOMIES[d][0] for n in ids]) for k,d in enumerate(DICH)}
    official={n:formm.score({i:('A' if formm.load_questions()[i-1]['options']['A']['scoreKey']==v else 'B') for i,v in R[n].items()}) for n in ids}
    human={d:(np.array([params[i]['params']['a'] for i in items_of(d)]),np.array([params[i]['params']['b'] for i in items_of(d)])) for d in DICH}
    names=['official','official, cut 0','human 2PL + EAP + cut','LLM 2PL + EAP + cut','item logistic']
    pred={m:{d:np.zeros(len(ids),bool) for d in DICH} for m in names}; cuts={m:{d:[] for d in DICH} for m in names[2:4]}
    for d in DICH:
        k=DICH.index(d); pred['official'][d]=np.array([official[n]['type'][k]==formm.DICHOTOMIES[d][0] for n in ids])
        pred['official, cut 0'][d]=np.array([official[n]['dichotomyResults'][d]['theta']>0 for n in ids])
    reps=5; agg={m:{'letter':{d:[] for d in DICH},'balanced':{d:[] for d in DICH},'exact':[],'crowd_exact':[]} for m in names}
    for r in range(reps):
        P={m:{d:np.zeros(len(ids),bool) for d in DICH} for m in names}
        for m in names[:2]: P[m]=pred[m]
        for tr,te in KFold(5,shuffle=True,random_state=r).split(ids):
            test_people=set(sel[te]); train_all=np.array([k for k in range(len(allids)) if k not in test_people])
            for d in DICH:
                a,b=human[d]; th=eap(U[d][sel],a,b); x=best_cut(th[tr],Y[d][tr]); P['human 2PL + EAP + cut'][d][te]=th[te]>x; cuts['human 2PL + EAP + cut'][d].append(x)
                a2,b2=fit_2pl(U[d][train_all]); th2=eap(U[d][sel],a2,b2)
                if np.corrcoef(th2,eap(U[d][sel],a,b))[0,1]<0: th2=-th2          # keep the E/S/T/J direction
                x2=best_cut(th2[tr],Y[d][tr]); P['LLM 2PL + EAP + cut'][d][te]=th2[te]>x2; cuts['LLM 2PL + EAP + cut'][d].append(x2)
                X=np.nan_to_num(U[d][sel],nan=.5); lr=LogisticRegression(C=.3,max_iter=3000).fit(X[tr],Y[d][tr]); P['item logistic'][d][te]=lr.predict(X[te]).astype(bool)
        for m in names:
            for d in DICH:
                y=Y[d]; q=P[m][d]; agg[m]['letter'][d].append(np.mean(q==y)); agg[m]['balanced'][d].append(np.mean([np.mean(q[y==v]==v) for v in (True,False)]))
            types=[''.join(formm.DICHOTOMIES[d][0] if P[m][d][i] else formm.DICHOTOMIES[d][1] for d in DICH) for i in range(len(ids))]
            agg[m]['exact'].append(np.mean([t==sv[n] for t,n in zip(types,ids)]))
            cc=[(t,crowd[n]) for t,n in zip(types,ids) if n in crowd]; agg[m]['crowd_exact'].append(np.mean([t==u for t,u in cc]))
    for m in names:
        o={'letter':{d:round(float(np.mean(v)),3) for d,v in agg[m]['letter'].items()},'balanced':{d:round(float(np.mean(v)),3) for d,v in agg[m]['balanced'].items()},
           'exact_vs_self':round(float(np.mean(agg[m]['exact'])),3),'exact_vs_crowd':round(float(np.mean(agg[m]['crowd_exact'])),3)}
        if m in cuts: o['division_point']={d:round(float(np.median(v)),2) for d,v in cuts[m].items()}
        res[m]=o; print(json.dumps({m:o}),flush=True)
    res['self_vote_shares']={d:round(float(Y[d].mean()),3) for d in DICH}
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
