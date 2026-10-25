"""All nine Enneagram cores and 27 instinctual subtypes: Big Five and MBTI. Database read-only; aggregates only.

  python tools/ennea_report.py --db data/pdb-top.duckdb

Labels (crowd, training-eligible only): core = modal ENNEAGRAM vote's digit, wing = its wing, dominant instinct = the
instinct with most first-position INSTINCT votes (so_votes.first_instinct), MBTI = modal MBTI vote.
Big Five: the Stage 3 agent's IPIP-NEO-120 percentiles (domains + 30 facets, prompt A) and the crowd's SLOAN letter shares.
Leak control: everything core-level is repeated on profiles whose raw text never mentions Enneagram vocabulary (the agent
read raw text, so it could otherwise echo a stated type). External check: PANDORA users who self-report both MBTI and
Enneagram (labels only, nothing leaves the machine).
Statistics: Kruskal-Wallis with epsilon-squared across cores; each core vs the rest by Cohen's d with Mann-Whitney
p, Benjamini-Hochberg over every core x scale test; multinomial logistic regression from Big Five to core (repeated
stratified 5-fold, balanced accuracy vs 1/9); MBTI letter shares per core with two-proportion z and Cramer's V;
instinct effects within core and pooled after removing core means; subtype cells with n >= 8 described against the
other two subtypes of their core.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ennea_compare import bh
from run_stage2 import readonly_store
from so5_big5 import SLOAN,cohen_d,first_instinct
from pdb_ingest import panel
from pdb_ingest.battery import corpus_text
from pdb_ingest.panel import DOMAINS,LETTERS,MBTI,modal

CORES='123456789'; INST=('sp','so','sx')
ENNEA_TALK=re.compile(r'ennea|\b[1-9]w[1-9]\b|\btype [1-9]\b|tritype|trifix|\b(sp|so|sx)[ /-]?(sp|so|sx|[1-9])\b|instinct',re.I)

def kruskal_eps(groups):
    from scipy.stats import kruskal
    groups=[g for g in groups if len(g)>=3]; n=sum(len(g) for g in groups); H,p=kruskal(*groups)
    return {'H':round(float(H),2),'p':float(p),'eps2':round(float((H-len(groups)+1)/(n-len(groups))),3)}

def core_vs_rest(X,core,names):
    from scipy.stats import mannwhitneyu
    rows=[]
    for c in CORES:
        m=core==c
        if m.sum()<5: continue
        for j,k in enumerate(names): rows.append([c,k,float(X[m,j].mean()),float(cohen_d(X[m,j],X[~m,j])),mannwhitneyu(X[m,j],X[~m,j]).pvalue])
    q=bh([r[4] for r in rows])
    return [{'core':c,'scale':k,'mean':round(m,1),'d':round(d,2),'q':round(float(qq),4)} for (c,k,m,d,_),qq in zip(rows,q)]

def classify(X,y,reps=10):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    out=[]
    for r in range(reps):
        p=np.empty(len(y),dtype=object)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(X,y):
            s=StandardScaler().fit(X[tr]); p[te]=LogisticRegression(C=.5,max_iter=3000,class_weight='balanced').fit(s.transform(X[tr]),y[tr]).predict(s.transform(X[te]))
        out.append(balanced_accuracy_score(y,p))
    return round(float(np.mean(out)),3)

def mbti_block(core,mbti):
    from scipy.stats import chi2_contingency
    ok=np.array([t is not None for t in mbti]); core=core[ok]; mbti=np.array(mbti)[ok]; out={'n':int(ok.sum())}
    tab=np.array([[np.sum((core==c)&(mbti==t)) for t in MBTI] for c in CORES]); tab=tab[tab.sum(1)>0][:,tab.sum(0)>0]
    chi2,p,_,_=chi2_contingency(tab); out['cramers_v']=round(float(np.sqrt(chi2/tab.sum()/(min(tab.shape)-1))),3); out['chi2_p']=float(p)
    base={i:np.mean([t[i]==a for t in mbti]) for i,(a,_) in enumerate(LETTERS)}; tbase=Counter(mbti)
    for c in CORES:
        m=core==c; n=int(m.sum())
        if n<5: continue
        lets={}
        for i,(a,b) in enumerate(LETTERS):
            s=np.mean([t[i]==a for t in mbti[m]]); p0=base[i]; z=(s-p0)/np.sqrt(p0*(1-p0)/n)
            lets[a+b]={'share_'+a:round(float(s),3),'z':round(float(z),2)}
        cnt=Counter(mbti[m]); lift=[(t,k,round(k/n/(tbase[t]/len(mbti)),2)) for t,k in cnt.most_common() if k>=3]
        out[c]={'n':n,'letters':lets,'top_types':cnt.most_common(4),'over_represented':sorted(lift,key=lambda x:-x[2])[:4]}
    return out

def main():
    from scipy.stats import chi2_contingency,kruskal
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/ennea_report.json'))
    args=p.parse_args(); store=readonly_store(args.db); c=store.connection
    rows=panel.load_rows(c,('top','top2','es')); lab={}
    for n,system,counts,eligible,total in c.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible,total_system_votes FROM vote_distributions').fetchall():
        if n not in rows or not counts: continue
        cnt=json.loads(counts); d=lab.setdefault(n,{})
        if system=='INSTINCT' and eligible: d['inst']=first_instinct(cnt)
        if system=='BIG5_SLOAN' and (total or 0)>=10:
            t=sum(cnt.values()); d['sloan']=[sum(v for code,v in cnt.items() if len(code)==5 and code[i]==ch)/t for k,(i,ch) in ((k,SLOAN[k]) for k in DOMAINS)]
    facets={}
    for n,f in c.execute("SELECT r.profile_num,e.ipip_facets_json FROM agent_runs r JOIN psychometric_evaluations e ON e.run_id=r.run_id WHERE r.complete AND r.prompt_version='stage3-v1' ORDER BY r.seed").fetchall():
        if n in rows and n not in facets and f: facets[n]={k:v['percentile'] for k,v in json.loads(f).items()}
    fk=sorted(next(iter(facets.values()))); names=list(DOMAINS)+fk
    P=[]
    for n,r in rows.items():
        e=modal(r.get('ennea_counts') or {})
        if not e or e[0] not in CORES: continue
        P.append({'n':n,'core':e[0],'wing':e[2] if len(e)>=3 and e[1]=='w' else None,'inst':lab.get(n,{}).get('inst'),'mbti':modal(r['mbti_counts']) if 'mbti_counts' in r else None,
                  'ipip':[r['pct'][k] for k in DOMAINS]+[facets[n][k] for k in fk] if 'pct' in r and n in facets else None,'sloan':lab.get(n,{}).get('sloan'),
                  'talk':bool(ENNEA_TALK.search(corpus_text(store.load_snapshot(n).segments,'raw')))})
    res={'profiles':len(P),'cores':{k:v for k,v in sorted(Counter(x['core'] for x in P).items())},
         'enneagram_talk_share':round(float(np.mean([x['talk'] for x in P])),3)}
    def big5_block(sub,tag):
        I=[x for x in sub if x['ipip']]; X=np.array([x['ipip'] for x in I]); core=np.array([x['core'] for x in I]); out={'n':len(I)}
        out['kruskal']={k:kruskal_eps([X[core==cc,j] for cc in CORES]) for j,k in enumerate(names)}
        out['means']={cc:{'n':int((core==cc).sum()),**{k:round(float(X[core==cc,j].mean()),1) for j,k in enumerate(DOMAINS)}} for cc in CORES if (core==cc).sum()}
        out['core_vs_rest']=core_vs_rest(X,core,names)
        out['classify_domains']=classify(X[:,:5],core); out['classify_facets']=classify(X,core); out['chance']=round(1/9,3)
        S=[x for x in sub if x['sloan']]; Y=np.array([x['sloan'] for x in S]); sc=np.array([x['core'] for x in S])
        out['sloan']={'n':len(S),'kruskal':{k:kruskal_eps([Y[sc==cc,j] for cc in CORES]) for j,k in enumerate(DOMAINS)},
                      'means':{cc:{k:round(float(Y[sc==cc,j].mean()),3) for j,k in enumerate(DOMAINS)} for cc in CORES if (sc==cc).sum()}}
        print(json.dumps({'block':tag,'n':len(I),'classify':[out['classify_domains'],out['classify_facets']]}),flush=True)
        return out
    res['big5_all']=big5_block(P,'all'); res['big5_no_talk']=big5_block([x for x in P if not x['talk']],'no_talk')
    res['mbti_all']=mbti_block(np.array([x['core'] for x in P]),[x['mbti'] for x in P])
    res['mbti_no_talk']=mbti_block(np.array([x['core'] for x in P if not x['talk']]),[x['mbti'] for x in P if not x['talk']])
    res['wings']={cc:dict(Counter(x['wing'] for x in P if x['core']==cc and x['wing']).most_common()) for cc in CORES}
    # PANDORA self-report MBTI x Enneagram
    import duckdb
    A=str(Path('data/external/pandora-full/author_profiles.csv')).replace('\\','/')
    pr=duckdb.connect().execute(f"select cast(cast(enneagram_type as int) as varchar), upper(mbti) from read_csv_auto('{A}') where enneagram_type is not null and length(mbti)=4 and mbti not ilike '%x%'").fetchall()
    res['pandora_mbti']=mbti_block(np.array([a for a,_ in pr]),[b for _,b in pr])
    # instincts and the 27 subtypes
    T=[x for x in P if x['inst']]; res['instinct_n']=len(T)
    tab=np.array([[sum(1 for x in T if x['core']==cc and x['inst']==i) for i in INST] for cc in CORES]); chi2,pv,_,_=chi2_contingency(tab)
    res['subtype_counts']={cc+i:int(tab[a,b]) for a,cc in enumerate(CORES) for b,i in enumerate(INST)}
    res['core_x_instinct']={'chi2_p':float(pv),'cramers_v':round(float(np.sqrt(chi2/tab.sum()/2)),3),
                            'instinct_share':{cc:{i:round(float(tab[a,b]/tab[a].sum()),2) for b,i in enumerate(INST)} for a,cc in enumerate(CORES)}}
    TI=[x for x in T if x['ipip']]; X=np.array([x['ipip'] for x in TI]); core=np.array([x['core'] for x in TI]); inst=np.array([x['inst'] for x in TI])
    R=X.copy()
    for cc in CORES:
        m=core==cc
        if m.any(): R[m]-=X[m].mean(0)
    pooled={k:{**kruskal_eps([R[inst==i,j] for i in INST]),**{i:round(float(R[inst==i,j].mean()),1) for i in INST}} for j,k in enumerate(names)}
    qs=bh([v['p'] for v in pooled.values()])
    for (k,v),q in zip(pooled.items(),qs): v['q']=round(float(q),4)
    res['instinct_pooled']=pooled
    res['instinct_mbti']={i:{a+b:round(float(np.mean([x['mbti'][k]==a for x in T if x['inst']==i and x['mbti']])),3) for k,(a,b) in enumerate(LETTERS)} for i in INST}
    sub={}
    for cc in CORES:
        for i in INST:
            m=(core==cc)&(inst==i); o=(core==cc)&(inst!=i)
            if m.sum()<8 or o.sum()<5: continue
            ds=sorted(((k,float(cohen_d(X[m,j],X[o,j]))) for j,k in enumerate(names)),key=lambda z:-abs(z[1]))
            sub[cc+i]={'n':int(m.sum()),**{k:round(float(X[m,j].mean()),1) for j,k in enumerate(DOMAINS)},'vs_same_core_top':[(k,round(d,2)) for k,d in ds[:5]],
                       'mbti_top':Counter(x['mbti'] for x in T if x['core']==cc and x['inst']==i and x['mbti']).most_common(3)}
    res['subtypes']=sub
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps({k:res[k] for k in ('profiles','cores','enneagram_talk_share','instinct_n')}))

if __name__=='__main__': main()
