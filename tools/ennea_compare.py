"""How do crowd-typed Enneagram core A profiles differ from core B? Database read-only; prints aggregates only.

  python tools/ennea_compare.py --db data/pdb-top.duckdb --a 3 --b 1

Core = modal training-eligible ENNEAGRAM vote. Compared on:
  crowd MBTI letters, dominant instinct and wing (Fisher exact),
  agent IPIP-NEO-120 domains and 30 facets (Cohen's d, Mann-Whitney, Benjamini-Hochberg over all 35),
  crowd SLOAN Big Five votes (letter shares, >= 10 votes),
  psycholinguistic document features from lexicons.doc_features (BH over 34),
  distinctive words: weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008) on document frequencies,
    words used by >= 5 profiles of the group, z >= 2,
  text separability: TF-IDF logistic regression, repeated stratified 5-fold AUC.
Text is the scrubbed encoder view, so typology talk is already removed.
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
from so5_big5 import SLOAN,cohen_d,first_instinct
from type_ablation import load
from pdb_ingest.lexicons import FEATURES,TOKEN,doc_features,load_lexicons
from pdb_ingest.panel import DOMAINS,LETTERS,modal

def bh(ps):
    ps=np.asarray(ps,float); o=np.argsort(ps); q=np.empty(len(ps)); run=1
    for r in range(len(ps)-1,-1,-1): i=o[r]; run=min(run,ps[i]*len(ps)/(r+1)); q[i]=run
    return q

def contrast(Xa,Xb,names):
    from scipy.stats import mannwhitneyu
    ps=[mannwhitneyu(Xa[:,j],Xb[:,j]).pvalue for j in range(len(names))]; q=bh(ps)
    rows=[{'name':k,'mean_a':round(float(Xa[:,j].mean()),4),'mean_b':round(float(Xb[:,j].mean()),4),'d':round(float(cohen_d(Xa[:,j],Xb[:,j])),2),'q':round(float(q[j]),4)} for j,k in enumerate(names)]
    return sorted(rows,key=lambda r:-abs(r['d']))

def log_odds(docs_a,docs_b,min_docs=5,top=25):
    da=Counter(w for d in docs_a for w in set(TOKEN.findall(d.lower()))); db=Counter(w for d in docs_b for w in set(TOKEN.findall(d.lower())))
    prior=da+db; a0=sum(prior.values()); na=sum(da.values()); nb=sum(db.values()); out=[]
    for w,p in prior.items():
        if max(da[w],db[w])<min_docs or len(w)<3: continue
        la=np.log((da[w]+p)/(na+a0-da[w]-p)); lb=np.log((db[w]+p)/(nb+a0-db[w]-p)); z=(la-lb)/np.sqrt(1/(da[w]+p)+1/(db[w]+p))
        out.append((w,float(z),da[w]/len(docs_a),db[w]/len(docs_b)))
    fmt=lambda r:{'word':r[0],'z':round(r[1],2),'share_a':round(r[2],2),'share_b':round(r[3],2)}
    s=sorted(out,key=lambda r:-r[1])
    return {'a':[fmt(r) for r in s[:top] if r[1]>=2],'b':[fmt(r) for r in s[::-1][:top] if r[1]<=-2]}

def text_auc(docs,y,reps=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    out=[]
    for r in range(reps):
        p=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(docs,y):
            v=TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True); m=LogisticRegression(C=4,max_iter=3000,class_weight='balanced')
            m.fit(v.fit_transform([docs[i] for i in tr]),y[tr]); p[te]=m.predict_proba(v.transform([docs[i] for i in te]))[:,1]
        out.append(roc_auc_score(y,p))
    return round(float(np.mean(out)),3)

def fisher_table(a,b):
    from scipy.stats import fisher_exact
    out={}
    for k in sorted(set(a)|set(b)):
        ca=sum(x==k for x in a); cb=sum(x==k for x in b)
        out[k]={'share_a':round(ca/len(a),3),'share_b':round(cb/len(b),3),'p':round(float(fisher_exact([[ca,len(a)-ca],[cb,len(b)-cb]])[1]),4)}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--a',default='3'); p.add_argument('--b',default='1')
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args(); A,B=args.a,args.b; conn=readonly_store(args.db).connection
    data=load(args.db,('top','top2')); ennea={}; inst={}; sloan={}; facets={}
    for n,system,counts,eligible,total in conn.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible,total_system_votes FROM vote_distributions').fetchall():
        if n not in data or not counts: continue
        c=json.loads(counts)
        if system=='ENNEAGRAM' and eligible: ennea[n]=modal(c)
        if system=='INSTINCT' and eligible: inst[n]=first_instinct(c)
        if system=='BIG5_SLOAN' and (total or 0)>=10:
            t=sum(c.values()); sloan[n]={k:sum(v for code,v in c.items() if len(code)==5 and code[i]==ch)/t for k,(i,ch) in SLOAN.items()}
    for n,f in conn.execute("SELECT r.profile_num,e.ipip_facets_json FROM agent_runs r JOIN psychometric_evaluations e ON e.run_id=r.run_id WHERE r.complete AND r.prompt_version='stage3-v1' ORDER BY r.seed").fetchall():
        if n in data and n not in facets and f: facets[n]={k:v['percentile'] for k,v in json.loads(f).items()}
    ga=[n for n in data if ennea.get(n) and ennea[n][0]==A]; gb=[n for n in data if ennea.get(n) and ennea[n][0]==B]
    res={'n':{A:len(ga),B:len(gb)}}
    res['mbti_letters']={f'{x}/{y}':fisher_table([data[n]['crowd'][i]==x for n in ga if data[n]['crowd']],[data[n]['crowd'][i]==x for n in gb if data[n]['crowd']])[True] for i,(x,y) in enumerate(LETTERS)}
    res['mbti_types_top']={f'core{c}':Counter(data[n]['crowd'] for n in g if data[n]['crowd']).most_common(5) for c,g in ((A,ga),(B,gb))}
    res['instinct']=fisher_table([inst[n] for n in ga if inst.get(n)],[inst[n] for n in gb if inst.get(n)])
    res['wing']={f'core{c}':dict(Counter(ennea[n] for n in g).most_common()) for c,g in ((A,ga),(B,gb))}
    fk=sorted(next(iter(facets.values()))) if facets else []
    ia=[n for n in ga if data[n]['ipip'] and n in facets]; ib=[n for n in gb if data[n]['ipip'] and n in facets]
    Xa=np.array([[data[n]['ipip'][k] for k in DOMAINS]+[facets[n][k] for k in fk] for n in ia]); Xb=np.array([[data[n]['ipip'][k] for k in DOMAINS]+[facets[n][k] for k in fk] for n in ib])
    res['agent_ipip']=contrast(Xa,Xb,list(DOMAINS)+fk)
    sa=[n for n in ga if n in sloan]; sb=[n for n in gb if n in sloan]
    res['crowd_sloan']=contrast(np.array([[sloan[n][k] for k in DOMAINS] for n in sa]),np.array([[sloan[n][k] for k in DOMAINS] for n in sb]),list(DOMAINS))
    lex=load_lexicons(); Fa=np.array([doc_features(data[n]['doc'],lex) for n in ga]); Fb=np.array([doc_features(data[n]['doc'],lex) for n in gb])
    res['psycholinguistic']=contrast(Fa,Fb,FEATURES)
    res['words']=log_odds([data[n]['doc'] for n in ga],[data[n]['doc'] for n in gb])
    res['text_auc']=text_auc([data[n]['doc'] for n in ga+gb],np.array([1]*len(ga)+[0]*len(gb)))
    out=args.output or Path(f'data/stage2/ennea_{A}_vs_{B}.json'); out.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(out)

if __name__=='__main__': main()
