"""Local student models trained on DeepSeek's PDB-voter labels vs on crowd votes. Database read-only; no API calls.

  python tools/teacher_student.py --db data/pdb-top.duckdb

Profiles: those with a cached teacher label (data/stage2/llm-voter/pdb<n>_scrubbed_s0.json) and a crowd MBTI vote.
Panel heads (TF-IDF + gte halves averaged; letters pooled with the 16-type head by letter_weight), 5-fold CV, targets:
  crowd:   the crowd modal type (the v0.3 crowd card),
  teacher: the teacher's 16-type distribution (letter heads on its letter marginals),
  mix:     0.5 * crowd one-hot + 0.5 * teacher.
Scored against the crowd modal type on full profiles and 250-word excerpts (exact, top-3, letter accuracy, S recall),
and against the teacher's top type. Big Five: ridge on agent IPIP raw domains (v0.3), on the teacher's percentiles, or
both averaged (each z-scored), trained on all of these profiles and scored on the Essays self-report median splits.
The ship block applies docs/superpowers/plans/2026-10-25-llm-teacher.md.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from short_text_experiment import vectors
from type_ablation import load
from pdb_ingest import embed
from pdb_ingest.encoder_text import clean
from pdb_ingest.panel import DOMAINS,LETTERS,MBTI,ENCODER_LETTER_C,ENCODER_TYPE_C,l2,letters_to_types,letter_weight,pool,soft_binary_fit,soft_fit,type_probs
from pdb_ingest.text import words

TEACHER=Path('data/stage2/llm-voter')

def main():
    import duckdb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression,RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/teacher_student.json'))
    args=p.parse_args()
    data=load(args.db,('top','top2','es')); teach={}
    for n,d in data.items():
        f=TEACHER/f'pdb{n}_scrubbed_s0.json'
        if d['crowd'] and f.exists(): teach[n]=json.loads(f.read_text(encoding='utf-8'))
    ids=sorted(teach); print(json.dumps({'profiles':len(ids)}),flush=True)
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); vpath=Path('data/stage2')/f'excerpt_vectors_{embed.model_slug(enc[2]["model"])}.npz'
    D=[data[n]['doc'] for n in ids]; F=l2(np,[data[n]['vec'] for n in ids]); C=[data[n]['crop'] for n in ids]; Ec=vectors(C,enc,vpath)
    y=np.array([data[n]['crowd'] for n in ids]); idx=np.array([MBTI.index(t) for t in y]); onehot=np.eye(16)[idx]
    T=np.array([teach[n]['probabilities'] for n in ids]); tt=T.argmax(1)
    lmarg=lambda M:np.array([[M[k,[j for j,t in enumerate(MBTI) if t[i]==a]].sum() for i,(a,_) in enumerate(LETTERS)] for k in range(len(M))])
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True); cv=list(KFold(5,shuffle=True,random_state=0).split(ids))
    sets={'full':(D,F),'crop':(C,Ec)}; res={'profiles':len(ids),'teacher_vs_crowd':{'exact':round(float(np.mean(tt==idx)),3)}}
    for name,M in (('crowd',onehot),('teacher',T),('mix',.5*onehot+.5*T)):
        Lt=lmarg(M); L={k:np.zeros((len(ids),4)) for k in sets}; P16={k:np.zeros((len(ids),16)) for k in sets}
        for tr,te in cv:
            for i in range(4):
                ms=[soft_binary_fit(np,pre,clf,X,Lt[tr,i]) for (pre,clf),X in zip(((tf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=ENCODER_LETTER_C,max_iter=3000))),([D[k] for k in tr],F[tr]))]
                for k,(Dx,Ex) in sets.items(): L[k][te,i]=sum(clf.predict_proba(pre.transform(X))[:,1] for (pre,clf),X in zip(ms,([Dx[j] for j in te],Ex[te])))/2
            ms=[soft_fit(np,pre,clf,X,M[tr]) for (pre,clf),X in zip(((tf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=ENCODER_TYPE_C,max_iter=3000))),([D[k] for k in tr],F[tr]))]
            for k,(Dx,Ex) in sets.items(): P16[k][te]=type_probs(np,ms,[Dx[j] for j in te],Ex[te])
        out={}
        for k,(Dx,_) in sets.items():
            P=pool(np,letters_to_types(np,L[k]),P16[k],letter_weight(np,[words(t) for t in Dx])); top=P.argmax(1)
            s=np.array([MBTI[i][1]=='S' for i in idx]); ps=np.array([MBTI[i][1]=='S' for i in top])
            out[k]={'exact':round(float(np.mean(top==idx)),3),'top3':round(float(np.mean([i in np.argsort(-q)[:3] for q,i in zip(P,idx)])),3),
                    'letters':{a+b:round(float(np.mean([MBTI[t][i]==MBTI[c][i] for t,c in zip(top,idx)])),3) for i,(a,b) in enumerate(LETTERS)},
                    'S_recall':round(float(ps[s].mean()),3),'agrees_with_teacher':round(float(np.mean(top==tt)),3),
                    'predicted':dict(Counter(MBTI[i] for i in top).most_common(5))}
        res[name]=out; print(json.dumps({'target':name,**out}),flush=True)
    # Big Five on Essays
    R=np.array([data[n]['ipip'] is not None for n in ids]); P=str(Path('data/external/essays-big5')/'*.parquet').replace('\\','/')
    rows=duckdb.connect().execute(f"select O,C,E,A,\"N\",text from read_parquet('{P}')").fetchall()
    ey=np.array([[int(v) for v in r[:5]] for r in rows]); et=[clean(r[5],True) for r in rows]; Ee=vectors(et,enc,vpath)
    TB={'O':'O','C':'C','E':'E','A':'A','N':'N'}
    def fit_pred(yv,mask):
        alphas=np.logspace(-1,4,12); a=tf().fit([D[k] for k in np.flatnonzero(mask)]); s=StandardScaler().fit(F[mask])
        m1=RidgeCV(alphas=alphas).fit(a.transform([D[k] for k in np.flatnonzero(mask)]),yv[mask]); m2=RidgeCV(alphas=alphas).fit(s.transform(F[mask]),yv[mask])
        return (m1.predict(a.transform(et))+m2.predict(s.transform(Ee)))/2
    z=lambda v:(v-v.mean())/v.std()
    big5={}
    for name in ('agent','teacher','both'):
        acc=[];auc=[]
        for i,k in enumerate(DOMAINS):
            ya=np.array([data[n]['ipip'][k] if data[n]['ipip'] else 0. for n in ids]); yt=np.array([teach[n]['big_five'][TB[k]] for n in ids])
            pe={'agent':lambda:fit_pred(ya,R),'teacher':lambda:fit_pred(yt,np.ones(len(ids),bool)),'both':lambda:z(fit_pred(ya,R))+z(fit_pred(yt,np.ones(len(ids),bool)))}[name]()
            acc.append(np.mean((pe>np.median(pe))==ey[:,i])); auc.append(roc_auc_score(ey[:,i],pe))
        big5[name]={'accuracy':round(float(np.mean(acc)),4),'auc':round(float(np.mean(auc)),4),'per_trait_acc':dict(zip(DOMAINS,np.round(acc,3).tolist()))}
    res['essays_big5']=big5; print(json.dumps({'essays_big5':big5}),flush=True)
    b=res['crowd']['crop']; ship={}
    for v in ('teacher','mix'):
        o=res[v]['crop']; ship[v]=bool((o['exact']-b['exact']>=.02 or o['top3']-b['top3']>=.02) and o['exact']>=b['exact'] and o['top3']>=b['top3'])
    ship['big5_'+max(('teacher','both'),key=lambda v:big5[v]['accuracy'])]=bool(max(big5['teacher']['accuracy'],big5['both']['accuracy'])-big5['agent']['accuracy']>=.01)
    res['ship']=ship; print(json.dumps({'ship':ship}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
