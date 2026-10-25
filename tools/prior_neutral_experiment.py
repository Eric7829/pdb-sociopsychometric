"""Current vs prior-neutral headline type for the crowd card. Database read-only.

  python tools/prior_neutral_experiment.py --db data/pdb-top.duckdb --portraits "<first-person folder>" "<third-person folder>"

The panel's headline pools the class-balanced letter heads with a 16-type head that carries the training type frequencies
(PDB: about 70% N). The prior-neutral variant divides the 16-type head's probabilities by the training frequency of each
type before pooling. Heads are the panel's (TF-IDF C=4 + encoder at panel C, averaged). Scored in 5-fold CV over
profiles against the crowd (full profiles, 250-word excerpts) and the creator's own badge, and on portraits.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portrait_test import portrait
from short_text_experiment import vectors
from type_ablation import load
from pdb_ingest import embed,panel
from pdb_ingest.encoder_text import clean
from pdb_ingest.panel import LETTERS,MBTI,ENCODER_LETTER_C,ENCODER_TYPE_C,l2,letters_to_types,letter_weight,pool,soft_binary_fit
from pdb_ingest.text import words

def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--portraits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=Path('data/stage2/prior_neutral_experiment.json'))
    args=p.parse_args()
    data=load(args.db,('top','top2')); ids=sorted(n for n in data if data[n]['crowd'])
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); vpath=Path('data/stage2')/f'excerpt_vectors_{embed.model_slug(enc[2]["model"])}.npz'
    D=[data[n]['doc'] for n in ids]; F=l2(np,[data[n]['vec'] for n in ids]); C=[data[n]['crop'] for n in ids]; Ec=vectors(C,enc,vpath)
    PT=[]; PY=[]
    for folder in args.portraits:
        for t in MBTI:
            f=next(iter(folder.glob(f'{t}.*')),None)
            if f: PT.append(clean(portrait(f,'strict'),True)); PY.append(t)
    Ep=l2(np,[embed.profile_vector(embed.embed_views([(t,[(0,len(t))])],enc)) for t in PT])
    crowd=np.array([data[n]['crowd'] for n in ids]); selfv=np.array([data[n]['self'] or '' for n in ids])
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True); cv=list(KFold(5,shuffle=True,random_state=0).split(ids))
    sets={'full':(D,F,len(ids)),'crop':(C,Ec,len(ids)),'port':(PT,Ep,len(PT))}
    L={k:np.zeros((n,4)) for k,(_,_,n) in sets.items()}; T={k:np.zeros((n,16)) for k,(_,_,n) in sets.items()}; prior={}
    for tr,te in cv+[(np.arange(len(ids)),None)]:
        keys=('port',) if te is None else ('full','crop')
        for i,(a,_) in enumerate(LETTERS):
            y=np.array([t[i]==a for t in crowd],dtype=float)
            ms=[soft_binary_fit(np,pre,clf,X,y[tr]) for (pre,clf),X in zip(((tf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=ENCODER_LETTER_C,max_iter=3000))),([D[k] for k in tr],F[tr]))]
            for k in keys:
                Dx,Ex,_=sets[k]; rows=slice(None) if te is None else te
                Dx=Dx if te is None else [Dx[j] for j in te]; Ex=Ex if te is None else Ex[te]
                L[k][rows,i]=sum(clf.predict_proba(pre.transform(X))[:,1] for (pre,clf),X in zip(ms,(Dx,Ex)))/2
        t=tf().fit([D[k] for k in tr]); mt=LogisticRegression(C=4,max_iter=3000).fit(t.transform([D[k] for k in tr]),crowd[tr])
        s=StandardScaler().fit(F[tr]); me=LogisticRegression(C=ENCODER_TYPE_C,max_iter=3000).fit(s.transform(F[tr]),crowd[tr]); cols=[MBTI.index(k) for k in mt.classes_]
        pr=np.array([np.mean(crowd[tr]==m) for m in MBTI])+1e-3
        for k in keys:
            Dx,Ex,_=sets[k]; rows=slice(None) if te is None else te
            Dx=Dx if te is None else [Dx[j] for j in te]; Ex=Ex if te is None else Ex[te]
            P=np.zeros((len(Ex),16)); P[:,cols]=(mt.predict_proba(t.transform(Dx))+me.predict_proba(s.transform(Ex)))/2
            T[k][rows]=(P+1e-4)/(P+1e-4).sum(1,keepdims=True)
            if te is None: prior['port']=pr
            else: prior.setdefault(k,np.zeros((len(ids),16)))[te]=pr
    def scores(P,truth,mask=None):
        m=np.ones(len(truth),bool) if mask is None else mask; idx=np.array([MBTI.index(t) if t else 0 for t in truth])[m]; top=P[m].argmax(1)
        recall=[float(np.mean(top[idx==j]==j)) for j in range(16) if (idx==j).any()]
        return {'exact':round(float(np.mean(top==idx)),3),'top3':round(float(np.mean([i in np.argsort(-q)[:3] for q,i in zip(P[m],idx)])),3),
                'mean_type_recall':round(float(np.mean(recall)),3),'S_predicted':int(sum(MBTI[k][1]=='S' for k in top)),'S_true':int(sum(MBTI[k][1]=='S' for k in idx))}
    res={}
    for name,neutral in (('current (PDB prior)',False),('prior-neutral',True)):
        out={}
        for k in ('full','crop','port'):
            Tk=T[k]/(prior[k] if neutral else 1); Tk=Tk/Tk.sum(1,keepdims=True)
            texts=sets[k][0]; P=pool(np,letters_to_types(np,L[k]),Tk,letter_weight(np,[words(x) for x in texts]))
            if k=='port': out['portraits']=scores(P,PY)
            else: out[f'{k}_vs_crowd']=scores(P,crowd); out[f'{k}_vs_self']=scores(P,selfv,selfv!='')
        res[name]=out; print(json.dumps({'variant':name,**out}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
