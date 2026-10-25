"""TF-IDF vs encoder weighting for the panel's MBTI heads, judged on real excerpts and on type portraits. Read-only.

  python tools/weight_experiment.py --db data/pdb-top.duckdb --portraits "data/portraits-first-person/Medium Level" ["<third-person folder>"]

The panel averages the TF-IDF and encoder halves 50/50. For encoder C in {.003, .01, .03} and encoder weight w in
{.3, .5, .7, .85, 1}, this scores p = w * p_encoder + (1 - w) * p_tfidf for the crowd card (letters + 16-type head, pooled as
in the panel) and the agent card (letters), on:
  - held-out 250-word excerpts in 5-fold CV over profiles (letter balanced accuracy, S/N, type exact and top-3),
  - 16-type portraits (strict scrub), with models fitted on every profile (letter accuracy, S/N, exact, S predicted).
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
from pdb_ingest.panel import LETTERS,MBTI,l2,letters_to_types,letter_weight,pool
from pdb_ingest.text import words

CS=(.003,.01,.03); WS=(.3,.5,.7,.85,1.)

def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--portraits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=Path('data/stage2/weight_experiment.json'))
    args=p.parse_args()
    data=load(args.db,('top','top2')); ids=sorted(data)
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); vpath=Path('data/stage2')/f'excerpt_vectors_{embed.model_slug(enc[2]["model"])}.npz'
    D=[data[n]['doc'] for n in ids]; F=l2(np,[data[n]['vec'] for n in ids]); C=[data[n]['crop'] for n in ids]; Ec=vectors(C,enc,vpath)
    PT=[]; PY=[]
    for folder in args.portraits:
        for t in MBTI:
            f=next(iter(folder.glob(f'{t}.*')),None)
            if f: PT.append(clean(portrait(f,'strict'),True)); PY.append(t)
    Ep=l2(np,[embed.profile_vector(embed.embed_views([(t,[(0,len(t))])],enc)) for t in PT]); PY=np.array(PY)
    v3b={int(f.stem):json.loads(f.read_text(encoding='utf-8'))['scored']['type'] for f in panel.AGENT_FORMM.glob('*.json')}
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True); cv=list(KFold(5,shuffle=True,random_state=0).split(ids))
    wc=letter_weight(np,[words(c) for c in C]); wp=letter_weight(np,[words(t) for t in PT])
    def letters_halves(lab,c):
        """Per letter: out-of-fold TF-IDF and encoder probabilities on excerpts, and full-data fits on portraits."""
        keep=np.array([lab[n] is not None for n in ids]); out={'tc':np.full((len(ids),4),.5),'ec':np.full((len(ids),4),.5),'tp':np.zeros((len(PT),4)),'ep':np.zeros((len(PT),4)),'keep':keep}
        for i,(a,_) in enumerate(LETTERS):
            y=np.array([lab[n] is not None and lab[n][i]==a for n in ids])
            for tr,te in cv+[(np.arange(len(ids)),None)]:
                tr=tr[keep[tr]]
                t=tf().fit([D[k] for k in tr]); mt=LogisticRegression(C=4,max_iter=3000,class_weight='balanced').fit(t.transform([D[k] for k in tr]),y[tr])
                s=StandardScaler().fit(F[tr]); me=LogisticRegression(C=c,max_iter=3000,class_weight='balanced').fit(s.transform(F[tr]),y[tr])
                if te is None: out['tp'][:,i]=mt.predict_proba(t.transform(PT))[:,1]; out['ep'][:,i]=me.predict_proba(s.transform(Ep))[:,1]
                else: out['tc'][te,i]=mt.predict_proba(t.transform([C[k] for k in te]))[:,1]; out['ec'][te,i]=me.predict_proba(s.transform(Ec[te]))[:,1]
        return out
    def type_halves(c):
        y=np.array([data[n]['crowd'] or '' for n in ids]); keep=y!=''; out={'tc':np.zeros((len(ids),16)),'ec':np.zeros((len(ids),16)),'tp':None,'ep':None}
        for tr,te in cv+[(np.arange(len(ids)),None)]:
            tr=tr[keep[tr]]
            t=tf().fit([D[k] for k in tr]); mt=LogisticRegression(C=4,max_iter=3000).fit(t.transform([D[k] for k in tr]),y[tr])
            s=StandardScaler().fit(F[tr]); me=LogisticRegression(C=c,max_iter=3000).fit(s.transform(F[tr]),y[tr]); cols=[MBTI.index(k) for k in mt.classes_]
            if te is None:
                out['tp']=np.zeros((len(PT),16)); out['ep']=np.zeros((len(PT),16))
                out['tp'][:,cols]=mt.predict_proba(t.transform(PT)); out['ep'][:,cols]=me.predict_proba(s.transform(Ep))
            else: out['tc'][np.ix_(te,cols)]=mt.predict_proba(t.transform([C[k] for k in te])); out['ec'][np.ix_(te,cols)]=me.predict_proba(s.transform(Ec[te]))
        return out
    def bal(p,truth,i):
        y=np.array([t[i]==LETTERS[i][0] for t in truth]); return float(np.mean([np.mean((p>=.5)[y==v]==v) for v in (True,False) if (y==v).any()]))
    def type_scores(P,truth):
        idx=np.array([MBTI.index(t) for t in truth]); top=P.argmax(1)
        return {'exact':round(float(np.mean(top==idx)),3),'top3':round(float(np.mean([i in np.argsort(-p)[:3] for p,i in zip(P,idx)])),3),
                'S_predicted':int(sum(MBTI[k][1]=='S' for k in top))}
    crowd={n:data[n]['crowd'] for n in ids}; agent={n:v3b.get(n) for n in ids}; res={}
    for c in CS:
        cl=letters_halves(crowd,c); al=letters_halves(agent,c); th=type_halves(c)
        truth_c=[crowd[n] for n in np.array(ids)[cl['keep']]]; truth_a=[agent[n] for n in np.array(ids)[al['keep']]]
        for w in WS:
            mix=lambda h,a,b:w*h[b]+(1-w)*h[a]
            Lc=mix(cl,'tc','ec'); La=mix(al,'tc','ec'); Lp=mix(cl,'tp','ep'); Lap=mix(al,'tp','ep')
            Tc=mix(th,'tc','ec'); Tp=mix(th,'tp','ep'); Tc=(Tc+1e-4)/(Tc+1e-4).sum(1,keepdims=True); Tp=(Tp+1e-4)/(Tp+1e-4).sum(1,keepdims=True)
            k=cl['keep']; Pc=pool(np,letters_to_types(np,Lc[k]),Tc[k],wc[k]); Pp=pool(np,letters_to_types(np,Lp),Tp,wp)
            r={'excerpt':{'crowd_letters':[round(bal(Lc[k][:,i],truth_c,i),3) for i in range(4)],
                          'agent_letters':[round(bal(La[al['keep']][:,i],truth_a,i),3) for i in range(4)],
                          'crowd_type':type_scores(Pc,truth_c)},
               'portraits':{'crowd_letters':[round(float(np.mean([(Lp[j,i]>=.5)==(PY[j][i]==LETTERS[i][0]) for j in range(len(PY))])),3) for i in range(4)],
                            'agent_letters':[round(float(np.mean([(Lap[j,i]>=.5)==(PY[j][i]==LETTERS[i][0]) for j in range(len(PY))])),3) for i in range(4)],
                            'crowd_type':type_scores(Pp,PY)}}
            r['excerpt']['crowd_mean']=round(float(np.mean(r['excerpt']['crowd_letters'])),3); r['excerpt']['agent_mean']=round(float(np.mean(r['excerpt']['agent_letters'])),3)
            res[f'C={c} w={w}']=r; print(json.dumps({'C':c,'w':w,**r}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
