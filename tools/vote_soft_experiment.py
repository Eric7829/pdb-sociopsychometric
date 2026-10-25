"""Crowd card trained on the full vote distribution instead of the modal vote. Database read-only.

  python tools/vote_soft_experiment.py --db data/pdb-top.duckdb --portraits "<first-person folder>" "<third-person folder>"

PDB voters' runner-up types follow cognitive-function closeness (mirror E/I pairs most, J/P flips least), while the modal
vote throws that away. Variants for the crowd card (heads as in the panel; 5-fold CV over profiles):
  modal:  letters and 16-type head on the modal type (v0.3),
  soft:   letters on each profile's vote share per letter, 16-type head on the vote distribution,
  mix:    0.5 * modal + 0.5 * distribution for both.
Scored against the modal type (full profiles and 250-word excerpts: exact, top-3, log loss), by the share of one-letter
errors that are J/P flips (functions-wise the furthest neighbour), and on portraits.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portrait_test import portrait
from short_text_experiment import vectors
from type_ablation import load
from run_stage2 import readonly_store
from pdb_ingest import embed,panel
from pdb_ingest.encoder_text import clean
from pdb_ingest.panel import MBTI,LETTERS,ENCODER_LETTER_C,ENCODER_TYPE_C,l2,letters_to_types,letter_weight,pool,soft_binary_fit,soft_fit,type_probs
from pdb_ingest.text import words

FLIPS=['E/I','S/N','T/F','J/P']

def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--portraits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=Path('data/stage2/vote_soft_experiment.json'))
    args=p.parse_args()
    rows=panel.load_rows(readonly_store(args.db).connection,('top','top2'))
    data=load(args.db,('top','top2')); ids=sorted(n for n in data if data[n]['crowd'] and 'mbti_counts' in rows[n])
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); vpath=Path('data/stage2')/f'excerpt_vectors_{embed.model_slug(enc[2]["model"])}.npz'
    D=[data[n]['doc'] for n in ids]; F=l2(np,[data[n]['vec'] for n in ids]); C=[data[n]['crop'] for n in ids]; Ec=vectors(C,enc,vpath)
    PT=[]; PY=[]
    for folder in args.portraits:
        for t in MBTI:
            f=next(iter(folder.glob(f'{t}.*')),None)
            if f: PT.append(clean(portrait(f,'strict'),True)); PY.append(t)
    Ep=l2(np,[embed.profile_vector(embed.embed_views([(t,[(0,len(t))])],enc)) for t in PT])
    y=np.array([data[n]['crowd'] for n in ids]); onehot=np.eye(16)[[MBTI.index(t) for t in y]]
    dist=np.array([[rows[n]['mbti_counts'].get(t,0) for t in MBTI] for n in ids],dtype=float); dist/=dist.sum(1,keepdims=True)
    lmarg=lambda T:np.array([[T[k,[j for j,t in enumerate(MBTI) if t[i]==a]].sum() for i,(a,_) in enumerate(LETTERS)] for k in range(len(T))])
    targets={'modal':onehot,'soft':dist,'mix':.5*onehot+.5*dist}
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True); cv=list(KFold(5,shuffle=True,random_state=0).split(ids))
    sets={'full':(D,F),'crop':(C,Ec),'port':(PT,Ep)}; res={}
    for name,T in targets.items():
        Lt=lmarg(T); L={k:np.zeros((len(v[1]),4)) for k,v in sets.items()}; P16={k:np.zeros((len(v[1]),16)) for k,v in sets.items()}
        for tr,te in cv+[(np.arange(len(ids)),None)]:
            keys=('port',) if te is None else ('full','crop')
            pick=lambda k:(sets[k][0],sets[k][1]) if te is None else ([sets[k][0][j] for j in te],sets[k][1][te])
            for i in range(4):
                ms=[soft_binary_fit(np,pre,clf,X,Lt[tr,i]) for (pre,clf),X in zip(((tf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=ENCODER_LETTER_C,max_iter=3000))),([D[k] for k in tr],F[tr]))]
                for k in keys:
                    Dx,Ex=pick(k); L[k][slice(None) if te is None else te,i]=sum(clf.predict_proba(pre.transform(X))[:,1] for (pre,clf),X in zip(ms,(Dx,Ex)))/2
            ms=[soft_fit(np,pre,clf,X,T[tr]) for (pre,clf),X in zip(((tf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=ENCODER_TYPE_C,max_iter=3000))),([D[k] for k in tr],F[tr]))]
            for k in keys:
                Dx,Ex=pick(k); P16[k][slice(None) if te is None else te]=type_probs(np,ms,Dx,Ex)
        out={}
        for k,truth in (('full',y),('crop',y),('port',np.array(PY))):
            P=pool(np,letters_to_types(np,L[k]),P16[k],letter_weight(np,[words(t) for t in sets[k][0]])); idx=np.array([MBTI.index(t) for t in truth]); top=P.argmax(1)
            flips=Counter(FLIPS[[i for i in range(4) if MBTI[a][i]!=MBTI[b][i]][0]] for a,b in zip(top,idx) if sum(MBTI[a][i]!=MBTI[b][i] for i in range(4))==1)
            out[k]={'exact':round(float(np.mean(top==idx)),3),'top3':round(float(np.mean([i in np.argsort(-q)[:3] for q,i in zip(P,idx)])),3),
                    'log_loss':round(float(-np.mean(np.log(np.clip(P[np.arange(len(idx)),idx],1e-9,1)))),3),
                    'one_letter_errors':{f:round(flips[f]/max(sum(flips.values()),1),2) for f in FLIPS}}
        res[name]=out; print(json.dumps({'variant':name,**out}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
