"""Ablation for the panel's MBTI heads (Gemini 4 Argon review, steps 1, 2 and 4); database read-only.

  python tools/type_ablation.py --db data/pdb-top.duckdb [--samples top,top2]

A (crowd modal type): soft-fit duplication bug vs fixed fit, Jev prior correction, class-balanced 16-type head, raw vs
L2-normalised profile vectors, and the letter weight of the pool, scored on full profiles and on 250-word crops (the first
250 pooled words of pinned comments and Google Docs, which mimic a pasted self-description). Models are always trained on
full profiles; crops are only ever test inputs.
B (creator self-vote type, the creator's own profile badge): letter heads trained on self-votes, crowd letter heads, and a
9-feature stack (out-of-fold IPIP predictions + crowd letter probabilities).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from pdb_ingest import embed,panel
from pdb_ingest.encoder_text import encoder_view
from pdb_ingest.panel import MBTI,LETTERS,letters_to_types,pool,soft_fit,type_probs,prior_correct,l2

CACHE=Path('data/stage2/emb')/embed.model_slug(embed.DEFAULT_MODEL)/'main'; CROP_WORDS=250

def self_votes(conn):
    out={}
    for n,ev in conn.execute("SELECT profile_num,evidence_json FROM vote_distributions WHERE system_name='MBTI' AND self_vote_status='SUBTRACTED'").fetchall():
        labels=[e['label'] for e in json.loads(ev or '[]') if e.get('confirmed') and e.get('label') in MBTI]
        if labels: out[n]=Counter(labels).most_common(1)[0][0]
    return out

def load(db,samples):
    store=readonly_store(db); rows=panel.load_rows(store.connection,samples); sv=self_votes(store.connection); data={}
    for n,r in rows.items():
        if 'mbti_counts' not in r and n not in sv: continue
        views=encoder_view(store,n); sha=hashlib.sha256(json.dumps(views,ensure_ascii=False).encode('utf-8')).hexdigest()
        f=CACHE/f'{n}_{sha[:16]}.npy'
        if not f.exists(): continue
        segs=store.load_snapshot(n).segments; pooled=[' '.join(t[a:b] for a,b in rs) for t,rs in views]
        own=' '.join(p for s,p in zip(segs,pooled) if s.source_type in ('PINNED_COMMENT','GOOGLE_DOC')).split() or ' '.join(pooled).split()
        data[n]={'doc':' '.join(pooled),'crop':' '.join(own[:CROP_WORDS]),'vec':np.load(f),'crowd':panel.modal(r['mbti_counts']) if 'mbti_counts' in r else None,
                 'self':sv.get(n),'ipip':r.get('ipip'),'jev':panel.jev_probs(n)}
    return data

def embed_crops(texts):
    enc=embed.hf_encoder(embed.DEFAULT_MODEL)
    return np.array([embed.profile_vector(embed.embed_views([(t,[(0,len(t))])],enc)) for t in texts])

def metrics(P,y,L=None):
    idx=np.array([MBTI.index(t) for t in y]); top=P.argmax(1)
    f1=[]
    for c in set(idx):
        tp=np.sum((top==c)&(idx==c)); f1.append(2*tp/max(np.sum(top==c)+np.sum(idx==c),1))
    out={'exact':float(np.mean(top==idx)),'top3':float(np.mean([i in np.argsort(-p)[:3] for p,i in zip(P,idx)])),
         'log_loss':float(-np.mean(np.log(np.clip(P[np.arange(len(y)),idx],1e-9,1)))),'macro_f1':float(np.mean(f1))}
    if L is not None:   # headline type contradicting a letter head that is >= 70% sure
        out['contradicts_confident_letter']=float(np.mean([any((p>=.7 and MBTI[t][i]!=a) or (p<=.3 and MBTI[t][i]==a) for i,((a,_),p) in enumerate(zip(LETTERS,l)))
                                                           for t,l in zip(top,L)]))
    return {k:round(v,4) for k,v in out.items()}

def main():
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression,RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--samples',default='top,top2')
    p.add_argument('--output',type=Path,default=Path('data/stage2/type_ablation.json'))
    args=p.parse_args()
    data=load(args.db,tuple(args.samples.split(','))); ids=sorted(data)
    crops_v=embed_crops([data[n]['crop'] for n in ids]); pos={n:i for i,n in enumerate(ids)}
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True)
    cv=KFold(5,shuffle=True,random_state=0); res={'profiles':len(ids)}

    def letter_heads(D,E,y_letters,tr):
        """Class-balanced TF-IDF + e5 binary heads (as in the panel); returns predict(D,E) -> (n,4) first-letter probabilities."""
        heads=[]
        for i in range(4):
            yl=y_letters[:,i][tr]
            mt=make_pipeline(tf(),LogisticRegression(C=4,max_iter=3000,class_weight='balanced')).fit([D[k] for k in tr],yl)
            me=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,class_weight='balanced')).fit(E[tr],yl)
            heads.append((mt,me))
        return lambda Dx,Ex:np.column_stack([(mt.predict_proba(Dx)[:,1]+me.predict_proba(Ex)[:,1])/2 for mt,me in heads])

    # A: crowd type
    A=[n for n in ids if data[n]['crowd']]; D=[data[n]['doc'] for n in A]; C=[data[n]['crop'] for n in A]
    y=np.array([data[n]['crowd'] for n in A]); onehot=np.eye(16)[[MBTI.index(t) for t in y]]
    has=np.array([data[n]['jev'] is not None for n in A]); J=np.array([data[n]['jev'] if data[n]['jev'] is not None else o for n,o in zip(A,onehot)])
    YL=np.array([[t[i]==a for i,(a,_) in enumerate(LETTERS)] for t in y])
    variants=('dup','fixed','fixed+prior','fixed+prior+balanced'); weights=(.4,.6,.8,1.)
    res['A']={}
    for vec_kind in ('raw','l2'):
        E=np.array([data[n]['vec'] for n in A]); Ec=crops_v[[pos[n] for n in A]]
        if vec_kind=='l2': E,Ec=l2(np,E),l2(np,Ec)
        oofL={b:np.zeros((len(A),4)) for b in ('full','crop')}; oofT={(v,b):np.zeros((len(A),16)) for v in variants for b in ('full','crop')}
        for tr,te in cv.split(A):
            lh=letter_heads(D,E,YL,tr); oofL['full'][te]=lh([D[k] for k in te],E[te]); oofL['crop'][te]=lh([C[k] for k in te],Ec[te])
            prior=onehot[tr].mean(0)+1e-3
            for v in variants:
                Jt=J[tr].copy()
                if 'prior' in v: Jt[has[tr]]=prior_correct(np,J[tr][has[tr]],prior)
                t=np.where(has[tr,None],.5*Jt+.5*onehot[tr],onehot[tr])
                if v=='dup':   # the original bug: rows duplicated before TF-IDF / scaler fitting
                    ms=[]
                    for pre,clf,X in ((tf(),LogisticRegression(C=4,max_iter=3000),D),(StandardScaler(),LogisticRegression(C=.3,max_iter=3000),E)):
                        rows=[(i,j,w) for i,tt in enumerate(t) for j,w in enumerate(tt) if w>1e-3]; idx=[tr[i] for i,_,_ in rows]
                        Xs=[X[k] for k in idx] if X is D else X[idx]; Z=pre.fit_transform(Xs)
                        clf.fit(Z,np.array(MBTI)[[j for _,j,_ in rows]],sample_weight=np.array([w for _,_,w in rows])); ms.append((pre,clf))
                else:
                    cw=(1/prior)/np.mean(1/prior) if 'balanced' in v else None
                    ms=[soft_fit(np,pre,clf,[X[k] for k in tr] if X is D else X[tr],t,cw)
                        for pre,clf,X in ((tf(),LogisticRegression(C=4,max_iter=3000),D),(StandardScaler(),LogisticRegression(C=.3,max_iter=3000),E))]
                oofT[(v,'full')][te]=type_probs(np,ms,[D[k] for k in te],E[te]); oofT[(v,'crop')][te]=type_probs(np,ms,[C[k] for k in te],Ec[te])
            print(json.dumps({'vectors':vec_kind,'fold_done':True}),flush=True)
        for b in ('full','crop'):
            LT=letters_to_types(np,oofL[b])
            res['A'][f'{vec_kind}/{b}/letters_only']=metrics(LT,y,oofL[b])
            for v in variants:
                for w in weights: res['A'][f'{vec_kind}/{b}/{v}/w{w}']=metrics(pool(np,LT,oofT[(v,b)],w) if w<1 else LT,y,oofL[b])
        # IPIP (agent target) Spearman r on full profiles and crops: Argon's step 2 criterion
        R=[n for n in A if data[n]['ipip']]; ri=[A.index(n) for n in R]; out={'full':[],'crop':[]}
        for k in 'OCEAN':
            yk=np.array([data[n]['ipip'][k] for n in R]); pf=np.zeros(len(R)); pc=np.zeros(len(R))
            for tr,te in cv.split(R):
                mt=make_pipeline(tf(),RidgeCV(alphas=np.logspace(-1,4,12))).fit([D[ri[i]] for i in tr],yk[tr])
                me=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,12))).fit(E[[ri[i] for i in tr]],yk[tr])
                pf[te]=(mt.predict([D[ri[i]] for i in te])+me.predict(E[[ri[i] for i in te]]))/2
                pc[te]=(mt.predict([C[ri[i]] for i in te])+me.predict(Ec[[ri[i] for i in te]]))/2
            out['full'].append(spearmanr(pf,yk)[0]); out['crop'].append(spearmanr(pc,yk)[0])
        res['A'][f'{vec_kind}/ipip_mean_spearman']={b:round(float(np.mean(v)),4) for b,v in out.items()}|{'n':len(R)}
        if vec_kind=='l2': crowd_oof=oofL

    # B: creator self-vote type (vectors L2-normalised)
    S=[n for n in A if data[n]['self']]; si=[A.index(n) for n in S]; ys=np.array([data[n]['self'] for n in S])
    YS=np.array([[t[i]==a for i,(a,_) in enumerate(LETTERS)] for t in ys]); E=l2(np,np.array([data[n]['vec'] for n in S]))
    Ec=l2(np,crops_v[[pos[n] for n in S]]); DS=[data[n]['doc'] for n in S]; CS=[data[n]['crop'] for n in S]
    bal=lambda P,Y:[round(float(np.mean([np.mean((P[:,i]>=.5)[Y[:,i]==v]==v) for v in (True,False)])),4) for i in range(4)]
    exact=lambda P,yy:round(float(np.mean([''.join(a if p>=.5 else b for (a,b),p in zip(LETTERS,row))==t for row,t in zip(P,yy)])),4)
    m1={b:np.zeros((len(S),4)) for b in ('full','crop')}
    for tr,te in cv.split(S):
        lh=letter_heads(DS,E,YS,tr); m1['full'][te]=lh([DS[k] for k in te],E[te]); m1['crop'][te]=lh([CS[k] for k in te],Ec[te])
    # IPIP predictions for every self-vote profile: out-of-fold where the agent label exists, fold-model mean otherwise
    R=[n for n in A if data[n]['ipip']]; EA=l2(np,np.array([data[n]['vec'] for n in A])); EcA=l2(np,crops_v[[pos[n] for n in A]])
    IP={b:np.zeros((len(A),5)) for b in ('full','crop')}; cnt=np.zeros(len(A))
    ri=np.array([A.index(n) for n in R]); unl=np.setdiff1d(np.arange(len(A)),ri)
    for k,dom in enumerate('OCEAN'):
        yk=np.array([data[n]['ipip'][dom] for n in R])
        for tr,te in cv.split(R):
            mt=make_pipeline(tf(),RidgeCV(alphas=np.logspace(-1,4,12))).fit([D[i] for i in ri[tr]],yk[tr])
            me=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,12))).fit(EA[ri[tr]],yk[tr])
            for b,docs_,Ev in (('full',D,EA),('crop',C,EcA)):
                tgt=np.concatenate([ri[te],unl]); pr=(mt.predict([docs_[i] for i in tgt])+me.predict(Ev[tgt]))/2
                IP[b][ri[te],k]=pr[:len(te)]; IP[b][unl,k]+=pr[len(te):]/5
    res['B']={'profiles':len(S),'self_type_counts':dict(Counter(ys).most_common()),
              'reference_crowd_modal_vs_self':{'letters_balanced':bal(np.array([[t[i]==a for i,(a,_) in enumerate(LETTERS)] for t in y[si]],dtype=float),YS),
                                               'exact':round(float(np.mean(y[si]==ys)),4)}}
    for b in ('full','crop'):
        m3=crowd_oof[b][si]; feats=np.column_stack([IP[b][si],m3]); m2=np.zeros((len(S),4))
        ftrain=np.column_stack([IP['full'][si],crowd_oof['full'][si]])   # stack always trained on full-profile features
        for tr,te in cv.split(S):
            for i in range(4):
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,class_weight='balanced')).fit(ftrain[tr],YS[tr,i])
                m2[te,i]=clf.predict_proba(feats[te])[:,1]
        for name,P in (('self_letter_heads',m1[b]),('stack_ipip+crowd_letters',m2),('crowd_letter_heads',m3)):
            res['B'][f'{b}/{name}']={'letters_balanced':bal(P,YS),'mean_letter':round(float(np.mean(bal(P,YS))),4),'exact':exact(P,ys)}
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res,indent=1))

if __name__=='__main__': main()
