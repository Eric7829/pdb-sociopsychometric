"""Psycholinguistic features and emotional-sentence filtering vs the v0.3 panel, on every harness. Read-only.

  python tools/upgrade_eval.py --db data/pdb-top.duckdb --portraits "<first-person folder>" [--limit 40]

Variants: baseline (TF-IDF + encoder halves averaged, as the panel), features (+ a third half on lexicons.doc_features,
weights 1/3 each), filter (lexicons.emotional_sentences applied to every text before TF-IDF and the encoder), both.
Document features are always computed on the unfiltered text (they are document-level, as in Majumder et al. 2017).
Filtered texts are embedded as plain text (cached by hash), not through encoder_view's span pooling.
Harnesses: PDB 5-fold CV (Big Five Spearman vs agent IPIP, crowd and agent letter balanced accuracy, crowd 16-type, on full
profiles and 250-word excerpts), Essays (median-split accuracy and AUC), PANDORA (500 test users, mean Spearman r),
portraits (crowd letter S/N accuracy), leak probe (mean crowd-letter shift from a typology-talk intro). The ship block
applies the pre-registered rule in docs/superpowers/plans/2026-10-25-psycholinguistic-upgrades.md.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from leak_probe import INTRO
from portrait_test import portrait
from short_text_experiment import vectors
from type_ablation import load
from pdb_ingest import embed,panel
from pdb_ingest.encoder_text import clean
from pdb_ingest.lexicons import doc_features,emotional_sentences,load_lexicons
from pdb_ingest.panel import LETTERS,MBTI,ENCODER_LETTER_C,ENCODER_TYPE_C,l2,letters_to_types,letter_weight,pool
from pdb_ingest.text import words

VARIANTS=('baseline','features','filter','features+filter')

def fit_halves(kind,D,E,F,y,feats):
    """Fitted (preprocessor, model) pairs for the TF-IDF, encoder and (optionally) feature halves."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression,RidgeCV
    from sklearn.preprocessing import StandardScaler
    alphas=np.logspace(-1,4,12)
    model={'reg':lambda c:RidgeCV(alphas=alphas),'bin':lambda c:LogisticRegression(C=c,max_iter=3000,class_weight='balanced'),
           'multi':lambda c:LogisticRegression(C=c,max_iter=3000)}[kind]
    enc_c=ENCODER_LETTER_C if kind=='bin' else ENCODER_TYPE_C
    halves=[(TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True),model(4),D),(StandardScaler(),model(enc_c),E)]
    if feats: halves.append((StandardScaler(),model(.1),F))
    out=[]
    for pre,m,X in halves: out.append((pre,m.fit(pre.fit_transform(X),y)))
    return out

def predict(kind,fitted,D,E,F):
    Xs=(D,E,F)[:len(fitted)]; out=[]
    for (pre,m),X in zip(fitted,Xs):
        Z=pre.transform(X)
        if kind=='reg': out.append(m.predict(Z))
        elif kind=='bin': out.append(m.predict_proba(Z)[:,1])
        else:
            P=np.zeros((Z.shape[0],16)); P[:,[MBTI.index(c) for c in m.classes_]]=m.predict_proba(Z); out.append(P)
    return np.mean(out,0)

def main():
    import duckdb
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--portraits',type=Path,required=True)
    p.add_argument('--limit',type=int,default=None); p.add_argument('--output',type=Path,default=Path('data/stage2/upgrade_eval.json'))
    args=p.parse_args(); lim=args.limit
    lex=load_lexicons(); enc=embed.hf_encoder(embed.DEFAULT_MODEL); vpath=Path('data/stage2')/f'excerpt_vectors_{embed.model_slug(enc[2]["model"])}.npz'
    data=load(args.db,('top','top2')); ids=sorted(data)[:lim] if lim else sorted(data)
    v3b={int(f.stem):json.loads(f.read_text(encoding='utf-8'))['scored']['type'] for f in panel.AGENT_FORMM.glob('*.json')}
    E_rows=duckdb.connect().execute(f"select O,C,E,A,\"N\",text from read_parquet('{str(Path('data/external/essays-big5')/'*.parquet').replace(chr(92),'/')}')").fetchall()[:lim*5 if lim else None]
    ey=np.array([[int(v) for v in r[:5]] for r in E_rows]); etext=[r[5] for r in E_rows]
    c=duckdb.connect(); P=str(Path('data/external/pandora-big5')/'test-*.parquet').replace('\\','/')
    users=c.execute(f'select O,C,E,A,"N" from read_parquet(\'{P}\') group by 1,2,3,4,5 having count(*)>=20').fetchall()
    random.Random(0).shuffle(users); users=users[:(lim or 500)]; ptext=[]; py=np.array(users,dtype=float)
    for u in users:
        t=[r[0] for r in c.execute(f'select text from read_parquet(\'{P}\') where O=? and C=? and E=? and A=? and "N"=?',list(u)).fetchall() if r[0]]
        random.Random(1).shuffle(t); doc=[]; w=0
        for x in t:
            doc.append(x); w+=len(x.split())
            if w>=1000: break
        ptext.append('\n'.join(doc))
    PT=[];PY=[]
    for t in MBTI:
        f=next(iter(args.portraits.glob(f'{t}.*')),None)
        if f: PT.append(clean(portrait(f,'strict'),True)); PY.append(t)
    LT=[clean(INTRO+x,True) for x in PT]
    crowd=[data[n]['crowd'] for n in ids]; agent=[v3b.get(n) for n in ids]
    def bits(types,i): return np.array([t is not None and t[i]==LETTERS[i][0] for t in types]),np.array([t is not None for t in types])
    res={}
    for variant in VARIANTS:
        feats='features' in variant; filt='filter' in variant
        prep=(lambda t:emotional_sentences(t,lex)) if filt else (lambda t:t)
        sets={}
        for name,texts,base_vec in (('full',[data[n]['doc'] for n in ids],None if filt else np.array([data[n]['vec'] for n in ids])),
                                    ('crop',[data[n]['crop'] for n in ids],None),('essays',[clean(t,True) for t in etext],None),
                                    ('pandora',[clean(t,True) for t in ptext],None),('portraits',PT,None),('leak',LT,None)):
            Dx=[prep(t) for t in texts]; Ex=l2(np,base_vec) if base_vec is not None else vectors(Dx,enc,vpath)
            sets[name]=(Dx,Ex,np.array([doc_features(t,lex) for t in texts]))
        out={}; cv=list(KFold(5,shuffle=True,random_state=0).split(ids)); oof={}
        # PDB cross-validation
        Dm,Em,Fm=sets['full']
        def cv_pred(kind,y,mask=None):
            got={b:np.full((len(ids),)+((16,) if kind=='multi' else ()),np.nan) for b in ('full','crop')}
            for tr,te in cv:
                tr=tr if mask is None else tr[mask[tr]]
                fit=fit_halves(kind,[Dm[k] for k in tr],Em[tr],Fm[tr],y[tr],feats)
                for b in ('full','crop'):
                    D_,E_,F_=sets[b]; got[b][te]=predict(kind,fit,[D_[k] for k in te],E_[te],F_[te])
            return got
        R=np.array([data[n]['ipip'] is not None for n in ids]); big5={b:[] for b in ('full','crop')}
        for i,k in enumerate('OCEAN'):
            y=np.array([data[n]['ipip'][k] if data[n]['ipip'] else 0. for n in ids]); g=cv_pred('reg',y,R)
            for b in big5: big5[b].append(spearmanr(g[b][R],y[R])[0])
        out['pdb_big5_r']={b:round(float(np.mean(v)),4) for b,v in big5.items()}
        for who,types in (('crowd',crowd),('agent',agent)):
            acc={b:[] for b in ('full','crop')}; L={b:np.zeros((len(ids),4)) for b in ('full','crop')}
            for i in range(4):
                y,m=bits(types,i); g=cv_pred('bin',y,m)
                for b in acc:
                    q=g[b][m]>=.5; acc[b].append(np.mean([np.mean(q[y[m]==v]==v) for v in (True,False)])); L[b][:,i]=g[b]
            out[f'pdb_{who}_letters']={b:round(float(np.mean(v)),4) for b,v in acc.items()}
            if who=='crowd': oof['L']=L
        M=np.array([t is not None for t in crowd]); yt=np.array([t or MBTI[0] for t in crowd]); g=cv_pred('multi',yt,M); idx=np.array([MBTI.index(t) for t in yt])[M]
        Pc=pool(np,letters_to_types(np,oof['L']['crop']),(g['crop']+1e-4)/(g['crop']+1e-4).sum(1,keepdims=True),letter_weight(np,[words(t) for t in sets['crop'][0]]))[M]
        out['pdb_crowd_type_crop']={'exact':round(float(np.mean(Pc.argmax(1)==idx)),4),'top3':round(float(np.mean([i in np.argsort(-q)[:3] for q,i in zip(Pc,idx)])),4)}
        # Fitted on all PDB profiles: Essays, PANDORA, portraits, leak probe
        allr=np.arange(len(ids))
        reg=[fit_halves('reg',[Dm[k] for k in allr[R]],Em[R],Fm[R],np.array([data[n]['ipip'][k] for n in np.array(ids)[R]]),feats) for k in 'OCEAN']
        Dx,Ex,Fx=sets['essays']; pe=np.column_stack([predict('reg',f,Dx,Ex,Fx) for f in reg])
        out['essays_accuracy']=round(float(np.mean([np.mean((pe[:,i]>np.median(pe[:,i]))==ey[:,i]) for i in range(5)])),4)
        out['essays_auc']=round(float(np.mean([roc_auc_score(ey[:,i],pe[:,i]) for i in range(5)])),4)
        Dx,Ex,Fx=sets['pandora']; pp=np.column_stack([predict('reg',f,Dx,Ex,Fx) for f in reg])
        out['pandora_r']=round(float(np.mean([spearmanr(pp[:,i],py[:,i])[0] for i in range(5)])),4)
        lets=[]
        for i in range(4):
            y,m=bits(crowd,i); lets.append(fit_halves('bin',[Dm[k] for k in allr[m]],Em[m],Fm[m],y[m],feats))
        Lp=np.column_stack([predict('bin',f,*sets['portraits']) for f in lets]); Ll=np.column_stack([predict('bin',f,*sets['leak']) for f in lets])
        out['portraits_SN']=round(float(np.mean((Lp[:,1]>=.5)==np.array([t[1]=='S' for t in PY]))),4)
        out['leak_shift']=round(float(np.mean(np.abs(Lp-Ll))),4)
        res[variant]=out; print(json.dumps({'variant':variant,**out}),flush=True)
    b=res['baseline']; res['ship']={}
    for v in VARIANTS[1:]:
        o=res[v]; gain=(o['essays_accuracy']-b['essays_accuracy']>=.005) or (o['pdb_big5_r']['crop']-b['pdb_big5_r']['crop']>=.01)
        drops=[o[k][s]-b[k][s] for k in ('pdb_big5_r','pdb_crowd_letters','pdb_agent_letters') for s in ('full','crop')]+[o['pdb_crowd_type_crop'][s]-b['pdb_crowd_type_crop'][s] for s in ('exact','top3')]
        res['ship'][v]={'gain':bool(gain),'no_drop':bool(min(drops)>=-.005),'leak_ok':bool(o['leak_shift']-b['leak_shift']<=.01)}
        res['ship'][v]['ship']=all(res['ship'][v].values())
    print(json.dumps({'ship':res['ship']}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
