"""Can the DeepSeek agent's S/N (Form M v3b) improve the crowd card's S/N? Database read-only.

  python tools/sn_teacher_experiment.py --db data/pdb-top.duckdb --portraits "<first-person folder>" "<third-person folder>"

Heads are the panel's (TF-IDF C=4 + encoder C=.003, averaged; class-balanced). Two ways to use the agent:
  blend b:  S/N probability = (1-b) * crowd head + b * agent head (agent head trained on v3b S/N), b in 0..1;
  soft a:   crowd S/N head trained on soft targets a * crowd + (1-a) * agent S/N where v3b exists (crowd alone elsewhere).
Scored in 5-fold CV over profiles against the crowd's S/N (full profiles and 250-word excerpts) and the creator's own badge,
and on portraits (S/N accuracy). The headline type pools the four letters (crowd letters with the new S/N) with the crowd
16-type head as in the panel; its excerpt exact/top-3 and the S types it predicts for portraits are reported too.
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
    p.add_argument('--output',type=Path,default=Path('data/stage2/sn_teacher_experiment.json'))
    args=p.parse_args()
    data=load(args.db,('top','top2')); ids=sorted(n for n in data if data[n]['crowd'])
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); vpath=Path('data/stage2')/f'excerpt_vectors_{embed.model_slug(enc[2]["model"])}.npz'
    D=[data[n]['doc'] for n in ids]; F=l2(np,[data[n]['vec'] for n in ids]); C=[data[n]['crop'] for n in ids]; Ec=vectors(C,enc,vpath)
    PT=[]; PY=[]
    for folder in args.portraits:
        for t in MBTI:
            f=next(iter(folder.glob(f'{t}.*')),None)
            if f: PT.append(clean(portrait(f,'strict'),True)); PY.append(t)
    Ep=l2(np,[embed.profile_vector(embed.embed_views([(t,[(0,len(t))])],enc)) for t in PT]); PY=np.array(PY)
    v3b={int(f.stem):json.loads(f.read_text(encoding='utf-8'))['scored']['type'] for f in panel.AGENT_FORMM.glob('*.json')}
    crowd=np.array([data[n]['crowd'] for n in ids]); agent=np.array([v3b.get(n,'') for n in ids]); selfv=np.array([data[n]['self'] or '' for n in ids])
    tf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True); cv=list(KFold(5,shuffle=True,random_state=0).split(ids))
    new=lambda:((tf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=ENCODER_LETTER_C,max_iter=3000)))
    def head(target,rows,i):
        """Letter head on soft targets for rows; returns predict(D,E)."""
        ms=[soft_binary_fit(np,pre,clf,X,target[rows]) for (pre,clf),X in zip(new(),([D[k] for k in rows],F[rows]))]
        return lambda Dx,Ex:sum(clf.predict_proba(pre.transform(X))[:,1] for (pre,clf),X in zip(ms,(Dx,Ex)))/2
    first=lambda types,i:np.array([t[i]==LETTERS[i][0] if t else np.nan for t in types],dtype=float)
    out={k:np.zeros((len(ids),4)) for k in ('crowd_full','crowd_crop','agent_full','agent_crop')}; port={'crowd':np.zeros((len(PT),4)),'agent':np.zeros((len(PT),4))}
    soft={a:{'full':np.zeros(len(ids)),'crop':np.zeros(len(ids)),'port':None} for a in (.75,.5)}
    for i in range(4):
        yc=first(crowd,i); ya=first(agent,i); has=~np.isnan(ya)
        for tr,te in cv+[(np.arange(len(ids)),None)]:
            hc=head(yc,tr,i); ha=head(ya,tr[has[tr]],i)
            if te is None: port['crowd'][:,i]=hc(PT,Ep); port['agent'][:,i]=ha(PT,Ep)
            else:
                out['crowd_full'][te,i]=hc([D[k] for k in te],F[te]); out['crowd_crop'][te,i]=hc([C[k] for k in te],Ec[te])
                out['agent_full'][te,i]=ha([D[k] for k in te],F[te]); out['agent_crop'][te,i]=ha([C[k] for k in te],Ec[te])
            if i==1:
                for a in soft:
                    tgt=np.where(has,a*yc+(1-a)*np.nan_to_num(ya),yc); hs=head(tgt,tr,i)
                    if te is None: soft[a]['port']=hs(PT,Ep)
                    else: soft[a]['full'][te]=hs([D[k] for k in te],F[te]); soft[a]['crop'][te]=hs([C[k] for k in te],Ec[te])
    # crowd 16-type head (encoder C as in the panel), out of fold on excerpts and full-data on portraits
    T={'crop':np.zeros((len(ids),16)),'port':np.zeros((len(PT),16))}
    for tr,te in cv+[(np.arange(len(ids)),None)]:
        t=tf().fit([D[k] for k in tr]); mt=LogisticRegression(C=4,max_iter=3000).fit(t.transform([D[k] for k in tr]),crowd[tr])
        s=StandardScaler().fit(F[tr]); me=LogisticRegression(C=ENCODER_TYPE_C,max_iter=3000).fit(s.transform(F[tr]),crowd[tr]); cols=[MBTI.index(k) for k in mt.classes_]
        Dx,Ex,key,rows=(PT,Ep,'port',slice(None)) if te is None else ([C[k] for k in te],Ec[te],'crop',te)
        P=np.zeros((len(Ex),16)); P[:,cols]=(mt.predict_proba(t.transform(Dx))+me.predict_proba(s.transform(Ex)))/2; T[key][rows]=(P+1e-4)/(P+1e-4).sum(1,keepdims=True)
    def bal(p,truth):
        y=first(truth,1); m=~np.isnan(y); y=y[m].astype(bool); q=p[m]>=.5
        return round(float(np.mean([np.mean(q[y==v]==v) for v in (True,False)])),3)
    wc=letter_weight(np,[words(c) for c in C]); wp=letter_weight(np,[words(t) for t in PT]); res={}
    def headline(L,Tm,w,truth):
        P=pool(np,letters_to_types(np,L),Tm,w); idx=np.array([MBTI.index(t) for t in truth]); top=P.argmax(1)
        return {'exact':round(float(np.mean(top==idx)),3),'top3':round(float(np.mean([i in np.argsort(-q)[:3] for q,i in zip(P,idx)])),3),'S_predicted':int(sum(MBTI[k][1]=='S' for k in top))}
    def record(name,sn_full,sn_crop,sn_port):
        Lc=out['crowd_crop'].copy(); Lc[:,1]=sn_crop; Lp=port['crowd'].copy(); Lp[:,1]=sn_port
        res[name]={'SN_vs_crowd_full':bal(sn_full,crowd),'SN_vs_crowd_excerpt':bal(sn_crop,crowd),'SN_vs_self_excerpt':bal(sn_crop,selfv),
                   'SN_portraits':round(float(np.mean((sn_port>=.5)==np.array([t[1]=='S' for t in PY]))),3),
                   'headline_excerpt':headline(Lc,T['crop'],wc,crowd),'headline_portraits':headline(Lp,T['port'],wp,PY)}
        print(json.dumps({'variant':name,**res[name]}),flush=True)
    for b in (0,.25,.5,.75,1):
        mix=lambda a,c:(1-b)*a+b*c
        record(f'blend b={b}',mix(out['crowd_full'][:,1],out['agent_full'][:,1]),mix(out['crowd_crop'][:,1],out['agent_crop'][:,1]),mix(port['crowd'][:,1],port['agent'][:,1]))
    for a,v in soft.items(): record(f'soft a={a}',v['full'],v['crop'],v['port'])
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
