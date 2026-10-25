"""Local personality panel: dataset aggregates plus a text assessor (TF-IDF + E5, trained at startup). Aggregates only;
assessed text is processed in memory and never stored."""
from collections import Counter
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
from .audits import letter_share,spearman
from .battery import load_ipip
from .encoder_text import clean,encoder_view
from .text import words

STATIC=Path(__file__).resolve().parent/'panel_static'
DOMAINS='OCEAN'
DOMAIN_NAMES={'O':'Openness','C':'Conscientiousness','E':'Extraversion','A':'Agreeableness','N':'Neuroticism'}
MBTI=[a+b+c+d for a in 'EI' for b in 'SN' for c in 'TF' for d in 'JP']
LETTERS=[('E','I'),('S','N'),('T','F'),('J','P')]
SAMPLE_JOIN="LEFT JOIN crawl_queue q ON q.profile_num=p.profile_num"
JEV_LABELS=Path('data/stage2/llm/typesafe--jev-1.14')
AGENT_FORMM=Path('data/stage2/formm-v3b-cap8k')   # agent Form M v3b results (tools/formm_v2.py); prompt A's form_m if too few
ABLATION=Path('data/stage2/type_ablation.json')

FIRST_PERSON=frozenset("i me my mine myself i'm im i've ive i'd i'll".split())

def style_stats(text):
    """(share of first-person words, share of repeated lines) of a raw text: cues that it is not a self-description."""
    toks=re.findall(r"[a-z']+",text.lower()); lines=[l.strip().lower() for l in text.splitlines() if l.strip()]
    fp=sum(t in FIRST_PERSON for t in toks)/max(len(toks),1)
    return fp,(1-len(set(lines))/len(lines) if len(lines)>=8 else 0.0)

def letter_weight(np,n_words):
    """Weight of the letter heads in the pool: 0.8 at <=300 words, 0.4 at >=2000 (tools/type_ablation.py: on 250-word texts
    the 16-type head contradicts a >=70%-sure letter head 37-40% of the time at 0.4, under 1% at 0.8)."""
    x=np.log10(np.clip(np.asarray(n_words,dtype=float),300,2000)/300)/np.log10(2000/300); return .8-.4*x

def jev_probs(n):
    f=JEV_LABELS/f'{n}.json'
    if not f.exists(): return None
    p=json.loads(f.read_text(encoding='utf-8'))['probabilities']; v=[max(float(p.get(t,0)),0) for t in MBTI]; s=sum(v)
    return [x/s for x in v] if s else None

def letters_to_types(np,P):
    """(n,4) first-letter probabilities -> (n,16) type probabilities, letters independent."""
    out=np.ones((len(P),16))
    for j,t in enumerate(MBTI):
        for i,(a,_) in enumerate(LETTERS): out[:,j]*=np.where(t[i]==a,P[:,i],1-P[:,i])
    return out

def pool(np,A,B,w):
    w=np.reshape(w,(-1,1)) if np.ndim(w) else w
    L=w*np.log(np.clip(A,1e-9,1))+(1-w)*np.log(np.clip(B,1e-9,1)); L-=L.max(1,keepdims=True); E=np.exp(L)
    return E/E.sum(1,keepdims=True)

def soft_fit(np,pre,clf,X,targets,class_weight=None):
    """Multinomial fit on soft targets. pre (TF-IDF or scaler) is fitted once on the unique profiles, then each (profile, type)
    row with target > 1e-3 is weighted by that target (times class_weight[type] if given). Returns (pre, clf)."""
    Z=pre.fit_transform(X); cw=np.ones(16) if class_weight is None else class_weight
    rows=[(i,j,w*cw[j]) for i,t in enumerate(targets) for j,w in enumerate(t) if w>1e-3]
    clf.fit(Z[[i for i,_,_ in rows]],np.array(MBTI)[[j for _,j,_ in rows]],sample_weight=np.array([w for _,_,w in rows]))
    return pre,clf

def soft_binary_fit(np,pre,clf,X,p):
    """Binary fit on soft labels p (probability of the first letter): each profile appears as a positive row weighted p and a
    negative row weighted 1-p, with classes balanced by their total weight. pre is fitted once on the unique profiles."""
    Z=pre.fit_transform(X); n=len(p); wp,wn=n/(2*max(p.sum(),1e-9)),n/(2*max((1-p).sum(),1e-9))
    idx=np.r_[np.arange(n),np.arange(n)]; y=np.r_[np.ones(n),np.zeros(n)]; w=np.r_[p*wp,(1-p)*wn]; keep=w>1e-6
    clf.fit(Z[idx[keep]],y[keep],sample_weight=w[keep]); return pre,clf

def binary_prob(models,D,E):
    return sum(clf.predict_proba(pre.transform(X))[:,1] for (pre,clf),X in zip(models,(D,E)))/2

def type_probs(np,models,D,E):
    T=np.zeros((len(E),16))
    for (pre,clf),X in zip(models,(D,E)): T[:,[MBTI.index(c) for c in clf.classes_]]+=clf.predict_proba(pre.transform(X))/2
    return (T+1e-4)/(T+1e-4).sum(1,keepdims=True)

def prior_correct(np,J,prior):
    """Rescale teacher probabilities so their mean matches prior (removes the teacher's type-frequency skew)."""
    R=J*(np.asarray(prior)/np.clip(J.mean(0),1e-6,None)); return R/R.sum(1,keepdims=True)

def l2(np,X):
    X=np.asarray(X,dtype=float); n=np.linalg.norm(X,axis=-1,keepdims=True); return X/np.where(n==0,1,n)

def modal(counts):
    return max(counts,key=lambda k:(counts[k],k)) if counts and sum(counts.values()) else None

def load_rows(conn,samples=('top',),prompt_version='stage3-v1'):
    """profile_num -> {'mbti_counts','ennea_counts','ipip' (raw domains), 'pct' (percentiles), 'form_m', 'agent_ennea'}."""
    marks=','.join('?'*len(samples)); rows={}
    for (n,) in conn.execute(f"SELECT p.profile_num FROM profiles p {SAMPLE_JOIN} WHERE p.complete AND coalesce(q.sample,'top') IN ({marks})",list(samples)).fetchall():
        rows[n]={}
    for n,system,counts,eligible in conn.execute('SELECT profile_num,system_name,vote_counts_json,training_eligible FROM vote_distributions').fetchall():
        if n in rows and eligible and system in ('MBTI','ENNEAGRAM'): rows[n]['mbti_counts' if system=='MBTI' else 'ennea_counts']=json.loads(counts)
    for n,domains,form_m,ennea in conn.execute('''SELECT r.profile_num,e.ipip_domains_json,e.form_m_mbti,e.agent_self_enneagram FROM agent_runs r
            JOIN psychometric_evaluations e ON e.run_id=r.run_id WHERE r.complete AND r.prompt_version=? ORDER BY r.seed''',[prompt_version]).fetchall():
        if n in rows and 'ipip' not in rows[n]:
            d=json.loads(domains)
            rows[n].update(ipip={k:d[k]['raw'] for k in DOMAINS},pct={k:d[k]['percentile'] for k in DOMAINS},form_m=form_m,agent_ennea=ennea)
    return rows

def agent_types(ids):
    """profile -> the agent's Form M type and its source: v3b results when they cover >= 200 of ids, else prompt A's."""
    v3b={int(f.stem):json.loads(f.read_text(encoding='utf-8'))['scored']['type'] for f in AGENT_FORMM.glob('*.json')} if AGENT_FORMM.exists() else {}
    return ({n:v3b[n] for n in ids if n in v3b},'Form M v3b') if sum(n in v3b for n in ids)>=200 else (None,'Form M prompt A')

def build_summary(conn,samples=('top',)):
    rowmap=load_rows(conn,samples); rows=rowmap.values(); agent,source=agent_types(rowmap)
    crowd_mbti=Counter(modal(r['mbti_counts']) for r in rows if 'mbti_counts' in r)
    form_m=Counter(agent.values()) if agent else Counter(r['form_m'] for r in rows if 'form_m' in r)
    crowd_core=Counter(modal(r['ennea_counts'])[0] for r in rows if 'ennea_counts' in r)
    agent_core=Counter(r['agent_ennea'][0] for r in rows if r.get('agent_ennea'))
    with_ipip=[r for r in rows if 'ipip' in r]
    hist={k:[0]*10 for k in DOMAINS}
    for r in with_ipip:
        for k in DOMAINS: hist[k][min(9,int(r['pct'][k]//10))]+=1
    big5_corr=[[spearman([r['ipip'][a] for r in with_ipip],[r['ipip'][b] for r in with_ipip]) for b in DOMAINS] for a in DOMAINS]
    both=[r for r in with_ipip if 'mbti_counts' in r]
    pairs=[('E-I','E'),('S-N','N'),('T-F','T'),('J-P','J')]
    letter_corr=[[spearman([letter_share(r['mbti_counts'],d,l) for r in both],[r['ipip'][k] for r in both]) for d,l in pairs] for k in DOMAINS]
    cross=Counter((modal(r['mbti_counts']),modal(r['ennea_counts'])[0]) for r in rows if 'mbti_counts' in r and 'ennea_counts' in r)
    return {'samples':list(samples),'profiles':len(rows),
            'mbti':{'types':MBTI,'crowd':[crowd_mbti.get(t,0) for t in MBTI],'form_m':[form_m.get(t,0) for t in MBTI],'form_m_source':source},
            'enneagram':{'cores':list('123456789'),'crowd':[crowd_core.get(c,0) for c in '123456789'],'agent':[agent_core.get(c,0) for c in '123456789']},
            'big5':{'domains':list(DOMAINS),'names':DOMAIN_NAMES,'n':len(with_ipip),'histograms':hist},
            'big5_correlation':{'domains':list(DOMAINS),'matrix':big5_corr,'n':len(with_ipip)},
            'crowd_letters_vs_big5':{'domains':list(DOMAINS),'letters':['% E votes','% N votes','% T votes','% J votes'],'matrix':letter_corr,'n':len(both)},
            'mbti_by_enneagram':{'types':MBTI,'cores':list('123456789'),'counts':[[cross.get((t,c),0) for c in '123456789'] for t in MBTI]}}

class Assessor:
    """Predicts agent-style IPIP domains, MBTI letters and Enneagram core from free text: TF-IDF and E5 models averaged.
    Accuracy shown to the user is 5-fold cross-validated with exactly this averaging (TF-IDF fitted inside each fold)."""
    def __init__(self,store,encoder,*,samples=('top',),cache_dir=None,embed_fn=None,folds=5):
        import hashlib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression,RidgeCV
        from sklearn.model_selection import KFold,StratifiedKFold,cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from . import embed
        self.np=np; self.encoder=encoder; self.embed_fn=embed_fn or self.embed
        rows=load_rows(store.connection,samples); docs={}; vecs={}
        for n,r in rows.items():
            if 'ipip' not in r and 'mbti_counts' not in r: continue
            views=encoder_view(store,n)
            docs[n]=' '.join(t[a:b] for t,rs in views for a,b in rs)
            if cache_dir is None: vecs[n]=self.embed_fn(views)
            else:
                sha=hashlib.sha256(json.dumps(views,ensure_ascii=False).encode('utf-8')).hexdigest()
                vecs[n]=embed.cached_vector(cache_dir,n,sha,lambda:self.embed_fn(views))
        ids=[n for n in docs if vecs[n] is not None and words(docs[n])>0]
        alphas=np.logspace(-1,4,12)
        tfidf=lambda:TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True)
        def data(sub): return [docs[n] for n in sub],l2(np,[vecs[n] for n in sub])
        cv=KFold(folds,shuffle=True,random_state=0)
        reg_ids=[n for n in ids if 'ipip' in rows[n]]; D,E=data(reg_ids)
        self.ipip={}; self.ipip_cv={}; self.ipip_sample={}; norms=load_ipip().load_norms()['scales']
        fp=[style_stats(docs[n])[0] for n in ids]; self.first_person_p5=float(np.quantile(fp,.05)) if fp else 0.0
        for k in DOMAINS:
            y=np.array([rows[n]['ipip'][k] for n in reg_ids])
            mt,me=make_pipeline(tfidf(),RidgeCV(alphas=alphas)),make_pipeline(StandardScaler(),RidgeCV(alphas=alphas))
            pred=(cross_val_predict(mt,D,y,cv=cv)+cross_val_predict(me,E,y,cv=cv))/2
            self.ipip_cv[k]={'spearman':spearman(list(pred),list(y)),'rmse':float(np.sqrt(np.mean((pred-y)**2))),'n':len(y)}
            self.ipip[k]=(mt.fit(D,y),me.fit(E,y),norms[k]); self.ipip_sample[k]=np.sort(y)
        def fit_letters(sub,P):
            """Per-letter heads trained on (n,4) first-letter probabilities P (0/1 for hard labels, a teacher's marginals for soft
            ones); returns {pair: models} and out-of-fold probabilities. Letters with too little of one class are left out."""
            D,E=data(sub); heads={}; oof=np.full((len(sub),4),.5)
            new=lambda:((tfidf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=1,max_iter=3000)))
            for i,(a,b) in enumerate(LETTERS):
                p=P[:,i]
                if min(p.sum(),(1-p).sum())<folds: continue
                for tr,te in cv.split(sub):
                    ms=[soft_binary_fit(np,pre,clf,X,p[tr]) for (pre,clf),X in zip(new(),([D[k] for k in tr],E[tr]))]
                    oof[te,i]=binary_prob(ms,[D[k] for k in te],E[te])
                heads[a+b]=[soft_binary_fit(np,pre,clf,X,p) for (pre,clf),X in zip(new(),(D,E))]
            return heads,oof
        def fit_types(sub,T):
            """16-type head on (n,16) soft targets T; returns models and out-of-fold probabilities."""
            D,E=data(sub); P=np.zeros((len(sub),16))
            new=lambda:((tfidf(),LogisticRegression(C=4,max_iter=3000)),(StandardScaler(),LogisticRegression(C=.3,max_iter=3000)))
            for tr,te in cv.split(sub):
                ms=[soft_fit(np,pre,clf,X,T[tr]) for (pre,clf),X in zip(new(),([D[i] for i in tr],E[tr]))]
                P[te]=type_probs(np,ms,[D[i] for i in te],E[te])
            return [soft_fit(np,pre,clf,X,T) for (pre,clf),X in zip(new(),(D,E))],P
        def scores(P,truth):
            """Exact / top-3 of (n,16) probabilities against hard types, plus per-letter balanced accuracy."""
            idx=np.array([MBTI.index(t) for t in truth]); top=P.argmax(1); bal=[]
            for i,(a,_) in enumerate(LETTERS):
                y=np.array([t[i]==a for t in truth]); hat=np.array([MBTI[k][i]==a for k in top])
                bal.append(float(np.mean([np.mean(hat[y==v]==v) for v in (True,False) if (y==v).any()])))
            return {'exact':float(np.mean(top==idx)),'top3':float(np.mean([i in np.argsort(-p)[:3] for p,i in zip(P,idx)])),
                    'letters_balanced':bal,'chance':max(Counter(truth).values())/len(truth),'n':len(truth)}
        def card(sub,letter_p,type_t,truths):
            """Letter heads (+ optional 16-type head) pooled by text length; CV scored against each named truth."""
            heads,loof=fit_letters(sub,letter_p); L=letters_to_types(np,loof)
            types,toof=fit_types(sub,type_t) if type_t is not None else (None,None)
            P=pool(np,L,toof,letter_weight(np,[words(docs[n]) for n in sub])) if types else L
            return {'letters':heads,'types':types,'cv':{name:scores(P,t) for name,t in truths.items()},'n':len(sub)}
        def letter_onehot(types): return np.array([[t[i]==a for i,(a,_) in enumerate(LETTERS)] for t in types],dtype=float)
        # Crowd card: modal crowd vote, letters + 16-type head.
        sub=[n for n in ids if 'mbti_counts' in rows[n]]; crowd=[modal(rows[n]['mbti_counts']) for n in sub]
        self.crowd=card(sub,letter_onehot(crowd),np.eye(16)[[MBTI.index(t) for t in crowd]],{'crowd':crowd})
        # Jev card: Jev 1.14's probabilities as the teacher (letter marginals and all 16), scored against Jev and the crowd.
        sub=[n for n in sub if jev_probs(n)]; self.jev=None
        if len(sub)>=folds*4:
            J=np.array([jev_probs(n) for n in sub])
            JL=np.array([[J[k,[j for j,t in enumerate(MBTI) if t[i]==a]].sum() for i,(a,_) in enumerate(LETTERS)] for k in range(len(sub))])
            self.jev=card(sub,JL,J,{'jev':[MBTI[k] for k in J.argmax(1)],'crowd':[modal(rows[n]['mbti_counts']) for n in sub]})
        # Agent card: the DeepSeek agent's Form M type (v3b when available for enough profiles), letters only.
        v3b,source=agent_types(ids); agent=(lambda n:v3b.get(n)) if v3b else (lambda n:rows[n].get('form_m'))
        sub=[n for n in ids if agent(n)]; at=[agent(n) for n in sub]
        self.agent=card(sub,letter_onehot(at),None,{'agent':at}); self.agent['source']=source
        def fit_cores(label):
            """Multiclass Enneagram-core head; (None, n, reason) when the data cannot support it."""
            sub=[n for n in ids if label(rows[n])]; D,E=data(sub); y=np.array([label(rows[n])[0] for n in sub]); counts=Counter(y)
            if len(counts)<2: return None,len(sub),f'only one class in training data ({next(iter(counts),"none")})'
            mt=make_pipeline(tfidf(),LogisticRegression(C=4,max_iter=3000)); me=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=3000))
            scv=StratifiedKFold(max(2,min(folds,min(counts.values()))),shuffle=True,random_state=0)
            p=(cross_val_predict(mt,D,y,cv=scv,method='predict_proba')+cross_val_predict(me,E,y,cv=scv,method='predict_proba'))/2
            cores=sorted(counts)
            cv_=({'top1':float(np.mean(np.array(cores)[p.argmax(1)]==y)),'chance':max(counts.values())/len(y),'n':len(y)})
            return (mt.fit(D,y),me.fit(E,y),cores,cv_),len(sub),None
        crowd_core=lambda r:modal(r['ennea_counts']) if 'ennea_counts' in r else None
        a=json.loads(ABLATION.read_text(encoding='utf-8'))['A'] if ABLATION.exists() else {}   # 250-word-crop accuracy, if measured
        self.short_cv={'type':a.get('l2/crop/fixed+prior/w0.8'),'ipip_spearman':(a.get('l2/ipip_mean_spearman') or {}).get('crop')} if a else None
        self.ennea,n_ennea,self.ennea_reason=fit_cores(crowd_core)
        self.agent_ennea,n_agent_ennea,self.agent_ennea_reason=fit_cores(lambda r:r.get('agent_ennea'))
        self.trained_on={'regression':len(reg_ids),'crowd_type':self.crowd['n'],'jev_type':self.jev['n'] if self.jev else 0,'agent_type':self.agent['n'],
                         'enneagram':n_ennea,'agent_enneagram':n_agent_ennea}

    def embed(self,views):
        from . import embed
        return embed.profile_vector(embed.embed_views(views,self.encoder))

    def assess(self,text):
        np=self.np; scrubbed=clean(text,True); n_words=words(scrubbed)
        if not n_words: return {'error':'No words left after removing typology labels.'}
        D=[scrubbed]; E=l2(np,[self.embed_fn([(scrubbed,[(0,len(scrubbed))])])])
        big5={}
        for k,(mt,me,norm) in self.ipip.items():
            raw=float((mt.predict(D)[0]+me.predict(E)[0])/2); rmse=self.ipip_cv[k]['rmse']
            sample=self.ipip_sample[k]; pdb=100*(np.searchsorted(sample,raw,'left')+np.searchsorted(sample,raw,'right'))/2/len(sample)
            big5[k]={'name':DOMAIN_NAMES[k],'raw':raw,'percentile':percentile(raw,norm),'low':percentile(raw-rmse,norm),
                     'high':percentile(raw+rmse,norm),'pdb_percentile':float(pdb),'cv':self.ipip_cv[k]}
        def cores(head,reason):
            if head is None: return {'unavailable':reason}
            mt,me,names,cv_=head; probs=(mt.predict_proba(D)[0]+me.predict_proba(E)[0])/2
            return {'cores':names,'probabilities':[float(x) for x in probs],'cv':cv_}
        def card(c):
            L=np.array([[float(binary_prob(c['letters'][a+b],D,E)[0]) if a+b in c['letters'] else .5 for a,b in LETTERS]])
            P=letters_to_types(np,L)
            if c['types']: P=pool(np,P,type_probs(np,c['types'],D,E),letter_weight(np,n_words))
            top=np.argsort(-P[0])[:3]; head=MBTI[top[0]]
            letter_type=''.join(a if L[0,i]>=.5 else b for i,(a,b) in enumerate(LETTERS))
            clash=[{'letter':letter_type[i],'p':float(max(L[0,i],1-L[0,i]))} for i in range(4) if abs(L[0,i]-.5)>=.2 and head[i]!=letter_type[i]]
            return {'type':head,'top':[{'type':MBTI[i],'p':float(P[0,i])} for i in top],'letter_type':letter_type,'contradicts':clash,
                    'letters':{a+b:float(L[0,i]) for i,(a,b) in enumerate(LETTERS)},'cv':c['cv'],'n':c['n']}
        agent=card(self.agent); agent['source']=self.agent['source']
        fp,rep=style_stats(text); genre=[]
        if rep>=.2: genre.append(f'{rep:.0%} of lines repeat, like lyrics or a poem')
        if fp<self.first_person_p5: genre.append(f'little first-person language ({fp:.1%} of words, below 95% of the PDB self-descriptions the models learned from)')
        return {'words':n_words,'short_text':n_words<150,'not_self_description':genre,'big5':big5,'short_text_cv':self.short_cv,
                'mbti':{'crowd':card(self.crowd),'jev':card(self.jev) if self.jev else None,'agent':agent},
                'enneagram':cores(self.ennea,self.ennea_reason),'agent_enneagram':cores(self.agent_ennea,self.agent_ennea_reason),
                'trained_on':self.trained_on}

MODEL_CACHE=Path('data/stage2/panel_models')

def fingerprint(db,samples,encoder_name):
    """Changes whenever anything the assessor learns from changes: the database file, teacher labels, agent Form M results,
    ablation numbers, the panel/embedding code, or library versions."""
    import hashlib,numpy,sklearn
    h=hashlib.sha256(); st=Path(db).stat()
    h.update(f'{Path(db).resolve()}|{st.st_size}|{st.st_mtime_ns}|{tuple(samples)}|{encoder_name}|{sklearn.__version__}|{numpy.__version__}'.encode())
    for f in (Path(__file__),Path(__file__).with_name('embed.py'),Path(__file__).with_name('encoder_text.py'),ABLATION):
        if f.exists(): h.update(f.read_bytes())
    for d in (JEV_LABELS,AGENT_FORMM):
        if d.exists(): h.update('|'.join(sorted(f'{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}' for f in d.glob('*.json'))).encode())
    return h.hexdigest()[:16]

def cached_assessor(store,encoder,db,samples,cache_dir,**kw):
    """Assessor trained once per fingerprint and pickled under MODEL_CACHE (only the newest is kept); later starts load it."""
    import pickle
    import numpy as np
    f=MODEL_CACHE/f'{fingerprint(db,samples,encoder[2]["model"])}.pkl'
    if f.exists(): a=Assessor.__new__(Assessor); a.__dict__.update(pickle.loads(f.read_bytes())); a.loaded_from=str(f)
    else:
        a=Assessor(store,encoder,samples=samples,cache_dir=cache_dir,**kw)
        MODEL_CACHE.mkdir(parents=True,exist_ok=True)
        for old in MODEL_CACHE.glob('*.pkl'): old.unlink()
        f.write_bytes(pickle.dumps({k:v for k,v in a.__dict__.items() if k not in ('np','encoder','embed_fn')})); a.loaded_from=None
    a.np=np; a.encoder=encoder; a.embed_fn=kw.get('embed_fn') or a.embed
    return a

def percentile(raw,norm):
    counts,minimum=norm['counts'],norm['minimum']
    r=min(max(int(round(raw)),minimum),minimum+len(counts)-1); i=r-minimum
    return 100*(sum(counts[:i])+counts[i]/2)/sum(counts)

def make_handler(summary,assessor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass  # never log request bodies or paths with text
        def send(self,code,body,ctype='application/json'):
            data=body if isinstance(body,bytes) else json.dumps(body).encode('utf-8')
            self.send_response(code); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path in ('/','/index.html'): return self.send(200,(STATIC/'index.html').read_bytes(),'text/html; charset=utf-8')
            if self.path=='/api/summary': return self.send(200,summary)
            self.send(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/api/assess': return self.send(404,{'error':'not found'})
            length=int(self.headers.get('Content-Length') or 0)
            if length>200_000: return self.send(413,{'error':'Text too long (200 kB max).'})
            try: text=json.loads(self.rfile.read(length) or b'{}').get('text','')
            except ValueError: return self.send(400,{'error':'Invalid JSON'})
            if not isinstance(text,str) or not text.strip(): return self.send(400,{'error':'Type a self-description first.'})
            self.send(200,assessor.assess(text))
    return Handler

def serve(summary,assessor,host='127.0.0.1',port=8765):
    server=ThreadingHTTPServer((host,port),make_handler(summary,assessor))
    print(json.dumps({'panel':f'http://{host}:{port}/'}),flush=True)
    server.serve_forever()
