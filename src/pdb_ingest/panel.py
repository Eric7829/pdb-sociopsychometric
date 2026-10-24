"""Local personality panel: dataset aggregates plus a text assessor (TF-IDF + E5, trained at startup). Aggregates only;
assessed text is processed in memory and never stored."""
from collections import Counter
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
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

def build_summary(conn,samples=('top',)):
    rows=load_rows(conn,samples).values()
    crowd_mbti=Counter(modal(r['mbti_counts']) for r in rows if 'mbti_counts' in r)
    form_m=Counter(r['form_m'] for r in rows if 'form_m' in r)
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
            'mbti':{'types':MBTI,'crowd':[crowd_mbti.get(t,0) for t in MBTI],'form_m':[form_m.get(t,0) for t in MBTI]},
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
        def data(sub): return [docs[n] for n in sub],np.array([vecs[n] for n in sub])
        cv=KFold(folds,shuffle=True,random_state=0)
        reg_ids=[n for n in ids if 'ipip' in rows[n]]; D,E=data(reg_ids)
        self.ipip={}; self.ipip_cv={}; norms=load_ipip().load_norms()['scales']
        for k in DOMAINS:
            y=np.array([rows[n]['ipip'][k] for n in reg_ids])
            mt,me=make_pipeline(tfidf(),RidgeCV(alphas=alphas)),make_pipeline(StandardScaler(),RidgeCV(alphas=alphas))
            pred=(cross_val_predict(mt,D,y,cv=cv)+cross_val_predict(me,E,y,cv=cv))/2
            self.ipip_cv[k]={'spearman':spearman(list(pred),list(y)),'rmse':float(np.sqrt(np.mean((pred-y)**2))),'n':len(y)}
            self.ipip[k]=(mt.fit(D,y),me.fit(E,y),norms[k])
        def fit_letters(label):
            """Per-letter binary heads for an MBTI label function; letters with one class in the data are left out."""
            sub=[n for n in ids if label(rows[n])]; D,E=data(sub); heads={}
            for i,(a,b) in enumerate(LETTERS):
                y=np.array([label(rows[n])[i]==a for n in sub])
                if len(set(y))<2 or min(y.sum(),(~y).sum())<folds: continue
                mt=make_pipeline(tfidf(),LogisticRegression(C=4,max_iter=3000,class_weight='balanced'))
                me=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,class_weight='balanced'))
                p=(cross_val_predict(mt,D,y,cv=cv,method='predict_proba')[:,1]+cross_val_predict(me,E,y,cv=cv,method='predict_proba')[:,1])/2
                heads[a+b]=(mt.fit(D,y),me.fit(E,y),{'balanced_accuracy':float(np.mean([np.mean((p>=.5)[y==v]==v) for v in (True,False)])),'n':len(y)})
            return heads,len(sub)
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
        crowd_type=lambda r:modal(r['mbti_counts']) if 'mbti_counts' in r else None
        crowd_core=lambda r:modal(r['ennea_counts']) if 'ennea_counts' in r else None
        self.letters,n_letters=fit_letters(crowd_type)
        self.ennea,n_ennea,self.ennea_reason=fit_cores(crowd_core)
        self.agent_letters,n_agent_letters=fit_letters(lambda r:r.get('form_m'))
        self.agent_ennea,n_agent_ennea,self.agent_ennea_reason=fit_cores(lambda r:r.get('agent_ennea'))
        self.trained_on={'regression':len(reg_ids),'letters':n_letters,'enneagram':n_ennea,'agent_letters':n_agent_letters,'agent_enneagram':n_agent_ennea}

    def embed(self,views):
        from . import embed
        return embed.profile_vector(embed.embed_views(views,self.encoder))

    def assess(self,text):
        np=self.np; scrubbed=clean(text,True); n_words=words(scrubbed)
        if not n_words: return {'error':'No words left after removing typology labels.'}
        D=[scrubbed]; E=np.array([self.embed_fn([(scrubbed,[(0,len(scrubbed))])])])
        big5={}
        for k,(mt,me,norm) in self.ipip.items():
            raw=float((mt.predict(D)[0]+me.predict(E)[0])/2); rmse=self.ipip_cv[k]['rmse']
            big5[k]={'name':DOMAIN_NAMES[k],'raw':raw,'percentile':percentile(raw,norm),'low':percentile(raw-rmse,norm),
                     'high':percentile(raw+rmse,norm),'cv':self.ipip_cv[k]}
        def letters(heads):
            out={pair:{'p_first':float((mt.predict_proba(D)[0,1]+me.predict_proba(E)[0,1])/2),'cv':cv_} for pair,(mt,me,cv_) in heads.items()}
            return out,''.join((out[a+b]['p_first']>=.5 and a or b) if a+b in out else 'X' for a,b in LETTERS)
        def cores(head,reason):
            if head is None: return {'unavailable':reason}
            mt,me,names,cv_=head; probs=(mt.predict_proba(D)[0]+me.predict_proba(E)[0])/2
            return {'cores':names,'probabilities':[float(x) for x in probs],'cv':cv_}
        crowd_letters,crowd_type=letters(self.letters); agent_letters,agent_type=letters(self.agent_letters)
        return {'words':n_words,'short_text':n_words<150,'big5':big5,'mbti_letters':crowd_letters,'mbti':crowd_type,
                'enneagram':cores(self.ennea,self.ennea_reason),'agent_mbti_letters':agent_letters,'agent_mbti':agent_type,
                'agent_enneagram':cores(self.agent_ennea,self.agent_ennea_reason),'trained_on':self.trained_on}

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
