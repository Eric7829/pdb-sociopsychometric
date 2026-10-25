"""Real self-report check on the Essays dataset (Pennebaker & King; 2,467 student stream-of-consciousness essays, ~660
words, Big Five self-report as above/below-median labels). Read-only; the essays stay in data/external.

  python tools/essays_check.py

Scores the panel's Big Five model and ppp57420/ocean-personality-distilbert by AUC per trait (0.5 = chance): how well
each model's continuous prediction ranks "high" writers above "low" ones. Neither model saw these essays.
"""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ocean_bert_compare import external_model
from pdb_ingest import embed,panel
from pdb_ingest.encoder_text import clean
from pdb_ingest.panel import l2

def main():
    import duckdb
    from sklearn.metrics import roc_auc_score
    P=str(Path('data/external/essays-big5')/'*.parquet').replace('\\','/')
    rows=duckdb.connect().execute(f'select O,C,E,A,"N",text from read_parquet(\'{P}\')').fetchall()
    y=np.array([[int(v) for v in r[:5]] for r in rows]); texts=[r[5] for r in rows]
    a=panel.Assessor.__new__(panel.Assessor); a.__dict__.update(pickle.loads(next(Path('data/stage2/panel_models').glob('*.pkl')).read_bytes()))
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); a.np=np; a.encoder=enc; a.embed_fn=a.embed; ext=external_model()
    ours=[]; theirs=[]
    for t in texts:
        s=clean(t,True); E=l2(np,[a.embed_fn([(s,[(0,len(s))])])])
        ours.append([float((mt.predict([s])[0]+me.predict(E)[0])/2) for mt,me,_ in (a.ipip[k] for k in 'OCEAN')]); theirs.append(ext(t))
    ours=np.array(ours); theirs=np.array(theirs)
    res={'essays':len(texts),'panel_auc':{},'external_auc':{}}
    for i,k in enumerate('OCEAN'):
        res['panel_auc'][k]=round(float(roc_auc_score(y[:,i],ours[:,i])),3); res['external_auc'][k]=round(float(roc_auc_score(y[:,i],theirs[:,i])),3)
    for m in ('panel_auc','external_auc'): res[m+'_mean']=round(float(np.mean(list(res[m].values()))),3)
    Path('data/stage2/essays_check.json').write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res,indent=1))

if __name__=='__main__': main()
