"""Does adding top2 agent labels improve Big Five prediction? Database read-only.

  python tools/more_data_ipip.py --db data/pdb-top.duckdb

Test folds are always Top profiles (5-fold); each fold is predicted by models trained on the other Top folds, either alone
or plus every top2 profile, so both arms are scored on identical held-out profiles. Models: TF-IDF ridge + e5 ridge averaged
(L2-normalised vectors), as in the panel. Scored on full profiles and on 250-word crops (tools/type_ablation.py).
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from type_ablation import load,embed_crops
from pdb_ingest.panel import l2

def main():
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/more_data_ipip.json'))
    args=p.parse_args()
    import duckdb
    sample=dict(duckdb.connect(str(args.db),read_only=True).execute("SELECT profile_num,coalesce(sample,'top') FROM crawl_queue").fetchall())
    data={n:d for n,d in load(args.db,('top','top2')).items() if d['ipip']}
    top=sorted(n for n in data if sample.get(n,'top')=='top'); extra=sorted(n for n in data if sample.get(n)=='top2')
    ids=top+extra; pos={n:i for i,n in enumerate(ids)}
    D=[data[n]['doc'] for n in ids]; C=[data[n]['crop'] for n in ids]
    E=l2(np,[data[n]['vec'] for n in ids]); Ec=l2(np,embed_crops(C))
    alphas=np.logspace(-1,4,12); res={'top_profiles':len(top),'top2_profiles':len(extra),'domains':{}}
    for k in 'OCEAN':
        y=np.array([data[n]['ipip'][k] for n in ids]); pred={(arm,b):np.zeros(len(top)) for arm in ('top_only','top+top2') for b in ('full','crop')}
        for tr,te in KFold(5,shuffle=True,random_state=0).split(top):
            for arm,train in (('top_only',list(tr)),('top+top2',list(tr)+[pos[n] for n in extra])):
                mt=make_pipeline(TfidfVectorizer(min_df=3,max_features=50000,sublinear_tf=True),RidgeCV(alphas=alphas)).fit([D[i] for i in train],y[train])
                me=make_pipeline(StandardScaler(),RidgeCV(alphas=alphas)).fit(E[train],y[train])
                pred[(arm,'full')][te]=(mt.predict([D[i] for i in te])+me.predict(E[te]))/2
                pred[(arm,'crop')][te]=(mt.predict([C[i] for i in te])+me.predict(Ec[te]))/2
        yt=y[:len(top)]
        res['domains'][k]={f'{arm}/{b}':round(float(spearmanr(v,yt)[0]),3) for (arm,b),v in pred.items()}
    res['mean']={key:round(float(np.mean([res['domains'][k][key] for k in 'OCEAN'])),3) for key in res['domains']['O']}
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res,indent=1))

if __name__=='__main__': main()
