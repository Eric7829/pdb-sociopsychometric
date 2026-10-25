"""Head-to-head on the Big Five: ppp57420/ocean-personality-distilbert vs this project's panel model. Read-only.

  python tools/ocean_bert_compare.py --db data/pdb-top.duckdb [--pandora-users 500]

Test A, PDB profiles: the external model's predictions vs the DeepSeek agent's IPIP (this project's training target), on
full profiles and 250-word excerpts. The panel's own numbers on the same target come from 5-fold CV
(data/stage2/short_text_experiment_main.json), since the panel was trained on these profiles.
Test B, PANDORA (Reddit, self-reported Big Five percentiles): the comment-level re-upload has no author id, so users are
the distinct (O, C, E, A, N) score tuples. For a random sample of test-split users, ~1,000 words of their comments are
scored by both models (the panel never saw PANDORA). Spearman r per trait.
The external model reads at most 512 tokens, so long texts are split into 512-token windows and the predictions averaged.
External files are read from data/external (downloaded from Hugging Face; weights as safetensors only).
"""
import argparse
import json
import pickle
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from type_ablation import load
from pdb_ingest import embed,panel
from pdb_ingest.encoder_text import clean
from pdb_ingest.panel import l2

EXT=Path('data/external'); TRAITS='OCEAN'

def external_model():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    tok=AutoTokenizer.from_pretrained(EXT/'ocean-distilbert'); dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=AutoModelForSequenceClassification.from_pretrained(EXT/'ocean-distilbert',use_safetensors=True).to(dev).eval()
    def predict(text,max_windows=16):
        ids=tok(text,add_special_tokens=False)['input_ids']; wins=[ids[i:i+510] for i in range(0,max(len(ids),1),510)][:max_windows] or [[]]
        out=[]
        for w in wins:
            x=torch.tensor([[tok.cls_token_id]+w+[tok.sep_token_id]],device=dev)
            with torch.no_grad(): out.append(m(input_ids=x).logits[0].float().cpu().numpy())
        return np.mean(out,0)
    return predict

def main():
    from scipy.stats import spearmanr
    import duckdb
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--pandora-users',type=int,default=500)
    p.add_argument('--output',type=Path,default=Path('data/stage2/ocean_bert_compare.json'))
    args=p.parse_args(); ext=external_model(); res={}
    # Test A: PDB
    data={n:d for n,d in load(args.db,('top','top2')).items() if d['ipip']}; ids=sorted(data)
    full=np.array([ext(data[n]['doc']) for n in ids]); crop=np.array([ext(data[n]['crop']) for n in ids])
    ref=json.loads(Path('data/stage2/short_text_experiment_main.json').read_text())['big5']['M0 RidgeCV']
    res['pdb']={'n':len(ids),'external_full':{},'external_crop':{},'panel_cv_full_mean':ref['full'],'panel_cv_crop_mean':ref['crop']}
    for i,k in enumerate(TRAITS):
        y=np.array([data[n]['ipip'][k] for n in ids])
        res['pdb']['external_full'][k]=round(float(spearmanr(full[:,i],y)[0]),3); res['pdb']['external_crop'][k]=round(float(spearmanr(crop[:,i],y)[0]),3)
    for s in ('full','crop'): res['pdb'][f'external_{s}_mean']=round(float(np.mean(list(res['pdb'][f'external_{s}'].values()))),3)
    print(json.dumps({'test':'pdb',**res['pdb']}),flush=True)
    # Test B: PANDORA test split, users = distinct score tuples
    c=duckdb.connect(); P=str(EXT/'pandora-big5'/'test-*.parquet').replace('\\','/')
    users=c.execute(f'select O,C,E,A,"N",count(*) n from read_parquet(\'{P}\') group by 1,2,3,4,5 having count(*)>=20').fetchall()
    random.Random(0).shuffle(users); users=users[:args.pandora_users]
    a=panel.Assessor.__new__(panel.Assessor); a.__dict__.update(pickle.loads(next((Path('data/stage2/panel_models')).glob('*.pkl')).read_bytes()))
    enc=embed.hf_encoder(embed.DEFAULT_MODEL); a.np=np; a.encoder=enc; a.embed_fn=a.embed
    ext_p=[]; ours=[]; truth=[]
    for O,C_,E,A_,N,_ in users:
        rows=c.execute(f'select text from read_parquet(\'{P}\') where O=? and C=? and E=? and A=? and "N"=?',[O,C_,E,A_,N]).fetchall()
        texts=[r[0] for r in rows if r[0]]; random.Random(1).shuffle(texts); doc=[]; w=0
        for t in texts:
            doc.append(t); w+=len(t.split())
            if w>=1000: break
        doc='\n'.join(doc); s=clean(doc,True)
        ext_p.append(ext(doc)); truth.append([O,C_,E,A_,N])
        E_=l2(np,[a.embed_fn([(s,[(0,len(s))])])])
        ours.append([float((mt.predict([s])[0]+me.predict(E_)[0])/2) for k,(mt,me,_) in ((k,a.ipip[k]) for k in TRAITS)])
    ext_p=np.array(ext_p); ours=np.array(ours); truth=np.array(truth)
    res['pandora']={'users':len(users),'words_per_user':1000,
                    'external':{k:round(float(spearmanr(ext_p[:,i],truth[:,i])[0]),3) for i,k in enumerate(TRAITS)},
                    'panel':{k:round(float(spearmanr(ours[:,i],truth[:,i])[0]),3) for i,k in enumerate(TRAITS)}}
    for m in ('external','panel'): res['pandora'][f'{m}_mean']=round(float(np.mean(list(res['pandora'][m].values()))),3)
    print(json.dumps({'test':'pandora',**res['pandora']}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
