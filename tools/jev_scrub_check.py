"""Does Jev's long-text advantage survive stricter scrubbing (Socionics, SLOAN, function positions)? Database read-only.

  python tools/jev_scrub_check.py --db data/pdb-top.duckdb [--n 100 --min-words 2500]

Relabels a random sample of long profiles with the current encoder_view (cached under data/stage2/llm-scrub2/) and compares
exact accuracy on the crowd's modal type with the original labels (data/stage2/llm/typesafe--jev-1.14/).
"""
import argparse
import asyncio
import json
from pathlib import Path
import random
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from distill_jev import MODEL,OUT,jev_decide
from run_stage2 import readonly_store
from pdb_ingest import panel
from pdb_ingest.encoder_text import EXTRA_JARGON,encoder_view
from pdb_ingest.llm import load_api_key

NEW=Path('data/stage2/llm-scrub2')/MODEL.replace('/','--')

async def relabel(docs):
    import httpx
    NEW.mkdir(parents=True,exist_ok=True); gate=asyncio.Semaphore(4); cost=[0.0]
    async with httpx.AsyncClient() as http:
        key=load_api_key()
        async def one(n):
            f=NEW/f'{n}.json'
            if f.exists(): return
            async with gate: r=await jev_decide(http,key,' '.join(docs[n].split()[:30000]))
            cost[0]+=(r['usage'] or {}).get('cost',0) or 0; f.write_text(json.dumps(r),encoding='utf-8')
        await asyncio.gather(*(one(n) for n in docs))
    return cost[0]

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--samples',default='top,top2')
    p.add_argument('--n',type=int,default=100); p.add_argument('--min-words',type=int,default=2500)
    args=p.parse_args()
    store=readonly_store(args.db); rows=panel.load_rows(store.connection,tuple(args.samples.split(',')))
    docs={}; y={}; hits={}
    for n,r in rows.items():
        if 'mbti_counts' not in r or not (OUT/f'{n}.json').exists(): continue
        views=encoder_view(store,n); doc=' '.join(t[a:b] for t,rs in views for a,b in rs)
        if len(doc.split())<args.min_words: continue
        docs[n]=doc; y[n]=panel.modal(r['mbti_counts'])
        raw=' '.join(s.raw_text for s in store.load_snapshot(n).segments); hits[n]=len(EXTRA_JARGON.findall(raw))
    pick=sorted(random.Random(0).sample(sorted(docs),min(args.n,len(docs))))
    cost=asyncio.run(relabel({n:docs[n] for n in pick}))
    top=lambda f:max((d:=json.loads(f.read_text(encoding='utf-8'))['probabilities']),key=lambda t:float(d.get(t,0)))
    old=[top(OUT/f'{n}.json')==y[n] for n in pick]; new=[top(NEW/f'{n}.json')==y[n] for n in pick]
    heavy=[n for n in pick if hits[n]>=5]
    res={'eligible_long_profiles':len(docs),'sampled':len(pick),'cost_usd':round(cost,4),
         'exact_original_scrub':round(float(np.mean(old)),3),'exact_strict_scrub':round(float(np.mean(new)),3),
         'profiles_with_5plus_new_jargon_hits':len(heavy),
         'heavy_original':round(float(np.mean([top(OUT/f'{n}.json')==y[n] for n in heavy])),3) if heavy else None,
         'heavy_strict':round(float(np.mean([top(NEW/f'{n}.json')==y[n] for n in heavy])),3) if heavy else None,
         'label_changed':round(float(np.mean([top(OUT/f'{n}.json')!=top(NEW/f'{n}.json') for n in pick])),3)}
    Path('data/stage2/jev_scrub_check.json').write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res,indent=1))

if __name__=='__main__': main()
