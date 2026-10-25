"""Embed every profile the panel learns from (agent IPIP or eligible crowd votes) with an encoder, into the shared cache.

  python tools/embed_all.py --db data/pdb-top.duckdb [--samples top,top2] [--model M] [--pooling mean]

Vectors are keyed by the encoder view's hash, so a scrubber or annotation change re-embeds only what changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from pdb_ingest import embed,panel
from pdb_ingest.encoder_text import encoder_view

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--samples',default='top,top2')
    p.add_argument('--model',default=embed.DEFAULT_MODEL); p.add_argument('--pooling',choices=('mean',),default=None)
    args=p.parse_args()
    store=readonly_store(args.db); rows=panel.load_rows(store.connection,tuple(args.samples.split(',')))
    enc=embed.hf_encoder(args.model,pooling=args.pooling); cache=Path('data/stage2/emb')/embed.model_slug(enc[2]['model'])/'main'
    need=sorted(n for n,r in rows.items() if 'ipip' in r or 'mbti_counts' in r); added=0; t=time.time()
    for n in need:
        v=encoder_view(store,n); sha=hashlib.sha256(json.dumps(v,ensure_ascii=False).encode('utf-8')).hexdigest()
        if (cache/f'{n}_{sha[:16]}.npy').exists(): continue
        embed.cached_vector(cache,n,sha,lambda:embed.profile_vector(embed.embed_views(v,enc))); added+=1
        if added%100==0: print(json.dumps({'embedded':added,'seconds':round(time.time()-t)}),flush=True)
    print(json.dumps({'model':enc[2]['model'],'profiles':len(need),'embedded':added,'cpu_fallbacks':enc[2]['cpu_fallbacks'],'seconds':round(time.time()-t)}))

if __name__=='__main__': main()
