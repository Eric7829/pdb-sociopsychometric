"""Run the Stage 2 baseline for one or more encoder views against a database opened READ-ONLY.

  python tools/run_stage2.py --db data/pdb-top.duckdb main unmasked raw
"""
import argparse
import json
from pathlib import Path
import sys
import duckdb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pdb_ingest import distill,embed
from pdb_ingest.store import Store

def readonly_store(path):
    store=Store.__new__(Store); store.connection=duckdb.connect(str(path),read_only=True)
    return store

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--out',type=Path,default=Path('data/stage2'))
    p.add_argument('--model',default=embed.DEFAULT_MODEL,help='Hugging Face encoder (e.g. jinaai/jina-embeddings-v5-text-small-classification)')
    p.add_argument('views',nargs='+',choices=sorted(distill.VIEWS))
    args=p.parse_args()
    store=readonly_store(args.db); encoder=embed.hf_encoder(args.model)
    for view in args.views:
        report=distill.run_distill(store,view,encoder=encoder,out_root=args.out)
        print(json.dumps({'view':view,'embedded':report['embedded'],'metrics':report['metrics']}),flush=True)

if __name__=='__main__': main()
