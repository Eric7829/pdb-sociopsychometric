"""Leak sensitivity: does prepending typology talk change the panel's types? Portraits stay local; nothing is stored.

  python tools/leak_probe.py "data/portraits-first-person/Medium Level" [--url http://127.0.0.1:8765]

For each portrait, assesses the text alone and with a typology-talk intro that names no type, and reports how often each
card's headline type and letters change, plus the mean shift of each letter probability.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portrait_test import assess,portrait
from pdb_ingest.panel import MBTI

INTRO=("I've been searching for my ennea type for a while, hesitating between type 4 and type 6, probably sx blind, and my "
       "tritype and cognitive functions confuse me too, so I'd like to be typed in any typology system. ")

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder',type=Path); p.add_argument('--url',default='http://127.0.0.1:8765')
    args=p.parse_args(); rows=[]
    for t in MBTI:
        f=next(iter(args.folder.glob(f'{t}.*')),None)
        if not f: continue
        text=portrait(f,'codes'); a=assess(args.url,text)['mbti']; b=assess(args.url,INTRO+text)['mbti']
        rows.append({c:(a[c],b[c]) for c in a if a[c]})
    out={}
    for c in rows[0]:
        pairs=[r[c] for r in rows]
        out[c]={'type_changed':sum(x['type']!=y['type'] for x,y in pairs),'portraits':len(pairs),
                'mean_abs_letter_shift':{k:round(float(np.mean([abs(x['letters'][k]-y['letters'][k]) for x,y in pairs])),3) for k in pairs[0][0]['letters']}}
    print(json.dumps(out,indent=1))

if __name__=='__main__': main()
