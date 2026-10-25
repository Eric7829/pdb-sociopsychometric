"""Sanity test: do the panel's models recover the type described in a 16-type portrait? Portraits stay local (read, sent to the
local panel on 127.0.0.1, never stored).

  python tools/portrait_test.py "<folder with ISTJ.md ... ENTJ.md>" [more folders] [--url http://127.0.0.1:8765]

Scrub levels: 'codes' drops header lines (type name, function stack) and type/function codes (the panel scrubs those too);
'strict' also removes MBTI pole words (introvert, sensing, thinking, judging, ...), since portraits name poles constantly.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pdb_ingest.panel import MBTI

HEADER=re.compile(r'(?i)(flesch|function|introverted|extraverted|extroverted|^\W*\*\*[IE][NS][TF][JP])')
POLES=re.compile(r'(?i)\b(extra|extro|intro)vert\w*|\b(sensing|sensor\w*|intuiti\w*|thinking|thinker\w*|feeling|feeler\w*|judging|judger\w*|perceiv\w*)\b')

def portrait(path,level):
    lines=[l for l in path.read_text(encoding='utf-8',errors='replace').splitlines() if not HEADER.search(l)]
    text=re.sub(r'(?<!\w)[IE][NS][TF][JP]s?(?!\w)','they',' '.join(lines))
    return POLES.sub('',text) if level=='strict' else text

def assess(url,text):
    req=urllib.request.Request(url+'/api/assess',json.dumps({'text':text}).encode(),{'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=300))

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders',nargs='+',type=Path); p.add_argument('--url',default='http://127.0.0.1:8765')
    p.add_argument('--output',type=Path,default=Path('data/stage2/portrait_test.json'))
    args=p.parse_args(); res={}
    for folder in args.folders:
        for level in ('codes','strict'):
            rows=[]
            for t in MBTI:
                f=next(iter(folder.glob(f'{t}.*')),None)
                if not f: continue
                r=assess(args.url,portrait(f,level))
                rows.append({'type':t,'words':r['words'],**{c:{'type':v['type'],'top3':[x['type'] for x in v['top']]} for c,v in r['mbti'].items() if v}})
            key=f'{folder.name}/{level}'; res[key]={'portraits':len(rows),'rows':rows}
            for c in ('crowd','jev','agent'):
                got=[x for x in rows if c in x]
                if not got: continue
                res[key][c]={'exact':round(float(np.mean([x[c]['type']==x['type'] for x in got])),3),
                             'top3':round(float(np.mean([x['type'] in x[c]['top3'] for x in got])),3),
                             'letters_E_S_T_J':[round(float(np.mean([x[c]['type'][i]==x['type'][i] for x in got])),3) for i in range(4)],
                             'S_predicted':sum(x[c]['type'][1]=='S' for x in got)}
            print(json.dumps({'set':key}|{c:res[key][c] for c in ('crowd','jev','agent') if c in res[key]}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
