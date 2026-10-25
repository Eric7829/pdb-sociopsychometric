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
                r=assess(args.url,portrait(f,level)); top=[x['type'] for x in r['mbti_type']['top']]
                rows.append({'type':t,'words':r['words'],'crowd_head':r['mbti'],'top3':top,'agent_letters':r['agent_mbti'],
                             'p_true_rank':top.index(t)+1 if t in top else None})
            key=f'{folder.name}/{level}'; n=len(rows)
            letters=lambda k:[round(float(np.mean([x[k][i]==x['type'][i] for x in rows])),3) for i in range(4)]
            res[key]={'portraits':n,'mean_words':int(np.mean([x['words'] for x in rows])),
                      'exact':round(float(np.mean([x['crowd_head']==x['type'] for x in rows])),3),
                      'top3':round(float(np.mean([x['p_true_rank'] is not None for x in rows])),3),
                      'letters_E_S_T_J':letters('crowd_head'),'agent_letters_exact':round(float(np.mean([x['agent_letters']==x['type'] for x in rows])),3),
                      'agent_letters_E_S_T_J':letters('agent_letters'),'rows':rows}
            print(json.dumps({k:v for k,v in res[key].items() if k!='rows'}|{'set':key}),flush=True)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8')

if __name__=='__main__': main()
