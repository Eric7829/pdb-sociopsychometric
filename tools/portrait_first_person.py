"""Rewrite 16-type portraits into first-person self-descriptions with the Stage 3 model (DeepSeek via OpenRouter), keeping
content. Input is the 'codes'-scrubbed portrait (no headers or type codes); output goes to data/portraits-first-person/<set>/
(git-ignored: the portraits are copyrighted and stay local).

  python tools/portrait_first_person.py "<folder with ISTJ.md ...>" [more folders]
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portrait_test import portrait
from pdb_ingest.llm import DEFAULT_MODEL,OpenRouter,load_api_key
from pdb_ingest.panel import MBTI

OUT=Path('data/portraits-first-person')
TASK='''Rewrite the description below as a first-person self-description, as if the person it describes wrote it about themself in plain prose. Keep every trait, strength, weakness and detail, in the same order; add nothing new and leave nothing out. Do not mention personality types, type names, letters, or psychological functions. Call {tool}.

<description>
{text}
</description>'''

def tool():
    return {'type':'function','function':{'name':'submit_rewrite','description':'Submit the first-person rewrite.','parameters':{
        'type':'object','additionalProperties':False,'required':['text'],'properties':{'text':{'type':'string'}}}}}

async def main_async(folders):
    import httpx
    cost=[0.0]
    async with httpx.AsyncClient(timeout=300) as http:
        llm=OpenRouter(http,load_api_key(),DEFAULT_MODEL,provider=None)   # any host: avoids competing with pinned runs
        gate=asyncio.Semaphore(2)
        async def one(folder,t):
            f=next(iter(folder.glob(f'{t}.*')),None); out=OUT/folder.name/f'{t}.txt'
            if not f or out.exists(): return
            async with gate: r=await llm.call_tool([{'role':'user','content':TASK.format(tool='submit_rewrite',text=portrait(f,'codes'))}],tool(),temperature=0,seed=0)
            cost[0]+=(r['usage'] or {}).get('cost',0) or 0
            out.parent.mkdir(parents=True,exist_ok=True); out.write_text(r['arguments']['text'],encoding='utf-8')
        await asyncio.gather(*(one(folder,t) for folder in folders for t in MBTI))
    print(json.dumps({'cost_usd':round(cost[0],4)}))

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders',nargs='+',type=Path); asyncio.run(main_async(p.parse_args().folders))

if __name__=='__main__': main()
