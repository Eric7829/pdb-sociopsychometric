"""Pilot: DeepSeek as a PDB voter. Live OpenRouter calls (costs money); results cached per call under data/stage2/llm-voter.

  python tools/llm_voter_pilot.py --db data/pdb-top.duckdb [--profiles 25] [--essays 100] [--samples 3] [--provider cloudflare] [--dry-run]

One forced tool call per (text, sample) returns a short function-theory rationale, a probability for each of the 16
MBTI types and Big Five percentile estimates. Samples (temperature .7, seeds 0..k-1) are averaged.
  PDB: a fixed random set of profiles with a training-eligible crowd MBTI vote, two conditions:
    scrubbed: the battery's scrubbed corpus with typology talk removed (EXTRA_JARGON), capped at 6000 words,
    raw: the creator's own text with self-typing left in (how much is leak, how much is reading).
  Essays (Pennebaker & King): a fixed random set; Big Five median-split accuracy and AUC, against our 56.1% / .586.
PANDORA is not used: its terms forbid transferring any part of the dataset, and an API call would.
"""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from pdb_ingest import panel
from pdb_ingest.battery import corpus_text
from pdb_ingest.encoder_text import EXTRA_JARGON
from pdb_ingest.llm import DEFAULT_MODEL,OpenRouter,ToolCallError,load_api_key
from pdb_ingest.panel import DOMAINS,LETTERS,MBTI

OUT=Path('data/stage2/llm-voter')
SYSTEM='''You are an experienced voter on the Personality Database (PDB). PDB voters type people with Jungian cognitive-function theory: they ask which function is dominant and auxiliary (Ni, Ne, Si, Se, Ti, Te, Fi, Fe) and map that stack to an MBTI type, so near-misses are usually types that share functions, not types that share letters.

Type the author of the text below the way an experienced PDB voter would. Judge only from how they think, write and what they care about. Ignore any type labels, self-typing or test results the text mentions: voters overrule self-types often.

Base rates among PDB community profiles (crowd-voted types): {base}. Use them as a prior, but let clear evidence override them; sensing types are under-typed on PDB, so do not default to intuition without evidence.

Also estimate the author's Big Five as population percentiles (0-100), from the same text.

<text>
{text}
</text>'''
TOOL={'type':'function','function':{'name':'submit_typing','description':'Submit the typing.','parameters':{'type':'object','properties':{
    'reasoning':{'type':'string','description':'At most 120 words: the dominant and auxiliary functions you see and the key evidence.'},
    'probabilities':{'type':'object','description':'Probability for each of the 16 types; they should sum to 1.','properties':{t:{'type':'number'} for t in MBTI},'required':MBTI},
    'big_five':{'type':'object','description':'Population percentile 0-100 for each domain.','properties':{k:{'type':'number'} for k in ('openness','conscientiousness','extraversion','agreeableness','neuroticism')},
                'required':['openness','conscientiousness','extraversion','agreeableness','neuroticism']}},
    'required':['reasoning','probabilities','big_five']}}}
B5={'O':'openness','C':'conscientiousness','E':'extraversion','A':'agreeableness','N':'neuroticism'}

def pdb_items(db,k,samples=('top','top2')):
    """The first k of a fixed shuffle of top/top2 profiles, then any extra samples' profiles; base rates stay top/top2's
    so the prompt is identical to the pilot's."""
    store=readonly_store(db); rows=panel.load_rows(store.connection,('top','top2'))
    ids=sorted(n for n,r in rows.items() if 'mbti_counts' in r); random.Random(11).shuffle(ids)
    base=Counter(panel.modal(rows[n]['mbti_counts']) for n in ids); tot=sum(base.values())
    extra=panel.load_rows(store.connection,samples); ids+=sorted(n for n,r in extra.items() if 'mbti_counts' in r and n not in rows); rows={**extra,**rows}
    items=[]
    for n in ids[:k]:
        segs=store.load_snapshot(n).segments
        items.append({'id':n,'crowd':panel.modal(rows[n]['mbti_counts']),
                      'scrubbed':EXTRA_JARGON.sub('',corpus_text(segs,'scrubbed',6000)),'raw':corpus_text(segs,'raw',6000)})
    return items,', '.join(f'{t} {base[t]/tot:.0%}' for t in MBTI)

def essay_items(k):
    import duckdb
    P=str(Path('data/external/essays-big5')/'*.parquet').replace('\\','/')
    rows=duckdb.connect().execute(f"select O,C,E,A,\"N\",text from read_parquet('{P}')").fetchall()
    random.Random(5).shuffle(rows)
    return [{'id':f'essay{i}','y':[int(v) for v in r[:5]],'text':r[5]} for i,r in enumerate(rows[:k])]

SPENT=[0.0]; BUDGET=[float('inf')]

async def ask(llm,sem,key,text,base,seed):
    f=OUT/f'{key}_s{seed}.json'
    if f.exists(): return json.loads(f.read_text(encoding='utf-8'))
    msgs=[{'role':'user','content':SYSTEM.format(base=base,text=text)}]
    for attempt in range(3):
        try:
            async with sem:
                if SPENT[0]>=BUDGET[0]: raise RuntimeError('budget reached')
                r=await llm.call_tool(msgs,TOOL,temperature=.7,seed=seed+100*attempt)
            SPENT[0]+=(r['usage'] or {}).get('cost',0) or 0
        except ToolCallError: continue
        a=r['arguments'] or {}; p=a.get('probabilities') or {}; b=a.get('big_five') or {}
        if set(MBTI)<=set(p) and all(isinstance(p[t],(int,float)) for t in MBTI) and sum(p[t] for t in MBTI)>0 and all(isinstance(b.get(v),(int,float)) for v in B5.values()):
            P=np.array([max(float(p[t]),0) for t in MBTI]); P/=P.sum()
            out={'probabilities':P.tolist(),'big_five':{k:float(b[v]) for k,v in B5.items()},'reasoning_words':len(str(a.get('reasoning','')).split()),
                 'cost_usd':(r['usage'] or {}).get('cost',0) or 0,'prompt_tokens':(r['usage'] or {}).get('prompt_tokens'),'provider':r.get('provider')}
            OUT.mkdir(parents=True,exist_ok=True); f.write_text(json.dumps(out),encoding='utf-8'); return out
    raise ValueError(f'{key}: no valid typing after 3 attempts')

def mbti_metrics(P,truth):
    idx=np.array([MBTI.index(t) for t in truth]); top=P.argmax(1)
    lets={a+b:round(float(np.mean([MBTI[p][i]==MBTI[t][i] for p,t in zip(top,idx)])),3) for i,(a,b) in enumerate(LETTERS)}
    return {'exact':round(float(np.mean(top==idx)),3),'top3':round(float(np.mean([i in np.argsort(-q)[:3] for q,i in zip(P,idx)])),3),
            'log_loss':round(float(-np.mean(np.log(np.clip(P[np.arange(len(idx)),idx],1e-9,1)))),3),'letters':lets,
            'predicted':dict(Counter(MBTI[i] for i in top).most_common(6))}

def main():
    from sklearn.metrics import roc_auc_score
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--profiles',type=int,default=25); p.add_argument('--essays',type=int,default=100)
    p.add_argument('--samples',type=int,default=3); p.add_argument('--provider',default='cloudflare'); p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--dry-run',action='store_true',help='one call: the first profile, scrubbed, seed 0')
    p.add_argument('--all',action='store_true',help='teacher labels: every top/top2/es profile, scrubbed only (use with --samples 1 --essays 0)')
    p.add_argument('--budget',type=float,default=float('inf'),help='stop making new calls once this many USD are spent')
    p.add_argument('--output',type=Path,default=Path('data/stage2/llm_voter_pilot.json'))
    args=p.parse_args(); BUDGET[0]=args.budget
    items,base=pdb_items(args.db,10**6 if args.all else 1 if args.dry_run else args.profiles,('top','top2','es') if args.all else ('top','top2'))
    essays=[] if args.dry_run else essay_items(args.essays)
    async def run():
        import httpx
        async with httpx.AsyncClient() as http:
            llm=OpenRouter(http,load_api_key(),DEFAULT_MODEL,provider=args.provider); sem=asyncio.Semaphore(args.concurrency)
            jobs=[]
            for it in items:
                for cond in (('scrubbed',) if args.dry_run or args.all else ('scrubbed','raw')):
                    for s in range(1 if args.dry_run else args.samples): jobs.append((('pdb',it['id'],cond),ask(llm,sem,f"pdb{it['id']}_{cond}",it[cond],base,s)))
            for it in essays:
                for s in range(args.samples): jobs.append((('essay',it['id'],'essay'),ask(llm,sem,it['id'],it['text'],base,s)))
            got=await asyncio.gather(*(j for _,j in jobs),return_exceptions=True)
            return [(k,g) for (k,_),g in zip(jobs,got)]
    got=asyncio.run(run()); fails=[str(g)[:120] for _,g in got if isinstance(g,Exception)]; ok=[(k,g) for k,g in got if not isinstance(g,Exception)]
    cost=sum(g['cost_usd'] for _,g in ok)
    if args.dry_run:
        print(json.dumps({'calls':len(ok),'failures':fails,'cost_usd':round(cost,5),'prompt_tokens':[g['prompt_tokens'] for _,g in ok],'provider':[g['provider'] for _,g in ok],
                          'crowd':items[0]['crowd'],'top':[MBTI[i] for i in np.argsort(-np.array(ok[0][1]['probabilities']))[:3]] if ok else None})); return
    agg={}
    for k,g in ok: agg.setdefault(k,[]).append(g)
    res={'samples':args.samples,'cost_usd':round(cost,4),'calls':len(ok),'failures':fails}
    for cond in ('scrubbed','raw'):
        keys=[it for it in items if ('pdb',it['id'],cond) in agg]
        if not keys: continue
        P=np.array([np.mean([g['probabilities'] for g in agg[('pdb',it['id'],cond)]],0) for it in keys])
        res[f'pdb_{cond}']={'n':len(keys),**mbti_metrics(P,[it['crowd'] for it in keys])}
    ek=[it for it in essays if ('essay',it['id'],'essay') in agg]
    if ek:
        S=np.array([[np.mean([g['big_five'][k] for g in agg[('essay',it['id'],'essay')]]) for k in DOMAINS] for it in ek]); Y=np.array([it['y'] for it in ek])
        res['essays']={'n':len(ek),'accuracy':{k:round(float(np.mean((S[:,i]>np.median(S[:,i]))==Y[:,i])),3) for i,k in enumerate(DOMAINS)},
                       'auc':{k:round(float(roc_auc_score(Y[:,i],S[:,i])),3) for i,k in enumerate(DOMAINS)}}
        res['essays']['mean_accuracy']=round(float(np.mean(list(res['essays']['accuracy'].values()))),3); res['essays']['mean_auc']=round(float(np.mean(list(res['essays']['auc'].values()))),3)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res))

if __name__=='__main__': main()
