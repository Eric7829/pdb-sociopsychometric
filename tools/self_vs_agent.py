"""The creator's own MBTI vs the DeepSeek agent's: its stated self-type and its Form M result. Database read-only.

  python tools/self_vs_agent.py --db data/pdb-top.duckdb

Labels per profile (each where present):
  self:        the creator's profile-specific self-vote badge (type_ablation.self_votes),
  crowd:       the crowd modal vote (training-eligible),
  agent_self:  the Stage 3 agent's stated type before the questionnaires (prompt A, stage3-v1, first seed),
  formm_A:     that agent's Form M result (same run),
  formm_v3b:   Form M v3b (scrubbed text, typology talk removed),
  voter:       the PDB-voter teacher's top type (scrubbed text).
For every pair: n, exact agreement, per-letter agreement and chance exact agreement (from the two label mixes).
For self vs each agent label, the same split by whether the text the agent read names an MBTI type (a leak check),
and the direction of self/agent letter disagreements.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from type_ablation import self_votes
from pdb_ingest import panel
from pdb_ingest.battery import corpus_text
from pdb_ingest.panel import LETTERS,MBTI

TYPE=re.compile(r'\b(?:'+'|'.join(MBTI)+r')s?\b',re.I)

def agree(a,b):
    keys=[n for n in a if n in b]
    if not keys: return {'n':0}
    pa=Counter(a[n] for n in keys); pb=Counter(b[n] for n in keys)
    return {'n':len(keys),'exact':round(float(np.mean([a[n]==b[n] for n in keys])),3),
            'chance_exact':round(sum(pa[t]*pb[t] for t in MBTI)/len(keys)**2,3),
            'letters':{x+y:round(float(np.mean([a[n][i]==b[n][i] for n in keys])),3) for i,(x,y) in enumerate(LETTERS)}}

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/self_vs_agent.json'))
    args=p.parse_args(); store=readonly_store(args.db); c=store.connection
    rows=panel.load_rows(c,('top','top2','es')); L={k:{} for k in ('self','crowd','agent_self','formm_A','formm_v3b','voter')}
    L['self']={n:t for n,t in self_votes(c).items() if n in rows}
    L['crowd']={n:panel.modal(r['mbti_counts']) for n,r in rows.items() if 'mbti_counts' in r}
    view={}
    for n,s,f,v in c.execute("""SELECT r.profile_num,e.agent_self_mbti,e.form_m_mbti,r.text_view FROM agent_runs r JOIN psychometric_evaluations e ON e.run_id=r.run_id
            WHERE r.complete AND r.prompt_version='stage3-v1' ORDER BY r.seed""").fetchall():
        if n in rows and n not in L['agent_self']:
            if s in MBTI: L['agent_self'][n]=s
            if f in MBTI: L['formm_A'][n]=f
            view[n]=v
    for f in panel.AGENT_FORMM.glob('*.json'):
        n=int(f.stem); t=json.loads(f.read_text(encoding='utf-8'))['scored']['type']
        if n in rows and t in MBTI: L['formm_v3b'][n]=t
    for f in Path('data/stage2/llm-voter').glob('pdb*_scrubbed_s0.json'):
        n=int(f.stem[3:].split('_')[0])
        if n in rows: L['voter'][n]=MBTI[int(np.argmax(json.loads(f.read_text(encoding='utf-8'))['probabilities']))]
    res={'counts':{k:len(v) for k,v in L.items()},'agent_text_views':dict(Counter(view.values())),
         'pairs':{f'{a} vs {b}':agree(L[a],L[b]) for a,b in combinations(L,2)}}
    mentions={}
    for n in L['self']:
        if n in view: mentions[n]=bool(TYPE.search(corpus_text(store.load_snapshot(n).segments,view[n] if view[n] in ('raw','scrubbed') else 'raw')))
    res['self_vs_agent_by_text_mention']={}
    for k in ('agent_self','formm_A','formm_v3b','voter'):
        res['self_vs_agent_by_text_mention'][k]={f'text names a type: {m}':agree({n:t for n,t in L['self'].items() if mentions.get(n) is m},L[k]) for m in (True,False)}
    flips={}
    for k in ('agent_self','formm_A','formm_v3b'):
        d=Counter()
        for n,t in L['self'].items():
            if n in L[k]:
                for i in range(4):
                    if t[i]!=L[k][n][i]: d[f'{t[i]}->{L[k][n][i]}']+=1
        flips[k]=dict(d.most_common())
    res['self_to_agent_letter_flips']=flips
    res['self_mix']=dict(Counter(L['self'].values()).most_common(8))
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res,indent=0))

if __name__=='__main__': main()
