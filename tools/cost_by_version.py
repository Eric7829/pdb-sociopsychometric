"""Mean cost and tokens per complete run, by prompt version, for a set of profiles (database read-only).

  python tools/cost_by_version.py --db data/pdb-top.duckdb --profiles-file data/sensitivity-profiles.txt stage3-predict-v1 stage3-predict-v2
"""
import argparse
import json
from pathlib import Path
import duckdb

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--profiles-file',type=Path,required=True); p.add_argument('versions',nargs='+')
    a=p.parse_args(); nums=[int(x) for x in a.profiles_file.read_text().split()]
    c=duckdb.connect(str(a.db),read_only=True); out={}
    for v in a.versions:
        rows=c.execute(f'''SELECT r.cost_usd,r.usage_json,(SELECT coalesce(sum(cost_usd),0) FROM agent_reflections f WHERE f.profile_num=r.profile_num
                           AND f.prompt_version=r.prompt_version) FROM agent_runs r WHERE r.complete AND r.prompt_version=? AND r.profile_num IN ({','.join('?'*len(nums))})''',
                       [v,*nums]).fetchall()
        if not rows: out[v]={'runs':0}; continue
        u=[json.loads(x[1]) for x in rows]
        out[v]={'runs':len(rows),'mean_cost_usd':round(sum(x[0] for x in rows)/len(rows),5),
                'mean_prompt_tokens':round(sum(x['prompt_tokens'] for x in u)/len(u)),'mean_completion_tokens':round(sum(x['completion_tokens'] for x in u)/len(u))}
    print(json.dumps(out,indent=1))

if __name__=='__main__': main()
