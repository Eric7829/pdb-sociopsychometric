"""A/B prompt variants of the Stage 3 battery without ground truth.

  select  --db D --n 25 --output F   first N complete profiles (crawl order) with >=200-word corpora
  report  --db D V1 V2 [V3 ...]      compare prompt versions on profiles with >=2 complete runs under every version
  shift   --db D BASE OTHER          single-run sensitivity check: how much OTHER's framing shifts scores vs BASE

Decision rule (fixed before any results): keep versions within .05 of the best mean IPIP domain test-retest r; among
those, keep versions within .05 of the best mean convergent validity (IPIP domain vs. Form M theta, sign-corrected);
if more than one remains, prefer stage3-v1, else the highest convergent validity.
"""
import argparse
import json
from pathlib import Path
import statistics as st
import sys
import duckdb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pdb_ingest.battery import corpus_text,load_ipip
from pdb_ingest.store import Store
from pdb_ingest.text import words

DOMAINS='NEOAC'
# Expected sign of r(IPIP domain, Form M theta); theta>0 means E, S, T, J.
CONVERGENT={'E':('E-I',1),'O':('S-N',-1),'A':('T-F',-1),'C':('J-P',1)}

def select(db,n,min_words=200):
    with Store(db) as store:
        nums=[r[0] for r in store.connection.execute('SELECT p.profile_num FROM profiles p LEFT JOIN crawl_queue q USING(profile_num) '
                                                    'WHERE p.complete ORDER BY q.discovery_rank NULLS LAST,p.profile_num').fetchall()]
        return [x for x in nums if words(corpus_text(store.load_snapshot(x).segments))>=min_words][:n]

def corr(xs,ys):
    try: return st.correlation(xs,ys)
    except st.StatisticsError: return float('nan')

def alpha(rows):
    k=len(rows[0])
    items=[st.pvariance([r[i] for r in rows]) for i in range(k)]
    total=st.pvariance([sum(r) for r in rows])
    return k/(k-1)*(1-sum(items)/total) if total else float('nan')

def mean(xs):
    xs=[x for x in xs if x==x]
    return st.fmean(xs) if xs else float('nan')

def load(c,version):
    rows=c.execute('''SELECT r.profile_num,r.seed,e.ipip_responses_json,e.form_m_continuous,e.form_m_mbti,e.response_style_json
                      FROM agent_runs r JOIN psychometric_evaluations e USING(run_id) WHERE r.complete AND r.prompt_version=?
                      ORDER BY r.profile_num,r.seed''',[version]).fetchall()
    runs={}
    for n,seed,resp,theta,mbti,style in rows:
        runs.setdefault(n,[]).append({'responses':json.loads(resp),'theta':json.loads(theta),'mbti':mbti,'style':json.loads(style)})
    return runs

def metrics(runs,profiles,c,keys):
    first=[runs[n][0] for n in profiles]; second=[runs[n][1] for n in profiles]
    def domains(run):
        d={k:0 for k in DOMAINS}
        for item,ans in zip(keys,run['responses']): d[item['facet'][0]]+=6-ans if item['reverse'] else ans
        return d
    d1=[domains(r) for r in first]; d2=[domains(r) for r in second]
    keyed=lambda run,dom:[6-a if it['reverse'] else a for it,a in zip(keys,run['responses']) if it['facet'][0]==dom]
    retest={k:corr([d[k] for d in d1],[d[k] for d in d2]) for k in DOMAINS}
    conv={k:sign*corr([d[k] for d in d1],[r['theta'][dich] for r in first]) for k,(dich,sign) in CONVERGENT.items()}
    crowd=[]
    for n,r in zip(profiles,first):
        row=c.execute("SELECT vote_counts_json FROM vote_distributions WHERE profile_num=? AND system_name='MBTI' AND training_eligible",[n]).fetchone()
        if row:
            counts=json.loads(row[0]); top=max(counts,key=counts.get)
            crowd.append(sum(a==b for a,b in zip(r['mbti'],top))/4)
    return {'profiles':len(profiles),
            'neutral_share':mean([r['style']['ipip_neutral_share'] for r in first+second]),
            'extreme_share':mean([r['style']['ipip_extreme_share'] for r in first+second]),
            'item_agreement':mean([sum(a==b for a,b in zip(x['responses'],y['responses']))/120 for x,y in zip(first,second)]),
            'retest_r':retest,'retest_mean':mean(retest.values()),
            'alpha':{k:alpha([keyed(r,k) for r in first]) for k in DOMAINS},
            'convergent_r':conv,'convergent_mean':mean(conv.values()),
            'between_profile_sd':{k:st.pstdev([d[k] for d in d1]) for k in DOMAINS},
            'form_m_retest_type_agreement':mean([x['mbti']==y['mbti'] for x,y in zip(first,second)]),
            'crowd_mbti_letter_agreement':mean(crowd),'crowd_profiles':len(crowd)}

def decide(results):
    score=lambda v,key:results[v][key] if results[v][key]==results[v][key] else float('-inf')
    candidates=list(results)
    for key in ('retest_mean','convergent_mean'):
        best=max(score(v,key) for v in candidates)
        candidates=[v for v in candidates if score(v,key)>=best-.05]
        if len(candidates)==1: return candidates[0],key
    if 'stage3-v1' in candidates: return 'stage3-v1','tie (kept stage3-v1)'
    return max(candidates,key=lambda v:score(v,'convergent_mean')),'tie (highest convergent)'

def report(db,*versions):
    c=duckdb.connect(str(db),read_only=True); keys=load_ipip().load_items()
    runs={v:load(c,v) for v in versions}
    profiles=sorted(set.intersection(*({n for n,r in runs[v].items() if len(r)>=2} for v in versions)))
    if len(profiles)<5: raise SystemExit(f'Only {len(profiles)} profiles have 2 complete runs under every version')
    results={v:metrics(runs[v],profiles,c,keys) for v in versions}
    winner,reason=decide(results)
    return {**results,'winner':winner,'decided_by':reason}

def shift(db,base,other):
    """Paired comparison of the first complete run per profile under each version (no decision rule)."""
    c=duckdb.connect(str(db),read_only=True); keys=load_ipip().load_items()
    a,b=load(c,base),load(c,other); profiles=sorted(a.keys()&b.keys())
    if not profiles: raise SystemExit('No profile has complete runs under both versions')
    def domains(run):
        d={k:0 for k in DOMAINS}
        for item,ans in zip(keys,run['responses']): d[item['facet'][0]]+=6-ans if item['reverse'] else ans
        return d
    da=[domains(a[n][0]) for n in profiles]; db_=[domains(b[n][0]) for n in profiles]
    pct=lambda version,n:{k:v['percentile'] for k,v in json.loads(c.execute('''SELECT e.ipip_domains_json FROM agent_runs r
        JOIN psychometric_evaluations e USING(run_id) WHERE r.profile_num=? AND r.prompt_version=? AND r.complete ORDER BY r.seed LIMIT 1''',[n,version]).fetchone()[0]).items()}
    pa=[pct(base,n) for n in profiles]; pb=[pct(other,n) for n in profiles]
    style=lambda runs,key:mean([runs[n][0]['style'][key] for n in profiles])
    out={'profiles':len(profiles),'domains':{}}
    for k in DOMAINS:
        out['domains'][k]={'mean_raw_'+base:mean([d[k] for d in da]),'mean_raw_'+other:mean([d[k] for d in db_]),
            'mean_shift_raw':mean([y[k]-x[k] for x,y in zip(da,db_)]),'mean_abs_shift_percentile':mean([abs(y[k]-x[k]) for x,y in zip(pa,pb)]),
            'rank_agreement_r':corr([d[k] for d in da],[d[k] for d in db_]),
            'between_profile_sd_'+base:st.pstdev([d[k] for d in da]),'between_profile_sd_'+other:st.pstdev([d[k] for d in db_])}
    for key in ('ipip_neutral_share','ipip_extreme_share'):
        out[key]={base:style(a,key),other:style(b,key)}
    out['form_m_type_agreement']=mean([a[n][0]['mbti']==b[n][0]['mbti'] for n in profiles])
    out['form_m_letter_agreement']=mean([sum(x==y for x,y in zip(a[n][0]['mbti'],b[n][0]['mbti']))/4 for n in profiles])
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    s=p.add_subparsers(dest='cmd',required=True)
    sel=s.add_parser('select'); sel.add_argument('--db',type=Path,required=True); sel.add_argument('--n',type=int,default=25); sel.add_argument('--output',type=Path,required=True)
    rep=s.add_parser('report'); rep.add_argument('--db',type=Path,required=True); rep.add_argument('versions',nargs='+')
    sh=s.add_parser('shift'); sh.add_argument('--db',type=Path,required=True); sh.add_argument('base'); sh.add_argument('other')
    args=p.parse_args()
    if args.cmd=='select':
        nums=select(args.db,args.n); args.output.write_text('\n'.join(map(str,nums))+'\n'); print(json.dumps({'selected':len(nums)}))
    elif args.cmd=='shift':
        rounded=lambda x:{k:rounded(v) for k,v in x.items()} if isinstance(x,dict) else round(x,3) if isinstance(x,float) else x
        print(json.dumps(rounded(shift(args.db,args.base,args.other)),indent=1))
    else:
        rounded=lambda x:{k:rounded(v) for k,v in x.items()} if isinstance(x,dict) else round(x,3) if isinstance(x,float) else x
        print(json.dumps(rounded(report(args.db,*args.versions)),indent=1))

if __name__=='__main__': main()
