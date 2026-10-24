"""Agent MBTI (Form M theta) vs agent Big Five (IPIP domains), and crowd letter shares vs agent Big Five, against the
latent correlations the owner cited (MBTI scored toward I, N, F, P: E -.72, O .70, C -.61, A .41, N ~0). Database read-only.

  python tools/mbti_big5.py --db data/pdb-top.duckdb
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from pdb_ingest import panel

# Observed (not latent) Form M correlations with Big Five factors scored from the Adjective Check List (John's mapping):
# MBTI Form M Manual Supplement (Schaubhut, Herk & Thompson, CPP 2009), Table 11, n = 185 adults, data from 2008-09.
# Positive = I, N, F, P.  Rows: Big Five factor; columns: E-I, S-N, T-F, J-P.
FORM_M_2009={'E':(-.70,.05,-.03,.16),'A':(-.31,.05,.47,.02),'C':(.15,-.21,-.04,-.54),'O':(-.20,.44,-.04,.17),'N':(.07,-.11,.03,.06)}
PAIRS=[('E-I','E',-.72),('S-N','O',.70),('J-P','C',-.61),('T-F','A',.41)]   # (dichotomy, Big Five domain, cited latent r)
SECOND={'E-I':'I','S-N':'N','T-F':'F','J-P':'P'}

def main():
    from scipy.stats import pearsonr,spearmanr
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/mbti_big5.json'))
    args=p.parse_args()
    store=readonly_store(args.db); c=store.connection; A={}; ip={}
    for n,cont,dom in c.execute("""SELECT e.profile_num,e.form_m_continuous,e.ipip_domains_json FROM psychometric_evaluations e JOIN agent_runs r USING(run_id)
                                   WHERE r.prompt_version='stage3-v1' AND r.complete ORDER BY r.seed""").fetchall():
        if n not in A: A[n]=json.loads(cont); ip[n]={k:v['raw'] for k,v in json.loads(dom).items()}
    V={int(f.stem):{d:-r['theta'] for d,r in json.loads(f.read_text(encoding='utf-8'))['scored']['dichotomyResults'].items()}
       for f in Path('data/stage2/formm-v2').glob('*.json')}
    rows=panel.load_rows(c,('top','top2'))
    both=sorted(set(V)&set(A)); res={'profiles_A_and_v2':len(both),'note':'theta sign flipped so positive = I, N, F, P'}
    def table(scores,ids):
        out={}
        for d,dom,lit in PAIRS:
            x=np.array([scores[n][d] for n in ids]); y=np.array([ip[n][dom] for n in ids])
            out[f'{SECOND[d]} vs {dom}']={'cited':lit,'pearson':round(float(pearsonr(x,y)[0]),3),'spearman':round(float(spearmanr(x,y)[0]),3)}
        out['full_matrix']={dom:[round(float(pearsonr([scores[n][d] for n in ids],[ip[n][dom] for n in ids])[0]),3) for d in SECOND] for dom in 'EACON'}
        out['full_matrix_mean_abs_gap_to_form_m_2009']=round(float(np.mean([abs(a-b) for dom in 'EACON' for a,b in zip(out['full_matrix'][dom],FORM_M_2009[dom])])),3)
        out['Neuroticism vs all four (max |pearson|)']={'cited':0.0,'pearson':round(float(max(abs(pearsonr([scores[n][d] for n in ids],[ip[n]['N'] for n in ids])[0]) for d in SECOND)),3)}
        return out
    res['prompt_A']=table({n:{d:-A[n][d] for d in SECOND} for n in both},both)
    res['formm_v2']=table(V,both)
    cr=[n for n in both if 'mbti_counts' in rows[n]]
    def sh(counts,i,l):
        tot=sum(counts.values()); return sum(v for t,v in counts.items() if t[i]==l)/tot
    share={n:{d:sh(rows[n]['mbti_counts'],i,SECOND[d]) for i,d in enumerate(SECOND)} for n in cr}
    res['crowd_vote_share']=table(share,cr); res['crowd_profiles']=len(cr)
    for k,v in res.items():
        if isinstance(v,dict) and 'I vs E' in v:
            dist=np.mean([abs(x['pearson']-x['cited']) for x in v.values() if isinstance(x,dict) and 'cited' in x]); v['mean_abs_gap_to_cited']=round(float(dist),3)
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res,indent=1))

if __name__=='__main__': main()
