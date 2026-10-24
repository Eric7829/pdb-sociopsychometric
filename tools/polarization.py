"""What drives the agent's Form M preference clarity? (database read-only; needs tools/formm_v2.py results)

  python tools/polarization.py --db data/pdb-top.duckdb

Polarisation = mean |theta| over the four dichotomies (theta is capped at +-3, so 3 means every item answered one way).
Tests, on profiles with both a prompt A run and a Form M v2 result:
1. Writing richness: Spearman of polarisation with scrubbed corpus words and with lexical diversity (type-token ratio over
   the first 500 words; profiles under 500 words excluded from that test).
2. Type identity: profiles whose own raw text states an MBTI code vs those that do not, for A (raw text, codes visible) and
   v2 (scrubbed, codes removed). Identity echo predicts a bigger gap under A.
3. Prompt effect: paired A vs v2 polarisation on the same profiles.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage2 import readonly_store
from pdb_ingest import formm

CODE=re.compile(r'(?<!\w)[IE][NS][TF][JP](?!\w)')

def main():
    from scipy.stats import mannwhitneyu,spearmanr,wilcoxon
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/stage2/polarization.json'))
    args=p.parse_args()
    store=readonly_store(args.db); c=store.connection; A={}
    for n,cont in c.execute("""SELECT e.profile_num,e.form_m_continuous FROM psychometric_evaluations e JOIN agent_runs r USING(run_id)
                               WHERE r.prompt_version='stage3-v1' AND r.complete ORDER BY r.seed""").fetchall(): A.setdefault(n,json.loads(cont))
    V={int(f.stem):json.loads(f.read_text(encoding='utf-8'))['scored']['dichotomyResults'] for f in Path('data/stage2/formm-v2').glob('*.json')}
    ids=sorted(set(A)&set(V)); pol=lambda th:float(np.mean([abs(th[d]) for d in formm.DICHOTOMIES]))
    pa=np.array([pol(A[n]) for n in ids]); pv=np.array([pol({d:V[n][d]['theta'] for d in formm.DICHOTOMIES}) for n in ids])
    words=[]; ttr=[]; states=[]
    for n in ids:
        segs=store.load_snapshot(n).segments
        toks=re.findall(r"[a-z']+",' '.join(s.scrubbed_text for s in segs).lower()); words.append(len(toks))
        ttr.append(len(set(toks[:500]))/500 if len(toks)>=500 else np.nan)
        states.append(bool(CODE.search(' '.join(s.raw_text for s in segs))))
    words=np.array(words); ttr=np.array(ttr); states=np.array(states); ok=~np.isnan(ttr)
    rho=lambda x,y:{'rho':round(float(spearmanr(x,y)[0]),3),'p':round(float(spearmanr(x,y)[1]),4),'n':int(len(x))}
    res={'profiles':len(ids),
         'mean_polarization':{'prompt_A':round(float(pa.mean()),3),'formm_v2':round(float(pv.mean()),3),
                              'paired_wilcoxon_p':float(wilcoxon(pa,pv).pvalue),'share_v2_lower':round(float(np.mean(pv<pa)),3)},
         'richness':{'A_vs_words':rho(pa,np.log1p(words)),'v2_vs_words':rho(pv,np.log1p(words)),
                     'A_vs_ttr500':rho(pa[ok],ttr[ok]),'v2_vs_ttr500':rho(pv[ok],ttr[ok])},
         'identity':{'profiles_stating_a_code':int(states.sum()),'profiles_not':int((~states).sum())}}
    for name,x in (('A',pa),('v2',pv)):
        res['identity'][name]={'stating':round(float(x[states].mean()),3),'not':round(float(x[~states].mean()),3),
                               'gap':round(float(x[states].mean()-x[~states].mean()),3),'mannwhitney_p':round(float(mannwhitneyu(x[states],x[~states]).pvalue),4)}
    args.output.write_text(json.dumps(res,indent=1),encoding='utf-8'); print(json.dumps(res,indent=1))

if __name__=='__main__': main()
