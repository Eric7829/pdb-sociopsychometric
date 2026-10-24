"""Agreement between two Form M simulation runs on the same profiles (e.g. capped corpus vs full, or seed 1 vs seed 0).

  python tools/formm_agreement.py data/stage2/formm-v3 data/stage2/formm-v3-cap8k
"""
import json
from pathlib import Path
import sys
import numpy as np

def main():
    a,b=Path(sys.argv[1]),Path(sys.argv[2])
    ids=sorted(int(f.stem) for f in b.glob('*.json') if (a/f.name).exists())
    load=lambda d,n:json.loads((d/f'{n}.json').read_text(encoding='utf-8'))
    items=[];types=[];letters=[];dtheta=[];cost=[0.0,0.0]
    for n in ids:
        x,y=load(a,n),load(b,n); cost[0]+=x['cost_usd']; cost[1]+=y['cost_usd']
        items.append(np.mean([x['answers'][k]==y['answers'][k] for k in x['answers']]))
        tx,ty=x['scored']['type'],y['scored']['type']; types.append(tx==ty); letters.append([p==q for p,q in zip(tx,ty)])
        dtheta.append([abs(x['scored']['dichotomyResults'][d]['theta']-y['scored']['dichotomyResults'][d]['theta']) for d in ('E-I','S-N','T-F','J-P')])
    print(json.dumps({'profiles':len(ids),'item_agreement':round(float(np.mean(items)),3),'type_agreement':round(float(np.mean(types)),3),
                      'letter_agreement':[round(float(v),3) for v in np.mean(letters,0)],'mean_abs_theta_diff':[round(float(v),2) for v in np.mean(dtheta,0)],
                      'cost_a':round(cost[0],3),'cost_b':round(cost[1],3)},indent=1))

if __name__=='__main__': main()
