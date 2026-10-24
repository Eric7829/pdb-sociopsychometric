from collections import Counter
import json
import math
import random
import pytest
from pdb_ingest import audits

def slow_loo(xs,ys):
    hits=0
    for i,(x,y) in enumerate(zip(xs,ys)):
        g=Counter(b for j,(a,b) in enumerate(zip(xs,ys)) if a==x and j!=i)
        o=Counter(b for j,b in enumerate(ys) if j!=i)
        src=g or o
        hits+=max(src,key=lambda k:(src[k],k))==y
    return hits/len(xs)

def test_loo_matches_brute_force():
    rng=random.Random(3)
    for _ in range(200):
        n=rng.randint(2,25); xs=[rng.choice('abc') for _ in range(n)]; ys=[rng.choice('pqrs') for _ in range(n)]
        assert audits.loo_accuracy(xs,ys)==pytest.approx(slow_loo(xs,ys))

def test_entropies():
    assert audits.entropy(Counter({'a':1,'b':1}))==pytest.approx(1)
    assert audits.conditional_entropy(['x','x','y','y'],['a','a','b','b'])==0
    assert audits.conditional_entropy(['x','y','x','y'],['a','a','b','b'])==pytest.approx(1)

def test_bundling_detects_dependence_and_not_independence():
    rng=random.Random(1)
    xs=[rng.choice('ABCD') for _ in range(300)]
    dependent=audits.bundling(xs,[{'A':'1','B':'2','C':'3','D':'4'}[x] if rng.random()<.9 else rng.choice('1234') for x in xs],permutations=200)
    assert dependent['p_accuracy']<.01 and dependent['loo_accuracy']>.8 and dependent['bundled_over_80pct']
    independent=audits.bundling(xs,[rng.choice('1234') for _ in xs],permutations=200)
    assert independent['p_H']>.01 and abs(independent['information_gain_over_null'])<.05

def test_spearman_ties_and_partial():
    assert audits.ranks([3,1,3,2])==[2.5,0,2.5,1]
    rng=random.Random(2); zs=[rng.gauss(0,1) for _ in range(400)]
    xs=[z+rng.gauss(0,.3) for z in zs]; ys=[z+rng.gauss(0,.3) for z in zs]
    assert audits.spearman(xs,ys)>.8 and abs(audits.partial_spearman(xs,ys,zs))<.15

def test_jungian_mapping():
    assert audits.jungian_to_mbti('IF(N)')=='INFP' and audits.jungian_to_mbti('EN(T)')=='ENTP'
    assert audits.jungian_to_mbti('IT(S)')=='ISTP' and audits.jungian_to_mbti('ES(F)')=='ESFP' and audits.jungian_to_mbti('EF(S)')=='ESFJ'
    assert audits.jungian_to_mbti('IN(S)') is None and audits.jungian_to_mbti('INFP') is None

def test_letter_share():
    assert audits.letter_share({'INFP':3,'INTP':1,'ENTJ':0},'T-F','T')==.25

async def test_run_audits_on_store(tmp_path):
    from pdb_ingest.store import Store
    from test_pipeline import Client,Docs
    from pdb_ingest.pipeline import process_profile
    rng=random.Random(5); types=['INFP','INTP','ENFP','ISTJ']; ennea={'INFP':'4w5','INTP':'5w4','ENFP':'7w6','ISTJ':'1w9'}
    class Many(Client):
        async def tallies(self,n):
            t=types[n%4]; return {'MBTI':{t:12,'ENTJ':rng.randint(0,3)},'ENNEAGRAM':{ennea[t]:11,'9w1':rng.randint(0,2)}}
    with Store(tmp_path/'a.duckdb') as store:
        for n in range(1,41): await process_profile(Many(),store,Docs(),n)
        result=audits.run_audits(store.connection,permutations=50)
    b=result['bundling']['MBTI->ENNEAGRAM']
    assert result['profiles']==40 and b['loo_accuracy']==1 and b['p_accuracy']<.05
    assert result['inventory']=={'skipped':'no complete stage3-v1 runs'}
    assert 'MBTI' in result['modality']
    json.dumps(result)
