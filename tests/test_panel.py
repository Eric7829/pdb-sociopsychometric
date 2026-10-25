import json
import random
import threading
import urllib.request
import pytest
from pdb_ingest import panel
from pdb_ingest.battery import assess
from pdb_ingest.models import Page,Comment
from pdb_ingest.pipeline import process_profile
from pdb_ingest.store import Store
from test_pipeline import Client,Docs
from test_battery import VaryingLlm

TYPES=['INFP','ENTJ','ISTP','ESFJ']; CORES=['4w5','8w7','5w6','2w1']
WORDS={'INFP':'dream feel poetry alone','ENTJ':'lead plan win organize','ISTP':'fix engine quiet tools','ESFJ':'friends party help family'}

class Many(Client):
    async def comments(self,n,token):
        t=TYPES[n%4]; rng=random.Random(n)
        return Page([Comment(f'c{n}','42','owner',' '.join(rng.choice(WORDS[t].split()) for _ in range(250)),True)],None,True)
    async def tallies(self,n):
        t=TYPES[n%4]; return {'MBTI':{t:12,'XNXX':1},'ENNEAGRAM':{CORES[n%4]:11}}

async def build(tmp_path,n=40):
    store=Store(tmp_path/'p.duckdb')
    for i in range(1,n+1): await process_profile(Many(),store,Docs(),i)
    await assess(store,VaryingLlm(),min_words=1)
    return store

def fake_embed(views):
    text=' '.join(t for t,_ in views)
    return [text.count(w) for w in ('dream','lead','fix','friends','poetry','plan','engine','party')]

async def test_summary_aggregates(tmp_path):
    store=await build(tmp_path)
    with store: s=panel.build_summary(store.connection)
    assert s['profiles']==40 and sum(s['mbti']['crowd'])==40 and s['mbti']['crowd'][panel.MBTI.index('INFP')]==10
    assert sum(s['enneagram']['crowd'])==40 and s['enneagram']['crowd'][3]==10
    assert sum(s['big5']['histograms']['O'])==40 and len(s['big5_correlation']['matrix'])==5
    assert len(s['crowd_letters_vs_big5']['matrix'][0])==4 and sum(map(sum,s['mbti_by_enneagram']['counts']))==40
    json.dumps(s)

async def test_assessor_predicts_with_cv_accuracy(tmp_path):
    pytest.importorskip('sklearn')
    store=await build(tmp_path)
    with store: a=panel.Assessor(store,None,embed_fn=fake_embed,folds=4)
    r=a.assess('I love to lead and plan, I organize teams and I always plan to win. ENTJ forever.')
    assert set(r['big5'])==set('OCEAN') and all(0<=v['percentile']<=100 and v['low']<=v['percentile']<=v['high'] for v in r['big5'].values())
    c=r['mbti']['crowd']; assert c['type'][0]=='E' and c['cv']['crowd']['letters_balanced'][0]>.9 and len(c['top'])==3
    assert abs(sum(r['enneagram']['probabilities'])-1)<1e-6 and r['short_text']
    assert a.assess('INFP ENTJ 4w5')=={'error':'No words left after removing typology labels.'}

async def test_http_endpoints(tmp_path):
    pytest.importorskip('sklearn')
    store=await build(tmp_path)
    with store:
        summary=panel.build_summary(store.connection); a=panel.Assessor(store,None,embed_fn=fake_embed,folds=4)
    server=panel.ThreadingHTTPServer(('127.0.0.1',0),panel.make_handler(summary,a)); port=server.server_address[1]
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        base=f'http://127.0.0.1:{port}'
        assert b'<html' in urllib.request.urlopen(base+'/').read().lower()
        assert json.loads(urllib.request.urlopen(base+'/api/summary').read())['profiles']==40
        req=urllib.request.Request(base+'/api/assess',data=json.dumps({'text':'quiet tools engine fix'}).encode(),headers={'Content-Type':'application/json'})
        assert 'big5' in json.loads(urllib.request.urlopen(req).read())
        bad=urllib.request.Request(base+'/api/assess',data=b'{"text":""}')
        with pytest.raises(urllib.error.HTTPError) as e: urllib.request.urlopen(bad)
        assert e.value.code==400
    finally: server.shutdown()

async def test_assessor_predicts_agent_types(tmp_path):
    pytest.importorskip('sklearn')
    store=await build(tmp_path)
    with store: a=panel.Assessor(store,None,embed_fn=fake_embed,folds=4)
    r=a.assess('quiet engine tools fix alone')
    agent=r['mbti']['agent']
    assert all(0<=v<=1 for v in agent['letters'].values()) and len(agent['type'])==4 and 'agent' in agent['cv']
    assert agent['source']=='Form M prompt A' and r['mbti']['jev'] is None   # no v3b or Jev labels for the fake profiles
    # The fake agent always self-types 4w5: a single class cannot be learned and is reported, not crashed on.
    assert r['agent_enneagram']=={'unavailable':'only one class in training data (4)'}

async def test_cached_assessor_trains_once_and_reloads(tmp_path,monkeypatch):
    pytest.importorskip('sklearn')
    store=await build(tmp_path); monkeypatch.setattr(panel,'MODEL_CACHE',tmp_path/'models')
    db=tmp_path/'fake.duckdb'; db.write_bytes(b'x')
    with store:
        a=panel.cached_assessor(store,(None,None,{'model':'fake'}),db,('top',),None,embed_fn=fake_embed,folds=4)
        b=panel.cached_assessor(store,(None,None,{'model':'fake'}),db,('top',),None,embed_fn=fake_embed,folds=4)
    assert a.loaded_from is None and b.loaded_from and len(list((tmp_path/'models').glob('*.pkl')))==1
    text='I love to lead and plan, I organize teams.'
    assert a.assess(text)['mbti']['crowd']['type']==b.assess(text)['mbti']['crowd']['type']
    before=panel.fingerprint(db,('top',),'fake'); db.write_bytes(b'changed')
    assert panel.fingerprint(db,('top',),'fake')!=before   # any database change retrains
