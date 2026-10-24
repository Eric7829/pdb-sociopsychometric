import asyncio
import json
from urllib.parse import urlencode
import pytest
from pdb_ingest.capture import Recorder
from pdb_ingest.client import PdbClient
from pdb_ingest.models import SchemaError
from pdb_ingest.pipeline import DocumentSource,process_profile,replay
from pdb_ingest.store import Store
from pdb_ingest.transport import JsonTransport

API='https://api.personality-database.com'
DETAIL=API+'/api/v1/profile/1'
def page_url(offset): return API+'/api/v1/comments/1?'+urlencode(dict(sort='NEW',offset=offset,range='all',limit=100,version='W3'))
def detail(n=10): return {'id':1,'user_id':42,'mbti_profile':'owner','breakdown_systems':{'1':[{'personality_type':'INFP','theCount':n}]}}
def page(cid,text,more):
    body={'comments':[{'id':cid,'user_id':42,'username':'owner','comment':text,'reply_count':0}],'has_more':more}
    if more: body['next_offset']=100
    return body

def texts(store): return sorted(s.raw_text for s in store.load_snapshot(1).segments)

def test_partial_refresh_never_borrows_older_pages(tmp_path):
    rec=Recorder(tmp_path/'run')
    rec.record('GET',DETAIL,None,200,detail(),attempt='a1')
    rec.record('GET',page_url(0),None,200,page(2,'old page 0',True),attempt='a1')
    rec.record('GET',page_url(100),None,200,page(3,'old terminal page',False),attempt='a1')
    rec.record('GET',DETAIL,None,200,detail(11),attempt='a2')
    rec.record('GET',page_url(0),None,200,page(4,'new partial refresh page',True),attempt='a2')
    with Store(tmp_path/'replay.duckdb') as store:
        assert replay(tmp_path/'run',store)=={'processed':1,'succeeded':1,'failed':0,'unavailable':0}
        assert texts(store)==['old page 0','old terminal page']

def test_only_partial_attempt_is_missing_evidence(tmp_path):
    rec=Recorder(tmp_path/'run')
    rec.record('GET',DETAIL,None,200,detail(),attempt='a1')
    rec.record('GET',page_url(0),None,200,page(2,'partial',True),attempt='a1')
    with Store(tmp_path/'replay.duckdb') as store:
        with pytest.raises(SchemaError,match='Missing replay'): replay(tmp_path/'run',store)

def test_documents_are_not_borrowed_across_attempts(tmp_path):
    rec=Recorder(tmp_path/'run')
    link='see https://docs.google.com/document/d/abc/edit'
    for attempt,text in (('a1','old '+link),('a2','new '+link)):
        rec.record('GET',DETAIL,None,200,detail(),attempt=attempt)
        rec.record('GET',page_url(0),None,200,page(2,text,False),attempt=attempt)
    row={'profile_num':1,'attempt':'a1','document':{'key':'abc','source_comment_ids':['2'],'status':'AVAILABLE','text':'old doc','error':None}}
    (tmp_path/'run'/'documents.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    with Store(tmp_path/'replay.duckdb') as store:
        assert replay(tmp_path/'run',store)['succeeded']==1
        assert texts(store)==['old doc','old '+link]

def test_capped_attempt_replays_with_same_cap(tmp_path):
    rec=Recorder(tmp_path/'run')
    rec.record('GET',DETAIL,None,200,detail(),attempt='a1')
    body=page(2,'kept',True); body['comments'].append({**body['comments'][0],'id':3,'comment':'beyond cap'})
    rec.record('GET',page_url(0),None,200,body,attempt='a1')
    with Store(tmp_path/'uncapped.duckdb') as store:
        with pytest.raises(SchemaError,match='Missing replay'): replay(tmp_path/'run',store)
    with Store(tmp_path/'capped.duckdb') as store:
        assert replay(tmp_path/'run',store,max_comments=1)['succeeded']==1
        assert texts(store)==['kept'] and store.load_snapshot(1).comments_truncated

def test_inactive_profile_replays_as_unavailable(tmp_path):
    rec=Recorder(tmp_path/'run')
    rec.record('GET',DETAIL,None,403,{'code':403,'message':'This profile is inactive now!'},attempt='a1')
    with Store(tmp_path/'replay.duckdb') as store:
        assert replay(tmp_path/'run',store)=={'processed':1,'succeeded':0,'failed':0,'unavailable':1}

def test_untagged_duplicate_requests_are_ambiguous(tmp_path):
    rec=Recorder(tmp_path/'run')
    rec.record('GET',DETAIL,None,200,detail(),)
    rec.record('GET',page_url(0),None,200,page(2,'old page 0',True))
    rec.record('GET',page_url(100),None,200,page(3,'old terminal page',False))
    rec.record('GET',DETAIL,None,200,detail(11))
    rec.record('GET',page_url(0),None,200,page(4,'new partial refresh page',True))
    with Store(tmp_path/'replay.duckdb') as store:
        with pytest.raises(SchemaError,match='Ambiguous replay'): replay(tmp_path/'run',store)

class Response:
    def __init__(self,payload): self.status=200; self.payload=payload; self.headers={'content-type':'application/json'}
    async def body(self): return json.dumps(self.payload).encode()
    async def dispose(self): pass

class Request:
    def __init__(self,pages): self.pages=pages
    async def get(self,url,**kwargs): return Response(self.pages[url])

def test_live_attempts_are_tagged_and_replayable(tmp_path):
    async def sleep(n): pass
    run=tmp_path/'run'
    first=Request({DETAIL:detail(),page_url(0):page(2,'first',False)})
    second=Request({DETAIL:detail(),page_url(0):page(2,'second',False)})
    async def live(store,request):
        transport=JsonTransport(request,run,sleep=sleep)
        docs=DocumentSource(None,run)
        assert await process_profile(PdbClient(transport),store,docs,1)
        assert transport.attempt is None and docs.attempt is None
    with Store(tmp_path/'live.duckdb') as store:
        for request in (first,second): asyncio.run(live(store,request))
    rows=[json.loads(x) for x in (run/'pdb.jsonl').read_text(encoding='utf-8').splitlines()]
    attempts=[r['attempt'] for r in rows]
    assert None not in attempts and len(set(attempts))==2
    with Store(tmp_path/'replay.duckdb') as store:
        assert replay(run,store)['succeeded']==1
        assert texts(store)==['second']
