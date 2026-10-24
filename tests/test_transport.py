import pytest
from pdb_ingest.transport import JsonTransport
from pdb_ingest.models import AccessBlocked

class Response:
    def __init__(self,status,payload,headers=None):
        self.status=status; self.payload=payload
        self.headers=headers or {'content-type':'application/json'}
    async def body(self):
        import json
        return json.dumps(self.payload).encode()
    async def dispose(self): pass

class Request:
    def __init__(self,responses): self.responses=iter(responses); self.calls=0
    async def get(self,*args,**kwargs): self.calls+=1; return next(self.responses)

async def test_429_retry_and_payload_401(tmp_path):
    slept=[]
    async def sleep(n): slept.append(n)
    request=Request([Response(429,{}, {'content-type':'application/json','retry-after':'3'}),Response(200,{'profiles':[]})])
    client=JsonTransport(request,tmp_path,sleep=sleep)
    assert (await client.get('/api/v1/profiles')).payload == {'profiles':[]}
    assert 3 in slept and request.calls==2
    request=Request([Response(200,{'code':401,'message':'Please log in'})])
    with pytest.raises(AccessBlocked): await JsonTransport(request,tmp_path,sleep=sleep).get('/api/v1/profile/1')

async def test_inactive_profile_is_not_a_session_block(tmp_path):
    import json
    from pdb_ingest.models import ProfileUnavailable
    async def sleep(n): pass
    inactive={'code':403,'message':'This profile is inactive now!'}
    friends={'code':403,'message':'The user has chosen to only allow friends to view this page.'}
    for status,payload in ((403,inactive),(200,inactive),(403,friends)):
        with pytest.raises(ProfileUnavailable):
            await JsonTransport(Request([Response(status,payload)]),tmp_path,sleep=sleep).get('/api/v1/profile/1')
    rows=[json.loads(x) for x in (tmp_path/'pdb.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['status'] for r in rows]==[403,200,403]
    for path,payload in (('/api/v1/profile/1',{'code':403,'message':'Please log in first'}),('/api/v1/profiles',friends)):
        with pytest.raises(AccessBlocked):
            await JsonTransport(Request([Response(403,payload)]),tmp_path,sleep=sleep).get(path)

async def test_challenge_and_retry_limit(tmp_path):
    async def sleep(n): pass
    with pytest.raises(AccessBlocked):
        await JsonTransport(Request([Response(403,{})]),tmp_path,sleep=sleep).get('/api/v1/profile/1')
    r=Request([Response(503,{})]*3)
    with pytest.raises(RuntimeError): await JsonTransport(r,tmp_path,sleep=sleep).get('/api/v1/profile/1')
    assert r.calls==3
