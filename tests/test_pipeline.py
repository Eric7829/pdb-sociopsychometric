from dataclasses import asdict
import json
import pytest
from pdb_ingest.models import Page, ProfileRef, Comment, SelfVote, Document
from pdb_ingest.capture import Capture
from pdb_ingest.store import Store
from pdb_ingest.pipeline import ingest_profile,run_crawl,replay

class Client:
    async def detail(self,n):
        return Capture('GET',f'/api/v1/profile/{n}',None,200,'2026-10-23T00:00:00+00:00',{'id':n,'user_id':42,'contributor':'owner','mbti_profile':'Person','systems':[],'breakdown_systems':{'1':[{'personality_type':'INFP','theCount':25}] }},'hash')
    async def tallies(self,n): return {'MBTI':{'INFP':25}}
    async def comments(self,n,token):
        if token is None:
            return Page([Comment('a','99','other','not mine',True,reply_count=1)],'20',False)
        return Page([Comment('b','42','owner','mine https://docs.google.com/document/d/abc/edit',True,self_votes=[SelfVote('MBTI','INFP','b',True)])],None,True)
    async def replies(self,n,c,token):
        return Page([Comment('r','42','owner','my answer',parent_id='a')],None,True)
    async def listing(self,token):
        return Page([ProfileRef(1,'url'),ProfileRef(2,'url2')],None,True)

class Docs:
    async def fetch(self,key,refs,n): return Document(key,refs,'UNAVAILABLE',None,'HTTP 403')

async def test_profile_reconstructs_creator_and_replies():
    s=await ingest_profile(Client(),1,Docs())
    assert s.complete
    assert {x.source_ref for x in s.segments}=={'b','r'}
    assert s.distributions[0].total_system_votes==24
    assert s.documents[0].status=='UNAVAILABLE'
    assert not any('not mine' in x.raw_text for x in s.segments)

async def test_bounded_run_and_resume(tmp_path):
    with Store(tmp_path/'data.duckdb') as store:
        stats=await run_crawl(Client(),store,Docs(),1)
        assert stats['processed']==1 and len(store.pending())==1 and not stats['full_crawl_complete']
        stats=await run_crawl(Client(),store,Docs(),None)
        assert stats['processed']==1 and stats['full_crawl_complete']

async def test_partial_does_not_claim_success(tmp_path):
    class Broken(Client):
        async def comments(self,n,token): raise RuntimeError('missing page')
    with Store(tmp_path/'data.duckdb') as store:
        stats=await run_crawl(Broken(),store,Docs(),1)
        assert stats['failed']==1 and not store.load_snapshot(1).complete
        assert len(store.pending())==2

def test_offline_replay_and_missing_page(tmp_path):
    from pdb_ingest.capture import Recorder
    from pdb_ingest.models import SchemaError
    from pdb_ingest.pipeline import request_key
    from urllib.parse import urlencode
    folder=tmp_path/'capture'
    rec=Recorder(folder)
    rec.record('GET','https://api.personality-database.com/api/v1/profile/1',None,200,
               {'id':1,'user_id':42,'mbti_profile':'owner','breakdown_systems':{'1':[{'personality_type':'INFP','theCount':10}]}})
    url='https://api.personality-database.com/api/v1/comments/1?'+urlencode(dict(sort='NEW',offset=0,range='all',limit=100,version='W3'))
    rec.record('GET',url,None,200,{'comments':[{'id':2,'user_id':42,'username':'owner','comment':'one two','reply_count':0}], 'has_more':False})
    with Store(tmp_path/'replay.duckdb') as store:
        assert replay(folder,store)['succeeded']==1
        assert store.load_snapshot(1).segments[0].raw_text=='one two'
    rec.record('GET',url,None,200,{'comments':[],'has_more':True,'next_offset':100})
    with Store(tmp_path/'missing.duckdb') as store:
        with pytest.raises(SchemaError,match='Ambiguous replay'): replay(folder,store)

async def test_comment_reply_without_count_is_not_recursed():
    from pdb_ingest.crawl import collect_comments
    client=Client()
    comments,truncated=await collect_comments(client,1)
    assert len(comments)==3 and not truncated

async def test_inactive_profile_is_skipped_not_fatal(tmp_path):
    from pdb_ingest.models import ProfileUnavailable
    class Inactive(Client):
        async def detail(self,n):
            if n==1: raise ProfileUnavailable('This profile is inactive now!')
            return await super().detail(n)
    with Store(tmp_path/'data.duckdb') as store:
        stats=await run_crawl(Inactive(),store,Docs(),None)
        assert stats['unavailable']==1 and stats['succeeded']==1 and stats['failed']==0
        assert stats['outstanding']==0 and stats['full_crawl_complete']
        assert not store.load_snapshot(1).complete
        assert (await run_crawl(Inactive(),store,Docs(),None))['processed']==0

async def test_comment_cap_limits_threads_and_is_recorded(tmp_path):
    from pdb_ingest.crawl import collect_comments
    comments,truncated=await collect_comments(Client(),1,1)
    assert truncated and {c.comment_id for c in comments}=={'a','r'}
    comments,truncated=await collect_comments(Client(),1,2)
    assert not truncated and len(comments)==3
    s=await ingest_profile(Client(),1,Docs(),1)
    assert s.complete and s.comment_limit==1 and s.comments_truncated
    assert [x.source_ref for x in s.segments]==['r'] and any('first 1 top-level' in i for i in s.issues)
    with Store(tmp_path/'cap.duckdb') as store:
        await run_crawl(Client(),store,Docs(),1,max_comments=50)
        with pytest.raises(ValueError,match='max_comments'): await run_crawl(Client(),store,Docs(),1)
        assert store.load_snapshot(store.connection.execute('select min(profile_num) from profiles').fetchone()[0]).comment_limit==50

async def test_scope_cap_preserves_listing_order(tmp_path):
    class Ranked(Client):
        calls=[]
        async def listing(self,token):
            offset=int(token or 0); self.calls.append(offset)
            return Page([ProfileRef(100-offset,'u'),ProfileRef(99-offset,'u')],str(offset+2),False)
    client=Ranked()
    with Store(tmp_path/'cap.duckdb') as store:
        await run_crawl(client,store,Docs(),1,max_profiles=5)
        assert [x.profile_num for x in store.pending()]==[99]
        stats=await run_crawl(client,store,Docs(),None,max_profiles=5)
        assert client.calls==[0,2,4]
        assert store.connection.execute('select profile_num from crawl_queue order by discovery_rank').fetchall()==[(100,),(99,),(98,),(97,),(96,)]
        assert stats['scope_complete'] and not stats['full_crawl_complete']

async def test_listing_sort_is_part_of_crawl_scope(tmp_path):
    class Top(Client): sort='top'
    with Store(tmp_path/'sort.duckdb') as store:
        await run_crawl(Top(),store,Docs(),1)
        with pytest.raises(ValueError,match='listing sort'): await run_crawl(Client(),store,Docs(),1)

async def test_unavailable_profiles_are_replaced_in_listing_order(tmp_path):
    from pdb_ingest.models import ProfileUnavailable
    class Ranked(Client):
        calls=[]
        async def listing(self,token):
            offset=int(token or 0); self.calls.append(offset)
            return Page([ProfileRef(100-offset,'u'),ProfileRef(99-offset,'u')],str(offset+2),False)
        async def detail(self,n):
            if n==96: raise ProfileUnavailable('The user has chosen to only allow friends to view this page.')
            return await super().detail(n)
    client=Ranked()
    with Store(tmp_path/'refill.duckdb') as store:
        stats=await run_crawl(client,store,Docs(),None,max_profiles=5)
        assert client.calls==[0,2,4,4]
        assert [r[0] for r in store.connection.execute('select profile_num from crawl_queue order by discovery_rank').fetchall()]==[100,99,98,97,96,95]
        assert stats['usable']==5 and stats['unavailable']==1 and stats['scope_complete']
        assert (await run_crawl(client,store,Docs(),None,max_profiles=5))['processed']==0 and client.calls==[0,2,4,4]

def test_parse_listing_keeps_consensus_type():
    from pdb_ingest.adapters import parse_listing
    page=parse_listing({'profiles':[{'id':5,'personality_type':'ESTP 7w8'},{'id':6}]},0,100)
    assert [r.personality_type for r in page.items]==['ESTP 7w8',None]

async def test_type_filtered_supplement_counts_assessable_profiles(tmp_path):
    words={1:30,2:30,3:2,4:30,5:30,6:30}
    types={1:'INFP 4w5',2:'ESTP 7w8',3:'ESFJ 2w1',4:'INTJ 5w6',5:'ESFP 7w6',6:'ESTJ 1w2'}
    class Typed(Client):
        calls=[]
        async def listing(self,token):
            offset=int(token or 0); self.calls.append(offset)
            ids=[offset+1,offset+2]
            return Page([ProfileRef(n,'u',types.get(n)) for n in ids if n in types],str(offset+2),offset+2>=6)
        async def comments(self,n,token):
            return Page([Comment(f'c{n}','42','owner',' '.join(['word']*words[n]),True)],None,True)
    with Store(tmp_path/'sup.duckdb') as store:
        store.checkpoint('scope','500',True)
        stats=await run_crawl(Typed(),store,Docs(),None,max_profiles=2,sample='es',type_filter=r'^ES[TF][JP]\b',min_words=10)
        rows=store.connection.execute('select profile_num,sample from crawl_queue order by discovery_rank').fetchall()
        assert rows==[(2,'es'),(3,'es'),(5,'es')]
        assert stats['usable']==2 and stats['scope_complete']
        assert store.load_checkpoint('scope')==('500',True) and store.load_checkpoint('scope:es')==('2',True)
        with pytest.raises(ValueError,match='max_profiles'):
            await run_crawl(Typed(),store,Docs(),None,max_profiles=3,sample='es',type_filter=r'^ES[TF][JP]\b',min_words=10)

def test_parse_profile_keeps_creation_date():
    from pdb_ingest.adapters import parse_profile
    p=parse_profile({'id':5,'mbti_profile':'x','contributor_create_date':'2026-09-28 16:49:43'})
    assert p.created=='2026-09-28 16:49:43'

async def test_created_after_and_require_eligible_scope_the_sample(tmp_path):
    created={1:'2026-08-01 10:00:00',2:'2026-09-20 10:00:00',3:'2026-09-21 10:00:00',4:'2026-09-22 10:00:00'}
    votes={1:30,2:30,3:5,4:30}
    class Recent(Client):
        calls=[]
        async def listing(self,token):
            offset=int(token or 0); self.calls.append(offset)
            return Page([ProfileRef(n,'u') for n in (offset+1,offset+2) if n in created],str(offset+2),offset+2>=4)
        async def detail(self,n):
            c=await super().detail(n); c.payload['contributor_create_date']=created[n]; return c
        async def tallies(self,n): return {'MBTI':{'INFP':votes[n]}}
    with Store(tmp_path/'recent.duckdb') as store:
        stats=await run_crawl(Recent(),store,Docs(),None,max_profiles=2,created_after='2026-09-15',require_eligible=('MBTI',))
        q=store.connection.execute
        assert q('select profile_num from crawl_queue where unavailable').fetchall()==[(1,)]
        assert stats['usable']==2 and stats['scope_complete']
        assert sorted(r[0] for r in q('select profile_num from crawl_queue where complete').fetchall())==[2,3,4]
        assert 'created before 2026-09-15' in store.load_snapshot(1).issues[0]
