from urllib.parse import urlencode
from pdb_ingest.adapters import parse_comment
from pdb_ingest.annotate import annotate
from pdb_ingest.capture import Recorder
from pdb_ingest.encoder_text import encoder_view
from pdb_ingest.models import Page
from pdb_ingest.pipeline import process_profile
from pdb_ingest.store import Store
from test_pipeline import Client,Docs

API='https://api.personality-database.com/api/v1/'
P={'id':10,'user_id':99,'username':'voter','comment':'why INFP?','reply_count':1}

async def view(tmp_path,answer,**kw):
    R={'id':11,'user_id':42,'username':'owner','comment':f'Q1: Why? {answer}','parent_id':10,'reply_to_user_id':99}
    class Thread(Client):
        async def comments(self,n,token): return Page([parse_comment(P)],None,True)
        async def replies(self,n,c,token): return Page([parse_comment(R)],None,True)
    rec=Recorder(tmp_path/'run')
    rec.record('GET',API+'comments/1?'+urlencode(dict(sort='NEW',offset=0,range='all',limit=100,version='W3')),None,200,{'comments':[P],'has_more':False})
    rec.record('GET',API+'comment_replies?'+urlencode(dict(limit=100,offset=0,comment_id=10)),None,200,{'comments':[R],'has_more':False})
    with Store(tmp_path/'e.duckdb') as store:
        await process_profile(Thread(),store,Docs(),1); annotate(store,tmp_path/'run')
        return encoder_view(store,1,**kw)

def pooled(text,ranges): return [text[a:b] for a,b in ranges]

async def test_markers_and_pooling(tmp_path):
    (text,ranges),=await view(tmp_path,'Because books.')
    assert text=='[CTX] why ? [/CTX] [Q] Q1: Why? [/Q] Because books.' and pooled(text,ranges)==['Because books.']

async def test_scrub_covers_context_quotes_and_prompts(tmp_path):
    # Review I2: a bidirectional encoder lets pooled tokens attend to unpooled text, so scrubbing must cover all of it.
    (text,ranges),=await view(tmp_path,'Because INFP books.')
    assert 'INFP' not in text and pooled(text,ranges)==['Because books.']
    (raw,ranges),=await view(tmp_path/'raw','Because INFP books.',scrub=False)
    assert raw.count('INFP')==2 and pooled(raw,ranges)==['Because INFP books.']

async def test_mask_prompts_false_pools_prompt(tmp_path):
    (text,ranges),=await view(tmp_path,'Because books.',mask_prompts=False)
    assert pooled(text,ranges)==['Q1: Why?','Because books.']

def test_clean_removes_typology_talk_but_not_ordinary_words():
    from pdb_ingest.encoder_text import clean
    out=clean("Searching my ennea type 4, sx blind, INFx or xNTP per my cognitive functions and MBTI; I type 40 wpm, the X-ray was fine.",True)
    for gone in ('ennea','type 4','sx','INFx','xNTP','cognitive functions','MBTI'): assert gone not in out
    for kept in ('I type 40 wpm','X-ray','blind'): assert kept in out
