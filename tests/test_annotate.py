from urllib.parse import urlencode
from pdb_ingest.adapters import parse_comment
from pdb_ingest.annotate import annotate
from pdb_ingest.capture import Recorder
from pdb_ingest.models import Page
from pdb_ingest.pipeline import process_profile
from pdb_ingest.store import Store
from test_pipeline import Client,Docs

P={'id':10,'user_id':99,'username':'voter','comment':'you are clearly a sensor because you love details','reply_count':1}
R={'id':11,'user_id':42,'username':'owner','comment':'"you are clearly a sensor because you love details" is wrong, I daydream.','parent_id':10,'reply_to_user_id':99}
API='https://api.personality-database.com/api/v1/'

class Thread(Client):
    async def comments(self,n,token): return Page([parse_comment(P)],None,True)
    async def replies(self,n,c,token): return Page([parse_comment(R)],None,True)

async def build(tmp_path):
    rec=Recorder(tmp_path/'run')
    rec.record('GET',API+'comments/1?'+urlencode(dict(sort='NEW',offset=0,range='all',limit=100,version='W3')),None,200,{'comments':[P],'has_more':False})
    rec.record('GET',API+'comment_replies?'+urlencode(dict(limit=100,offset=0,comment_id=10)),None,200,{'comments':[R],'has_more':False})
    store=Store(tmp_path/'a.duckdb')
    await process_profile(Thread(),store,Docs(),1)
    return store

async def test_annotate_writes_spans_quotes_context(tmp_path):
    store=await build(tmp_path)
    with store:
        stats=annotate(store,tmp_path/'run')
        q=store.connection.execute
        kinds=q("select kind from text_spans where source_ref='11' order by ordinal").fetchall()
        assert [k for (k,) in kinds]==['QUOTE','PROSE'] and stats['quotes']==1
        assert q("select context_comment_id,context_user_id,context_text from reply_context where comment_id='11'").fetchone()==('10','99',P['comment'])

async def test_annotate_is_idempotent(tmp_path):
    store=await build(tmp_path)
    with store:
        annotate(store,tmp_path/'run'); first=[store.connection.execute(f'select count(*) from {t}').fetchone() for t in ('text_spans','reply_context')]
        annotate(store,tmp_path/'run'); assert [store.connection.execute(f'select count(*) from {t}').fetchone() for t in ('text_spans','reply_context')]==first

async def test_raw_text_invariant(tmp_path):
    store=await build(tmp_path)
    with store: assert annotate(store,tmp_path/'run')['raw_text_mismatches']==0

async def test_cli_annotate(tmp_path,capsys):
    import json
    from pdb_ingest.cli import main
    store=await build(tmp_path); store.connection.close()
    assert main(['annotate','--db',str(tmp_path/'a.duckdb'),'--capture-dir',str(tmp_path/'run')])==0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['quotes']==1

async def test_unicode_line_separators_inside_captured_text(tmp_path):
    from pdb_ingest.annotate import captured_comments
    rec=Recorder(tmp_path/'run')
    rec.record('GET',API+'comment_replies?'+urlencode(dict(limit=100,offset=0,comment_id=10)),None,200,
               {'comments':[{**R,'comment':'line one line two\x85end'}],'has_more':False})
    assert captured_comments(tmp_path/'run')['11'].text=='line one line two\x85end'

async def test_later_comment_quoting_the_creator_does_not_mark_creator_text(tmp_path):
    later={'id':12,'user_id':99,'username':'voter','comment':'you said you love details is wrong I daydream, no way','parent_id':10,'reply_to_user_id':42}
    store=await build(tmp_path)
    Recorder(tmp_path/'run').record('GET',API+'comment_replies?'+urlencode(dict(limit=100,offset=100,comment_id=10)),None,200,{'comments':[later],'has_more':False})
    with store:
        annotate(store,tmp_path/'run')
        spans=store.connection.execute("select kind,start_pos,end_pos from text_spans where source_ref='11' order by ordinal").fetchall()
    text=R['comment']
    assert [(k,text[a:b]) for k,a,b in spans]==[('QUOTE','"you are clearly a sensor because you love details"'),('PROSE','is wrong, I daydream.')]
