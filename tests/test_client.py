import pytest
from pdb_ingest.client import PdbClient
from pdb_ingest.capture import Capture
from pdb_ingest.models import SchemaError

class Transport:
    async def get(self,path):
        if '/comments/' in path:
            return Capture('GET',path,None,200,'now',{'comments':[{'id':1,'user_id':4,'comment':'hello','reply_count':0}], 'has_more':True,'next_offset':20},'hash')
        return Capture('GET',path,None,200,'now',{'comments':[], 'has_more':False,'next_offset':10},'hash')

async def test_comment_terminal_signals():
    client=PdbClient(Transport())
    page=await client.comments(1,None)
    assert page.next_token=='20' and not page.terminal
    assert (await client.replies(1,'1',None)).terminal

async def test_missing_pagination_signal_is_error():
    with pytest.raises(SchemaError): PdbClient.comment_page({'comments':[]})

async def test_listing_sort_matches_site_request():
    from pdb_ingest.pipeline import request_key
    seen=[]
    class Listing:
        async def get(self,path):
            seen.append(path); return Capture('GET',path,None,200,'now',{'profiles':[]},'hash')
    await PdbClient(Listing()).listing(None)
    await PdbClient(Listing(),sort='top').listing('100')
    assert 'sort' not in seen[0]
    site='https://api.personality-database.com/api/v1/profiles?offset=100&limit=100&cid=15&pid=1&sub_cat_id=4065&sort=top&cat_id=15&property_id=1'
    assert request_key(seen[1])==request_key(site)
