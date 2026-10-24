import httpx
import pytest
from pdb_ingest.models import Comment
from pdb_ingest.gdocs import extract_documents, fetch_document


def test_deduplicate_links():
    comments = [Comment(str(i), None, 'user', 'https://docs.google.com/document/d/abc_123/edit?usp=sharing') for i in range(2)]
    assert extract_documents(comments) == {'abc_123': ['0','1']}


@pytest.mark.parametrize('status', [401,403,404,410])
async def test_unavailable(status):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status))) as client:
        d = await fetch_document('abc_123', ['0'], client)
        assert d.status == 'UNAVAILABLE' and d.text is None


async def test_export_and_html_rejection():
    def handler(request):
        assert str(request.url) == 'https://docs.google.com/document/d/abc_123/export?format=txt'
        return httpx.Response(200, text='\ufeffcaf\u00e9', headers={'content-type': 'text/plain; charset=utf-8'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (await fetch_document('abc_123', ['0'], client)).text == 'caf\u00e9'
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200,text='<html>Login</html>',headers={'content-type':'text/html'}))) as client:
        assert (await fetch_document('abc_123', ['0'], client)).status == 'UNAVAILABLE'


async def test_size_limit_and_redirect():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200,content=b'x'*21,headers={'content-type':'text/plain'}))) as client:
        assert (await fetch_document('abc', [], client, max_bytes=20)).status == 'ERROR'
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(302,headers={'location':'https://evil.example/a'}))) as client:
        assert (await fetch_document('abc', [], client)).status == 'UNAVAILABLE'
