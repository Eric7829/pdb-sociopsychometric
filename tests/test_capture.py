import json
import pytest
from pdb_ingest.capture import allowed_url, sanitize_url, Recorder


def test_capture_filter_and_redaction():
    assert allowed_url('https://api.personality-database.com/api/v1/example')
    assert not allowed_url('https://api.personality-database.com.evil.test/api/v1/example')
    assert not allowed_url('http://api.personality-database.com/api/v1/example')
    assert not allowed_url('https://api.personality-database.com/api/v2/oauth/email/send_digits')
    assert 'secret' not in sanitize_url('https://api.personality-database.com/api/v1/example?token=secret')


def test_duplicate_url_keeps_distinct_pages(tmp_path):
    recorder = Recorder(tmp_path)
    for page in (1, 2):
        recorder.record('POST', 'https://api.personality-database.com/api/v1/example', {'page': page}, 200, {'items': [page]})
    captures = [json.loads(line) for line in (tmp_path / 'pdb.jsonl').read_text().splitlines()]
    assert len(captures) == 2
    assert captures[0]['request_body'] != captures[1]['request_body']


def test_json_size_limit(tmp_path):
    with pytest.raises(ValueError, match='limit'):
        Recorder(tmp_path, max_bytes=20).record('GET', 'https://api.personality-database.com/api/v1/a', None, 200, {'x': 'x' * 30})

async def test_closed_browser_still_writes_summary(tmp_path):
    from pdb_ingest.capture import finalize_capture
    class ClosedPage:
        url='https://www.personality-database.com/profile/1'
        def is_closed(self): return True
    await finalize_capture(Recorder(tmp_path),ClosedPage(),None)
    assert json.loads((tmp_path/'summary.json').read_text())['browser_closed']
