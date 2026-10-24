from dataclasses import asdict
import pytest
from pdb_ingest.models import Profile, ProfileRef, Snapshot, CreatorMatch
from pdb_ingest.store import Store
from pdb_ingest.text import make_segment

def snap(complete=True):
    return Snapshot(Profile(1,'person'), [make_segment('AUTHOR_COMMENT','a','one two')], [], [], CreatorMatch([], 'NONE','UNKNOWN'), 1., complete, [], '2026-10-23T00:00:00+00:00')

def test_store_roundtrip_queue_export(tmp_path):
    path=tmp_path/'test.duckdb'
    with Store(path) as store:
        store.enqueue([ProfileRef(1,'url'),ProfileRef(2,'url2')])
        store.checkpoint('listing','100',False)
        store.save_snapshot(snap())
        store.save_snapshot(snap())
        assert len(store.pending()) == 1
        assert asdict(store.load_snapshot(1)) == asdict(snap())
        assert store.connection.execute('select count(*) from text_segments').fetchone()[0] == 1
        store.save_snapshot(snap(False))
        assert store.load_snapshot(1).complete
        assert store.export_jsonl(tmp_path/'out.jsonl') == 1
    with Store(path) as store:
        assert store.load_checkpoint('listing') == ('100',False)
        assert len(store.pending()) == 1

def test_failed_replace_rolls_back(tmp_path):
    with Store(tmp_path/'test.duckdb') as store:
        store.save_snapshot(snap())
        changed=snap()
        changed.segments[0].source_type='BAD'
        with pytest.raises(Exception): store.save_snapshot(changed)
        assert store.load_snapshot(1).segments[0].source_type == 'AUTHOR_COMMENT'
