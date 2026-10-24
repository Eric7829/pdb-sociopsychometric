from pdb_ingest.cli import main

def test_help(capsys):
    import pytest
    with pytest.raises(SystemExit) as e: main(['--help'])
    assert e.value.code==0
    text=capsys.readouterr().out
    assert all(x in text for x in ['capture','crawl','profile','replay','export'])

def test_invalid_limit():
    import pytest
    with pytest.raises(SystemExit) as e: main(['crawl','--limit','0'])
    assert e.value.code==2
