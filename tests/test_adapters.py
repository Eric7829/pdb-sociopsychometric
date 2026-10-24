import pytest
from pdb_ingest.adapters import parse_profile, parse_counts, parse_comment, parse_listing
from pdb_ingest.models import SchemaError

def test_verified_profile_fields():
    d={'id':1912391,'mbti_profile':'Esqueleto Narrativo','user_id':4957070,'contributor':'Esqueleto Narrativo','total_vote_counts':301,'personality_type':'IF(N)','wiki_description':'quote','breakdown_systems':{'1':[{'personality_type':'INFP','theCount':25}]},'systems':[{'id':1,'system_vote_count':25}]}
    p=parse_profile(d)
    assert p.creator_id=='4957070' and p.total_votes_all==301
    assert parse_counts(d)=={'MBTI':{'INFP':25}}

def test_comment_badge_no_cross_system_inference():
    d={'id':1,'user_id':4957070,'username':'Esqueleto Narrativo','comment':'hello','is_pinned':True,'parent_id':0,'vote_types':'IF(N) - Melancholic [Dominant] - Extraversion 0%','user_mbti':'XXXX'}
    c=parse_comment(d)
    assert c.author_id=='4957070' and c.is_pinned
    assert {v.system_name for v in c.self_votes}=={'JUNGIAN','TEMPERAMENT','BIG5_E'}
    assert all(v.confirmed for v in c.self_votes)
    assert c.flair=='XXXX'

def test_ap_badge_requires_permutation():
    from pdb_ingest.adapters import badge_system
    assert badge_system('ELVF')=='AP' and badge_system('VLFE')=='AP'
    assert badge_system('FEEL')=='UNMAPPED_BADGE' and badge_system('FELL')=='UNMAPPED_BADGE'

def test_listing_and_schema_drift():
    page=parse_listing({'profiles':[{'id':1}]},0,100)
    assert page.terminal and page.items[0].profile_num==1
    with pytest.raises(SchemaError): parse_listing({'data':[]},0,100)
