import math
import pytest
from pdb_ingest.models import Comment, SelfVote, Profile, Page, PaginationError
from pdb_ingest.crawl import collect_pages, match_creator
from pdb_ingest.votes import build_distributions
from pdb_ingest.text import make_segment, ramble_ratio


async def test_all_pages_and_overlap():
    pages = {None: Page([1, 2], 'a', False), 'a': Page([2], 'b', False), 'b': Page([3], None, True)}
    async def fetch(token): return pages[token]
    assert await collect_pages(fetch) == [1, 2, 3]


async def test_repeated_token_raises():
    async def fetch(token): return Page([1], 'a', False)
    with pytest.raises(PaginationError): await collect_pages(fetch)


def test_creator_identity():
    p = Profile(1, 'person', '42', 'old')
    a = Comment('a', '42', 'renamed', 'mine')
    b = Comment('b', '99', 'old', 'theirs', True)
    assert match_creator(p, [a, b]).comments == [a]
    assert match_creator(Profile(2, 'x'), [a]).status == 'UNKNOWN'
    assert match_creator(Profile(2, 'x', None, 'Caf\u00e9'), [Comment('c', None, 'Cafe\u0301', 'yes')]).status == 'MATCHED'


def test_votes_are_auditable():
    counts = {'MBTI': {'INFP': 25, 'INTP': 5, 'INFJ': 1}}
    evidence = [SelfVote('MBTI', 'INFP', 'a', True)] * 2
    d = build_distributions(counts, evidence)[0]
    assert d.observed_counts == counts['MBTI']
    assert d.adjusted_counts == {'INFP': 24, 'INTP': 5, 'INFJ': 1}
    assert d.total_system_votes == 30
    assert sum(d.simplex.values()) == pytest.approx(1)
    assert d.shannon_entropy == pytest.approx(-sum(p*math.log2(p) for p in d.simplex.values()))
    assert build_distributions(counts, [SelfVote('JUNGIAN', 'IF(N)', 'a', True)])[0].total_system_votes == 31
    conflict = evidence + [SelfVote('MBTI', 'INTP', 'b', True)]
    assert build_distributions(counts, conflict)[0].self_vote_status == 'CONFLICT'


def test_zero_sparse_invalid_votes():
    d = build_distributions({'MBTI': {'INFP': 1}}, [SelfVote('MBTI', 'INFP', 'a', True)])[0]
    assert d.simplex == {} and d.shannon_entropy is None
    for n in (-1, 1.5, True):
        with pytest.raises(ValueError): build_distributions({'MBTI': {'INFP': n}}, [])
    assert not build_distributions({'MBTI': {'INFP': 9}}, [])[0].training_eligible
    assert build_distributions({'MBTI': {'INFP': 10}}, [])[0].training_eligible
    assert not build_distributions({'BIG5_E': {'0%': 10}}, [])[0].training_eligible


def test_text_views():
    s = make_segment('AUTHOR_COMMENT', '1', 'Q1: What do you fear? I fear isolation.\n\nINFP')
    assert s.modality == 'QUESTIONNAIRE'
    assert 'I fear isolation.' in s.raw_text and 'What do you fear?' not in s.raw_text
    assert 'INFP' in s.raw_text and 'INFP' not in s.scrubbed_text
    codes = 'INFP INTP XXXX Fi Ne Si Te IF(N) IT(N) 4w5 5w4 SO4 sx/sp 459 ELVF LEVF 1E 3V Melancholic [Dominant] Melancholic-Phlegmatic'
    assert not make_segment('AUTHOR_COMMENT', '2', codes).scrubbed_text.strip()
    assert make_segment('AUTHOR_COMMENT', '3', 'fine neon sister team').scrubbed_text == 'fine neon sister team'
    a = make_segment('AUTHOR_COMMENT', '4', 'one two three')
    b = make_segment('AUTHOR_COMMENT', '5', 'Q1: Why? Answer')
    assert ramble_ratio([a,b]) == .75
    assert ramble_ratio([]) is None
