from pdb_ingest.text import segment_spans,make_segment
def kinds(text): return [(s.kind,text[s.start:s.end]) for s in segment_spans(text)]
def test_questionnaire_spans():
    t='Q1: Describe yourself. Answer: I love books.\nQ2: What do you fear? Isolation.'
    assert kinds(t)==[('PROMPT','Q1: Describe yourself.'),('ANSWER_LABEL','Answer:'),('ANSWER','I love books.'),
                      ('PROMPT','Q2: What do you fear?'),('ANSWER','Isolation.')]
def test_prose_is_one_span_per_paragraph():
    assert kinds('I write.\n\nI read.')==[('PROSE','I write.'),('PROSE','I read.')]
def test_offsets_survive_emoji():
    t='Q1: Why? 💀🎉 because.\n\nQ2: How? Slowly.\n\nQ3: When? Never.'
    assert ('ANSWER','Slowly.') in kinds(t)
def test_spans_match_raw_text():
    t='intro words\n\nQUESTIONNAIRE:\n\nQ1: Why? Answer'
    assert '\n\n'.join(t[s.start:s.end] for s in segment_spans(t) if s.kind in ('PROSE','ANSWER'))==make_segment('X','x',t).raw_text

from pdb_ingest.quotes import quote_ranges,merge_quotes
def test_gt_lines_are_quotes():
    t='> are you sure you are an introvert\nYes, very sure.'
    assert [t[a:b] for a,b in quote_ranges(t,[])]==['> are you sure you are an introvert']
def test_shared_six_word_run_is_quote():
    other='I think you are clearly a sensor because you love details'
    t='You said "you are clearly a sensor because you love" but no.'
    (a,b),=quote_ranges(t,[other])
    assert 'you are clearly a sensor because' in t[a:b] and 'but no' not in t[a:b]
def test_five_shared_words_are_not_a_quote():
    assert quote_ranges('we both love long walks today',['i love long walks today too'])==[]
def test_quote_inside_answer_splits_span():
    t='Q1: Why? Because "you are clearly a sensor because you love details" is wrong.\n\nQ2: Ok? Yes.\n\nQ3: No? No.'
    spans=merge_quotes(segment_spans(t),quote_ranges(t,['you are clearly a sensor because you love details']))
    assert [s.kind for s in spans if s.start>=t.index('Because')][:3]==['ANSWER','QUOTE','ANSWER']
    assert all(x.end<=y.start for x,y in zip(spans,spans[1:]))
from pdb_ingest.models import Comment
from pdb_ingest.context import reply_context
P=Comment('1','9','voter','you seem INTP',parent_id=None)
A=Comment('2','42','me','no, INFP',parent_id='1',reply_to_user_id='9')
B=Comment('3','7','other','why INFP?',parent_id='1',reply_to_user_id='42')
C=Comment('4','42','me','because Fi',parent_id='1',reply_to_user_id='7')
def test_context_is_latest_earlier_comment_by_target(): assert reply_context(C,[P,A,B,C]) is B
def test_falls_back_to_parent(): assert reply_context(A,[P,A,B,C]) is P
def test_missing_reply_target_gives_no_context():
    assert reply_context(Comment('5','42','me','hm',parent_id='1',reply_to_user_id='555'),[P]) is None
def test_self_reply_has_no_context():
    assert reply_context(Comment('6','42','me','also',parent_id='1',reply_to_user_id='42'),[P,A]) is None
def test_parse_comment_keeps_reply_target():
    from pdb_ingest.adapters import parse_comment
    assert parse_comment({'id':2,'user_id':42,'comment':'x','parent_id':1,'reply_to_user_id':9}).reply_to_user_id=='9'
