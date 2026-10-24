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
