from pdb_ingest.text import make_segment

def test_exact_questionnaire_threshold():
    text='\n\n'.join(['Why?']*3+['I enjoy writing.']*17)
    assert make_segment('AUTHOR_COMMENT','x',text).modality=='RAMBLE'

def test_lists_are_not_question_prompts():
    text='Books I love:\n\n- Anna Karenina\n\n- Confessions'
    assert make_segment('AUTHOR_COMMENT','x',text).raw_text==text

def test_mixed_sections_preserve_prose():
    s=make_segment('GOOGLE_DOC','x','one two three\n\nQUESTIONNAIRE:\n\nQ1: Why? Answer')
    assert s.modality=='MIXED'
    assert s.ramble_word_count==3 and s.word_count==4

def test_explicit_answer_delimiter_without_question_mark():
    s=make_segment('AUTHOR_COMMENT','x','Q1: Describe yourself. Answer: I love books.')
    assert s.modality=='QUESTIONNAIRE'
    assert s.raw_text=='I love books.' and s.scrubbed_text=='I love books.'

def test_ambiguous_explicit_prompt_is_preserved():
    s=make_segment('AUTHOR_COMMENT','x','Question: Describe your childhood. It was quiet.')
    assert s.raw_text=='Describe your childhood. It was quiet.'

def test_answer_label_on_its_own_line_is_stripped():
    s=make_segment('AUTHOR_COMMENT','x','Q1: What do you fear?\nA1: Isolation.')
    assert s.raw_text=='Isolation.'

def test_ap_scrub_requires_permutations():
    text='I FEEL very happy. It FELL. VELL LLLL'
    assert make_segment('AUTHOR_COMMENT','x',text).scrubbed_text==text
    assert make_segment('AUTHOR_COMMENT','x','I am VLFE and FVLE.').scrubbed_text=='I am and .'

def test_numbers_not_valid_tritypes_are_preserved():
    assert make_segment('AUTHOR_COMMENT','x','I counted 999 objects.').scrubbed_text=='I counted 999 objects.'
