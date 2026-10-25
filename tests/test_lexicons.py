from pathlib import Path
import time
import numpy as np
from pdb_ingest import lexicons
from pdb_ingest.lexicons import FEATURES,doc_features,load_lexicons

FIX=Path(__file__).resolve().parent/'fixtures'/'lexicons'
f=lambda v,name:v[FEATURES.index(name)]

def test_features_are_finite_without_hits():
    v=doc_features('zxq vbn 123 🤖',load_lexicons(FIX))
    assert len(v)==len(FEATURES) and np.isfinite(v).all() and f(v,'concreteness_mean')==3.0

def test_pronoun_and_emotion_rates():
    v=doc_features('I hate my job. I am so angry.',load_lexicons(FIX))
    assert f(v,'i_rate')==3/8 and f(v,'nrc_anger_rate')==2/8 and f(v,'nrc_negative_rate')==2/8

def test_concreteness_mean():
    v=doc_features('stone idea',load_lexicons(FIX))
    assert f(v,'concreteness_mean')==3.0 and f(v,'concrete_share')==.5 and f(v,'lexicon_coverage')==1.0

def test_glasgow_means_and_midpoint():
    v=doc_features('angry stone',load_lexicons(FIX)); w=doc_features('nothing here',load_lexicons(FIX))
    assert f(v,'glasgow_arousal')==4.0 and f(w,'glasgow_arousal')==4.0 and f(v,'glasgow_imageability')==6.0

def test_missing_lexicons(tmp_path,monkeypatch):
    monkeypatch.setattr(lexicons,'LEXICON_ROOT',tmp_path)
    lex=load_lexicons(); v=doc_features('I hate my job.',lex)
    assert lex.available is False and len(v)==len(FEATURES) and np.isfinite(v).all()

def test_long_text_is_fast():
    lex=load_lexicons(FIX); t=time.perf_counter(); doc_features(' '.join(['I hate my stone job.']*20000),lex)
    assert time.perf_counter()-t<2

def test_plural_falls_back_to_singular():
    v=doc_features('stones ideas',load_lexicons(FIX))
    assert f(v,'concreteness_mean')==3.0 and f(v,'lexicon_coverage')==1.0

def test_plural_fallback_does_not_touch_function_words():
    lex=load_lexicons(FIX); lex.concreteness['i']=1.0; lex.concreteness['hi']=2.0
    v=doc_features('his is',lex)
    assert f(v,'i_rate')==0 and f(v,'they_rate')==.5
