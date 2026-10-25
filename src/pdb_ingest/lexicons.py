"""Psycholinguistic document features from free word norms (Mairesse et al. 2007 style, without the commercial LIWC):
Brysbaert et al. (2014) concreteness, the Glasgow Norms (Scott et al. 2019) and the NRC Word-Emotion Association Lexicon
(Mohammad & Turney 2013), plus built-in function-word lists. Lexicon files live under data/external/lexicons (git-ignored);
without them every lexicon feature falls back to its scale midpoint or zero."""
from collections import Counter
import csv
from dataclasses import dataclass,field
from pathlib import Path
import re
import numpy as np

LEXICON_ROOT=Path('data/external/lexicons')
EMOTIONS=['anger','anticipation','disgust','fear','joy','sadness','surprise','trust','positive','negative']
GLASGOW={'AROU':'arousal','VAL':'valence','DOM':'dominance','IMAG':'imageability','FAM':'familiarity','AOA':'aoa'}
WORDS={'i':"i me my mine myself i'm i've i'd i'll im ive",'we':"we us our ours ourselves we're we've",
       'you':"you your yours yourself yourselves you're u ur",'they':"they them their theirs themselves he she him her his hers they're",
       'negation':"no not never none nothing nobody neither nor cannot don't can't won't isn't didn't doesn't wasn't aren't couldn't wouldn't shouldn't",
       'article':'a an the','preposition':'in on at of to for with about from by into over under between through during after before without within across',
       'auxiliary':'am is are was were be been being have has had do does did will would shall should can could may might must',
       'tentative':'maybe perhaps possibly probably might guess seems seem unsure somewhat kinda sorta apparently',
       'certainty':'always never definitely certainly absolutely sure obviously clearly undoubtedly totally completely'}
WORDS={k:frozenset(v.split()) for k,v in WORDS.items()}; WORDS_ALL=frozenset().union(*WORDS.values())
FEATURES=([f'{k}_rate' for k in WORDS]+[f'nrc_{e}_rate' for e in EMOTIONS]+['concreteness_mean','concrete_share']+
          [f'glasgow_{v}' for v in GLASGOW.values()]+['mean_word_length','words_per_sentence','ttr_500','question_rate','exclamation_rate','lexicon_coverage'])
TOKEN=re.compile(r"[a-z']+")

@dataclass
class Lexicons:
    concreteness:dict=field(default_factory=dict); glasgow:dict=field(default_factory=dict)
    emotions:dict=field(default_factory=dict); available:bool=False

def load_lexicons(root=None):
    root=Path(root or LEXICON_ROOT); lex=Lexicons()
    f=root/'brysbaert-concreteness'/'concreteness.tsv'
    if f.exists():
        for r in csv.DictReader(f.open(encoding='utf-8'),delimiter='\t'):
            try: lex.concreteness[r['Word'].lower()]=float(r['Conc.M'])
            except (TypeError,ValueError): pass
    f=root/'glasgow-norms'/'glasgow_norms.csv'
    if f.exists():
        rows=list(csv.reader(f.open(encoding='utf-8-sig'))); head=rows[0]; col={GLASGOW[h]:i for i,h in enumerate(head) if h in GLASGOW}
        for r in rows[2:]:
            try: lex.glasgow[r[0].lower()]={k:float(r[i]) for k,i in col.items()}
            except (IndexError,ValueError): pass
    f=root/'nrc-emolex'/'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    if f.exists():
        sets={e:set() for e in EMOTIONS}
        for line in f.open(encoding='utf-8'):
            p=line.strip().split('\t')
            if len(p)==3 and p[2]=='1' and p[1] in sets: sets[p[1]].add(p[0].lower())
        lex.emotions={e:frozenset(s) for e,s in sets.items()}
    lex.available=bool(lex.concreteness or lex.glasgow or lex.emotions)
    return lex

def lemma(w,table):
    """w, or its singular when only that is in the norms (Brysbaert lists lemmas: 'bike', not 'bikes')."""
    if w in table: return w
    for s in (w[:-1] if w.endswith('s') else None,w[:-2] if w.endswith('es') else None):
        if s and s in table: return s
    return w

def emotional_sentences(text,lex,min_share=.3,min_words=50):
    """Only the sentences holding at least one NRC emotion word (the eight emotions, not bare positive/negative), as in
    Majumder et al. (2017); the full text when that keeps under min_share of the words or under min_words words."""
    emo=frozenset().union(*(lex.emotions.get(e,frozenset()) for e in EMOTIONS[:8]))
    if not emo or not text.strip(): return text
    parts=re.findall(r'[^.!?\n]+[.!?\n]*',text); kept=[p.strip() for p in parts if emo & set(TOKEN.findall(p.lower()))]
    k=sum(len(TOKEN.findall(p.lower())) for p in kept); n=len(TOKEN.findall(text.lower()))
    return ' '.join(kept) if kept and k>=min_words and k>=min_share*n else text

def doc_features(text,lex):
    """Fixed-length, always finite feature vector (order: FEATURES)."""
    toks=TOKEN.findall(text.lower()); n=max(len(toks),1); out=[]; raw=Counter(toks)
    c=Counter()   # lexicon lookups use singulars; function-word and style counts use the raw tokens
    for w,v in raw.items(): c[w if w in WORDS_ALL else lemma(w,lex.concreteness)]+=v
    out+=[sum(v for w,v in raw.items() if w in s)/n for s in WORDS.values()]
    out+=[sum(v for w,v in c.items() if w in lex.emotions.get(e,()))/n for e in EMOTIONS]
    hits=[(lex.concreteness[w],v) for w,v in c.items() if w in lex.concreteness]; k=sum(v for _,v in hits)
    out+=[sum(s*v for s,v in hits)/k if k else 3.0,sum(v for s,v in hits if s>=4)/k if k else 0.0]
    g=[(lex.glasgow[w],v) for w,v in c.items() if w in lex.glasgow]; m=sum(v for _,v in g)
    out+=[sum(d[k_]*v for d,v in g)/m if m else 4.0 for k_ in GLASGOW.values()]
    sentences=[s for s in re.split(r'[.!?\n]+',text) if TOKEN.search(s.lower())]
    out+=[sum(len(w)*v for w,v in raw.items())/n,len(toks)/max(len(sentences),1),len(set(toks[:500]))/max(min(len(toks),500),1),
          text.count('?')/n,text.count('!')/n,k/n]
    return np.array(out,dtype=float)
