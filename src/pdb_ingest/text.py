"""Conservative heuristic text views; original evidence is never overwritten."""
from itertools import permutations
import re
from .models import Segment, Span

AP_TYPES = frozenset(''.join(p) for p in permutations('ELVF'))
PROMPT = re.compile(r'^\s*(?:(?:[-*•]\s*)?(?:Q\d+\s*[:.)]|\d+[.)]|(?:\*\*)?Question(?:\*\*)?\s*:)|[-*•]\s+)', re.I)
EXPLICIT_PROMPT = re.compile(r'^\s*(?:Q\d+\s*:|(?:\*\*)?Question(?:\*\*)?\s*:)\s*', re.I)
ANSWER = re.compile(r'(?:^|(?<=\s))(?:\*\*)?(?:Answer|A\d*)(?:\*\*)?\s*:\s*', re.I)
JARGON = re.compile(
    r'(?<!\w)(?:[IE][NS][TF][JP]|XXXX|[FTNS][ie]|[IE][FTNS]\([FTNS]\)|'
    r'[1-9]w[1-9]|(?:SO|SP|SX|so|sp|sx)[1-9]|(?:sx|sp|so)/(?:sx|sp|so)|'
    r'(?:[234][567][891]|[234][891][567]|[567][234][891]|[567][891][234]|[891][234][567]|[891][567][234])|'
    + '|'.join(sorted(AP_TYPES)) + r'|[1-4][ELVF]|'
    r'(?:Melancholic|Phlegmatic|Sanguine|Choleric)(?:\s*\[Dominant\]|-(?:Melancholic|Phlegmatic|Sanguine|Choleric))?)(?!\w)')

def words(text):
    return len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text, re.UNICODE))

def prompt_answer(p):
    """Answer text on a prompt line; ambiguous explicit prompts keep their content."""
    delimiter = ANSWER.search(p, EXPLICIT_PROMPT.match(p).end() if EXPLICIT_PROMPT.match(p) else 0)
    if delimiter:
        return p[delimiter.end():].strip()
    _, mark, answer = p.partition('?')
    if mark:
        return answer.strip()
    return EXPLICIT_PROMPT.sub('', p, count=1).strip()

def make_segment(source_type, source_ref, text):
    split = re.split(r'\n\s*(?:#+\s*)?QUESTIONNAIRE\s*:\s*\n', text, maxsplit=1, flags=re.I)
    if len(split)==2 and split[0].strip():
        prose=make_segment(source_type,source_ref,split[0].strip())
        answers=make_segment(source_type,source_ref,split[1].strip())
        raw=prose.raw_text+'\n\n'+answers.raw_text
        clean=prose.scrubbed_text+'\n\n'+answers.scrubbed_text
        return Segment(source_type,source_ref,text,raw,clean,'MIXED',prose.word_count+answers.word_count,prose.word_count)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\r?\n', text) if p.strip()]
    flags = [bool((PROMPT.match(p) and '?' in p) or p.rstrip('* ').endswith('?') or re.match(r'^\s*(?:Q\d+\s*:|(?:\*\*)?Question(?:\*\*)?\s*:)',p,re.I)) for p in paragraphs]
    questionnaire = bool(paragraphs) and sum(flags) / len(paragraphs) > .15
    retained = []
    ramble_words = 0
    # Prompts define an answer span until the next prompt; preserve inline answers.
    for p, is_prompt in zip(paragraphs, flags):
        if questionnaire and is_prompt:
            answer = prompt_answer(p)
            if answer:
                retained.append(answer)
        elif questionnaire and ANSWER.match(p):
            if ANSWER.sub('', p, count=1).strip():
                retained.append(ANSWER.sub('', p, count=1).strip())
        else:
            retained.append(p)
            if not questionnaire:
                ramble_words += words(p)
    raw = '\n\n'.join(retained)
    clean = JARGON.sub('', raw)
    clean = '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in clean.splitlines()).strip()
    return Segment(source_type, source_ref, text, raw, clean,
                   'QUESTIONNAIRE' if questionnaire else 'RAMBLE', words(raw), ramble_words)

HEADER = re.compile(r'\n\s*(?:#+\s*)?QUESTIONNAIRE\s*:\s*\n', re.I)

def segment_spans(text, offset=0):
    """Character spans over text mirroring make_segment: PROSE/ANSWER spans joined by blank lines reproduce raw_text."""
    spans = []
    def add(kind, a, b):
        piece = text[a:b]
        a += len(piece) - len(piece.lstrip()); b -= len(piece) - len(piece.rstrip())
        if b > a: spans.append(Span(kind, offset + a, offset + b))
    head = HEADER.search(text)
    if head and text[:head.start()].strip():
        spans += segment_spans(text[:head.start()], offset)
        add('PROMPT', head.start(), head.end())
        return spans + segment_spans(text[head.end():], offset + head.end())
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\r?\n', text) if p.strip()]
    flags = [bool((PROMPT.match(p) and '?' in p) or p.rstrip('* ').endswith('?') or re.match(r'^\s*(?:Q\d+\s*:|(?:\*\*)?Question(?:\*\*)?\s*:)',p,re.I)) for p in paragraphs]
    questionnaire = bool(paragraphs) and sum(flags) / len(paragraphs) > .15
    cursor = 0
    for p, is_prompt in zip(paragraphs, flags):
        s = text.find(p, cursor); cursor = s + len(p)
        if questionnaire and is_prompt:
            explicit = EXPLICIT_PROMPT.match(p)
            delimiter = ANSWER.search(p, explicit.end() if explicit else 0)
            if delimiter:
                cut, label_end = delimiter.start(), delimiter.end()
            elif '?' in p:
                cut = label_end = p.index('?') + 1
            else:
                cut = label_end = explicit.end() if explicit else len(p)
            add('PROMPT', s, s + cut); add('ANSWER_LABEL', s + cut, s + label_end); add('ANSWER', s + label_end, s + len(p))
        elif questionnaire and ANSWER.match(p):
            m = ANSWER.match(p); add('ANSWER_LABEL', s, s + m.end()); add('ANSWER', s + m.end(), s + len(p))
        else:
            add('PROSE', s, s + len(p))
    return spans

def ramble_ratio(segments):
    total = sum(s.word_count for s in segments)
    return sum(s.ramble_word_count for s in segments) / total if total else None
