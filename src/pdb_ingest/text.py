"""Conservative heuristic text views; original evidence is never overwritten."""
from itertools import permutations
import re
from .models import Segment

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

def ramble_ratio(segments):
    total = sum(s.word_count for s in segments)
    return sum(s.ramble_word_count for s in segments) / total if total else None
