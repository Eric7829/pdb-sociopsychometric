"""Encoder input per segment: marked text plus the character ranges an encoder should pool (the creator's own words)."""
import re
from .text import JARGON,segment_spans

# Encoder-only extras (stored scrubbed_text is unchanged): Socionics codes, SLOAN codes, function-position words, partial MBTI codes
# (INFx, xNTP) and typology talk. PDB writers who talk typology are mostly voted N/P, so that talk acts as a hidden label.
SOCIONICS='ILE|SEI|ESE|LII|EIE|LSI|SLE|IEI|SEE|ILI|LIE|ESI|LSE|EII|IEE|SLI'
EXTRA_JARGON=re.compile(r'(?<!\w)(?:'+SOCIONICS+r'|[RSrs][CLcl][OUou][AEae][INin]|(?i:dom|aux|tert|auxiliary|tertiary|polr|blindspot)|'
                        r'(?i:(?:dominant|inferior|demon|trickster|critic|nemesis)\s+function)|'
                        r'(?=[A-Za-z]{0,3}[Xx])[IEXx][NSXx][TFXx][JPXx]|'
                        r'(?i:ennea\w*|tritype\w*|trifix\w*|mbti|myers[- ]briggs|socionics|typolog\w*|16\s?personalities|sakinorva|keirsey|'
                        r'sx|sp|instinctual\s+variants?|(?:cognitive\s+)?function\s+stacks?|cognitive\s+functions?|type\s+[1-9](?!\d)))(?!\w)')

def clean(piece,scrub):
    return re.sub(r'\s+',' ',EXTRA_JARGON.sub('',JARGON.sub('',piece)) if scrub else piece).strip()

def encoder_view(store,profile_num,*,scrub=True,with_context=True,mask_prompts=True):
    """[(text, pooled_ranges)] in snapshot segment order. Prompts are wrapped [Q]..[/Q], quotes [QUOTE]..[/QUOTE], reply
    context [CTX]..[/CTX] (prepended, never pooled); answer labels are dropped. scrub applies to every piece: pooled tokens
    attend to unpooled context in a bidirectional encoder, so unscrubbed context would leak type labels. Unannotated segments fall back to
    on-the-fly questionnaire spans without quotes or context."""
    q=store.connection.execute; out=[]
    for s in store.load_snapshot(profile_num).segments:
        rows=q('SELECT kind,start_pos,end_pos FROM text_spans WHERE profile_num=? AND source_type=? AND source_ref=? ORDER BY ordinal',
               [profile_num,s.source_type,s.source_ref]).fetchall()
        spans=rows or [(x.kind,x.start,x.end) for x in segment_spans(s.original_text)]
        ctx=q('SELECT context_text FROM reply_context WHERE comment_id=?',[s.source_ref]).fetchone() if with_context and s.source_type!='GOOGLE_DOC' else None
        text=''; ranges=[]
        def put(piece,pool):
            nonlocal text
            if not piece: return
            if text: text+=' '
            if pool: ranges.append((len(text),len(text)+len(piece)))
            text+=piece
        if ctx and clean(ctx[0],scrub): put('[CTX] '+clean(ctx[0],scrub)+' [/CTX]',False)
        for kind,a,b in spans:
            piece=s.original_text[a:b]
            if kind=='ANSWER_LABEL': continue
            if kind=='QUOTE':
                if clean(piece,scrub): put('[QUOTE] '+clean(piece,scrub)+' [/QUOTE]',False)
            elif kind=='PROMPT':
                body=clean(piece,scrub)
                if not body: continue
                put('[Q]',False); put(body,not mask_prompts); put('[/Q]',False)
            else: put(clean(piece,scrub),True)
        out.append((text,ranges))
    return out
