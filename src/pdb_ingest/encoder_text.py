"""Encoder input per segment: marked text plus the character ranges an encoder should pool (the creator's own words)."""
import re
from .text import JARGON,segment_spans

def clean(piece,scrub):
    return re.sub(r'\s+',' ',JARGON.sub('',piece) if scrub else piece).strip()

def encoder_view(store,profile_num,*,scrub=True,with_context=True,mask_prompts=True):
    """[(text, pooled_ranges)] in snapshot segment order. Prompts are wrapped [Q]..[/Q], quotes [QUOTE]..[/QUOTE], reply
    context [CTX]..[/CTX] (prepended, never scrubbed or pooled); answer labels are dropped. Unannotated segments fall back to
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
        if ctx and ctx[0].strip(): put('[CTX] '+clean(ctx[0],False)+' [/CTX]',False)
        for kind,a,b in spans:
            piece=s.original_text[a:b]
            if kind=='ANSWER_LABEL': continue
            if kind=='QUOTE': put('[QUOTE] '+clean(piece,False)+' [/QUOTE]',False)
            elif kind=='PROMPT':
                body=clean(piece,scrub and not mask_prompts)
                if not body: continue
                put('[Q]',False); put(body,not mask_prompts); put('[/Q]',False)
            else: put(clean(piece,scrub),True)
        out.append((text,ranges))
    return out
