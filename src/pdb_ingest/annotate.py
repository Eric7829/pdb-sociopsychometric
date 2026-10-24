"""Backfill span annotations and reply context for stored segments, offline from a capture directory."""
from collections import defaultdict
import json
from .adapters import parse_comment
from .capture import jsonl_rows
from .context import reply_context
from .quotes import merge_quotes,quote_ranges
from .text import segment_spans

def captured_comments(capture_dir):
    """comment_id -> Comment from every captured comment/reply page; later captures win."""
    out={}
    for row in jsonl_rows(capture_dir/'pdb.jsonl'):
        if ('/api/v1/comments/' in row['url'] or '/api/v1/comment_replies' in row['url']) and isinstance(row['payload'],dict):
            for c in row['payload'].get('comments') or []: out[str(c['id'])]=parse_comment(c)
    return out

def annotate(store,capture_dir):
    comments=captured_comments(capture_dir); threads=defaultdict(list)
    for c in comments.values(): threads[c.parent_id or c.comment_id].append(c)
    stats={'profiles':0,'segments':0,'spans':0,'quotes':0,'contexts':0,'raw_text_mismatches':0}
    con=store.connection
    for (n,) in con.execute('SELECT profile_num FROM profiles WHERE complete ORDER BY profile_num').fetchall():
        snap=store.load_snapshot(n); creator=snap.profile.creator_id; span_rows=[]; ctx_rows=[]
        for s in snap.segments:
            base=segment_spans(s.original_text)
            if '\n\n'.join(s.original_text[x.start:x.end] for x in base if x.kind in ('PROSE','ANSWER'))!=s.raw_text: stats['raw_text_mismatches']+=1
            own=comments.get(s.source_ref) if s.source_type!='GOOGLE_DOC' else None
            thread=threads[own.parent_id or own.comment_id] if own else []
            # Only earlier comments can be quoted; later ones that quote the creator must not mark the creator's own words.
            others=[c.text for c in thread if c.author_id!=creator and c.text.strip() and int(c.comment_id)<int(own.comment_id)] if own else []
            quotes=quote_ranges(s.original_text,others)
            spans=merge_quotes(base,quotes,s.original_text)
            span_rows+=[(n,s.source_type,s.source_ref,i,x.kind,x.start,x.end) for i,x in enumerate(spans)]
            stats['quotes']+=sum(x.kind=='QUOTE' for x in spans)
            if own and own.parent_id:
                ctx=reply_context(own,thread)
                if ctx: ctx_rows.append((n,own.comment_id,ctx.comment_id,ctx.author_id,ctx.text))
            stats['segments']+=1
        con.execute('BEGIN')
        try:
            con.execute('DELETE FROM text_spans WHERE profile_num=?',[n]); con.execute('DELETE FROM reply_context WHERE profile_num=?',[n])
            if span_rows: con.executemany('INSERT INTO text_spans VALUES (?,?,?,?,?,?,?)',span_rows)
            if ctx_rows: con.executemany('INSERT INTO reply_context VALUES (?,?,?,?,?)',ctx_rows)
            con.execute('COMMIT')
        except BaseException:
            con.execute('ROLLBACK'); raise
        stats['profiles']+=1; stats['spans']+=len(span_rows); stats['contexts']+=len(ctx_rows)
    return stats
