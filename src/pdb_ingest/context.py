"""The comment a creator reply answers: shown to the encoder as context, never pooled as the creator's own words."""

def reply_context(reply,thread):
    """Latest earlier comment in the thread by the replied-to user; the top-level comment only when no target is recorded.
    None for self-replies, missing targets and blank (hidden/deleted) text."""
    target=reply.reply_to_user_id
    if target is not None and target==reply.author_id: return None
    earlier=[c for c in thread if target is not None and c.author_id==target and int(c.comment_id)<int(reply.comment_id)]
    if earlier: found=max(earlier,key=lambda c:int(c.comment_id))
    elif target is None: found=next((c for c in thread if c.comment_id==reply.parent_id),None)
    else: return None
    if found is None or found.author_id==reply.author_id or not found.text.strip(): return None
    return found
