from __future__ import annotations
import unicodedata
from .models import CreatorMatch, PaginationError, SchemaError


async def collect_pages(fetch, limit=None):
    token, seen_tokens, seen_ids, items = None, set(), set(), []
    while True:
        page = await fetch(token)
        for item in page.items:
            key = getattr(item, 'profile_num', getattr(item, 'comment_id', None))
            if key is None:
                key = item
            if key not in seen_ids:
                seen_ids.add(key)
                items.append(item)
        if limit is not None and len(items) >= limit:
            return items[:limit]
        if page.terminal:
            if page.next_token is not None:
                raise SchemaError('Terminal page also contains a continuation token')
            return items
        if page.next_token is None:
            raise SchemaError('No pagination terminal signal or continuation token')
        if page.next_token in seen_tokens:
            raise PaginationError('Repeated pagination token')
        seen_tokens.add(page.next_token)
        token = page.next_token


def match_creator(profile, comments):
    unique = {c.comment_id: c for c in comments}.values()
    if profile.creator_id is not None:
        return CreatorMatch([c for c in unique if c.author_id == profile.creator_id], 'ACCOUNT_ID', 'MATCHED')
    if not profile.creator_username:
        return CreatorMatch([], 'NONE', 'UNKNOWN')
    name = unicodedata.normalize('NFC', profile.creator_username)
    matching = [c for c in unique if c.author_username and unicodedata.normalize('NFC', c.author_username) == name]
    ids = {c.author_id for c in matching if c.author_id is not None}
    if len(ids) > 1:
        return CreatorMatch([], 'USERNAME', 'AMBIGUOUS')
    return CreatorMatch(matching, 'USERNAME', 'MATCHED')


async def collect_comments(client, profile_num, max_comments=None):
    """Returns (comments, truncated). max_comments caps top-level threads (pinned ones come first in the stream);
    replies are fetched only for kept threads."""
    limit = None if max_comments is None else max_comments + 1
    comments = await collect_pages(lambda token: client.comments(profile_num, token), limit)
    truncated = max_comments is not None and len(comments) > max_comments
    comments = comments[:max_comments] if truncated else comments
    all_comments = {c.comment_id: c for c in comments}
    queue = list(comments)
    visited = set()
    while queue:
        comment = queue.pop(0)
        if comment.comment_id in visited:
            continue
        visited.add(comment.comment_id)
        if not comment.reply_count:
            continue
        for reply in await collect_pages(lambda token: client.replies(profile_num, comment.comment_id, token)):
            if reply.comment_id not in all_comments:
                queue.append(reply)
                all_comments[reply.comment_id] = reply
    return list(all_comments.values()), truncated
