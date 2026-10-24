from __future__ import annotations
from dataclasses import asdict
from datetime import datetime,timezone
import asyncio
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit,parse_qsl
import uuid
from .adapters import parse_profile
from .capture import Capture
from .client import PdbClient
from .crawl import collect_comments,match_creator
from .gdocs import extract_documents,fetch_document
from .models import (Profile,Snapshot,CreatorMatch,Document,AccessBlocked,PaginationError,SchemaError,MissingEvidence,
                     ProfileUnavailable,profile_refusal)
from .text import make_segment,ramble_ratio
from .votes import build_distributions

class DocumentSource:
    def __init__(self,http_client,capture_dir):
        self.http_client=http_client
        self.path=capture_dir/'documents.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.cache={}
        self.attempt=None

    async def fetch(self,key,refs,profile_num):
        if key not in self.cache:
            self.cache[key]=await fetch_document(key,[],self.http_client)
        cached=self.cache[key]
        doc=Document(key,refs,cached.status,cached.text,cached.error)
        with self.path.open('a',encoding='utf-8') as out:
            out.write(json.dumps({'profile_num':profile_num,'attempt':self.attempt,'document':asdict(doc)},ensure_ascii=False)+'\n')
        return doc

async def ingest_profile(client,profile_num,docs_client,max_comments=None):
    detail=await client.detail(profile_num)
    profile=parse_profile(detail)
    comments,truncated=await collect_comments(client,profile_num,max_comments)
    creator=match_creator(profile,comments)
    authored=sorted(creator.comments,key=lambda c:(not c.is_pinned,c.comment_id))
    segments=[make_segment('PINNED_COMMENT' if c.is_pinned else 'AUTHOR_COMMENT',c.comment_id,c.text) for c in authored if c.text.strip()]
    docs=[]; issues=[]
    complete=creator.status=='MATCHED'
    if not complete: issues.append('Creator identity '+creator.status)
    if truncated: issues.append(f'Comments limited to the first {max_comments} top-level threads')
    for key,refs in sorted(extract_documents(authored).items()):
        doc=await docs_client.fetch(key,refs,profile_num)
        docs.append(doc)
        if doc.status=='AVAILABLE' and doc.text is not None:
            segments.append(make_segment('GOOGLE_DOC',key,doc.text))
        else:
            issues.append(f'Google document {key}: {doc.status}')
            if doc.status=='ERROR': complete=False
    distributions=build_distributions(await client.tallies(profile_num),[v for c in authored for v in c.self_votes])
    return Snapshot(profile,segments,docs,distributions,creator,ramble_ratio(segments),complete,issues,detail.captured_at,
                    max_comments,truncated)

def failed_snapshot(profile_num,error):
    return Snapshot(Profile(profile_num,str(profile_num)),[],[],[],CreatorMatch([],'NONE','UNKNOWN'),None,False,
                    [f'{type(error).__name__}: {error}'],datetime.now(timezone.utc).isoformat())

async def process_profile(client,store,docs_client,profile_num,max_comments=None):
    # Tag every recorded response so replay can reconstruct exactly one attempt.
    targets=[t for t in (getattr(client,'transport',None),docs_client) if t is not None]
    for t in targets: t.attempt=f'{profile_num}:{uuid.uuid4().hex}'
    try:
        snapshot=await ingest_profile(client,profile_num,docs_client,max_comments)
    except Exception as error:
        store.save_snapshot(failed_snapshot(profile_num,error))
        if isinstance(error,(AccessBlocked,SchemaError,PaginationError,ProfileUnavailable)): raise
        return False
    finally:
        for t in targets: t.attempt=None
    store.save_snapshot(snapshot)
    return snapshot.complete

async def run_crawl(client,store,docs_client,limit,*,max_profiles=500,max_comments=None):
    if limit is not None and limit<=0: raise ValueError('limit must be positive')
    if max_profiles<=0: raise ValueError('max_profiles must be positive')
    old_scope,_=store.load_checkpoint('scope')
    if old_scope is not None and old_scope!=str(max_profiles):
        raise ValueError('Use the same max_profiles when resuming, or start a new database')
    store.checkpoint('scope',str(max_profiles),True)
    # Older databases have no comment scope; they were crawled without a cap.
    old_comments,_=store.load_checkpoint('comment_scope')
    comment_scope=str(max_comments or 'all')
    if old_comments is not None and old_comments!=comment_scope:
        raise ValueError('Use the same max_comments when resuming, or start a new database')
    store.checkpoint('comment_scope',comment_scope,True)
    listing_sort=getattr(client,'sort',None) or 'hot'
    old_sort,_=store.load_checkpoint('listing_sort')
    if old_sort is not None and old_sort!=listing_sort:
        raise ValueError('Use the same listing sort when resuming, or start a new database')
    store.checkpoint('listing_sort',listing_sort,True)
    attempted=set(); succeeded=0; failed=0; unavailable=0; seen_tokens=set()
    # max_profiles counts usable profiles: unavailable ones (friends-only, deactivated) are replaced from the listing.
    usable=lambda:store.connection.execute('SELECT count(*) FROM crawl_queue WHERE NOT unavailable').fetchone()[0]
    while True:
        for ref in store.pending():
            if ref.profile_num in attempted: continue
            if limit is not None and len(attempted)>=limit: break
            attempted.add(ref.profile_num)
            print(json.dumps({'event':'profile','profile_num':ref.profile_num,'processed':len(attempted)}),flush=True)
            try:
                if await process_profile(client,store,docs_client,ref.profile_num,max_comments): succeeded+=1
                else: failed+=1
            except ProfileUnavailable:
                store.mark_unavailable(ref.profile_num); unavailable+=1
        if limit is not None and len(attempted)>=limit: break
        token,complete=store.load_checkpoint('listing')
        if complete and (store.load_checkpoint('category_terminal')[1] or usable()>=max_profiles): break
        if token in seen_tokens: raise PaginationError('Listing pagination token cycle')
        seen_tokens.add(token)
        page=await client.listing(token)
        if not page.terminal and (page.next_token is None or page.next_token==token):
            raise PaginationError('Listing pagination failed to advance')
        # Queue insertion and advancement are one durable operation.
        store.connection.execute('BEGIN')
        try:
            existing={r[0] for r in store.connection.execute('SELECT profile_num FROM crawl_queue').fetchall()}
            count=usable(); fresh=[]; cut=False
            for ref in page.items:
                if ref.profile_num not in existing:
                    if count>=max_profiles: cut=True; break
                    fresh.append(ref); existing.add(ref.profile_num); count+=1
            store.enqueue(fresh)
            capped=count>=max_profiles
            # A page cut short by the cap is re-read on refill (queued items dedupe), so listing order is kept.
            if cut: seen_tokens.discard(token)
            store.checkpoint('listing',token if cut else page.next_token,page.terminal or capped)
            store.checkpoint('category_terminal',None,page.terminal)
            store.connection.execute('COMMIT')
        except BaseException:
            store.connection.execute('ROLLBACK'); raise
    outstanding=len(store.pending())
    return {'processed':len(attempted),'succeeded':succeeded,'failed':failed,'unavailable':unavailable,'outstanding':outstanding,
            'max_profiles':max_profiles,'usable':usable(),
            'scope_complete':store.load_checkpoint('listing')[1] and outstanding==0 and (usable()>=max_profiles or store.load_checkpoint('category_terminal')[1]),
            'full_crawl_complete':store.load_checkpoint('category_terminal')[1] and outstanding==0}

def request_key(url):
    parts=urlsplit(url)
    return parts.path,tuple(sorted(parse_qsl(parts.query)))

def single(rows,name):
    # Untagged (legacy) evidence may hold several responses for one request; never guess between them.
    if len({json.dumps(r,sort_keys=True) if isinstance(r,dict) else r.sha256 for r in rows})>1:
        raise SchemaError(f'Ambiguous replay {name}; record attempts or use one capture directory per run')
    return rows[-1]

class ReplayTransport:
    def __init__(self,rows): self.rows=rows

    async def get(self,path):
        key=request_key(path)
        if key not in self.rows: raise MissingEvidence('Missing replay request: '+path)
        c=single(self.rows[key],'request: '+path)
        if profile_refusal(c.url,c.payload): raise ProfileUnavailable(c.payload['message'])
        if c.status!=200 or (isinstance(c.payload,dict) and c.payload.get('code') in (401,403)):
            raise AccessBlocked('Recorded API response is blocked')
        return c

class ReplayDocuments:
    def __init__(self,rows): self.rows=rows
    async def fetch(self,key,refs,profile_num):
        if key not in self.rows: raise MissingEvidence('Missing Google document replay result')
        row=single(self.rows[key],'Google document: '+key)
        return Document(key,refs,row['status'],row['text'],row['error'])

def load_attempts(capture_dir):
    """Group evidence by recorded attempt (None = untagged legacy), in first-seen order."""
    attempts={}
    for line in (capture_dir/'pdb.jsonl').read_text(encoding='utf-8').splitlines():
        c=Capture(**json.loads(line))
        digest=hashlib.sha256(json.dumps(c.payload,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
        if digest!=c.sha256: raise SchemaError('Capture integrity mismatch')
        attempts.setdefault(c.attempt,{}).setdefault(request_key(c.url),[]).append(c)
    documents={}
    path=capture_dir/'documents.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            documents.setdefault((row.get('attempt'),row['profile_num']),{}).setdefault(row['document']['key'],[]).append(row['document'])
    return attempts,documents

def profile_ids(rows):
    return {int(key[0].rsplit('/',1)[1]) for key in rows if re.fullmatch(r'/api/v1/profile/\d+',key[0])}

def replay(capture_dir,store,max_comments=None):
    attempts,documents=load_attempts(capture_dir)
    ids=sorted(set().union(*map(profile_ids,attempts.values())))
    async def run():
        successful=unavailable=0
        for n in ids:
            # Replay each attempt alone, oldest first, as the live store saw them. Attempts with
            # missing evidence were interrupted and are skipped; responses are never mixed.
            names=[a for a in attempts if a is None]+[a for a in attempts if a is not None]
            names=[a for a in names if n in profile_ids(attempts[a])]
            usable=[]
            for a in names:
                try:
                    await ingest_profile(PdbClient(ReplayTransport(attempts[a])),n,ReplayDocuments(documents.get((a,n),{})),max_comments)
                except MissingEvidence:
                    continue
                except Exception:
                    pass
                usable.append(a)
            if not usable:
                a=names[-1]
                await process_profile(PdbClient(ReplayTransport(attempts[a])),store,ReplayDocuments(documents.get((a,n),{})),n,max_comments)
            ok=False
            try:
                for a in usable:
                    ok=await process_profile(PdbClient(ReplayTransport(attempts[a])),store,ReplayDocuments(documents.get((a,n),{})),n,max_comments) or ok
            except ProfileUnavailable:
                unavailable+=1; continue
            successful+=ok
        return {'processed':len(ids),'succeeded':successful,'failed':len(ids)-successful-unavailable,'unavailable':unavailable}
    return asyncio.run(run())
