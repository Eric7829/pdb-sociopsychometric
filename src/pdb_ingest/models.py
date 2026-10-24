from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, TypeVar
from .capture import Capture

class SchemaError(ValueError): pass
class MissingEvidence(SchemaError): pass
class PaginationError(RuntimeError): pass
class AccessBlocked(RuntimeError): pass
class ProfileUnavailable(RuntimeError): pass  # profile-specific (deactivated, friends-only); the session is fine

PROFILE_PATHS=('/api/v1/profile/','/api/v1/comments/','/api/v1/comment_replies')

def profile_refusal(url,payload):
    """A JSON 403 with a reason on a profile-scoped endpoint, e.g. "This profile is inactive now!" or
    "The user has chosen to only allow friends to view this page." Login prompts stay session blocks."""
    if not isinstance(payload,dict) or payload.get('code')!=403: return False
    message=str(payload.get('message','')).lower()
    return bool(message.strip()) and any(p in url for p in PROFILE_PATHS) and not any(w in message for w in ('log in','login','sign in','signin'))

T = TypeVar('T')
@dataclass
class Page(Generic[T]):
    items: list[T]
    next_token: str | None
    terminal: bool

@dataclass
class ProfileRef:
    profile_num: int
    url: str

@dataclass
class Profile:
    profile_num: int
    profile_slug: str
    creator_id: str | None = None
    creator_username: str | None = None
    header_summary: str | None = None
    bio_quote: str | None = None
    total_votes_all: int | None = None

@dataclass
class SelfVote:
    system_name: str
    label: str
    evidence_ref: str
    confirmed: bool

@dataclass
class Comment:
    comment_id: str
    author_id: str | None
    author_username: str | None
    text: str
    is_pinned: bool = False
    parent_id: str | None = None
    self_votes: list[SelfVote] = field(default_factory=list)
    flair: str | None = None
    reply_count: int = 0

@dataclass
class CreatorMatch:
    comments: list[Comment]
    method: str
    status: str

@dataclass
class Document:
    key: str
    source_comment_ids: list[str]
    status: str
    text: str | None
    error: str | None = None

@dataclass
class Segment:
    source_type: str
    source_ref: str
    original_text: str
    raw_text: str
    scrubbed_text: str
    modality: str
    word_count: int
    ramble_word_count: int

@dataclass
class Distribution:
    system_name: str
    observed_counts: dict[str,int]
    adjusted_counts: dict[str,int]
    total_system_votes: int
    simplex: dict[str,float]
    shannon_entropy: float | None
    self_vote_status: str
    training_eligible: bool
    evidence: list[SelfVote] = field(default_factory=list)

@dataclass
class Snapshot:
    profile: Profile
    segments: list[Segment]
    documents: list[Document]
    distributions: list[Distribution]
    creator_match: CreatorMatch
    ramble_word_ratio: float | None
    complete: bool
    issues: list[str]
    captured_at: str
    comment_limit: int | None = None
    comments_truncated: bool = False
