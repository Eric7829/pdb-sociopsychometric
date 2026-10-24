# PDB scraping and ingestion scope

Source: the user's technical specification supplied in this chat on October 23, 2026. This document preserves the requirements for the first independently usable subsystem. Modeling, psychometric agents, and statistical audits need separate plans after ingestion works.

## Required outcome

Updated scope during execution: the user requested **only the top 500 profiles**. Interpret this as the first 500 unique profiles in PDB's default listing order; preserve discovery rank on resume. Stop discovery at 500. Report completion for this scope separately from completion of the remote category.

Enumerate every page of PDB's Pdb Profiles subcategory (`pid=1`, `cid=15`, `sub_cat_id=4065`); collect profile metadata, per-system vote counts, creator-authored comments including pinned comments and replies, and publicly accessible linked Google Documents. Produce repeatable local records with source provenance, original and normalized texts, raw and adjusted vote distributions, and explicit completeness and uncertainty statuses.

Use Python Playwright to capture PDB's internal REST JSON responses from `api.personality-database.com/api/v1/` and `/api/v2/`. Do not parse PDB HTML with BeautifulSoup. Discover actual request methods, endpoint paths, pagination fields, response shapes, creator identity, author vote evidence, and reply-loading behavior in a browser before implementing field mappings. Browser operation must allow a human to resolve a challenge; a challenge is an interruption, never an empty successful page.

Starting URLs:
- https://www.personality-database.com/profile?pid=1&cid=15&sub_cat_id=4065
- https://www.personality-database.com/profile/1912391/esqueleto-narrativo-pdb-profiles-mbti-personality-type

Creator matching uses stable account IDs where available, otherwise exact Unicode-normalized username matching. Preserve the match method. Never assume a pinned comment is creator-authored. Preserve flair and explicit self-claims separately from evidence of an actual vote. Only a confirmed vote for the same system can cause a single subtraction: `max(0, count - 1)`. Repeated badges do not imply repeated votes. Jungian `IF(N)` does not establish that an MBTI INFP vote was cast; no cross-system inference may alter counts. Missing or conflicting evidence must be recorded as uncertainty.

Extract Google Document keys using `https://docs.google.com/document/d/([a-zA-Z0-9_-]+)` from creator comments, fetch `/export?format=txt`, and retain each source comment association. Public link parameters do not prove public permissions. Treat 401/403/404 as `UNAVAILABLE`, transient exhaustion as `ERROR`, and non-text login/challenge pages as unavailable content. Preserve inline comments when exports fail.

Per segment, classify questionnaires when more than 15% of nonempty paragraphs match prompt syntax (question-mark endings or numbered/bulleted questions, `Q1:`, `Question:`). Remove template question text from training views while retaining the original source text and first-person answers. Continuous prose is `RAMBLE`; mixed questionnaire/answer documents are `QUESTIONNAIRE` for ratio calculation, reserving `MIXED` for prose with separately identifiable embedded questionnaire blocks. Compute ramble word ratio from retained answer/prose words, excluding prompts; count only ramble words in the numerator. Empty corpora yield null ratio.

Preserve the question-stripped raw view and produce TypoScrub's clean view removing explicit Four Letter/Jungian, Enneagram, instinct, tritype, AP/psychosophy and temperament labels. Preserve ordinary prose words and use case-sensitive token rules for ambiguous function abbreviations such as `Ne` and `Si`.

Store profiles, text_segments and vote_distributions with the user's field names in DuckDB. MBTI and ENNEAGRAM are core systems; INSTINCT, TRITYPE and AP are tier 2. Retain other observed systems, including individual Big Five sliders, with training eligibility false for Big Five. Store observed and adjusted counts separately; calculate simplex and Shannon entropy in bits from adjusted counts. Zero remaining votes produce an empty simplex and null entropy. Keep per-system totals distinct from the all-systems vote banner. Fewer than 10 adjusted votes means ineligible for that training head, without discarding observations.

## Implementation choices for review

- Standalone Python 3.11+ package, Playwright, httpx, DuckDB, pytest; leave `IPIP-NEO-120-compact/` intact. No agent execution or PDF analysis is needed for this subsystem.
- Browser-backed PDB access through the same authenticated browser context; direct httpx access only for public Google exports. No credentials or cookies in recorded fixtures.
- Persist raw JSON response envelopes locally for replay. Keep method, sanitized URL, request pagination body when needed, status, UTC timestamp, payload hash and JSON body; omit authorization and cookie headers.
- Default sequential requests with at least 2 seconds between PDB requests, 3 attempts for transport errors/429/5xx, capped exponential backoff (2, 4, 8 seconds) respecting Retry-After. Challenges and 401/403 stop PDB collection for human attention. Limit request time to 30 seconds, page navigation to 60 seconds, Google export to 5 MiB and PDB JSON to 20 MiB.
- Transactional profile replacement, persisted discovery queue and checkpoints, JSONL export, deterministic offline replay, and nonzero exit status for incomplete collection. Partial snapshots remain inspectable but cannot silently overwrite complete records.
- Exact PDB endpoint paths and fields are unknown as of planning. The web reader returned no usable listing data and could not access the example profile. Example numbers in the supplied spec are illustrative, not fixtures verified from PDB.

## Acceptance

Offline tests cover multi-page listings/comments/replies, identity ambiguity, evidence-preserving subtraction, Google export failures, text views, storage transactions and interruption/resumption. A bounded live smoke run captures the example profile plus at least two listing pages, or reports the specific access block. A full crawl is successful only when every discovered profile and pagination stream reaches a verified terminal state. Replaying a completed capture yields the same normalized records without network access.
