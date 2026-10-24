# PDB Scraping and Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable PDB subcategory scraper that reconstructs creator corpora and auditable crowd distributions in DuckDB.

**Architecture:** Discover and capture actual PDB JSON contracts in Playwright, then normalize them through versioned adapters. An orchestration layer exhausts pagination, downloads public Google exports, processes texts and votes, and commits complete profile snapshots transactionally. Raw captures support deterministic offline replay.

**Tech Stack:** Python 3.11+, Playwright Chromium, httpx, DuckDB, pytest, pytest-asyncio; Windows PowerShell CLI.

**Spec:** `docs/superpowers/specs/2026-10-23-pdb-ingestion-design.md` (read before execution).

## Global Constraints

- `pid=1`, `cid=15`, `sub_cat_id=4065`; enumerate multiple pages.
- "do not use HTML DOM parsing (`BeautifulSoup`) on PDB URLs."
- PDB data comes from internal JSON, never inferred from rendered summary strings.
- Google exports use `/export?format=txt`; HTTP `401/403/404` means `UNAVAILABLE`.
- Subtract at most one confirmed same-system creator vote per class: `max(0, count - 1)`.
- Questionnaire heuristic threshold is `>15%`; preserve answers and original source evidence.
- Core systems MBTI and ENNEAGRAM; tier 2 INSTINCT, TRITYPE, AP; Big Five sliders excluded from training targets.
- Preserve existing IPIP files. Models, inventories, and audits are out of scope.
- All shell commands use `rtk`; PowerShell-specific operations use `rtk proxy powershell -NoProfile -Command ...`.
- This workspace has no Git repository. Commit steps apply only if a repository exists by execution time; otherwise report them as unavailable and retain files. Do not stage unrelated questionnaire assets.

## Review Focus

- Repeated pagination cursors or overlapping pages must not truncate the crawl or loop forever (Tasks 1, 2, 7).
- Renamed accounts, lookalike usernames, pinned third-party comments, and contradictory badges must not contaminate identity or subtraction (Tasks 3, 5).
- HTTP 200 Google login pages and unavailable documents must not enter a person's corpus (Task 4).
- Combined prompt/answer lines and ordinary words resembling typology codes must retain meaningful answers (Task 5).
- A crash during replacement or a partial refresh must not destroy a complete snapshot or mark missing pages complete (Tasks 6, 7).

---

## File map and shared contracts

Create `pyproject.toml`, `.gitignore`, `README.md`, and `src/pdb_ingest/__init__.py`, `__main__.py`.

Focused modules:
- `models.py`: dataclasses and enums shared across boundaries.
- `capture.py`: browser response capture and sanitized response envelopes.
- `transport.py`: browser-context API requests, pacing/retries and access errors.
- `adapters.py`: observed PDB schema to typed pages/profile/comments; reject schema drift.
- `crawl.py`: pagination and creator comment/reply traversal.
- `gdocs.py`: URL extraction and bounded plaintext retrieval.
- `text.py`: original/raw/clean views and modality/word ratios.
- `votes.py`: evidence resolution, adjusted counts, simplex and entropy.
- `schema.sql`, `store.py`: DuckDB schema and transactions/checkpoints.
- `pipeline.py`, `cli.py`: ingestion, replay and CLI commands.

Shared dataclasses (define in Task 1; JSON serialization uses named fields):
- `Capture(method: str, url: str, request_body: dict | None, status: int, captured_at: str, payload: dict | list, sha256: str)`.
- `Page[T](items: list[T], next_token: str | None, terminal: bool)`; terminal requires observed server semantics, never merely zero new IDs.
- `ProfileRef(profile_num: int, url: str)`.
- `Profile(profile_num: int, profile_slug: str, creator_id: str | None, creator_username: str | None, header_summary: str | None, bio_quote: str | None, total_votes_all: int | None)`.
- `SelfVote(system_name: str, label: str, evidence_ref: str, confirmed: bool)`.
- `Comment(comment_id: str, author_id: str | None, author_username: str | None, text: str, is_pinned: bool, parent_id: str | None, self_votes: list[SelfVote], flair: str | None)`.
- `CreatorMatch(comments: list[Comment], method: str, status: str)` with status MATCHED/UNKNOWN/AMBIGUOUS.
- `Document(key: str, source_comment_ids: list[str], status: str, text: str | None, error: str | None)`.
- `Segment(source_type: str, source_ref: str, original_text: str, raw_text: str, scrubbed_text: str, modality: str, word_count: int, ramble_word_count: int)`.
- `Distribution(system_name: str, observed_counts: dict[str,int], adjusted_counts: dict[str,int], total_system_votes: int, simplex: dict[str,float], shannon_entropy: float | None, self_vote_status: str, training_eligible: bool)`.
- `Snapshot(profile: Profile, segments: list[Segment], documents: list[Document], distributions: list[Distribution], creator_match: CreatorMatch, ramble_word_ratio: float | None, complete: bool, issues: list[str], captured_at: str)`.

### Task 1: Discover and record the PDB API contract

**Files:** Create `pyproject.toml`, `.gitignore`, `src/pdb_ingest/{__init__,models,capture}.py`, `tests/test_capture.py`, `tests/fixtures/pdb/`, `docs/pdb-api-contract.md`.

**Interfaces:** Produces the shared dataclasses above, `allowed_url(url: str) -> bool`, `sanitize_url(url: str) -> str`, and `async capture_session(start_url: str, output_dir: Path, user_data_dir: Path) -> list[Capture]`. Capture command runs as `rtk python -m pdb_ingest.capture --url URL --output data/captures --session data/browser`. Error types `SchemaError`, `PaginationError` and `AccessBlocked` live in models.py. A versioned JSON contract loaded from `src/pdb_ingest/pdb_contract.json` records the observed method/path, request and field mappings; Task 1 creates it only from verified requests and Task 2 consumes it.

- [ ] **Step 1: Write failing capture tests.** Include assertions:
  ```python
  def test_capture_filter_and_redaction():
      assert allowed_url("https://api.personality-database.com/api/v1/example")
      assert not allowed_url("https://api.personality-database.com.evil.test/api/v1/example")
      assert "token=" not in sanitize_url("https://api.personality-database.com/api/v1/example?token=secret")
  ```
  Add `test_duplicate_url_keeps_distinct_pages`: two POST responses with the same URL but different pagination bodies remain two captures; `test_json_size_limit` rejects >20 MiB.
- [ ] **Step 2: Run** `rtk python -m pytest tests/test_capture.py -v`; expect import failure before implementation. Create the package/dependency metadata as part of this test cycle; install editable package with development dependencies using `rtk python -m pip install -e '.[dev]'` and browser with `rtk python -m playwright install chromium`.
- [ ] **Step 3: Implement** shared models and capture functions. Register response handlers before navigation, track and drain asynchronous handlers before closing, capture only exact API hostname/version paths and JSON, persist append-only sanitized envelopes with UTC time and SHA-256. Never record request cookies/authorization or browser storage in fixtures. Ignore runtime database/capture/browser directories in `.gitignore`.
- [ ] **Step 4: Run** `rtk python -m pytest tests/test_capture.py -v`; expect PASS.
- [ ] **Step 5: Discover live contracts.** Launch headful capture on the listing, navigate to profile 1912391, load two listing pages, all comment sorts needed for pinned/ordinary comments and nested replies, and per-system vote panels. Document exact methods, paths, parameters, terminal pagination signals, creator identity fields, tally/class fields and self-vote evidence fields. Store minimal sanitized real fixtures plus separately labeled synthetic edge cases. If blocked, document the observed block; do not invent endpoint paths or proceed with production mappings.
- [ ] **Step 6: Commit** only Task 1 files with `rtk git add ...` then `rtk git commit -m "feat: capture and document PDB JSON contracts"` when Git is available.

### Task 2: Browser transport and exhaustive pagination

**Files:** Create `src/pdb_ingest/{transport,adapters,crawl}.py`, `tests/test_transport.py`, `tests/test_pagination.py`.

**Interfaces:** Consumes Capture and Page. Produces `PdbClient(context, contract)` with `async listing(token: str | None) -> Page[ProfileRef]`, `async detail(profile_num: int) -> Capture`, `async comments(profile_num: int, token: str | None) -> Page[Comment]`, `async replies(profile_num: int, comment_id: str, token: str | None) -> Page[Comment]`, `async tallies(profile_num: int) -> dict[str,dict[str,int]]`; `async collect_pages(fetch: Callable[[str | None], Awaitable[Page[T]]]) -> list[T]`.

- [ ] **Step 1: Write failing tests.** `test_all_pages_and_overlap` asserts three pages yield unique IDs `[1,2,3,4]`; `test_duplicate_only_page_with_next_token_continues` still reaches page three; `test_repeated_token_raises` expects `PaginationError`; `test_missing_terminal_signal_raises` expects `SchemaError`. Transport tests assert 429 Retry-After is honored, a 503 retries at most 3 attempts, 403 raises `AccessBlocked`, 200 challenge HTML raises `AccessBlocked`, and unknown tally shape raises `SchemaError`. Inject clocks/sleep rather than waiting in tests.
- [ ] **Step 2: Run** `rtk python -m pytest tests/test_transport.py tests/test_pagination.py -v`; expect missing implementations.
- [ ] **Step 3: Implement** PdbClient against Task 1's recorded contract using the established browser session/context; preserve every JSON response through capture. Enforce 2-second minimum PDB request spacing, configured 30-second requests and 60-second navigation, 3 attempts for transient failures with capped 2/4/8-second backoff and Retry-After, and stop on access challenge. Adapt only verified schemas. Exhaust each stream using documented terminal signals, deduplicate by stable IDs, reject token cycles.
- [ ] **Step 4: Run** both test files; expect PASS with all requests mocked/offline.
- [ ] **Step 5: Commit** Task 2 files, message `feat: paginate PDB API through browser session`.

### Task 3: Profile ownership and complete creator comments

**Files:** Extend `adapters.py`, `crawl.py`; create `tests/test_creator.py`.

**Interfaces:** Produces `parse_profile(capture: Capture) -> Profile`, `match_creator(profile: Profile, comments: list[Comment]) -> CreatorMatch`, `async collect_comments(client: PdbClient, profile_num: int) -> list[Comment]`. Uses Task 2 page traversal for all top-level and reply streams.

- [ ] **Step 1: Write failing tests.** `test_creator_id_beats_changed_name` matches author ID despite renamed username; `test_id_mismatch_cannot_fall_back_to_name` excludes a different ID with the same name; `test_username_fallback_is_exact_normalized` accepts NFC-equivalent names and rejects substrings/lookalikes. `test_unknown_creator_is_not_guessed` gives UNKNOWN and no authored segments. `test_pinned_noncreator_is_excluded` excludes its text. `test_author_reply_on_last_page_is_included_once` includes a creator reply also present in a pinned stream only once.
- [ ] **Step 2: Run** `rtk python -m pytest tests/test_creator.py -v`; expect failures for missing matching/traversal behavior.
- [ ] **Step 3: Implement** these three interfaces from real field mappings. Match stable IDs first, exact NFC-normalized usernames only where IDs cannot resolve identity; conflicting accounts produce AMBIGUOUS. Traverse replies to noncreator comments as well as creator threads. Retain original comment text, vote evidence and flair separately; emit PINNED_COMMENT only for creator-pinned text, otherwise AUTHOR_COMMENT.
- [ ] **Step 4: Run** `rtk python -m pytest tests/test_creator.py -v`; expect PASS.
- [ ] **Step 5: Commit** Task 3 files, message `feat: reconstruct creator comment corpus`.

### Task 4: Public Google Document ingestion

**Files:** Create `src/pdb_ingest/gdocs.py`, `tests/test_gdocs.py`.

**Interfaces:** `extract_documents(comments: list[Comment]) -> dict[str,list[str]]` maps document keys to source comment IDs; `async fetch_document(key: str, source_comment_ids: list[str], client: httpx.AsyncClient) -> Document`.

- [ ] **Step 1: Write failing tests.** Assert repeated edit/share links produce one key with both comment IDs; export URL is exactly `https://docs.google.com/document/d/abc_123/export?format=txt`; 401/403/404 give UNAVAILABLE and null text; UTF-8/BOM accents survive; HTTP 200 HTML login yields UNAVAILABLE; exhausted timeout/429/5xx yields ERROR; >5 MiB streamed response stops with ERROR. Only keys extracted from creator comments are fetched.
- [ ] **Step 2: Run** `rtk python -m pytest tests/test_gdocs.py -v`; expect missing module failure.
- [ ] **Step 3: Implement** extraction with the spec regex and httpx streaming exports, 30-second timeout, bounded redirects and download size. Follow only HTTPS Google Docs/Drive/Googleusercontent export redirects; reject other hosts and login destinations. Accept validated plaintext, not HTML. Retry transient failures up to 3 attempts, respecting Retry-After; never retry unavailable exports. Deduplicate fetched keys without losing source associations.
- [ ] **Step 4: Run** `rtk python -m pytest tests/test_gdocs.py -v`; expect PASS without network.
- [ ] **Step 5: Commit** Task 4 files, message `feat: fetch linked public Google document text`.

### Task 5: Text views and auditable vote decontamination

**Files:** Create `src/pdb_ingest/{text,votes}.py`, `tests/test_text.py`, `tests/test_votes.py`.

**Interfaces:** `make_segment(source_type: str, source_ref: str, text: str) -> Segment`; `ramble_ratio(segments: list[Segment]) -> float | None`; `build_distributions(counts: dict[str,dict[str,int]], evidence: list[SelfVote]) -> list[Distribution]`. Segment concatenation in stored order defines corpus views.

- [ ] **Step 1: Write failing text tests.** Assert `"Q1: What do you fear? I fear isolation."` retains `"I fear isolation."` but removes the prompt, >15% tags QUESTIONNAIRE whereas exactly 15% does not, and continuous prose tags RAMBLE. Assert original_text remains unchanged and raw_text retains `INFP` while scrubbed_text removes it. Cover `INFP`, `INTP`, `XXXX`, `Fi`, `Ne`, `Si`, `Te`, `IF(N)`, `IT(N)`, `4w5`, `5w4`, `SO4`, `sx/sp`, `459`, `ELVF`, `LEVF`, `1E`, `3V`, `Melancholic [Dominant]`, `Melancholic-Phlegmatic`. Assert `"fine neon sister team"` survives. For 3 ramble words plus 1 questionnaire answer word assert ratio `0.75`; empty text gives None. MIXED fixtures count only prose words in numerator.
- [ ] **Step 2: Write failing vote tests.** Assert MBTI observed `{INFP:25, INTP:5, INFJ:1}` plus confirmed MBTI INFP evidence gives adjusted `{INFP:24, INTP:5, INFJ:1}`, total 30, simplex sum approximately 1 and entropy equals `-sum(p*log2(p))`. Duplicate evidence subtracts once; conflicting confirmed labels for one system subtract nothing and flag CONFLICT. A Jungian IF(N) claim or flair XXXX changes no MBTI count. Unknown/missing labels produce explicit status, not guessed mappings. Negative/nonintegral counts fail validation; zero adjusted total gives `{}` and null entropy; 9 votes is ineligible, 10 eligible, Big Five always ineligible. Observed counts remain unchanged.
- [ ] **Step 3: Run** `rtk python -m pytest tests/test_text.py tests/test_votes.py -v`; expect missing implementations.
- [ ] **Step 4: Implement** text interfaces with explicit prompt spans, retention of same-line answers, Unicode-safe token boundaries and conservative case-sensitive short-code matching. Implement vote interfaces using system IDs/classes from discovery, no cross-system inference, per-system evidence resolution and base-2 entropy. Missing tallies stay missing rather than fabricated zeros; retain unmapped systems with eligibility false. Preserve evidence references/status for every adjustment.
- [ ] **Step 5: Run** both test files; expect PASS.
- [ ] **Step 6: Commit** Task 5 files, message `feat: normalize corpus and audit creator vote subtraction`.

### Task 6: Transactional DuckDB snapshots and restart state

**Files:** Create `src/pdb_ingest/{schema.sql,store.py}`, `tests/test_store.py`.

**Interfaces:** `Store(path: Path)` with `save_snapshot(snapshot: Snapshot) -> None`, `load_snapshot(profile_num: int) -> Snapshot | None`, `enqueue(refs: list[ProfileRef]) -> None`, `pending() -> list[ProfileRef]`, `checkpoint(stream: str, token: str | None, complete: bool) -> None`, `load_checkpoint(stream: str) -> tuple[str | None,bool]`, `export_jsonl(path: Path) -> int`.

- [ ] **Step 1: Write failing tests.** Assert saving identical snapshots twice produces one profile and stable segment count, replacing a snapshot deletes stale segments/doc associations, and original/normalized text, observed/adjusted counts and document errors round-trip. Inject failure between deletes and inserts and assert full rollback. Assert incomplete refresh preserves prior complete snapshot and records an attempt; queue checkpoint survives close/reopen, completed queue entries are not pending, failed entries remain pending. JSONL emits exactly one full record per complete profile.
- [ ] **Step 2: Run** `rtk python -m pytest tests/test_store.py -v`; expect missing module failure.
- [ ] **Step 3: Implement** DuckDB-compatible versions of profiles/text_segments/vote_distributions using JSON rather than JSONB and sequence-generated segment IDs rather than BIGSERIAL. Include the user's profile fields (`self_vote_raw`, `last_updated_date`) and additional creator ID, capture time/completeness, original text, observed counts, evidence status and eligibility. Add documents/source associations, ingestion_attempts, crawl_queue and crawl_checkpoints tables, plus schema_version. Save normalized snapshots and mark queue complete in one transaction; preserve incomplete attempts separately. Use bound SQL parameters and deterministic JSONL order. Package schema.sql as package data. Do not create psychometric_evaluations until that subsystem exists.
- [ ] **Step 4: Run** `rtk python -m pytest tests/test_store.py -v`; expect PASS.
- [ ] **Step 5: Commit** Task 6 files, message `feat: persist resumable ingestion snapshots in DuckDB`.

### Task 7: CLI orchestration, replay and end-to-end acceptance

**Files:** Create `src/pdb_ingest/{pipeline,cli,__main__}.py`, `README.md`, `tests/test_pipeline.py`, `tests/test_cli.py`; extend capture/adapters for replay indexing.

**Interfaces:** `async ingest_profile(client: PdbClient, profile_num: int, docs_client: httpx.AsyncClient) -> Snapshot`; `async run_crawl(client: PdbClient, store: Store, docs_client: httpx.AsyncClient, limit: int | None) -> dict[str,int]`; `replay(capture_dir: Path, store: Store) -> dict[str,int]`; `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write failing integration tests.** A fixture with 3 listing pages, overlapping IDs, paginated comments/replies and unavailable Google export yields unique profiles, creator-only segments and correct clean distributions. Crash after discovery checkpoint and before profile commit, then resume: all pending IDs are processed and none duplicated. Repeated-token/schema error yields nonzero status and incomplete attempt. Unavailable docs retain inline text and document status; unknown creator cannot yield a complete corpus. A `--limit 1` run leaves the remaining queue pending and reports bounded status. Replay fixtures include Google export results, make zero network calls and match live normalized output excluding generated segment IDs.
- [ ] **Step 2: Run** `rtk python -m pytest tests/test_pipeline.py tests/test_cli.py -v`; expect missing orchestrator failures.
- [ ] **Step 3: Implement** orchestrator interfaces and CLI: `capture --url URL --session PATH --output DIR`, `crawl --db PATH --session PATH [--limit N]`, `profile ID --db PATH --session PATH`, `replay DIR --db PATH`, `export --db PATH --output PATH`. Discover and persist queue entries before atomically advancing listing checkpoint; then process pending profiles. Preserve partial issues and successful profiles after failures. Use exit codes 0 for requested work complete (including explicit bounded runs), 1 for incomplete/error, 2 for invalid arguments. Emit machine-readable summary counts, outstanding queue count and full-crawl completeness separately. Export full original/raw/clean text views and provenance. Replay must detect missing request pages/exports rather than consult network. Await in-flight captures and close browser/httpx/database on cancellation.
- [ ] **Step 4: Document** installation, headful challenge resolution, capture/discovery, full crawl, single profile, resume, replay, export, statuses and snapshot field semantics in README. Include examples using the example profile ID 1912391. Explain that observed self-vote evidence may be unavailable and that no subtraction occurs without it. Document raw capture retention and exclusion from source control.
- [ ] **Step 5: Run** `rtk python -m pytest tests -v`, then `rtk proxy powershell -NoProfile -Command "Push-Location IPIP-NEO-120-compact; python -m unittest -v; Pop-Location"`; expect both suites PASS. Run `rtk python -m pdb_ingest --help`; expect all five commands listed.
- [ ] **Step 6: Run bounded live acceptance** using the discovered contract: profile 1912391 and at least two listing pages; verify terminal comment/reply pagination, actual creator ID match, distinct system totals, export statuses and adjustment provenance in DuckDB. Replay the captures into a fresh DB and compare canonical normalized records. Report concrete access blocks or absent upstream evidence; never label fixture-only success as live scraper success. A full crawl remains an explicit subsequent run, not part of bounded acceptance.
- [ ] **Step 7: Commit** Task 7 files, message `feat: ship PDB crawl replay and export commands`.

## Self-review and handoff

Coverage: the ingestion scope is covered by Tasks 1–7; normalization and schema portions needed for usable records are included. Neural distillation, agent assessment, PDF interpretation and audits are deliberately deferred to separate plans. Real API discovery is a dependency gate, not a claim of verified endpoints. All five Review Focus items have named tests. Shared dataclass names/signatures and zero/unknown semantics were checked across tasks. No implementation code or dependencies have been created or installed during planning.

Recommend Native execution: seven tightly connected tasks benefit from keeping the discovered API schema in one session. The requested writing-plans workflow requires plan review and execution-method selection before implementation.
