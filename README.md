# PDB profile ingestion

Scrape PDB's **Pdb Profiles** community category through its JSON API, reconstruct creator-authored comments/replies and linked public Google Documents, and store auditable text views and vote distributions in DuckDB.

The crawl is capped at the **first 500 unique profiles in PDB's default listing order**. This is the website's ordering, not a new ranking by vote count. Completed profiles are skipped on resume. The existing `IPIP-NEO-120-compact/` questionnaire remains independent.

## Setup (Windows PowerShell)

```powershell
rtk python -m venv .venv
rtk .venv\Scripts\python -m pip install -e '.[dev]'
rtk .venv\Scripts\python -m playwright install chromium
```

## Run

PDB requires login to view community profiles. First use a headful capture to sign in directly in the browser; never paste credentials into the terminal. The session is saved in `data/browser`. Close the window when done; the recorder handles manual closure.

```powershell
rtk .venv\Scripts\python -m pdb_ingest capture --url 'https://www.personality-database.com/profile/1912391/esqueleto-narrativo-pdb-profiles-mbti-personality-type' --seconds 600
```

Start/resume the first 500 profiles:

```powershell
rtk .venv\Scripts\python -m pdb_ingest crawl --max-profiles 500 --db data/pdb.duckdb --capture-dir data/run
```

`--sort top` crawls PDB's Top (All Time) order (`sort=top`); the default `hot` is the site's default listing. Top profiles are far more active: on the first page the median was 441 votes and 543 comments, against 111 and 199 for HOT. `--max-comments N` fetches reply threads only for the first N top-level comments (pinned comments come first, and their Google Docs are always followed). Snapshots record `comment_limit`/`comments_truncated`. Sort, cap and `--max-profiles` are all stored with the crawl, and resuming with different values is refused. The current main dataset was produced with `--sort top --max-profiles 500 --max-comments 25 --db data/pdb-top.duckdb --capture-dir data/run-top`. Profiles PDB refuses individually (a JSON 403 with a reason, such as "inactive" or "friends only") are recorded as unavailable and skipped without stopping the crawl. Login prompts, 401s and Cloudflare pages still stop it.

For a shorter invocation add `--limit 5`; this limits profile attempts in that invocation without changing the saved 500-profile scope. Re-run without it to continue. The scope is stored with the queue; changing `--max-profiles` requires a separate database. Category listing pages are fetched incrementally as needed. Completed snapshots and queue advancement are transactional. An interrupted profile starts over; its already captured responses remain available for inspection. Cloudflare checks must be completed manually in the browser within two minutes; otherwise the command exits with a recoverable error. Avoid concurrent commands against the same browser session or database.

Single-profile verification:

```powershell
rtk .venv\Scripts\python -m pdb_ingest profile 1912391 --db data/pdb.duckdb --capture-dir data/run
```

Offline replay and complete-profile JSONL export:

```powershell
rtk .venv\Scripts\python -m pdb_ingest replay data/run --db data/replay.duckdb
rtk .venv\Scripts\python -m pdb_ingest export --db data/pdb.duckdb --output data/profiles.jsonl
```

Replay requires all request pages and Google export results for each captured profile; missing evidence is an error and never triggers network access. Manual `capture` is for API discovery, not by itself a complete replay dataset. Live `crawl` and `profile` record replayable API envelopes in `pdb.jsonl` and document outcomes in `documents.jsonl`. Captures use a payload hash and canonical query matching. Each live profile attempt tags its API responses and document outcomes with an `attempt` ID; replay reconstructs every attempt on its own, oldest first, and never fills one attempt's gaps with another's responses. Attempts with missing evidence (interrupted refreshes) are skipped, so an earlier complete snapshot survives as it does live. Untagged captures from older versions are replayed only when each request has one distinct response; otherwise replay stops with an `Ambiguous replay` error. Keep capture directories scoped to a run/database.

Exit code 0 means the requested invocation completed, including an explicitly bounded invocation; 1 means interrupted, blocked, incomplete, or failed; 2 means invalid arguments. Final crawl summaries distinguish `scope_complete` (the selected top 500) from `full_crawl_complete` (the entire remote category). `outstanding` counts queued incomplete profiles, not profiles still awaiting discovery. The category currently reports roughly 978,000 profiles, so the default scope intentionally stops far earlier.

## Stored data

- `profiles`: original header metadata, creator identity, all-systems vote banner, ramble ratio, capture timestamp, and a complete structured snapshot.
- `text_segments`: source reference, unchanged `original_text`, question-stripped `raw_text`, typology-scrubbed `scrubbed_text`, modality, and word counts. Source types are `PINNED_COMMENT`, `AUTHOR_COMMENT`, `GOOGLE_DOC`.
- `vote_distributions`: observed counts, adjusted counts, normalized simplex, Shannon entropy in bits, self-vote evidence/status, and per-head eligibility.
- `documents`: export statuses and all referring creator comment IDs. `UNAVAILABLE` exports do not remove inline text; transient `ERROR` outcomes leave the snapshot incomplete.
- `ingestion_attempts`, `crawl_queue`, `crawl_checkpoints`: failure evidence and durable restart state.

The exported JSONL includes full source records plus concatenated `raw_corpus` and `clean_corpus`. Only complete profiles are exported. An incomplete refresh cannot replace a previously complete snapshot. Unknown creator identity remains explicit and is not included as a complete corpus. Hidden/deleted comment bodies are excluded from text while their API evidence remains in the local capture.

Self-votes come from the creator's **profile-specific comment vote badge**, matched by stable account ID where available. Account flair and unstructured type claims do not cause subtraction. Duplicate badges subtract once per system; contradictions are retained as `CONFLICT`. `IF(N)` is a Classic Jungian label, so it does **not** subtract an MBTI INFP vote. If a matching class is absent/zero, status is `INCONSISTENT` and counts stay unchanged. Systems with fewer than 10 adjusted votes are retained but excluded for their training head; PDB Big Five targets are always excluded. Zero totals yield an empty simplex and null entropy.

Text processing is heuristic: question-mark endings and explicit prompt prefixes identify questionnaires only above the 15% paragraph threshold. Ordinary book/bullet lists are preserved. An explicit `QUESTIONNAIRE:` section can mark a mixed document. Ambiguous short typology tokens use case-sensitive rules. Original text is always available for future parser improvements; these heuristics do not claim perfect questionnaire detection.

`data/` is ignored by Git. It contains authenticated browser state and original personal writings: keep it local unless intentionally sharing a reviewed subset. The capture filter excludes login/verification endpoints and request cookies/authorization; minimal fixtures contain only selected source fields. Data is not automatically deleted.

## Span annotations and encoder input

`pdb_ingest annotate --db D --capture-dir C` works offline from the captures. It tags every stored segment's `original_text` with character spans in `text_spans`, and records in `reply_context` which comment each creator reply answers.
- **Span kinds:** `PROSE`, `PROMPT`, `ANSWER`, `ANSWER_LABEL` and `QUOTE`.
- **Quotes:** `>` lines, and runs of 6 or more words shared with an *earlier* comment by another user in the same thread.
- **Reply context:** the latest earlier comment by the replied-to user. Self-replies and hidden targets get none.
- **Stored views unchanged:** `raw_text`/`scrubbed_text` are not modified. The command checks that PROSE+ANSWER spans rebuild `raw_text` exactly, and exits 1 on any mismatch.

`encoder_text.encoder_view(store, n, scrub=True, with_context=True, mask_prompts=True)` returns one `(text, pooled_ranges)` per segment, with `[CTX]…[/CTX]`, `[Q]…[/Q]` and `[QUOTE]…[/QUOTE]` markers. Only the creator's own prose and answers are pooled; `mask_prompts=False` is the Stage 2 ablation.

On the Top database: 574 profiles, 31,175 segments, 650 quotes and 11,288 reply contexts (every reply whose target is still visible), with 0 `raw_text` mismatches. Creator words split as 90% prose, 5% prompts, 4% answers and 0.6% quotes.

## Stage 3: psychometric agent battery

Each complete profile's creator-authored corpus (`raw_text` by default, `--text-view scrubbed` for the jargon-free view) is placed inside `<biographical_corpus>` in a fixed persona-simulation system prompt. The model then answers through forced tool calls: five 24-item IPIP-NEO-120 blocks (1–5), two MBTI Form M blocks (A/B), and one self-typing call (MBTI, Enneagram with wing, first-person rationale). Calls are independent, so no block sees earlier answers, and they share the same prefix. Answers that are invalid or incomplete are retried up to twice. Scoring is local: `IPIP-NEO-120-compact/ipip_neo.py` and `pdb_ingest/formm.py`, a Python port of `MBTI Form M/scorer.js` (a parity test runs when Node is installed).

```powershell
rtk .venv\Scripts\python -m pdb_ingest assess --db data/pdb.duckdb --limit 5
rtk .venv\Scripts\python -m pdb_ingest assess 1912391 --runs 3 --provider <upstream>
```

The default model is `deepseek/deepseek-v4-flash` via OpenRouter. The key comes from `OPEN_ROUTER_API_KEY` or `OPEN_ROUTER_API_KEY.txt` (git-ignored), and is never logged or stored. `--runs N` keeps adding runs until each profile has N complete ones for the same model, prompt version and text view, with seeds offset per run. `--provider` defaults to `deepinfra/fp8`: of the cheap fp8 upstreams tested, it was the only one that honoured both forced `tool_choice` and `seed`, and its prefix cache hit (second call 99% cached, 77% cheaper). `--provider any` lets OpenRouter route each call, so weights can differ between calls and the cache rarely hits. Profiles below `--min-words` (default 200) are skipped. HTTP, auth and credit failures stop the invocation.

Stored: `agent_runs` (model, prompt version, seed, corpus hash/size, cost, error), `agent_calls` (every attempt's prompt, arguments, reasoning, provider and usage), and `psychometric_evaluations` (one row per complete run: agent self-type, Form M type and thetas, IPIP domains/facets with percentiles, raw responses, and response-style diagnostics such as neutral and extreme answer shares).

## Verify

```powershell
rtk .venv\Scripts\python -m pytest tests -q
rtk .venv\Scripts\python -m unittest discover -s IPIP-NEO-120-compact -v
```

See [API contract](docs/pdb-api-contract.md), [design](docs/superpowers/specs/2026-10-23-pdb-ingestion-design.md), and [implementation plan](docs/superpowers/plans/2026-10-23-pdb-ingestion.md). Modeling, psychometric agents and statistical audits are separate future subsystems.
