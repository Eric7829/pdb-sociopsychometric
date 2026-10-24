# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pdb_ingest` (in `src/pdb_ingest/`) scrapes the Personality Database (PDB) "Pdb Profiles" community category via its JSON API, reconstructs the profile creator's own comments/replies plus linked public Google Docs, and stores auditable text views and vote distributions in DuckDB. The crawl scope is the first N (default 500) profiles in PDB's default listing order.

Two instruments also live here: `IPIP-NEO-120-compact/` (stdlib-only Python Big Five questionnaire and scorer, with its own README and tests) and `MBTI Form M/` (JS/TS scorer). The Stage 3 agent battery uses both. The Stage 2 distillation model and the Phase 5 audits are not built yet.

## Commands (Windows PowerShell, venv at `.venv`)

```powershell
rtk .venv\Scripts\python -m pip install -e '.[dev]'
rtk .venv\Scripts\python -m playwright install chromium

rtk .venv\Scripts\python -m pytest tests -q                              # all pipeline tests
rtk .venv\Scripts\python -m pytest tests/test_pipeline.py::test_bounded_run_and_resume -q   # single test
rtk .venv\Scripts\python -m unittest discover -s IPIP-NEO-120-compact -v  # IPIP tests

rtk .venv\Scripts\python -m pdb_ingest crawl --max-profiles 500 --db data/pdb.duckdb --capture-dir data/run [--limit 5]
rtk .venv\Scripts\python -m pdb_ingest profile 1912391 --db data/pdb.duckdb --capture-dir data/run
rtk .venv\Scripts\python -m pdb_ingest replay data/run --db data/replay.duckdb
rtk .venv\Scripts\python -m pdb_ingest export --db data/pdb.duckdb --output data/profiles.jsonl
rtk .venv\Scripts\python -m pdb_ingest capture --url <profile-url> --seconds 600   # headful login / API discovery
```

pytest config (`pyproject.toml`) sets `pythonpath=src` and `asyncio_mode=auto`, so async tests need no decorator. Tests use fake client/docs classes and `tmp_path` DuckDB files; they never touch the network. No linter/formatter is configured.

Live commands (`crawl`, `profile`, `capture`) require a logged-in browser session in `data/browser` (the user signs in manually in a headful window; never handle credentials) and may pause for a manual Cloudflare check. Don't run them unprompted. Exit codes: 0 done, 1 interrupted/blocked/incomplete/failed, 2 bad args.

## Architecture

Data flow: `cli.py` → `browser.py` (Playwright persistent context; `BrowserRequest` runs `fetch` inside the page so PDB sees Chromium's own cookies/TLS) → `transport.py` `JsonTransport` (allowlisted URLs, ≥2 s between requests, retry/backoff on 429/5xx, raises `AccessBlocked` on 401/403/non-JSON/error envelopes, records every response via `capture.Recorder` to `<capture-dir>/pdb.jsonl`) → `client.py` `PdbClient` (builds the four endpoint queries; parsing delegated to `adapters.py`) → `pipeline.py`.

- `pipeline.ingest_profile`: detail → `crawl.collect_comments` (comments + reply streams, paginated by `has_more`/`next_offset`) → `crawl.match_creator` (comments owned by the profile creator) → `text.make_segment` per authored comment → `gdocs.extract_documents`/`fetch_document` for linked Google Docs (via `DocumentSource`, logged to `documents.jsonl`) → `votes.build_distributions` (subtracts the creator's profile-specific self-vote badge). Returns a `models.Snapshot`.
- `pipeline.run_crawl`: drains `store.pending()` then fetches the next listing page; queue insertion + checkpoint advance happen in one DuckDB transaction. Scope (`max_profiles`) is persisted and must match on resume.
- `pipeline.replay`: `ReplayTransport`/`ReplayDocuments` substitute for the live transport using recorded JSONL (verifying payload sha256, matching by path + canonical sorted query). Missing evidence is an error, never a network fallback. The same `PdbClient` + `process_profile` code path runs live and offline, so keep new network access behind the transport/document-source interfaces.
- `store.py` + `schema.sql`: DuckDB tables `profiles`, `text_segments`, `vote_distributions`, `documents`, `ingestion_attempts`, `crawl_queue`, `crawl_checkpoints`. `save_snapshot` always logs an attempt, but an incomplete snapshot never overwrites a previously complete one. `profiles.snapshot_json` holds the full `Snapshot` and is round-tripped by `snapshot_from_dict`; changing dataclasses in `models.py` requires keeping that in sync.
- Error semantics: `AccessBlocked`, `SchemaError`, `PaginationError` abort the run (recoverable by re-running); other per-profile exceptions record a failed snapshot and continue.

Stage 3 (`battery.py`, `llm.py`, `formm.py`, `assess` subcommand): a persona prompt over the corpus, then 8 independent forced tool calls through OpenRouter (`llm.OpenRouter`, default model `deepseek/deepseek-v4-flash`; don't use Claude for this stage). Scoring is local. `formm.py` must stay output-identical to `MBTI Form M/scorer.js` (parity test in `tests/test_battery.py`). Changing prompt wording means bumping `battery.PROMPT_VERSION`, because resume logic and analyses group runs by it. The OpenRouter key lives in git-ignored `OPEN_ROUTER_API_KEY.txt`; never print it. Live `assess` calls cost money, so run them only when asked.

## Domain rules to preserve

The README's "Stored data" section is the spec; key invariants:
- Text segments keep unchanged `original_text`, plus `raw_text` (questionnaire prompts stripped) and `scrubbed_text` (typology terms removed). Source types: `PINNED_COMMENT`, `AUTHOR_COMMENT`, `GOOGLE_DOC`. Hidden/deleted comment bodies are excluded from text.
- Self-votes come only from the comment's profile-specific `vote_types`/`theVote` badge, matched by account ID, not `user_mbti` flair or text claims. Duplicates subtract once; contradictions → `CONFLICT`; missing class → `INCONSISTENT` with counts unchanged. `IF(N)` is Classic Jungian, not MBTI.
- Systems with <10 adjusted votes and all PDB Big Five systems are not training-eligible. Zero totals → empty simplex, null entropy.
- Vote banner is `total_vote_counts`, not legacy `vote_count*`. Endpoint shapes, field mappings and system IDs are in `docs/pdb-api-contract.md` and `src/pdb_ingest/pdb_contract.json`.

Design and plan: `docs/superpowers/specs/2026-10-23-pdb-ingestion-design.md`, `docs/superpowers/plans/2026-10-23-pdb-ingestion.md`.

## Conventions

- Code style is deliberately terse: minimal whitespace (`a=b`, no spaces after commas), short one-line `if`/method bodies, few comments. Match it.
- `data/` (git-ignored) holds authenticated browser state, captures, and personal writings. Keep it local; don't commit or publish it. `data/discovery/` and `data/smoke-captures/` are raw API evidence used to derive the contract.
- The project directory is not currently a git repository.
