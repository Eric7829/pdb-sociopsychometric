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

Supplemental samples get their own scope and checkpoints with `--sample NAME`, and queue rows carry the tag. In `data/pdb-top.duckdb`:
- `top`: the first 500 usable Top profiles.
- `top2`: the next 500 (`--sample top2`).
- `es`: 59 assessable ESTP/ESFP/ESTJ/ESFJ profiles added with `--type-filter '^ES[TF][JP]' --min-words 200`. Its type mix is filtered by construction, so leave it out of type-classifier training.

Requests are spaced by `--interval` seconds (default 1).

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

## Personality panel

```powershell
rtk .venv\Scripts\python -m pdb_ingest panel --db data/pdb-top.duckdb --samples top top2   # then open http://127.0.0.1:8765/
```

A local web page that only listens on 127.0.0.1. It opens the database read-only and shows aggregate charts only:
- crowd vs agent MBTI and Enneagram distributions
- agent Big Five percentile histograms
- crowd vote share vs Big Five correlations
- Big Five intercorrelations
- the MBTI × Enneagram bundling heatmap

The **assessor** scores a typed self-description with typology labels scrubbed. It averages TF-IDF and `gte-modernbert-base` models trained at startup on the chosen samples. Each output comes with its cross-validated accuracy:
- **Big Five:** percentiles with ±1 cross-validated error bands.
- **Crowd MBTI type:** the top 3 of 16 types, as PDB voters would type the text. A 16-type head distilled from Jev (below) is pooled with four letter heads (log-linear pool, letter weight 0.4).
- **Crowd MBTI letters and Enneagram core.**
- **Agent MBTI and Enneagram:** the LLM agent's Form M result and self-typed core.

Assessed text is never stored or logged.

Predictions mimic the *agent's* reading for the Big Five and the *crowd's* for types. The agent's reading is skewed in this sample (high O and N, low C, E and A), so new texts are pulled toward that profile. Startup trains every head with 5-fold CV, which takes about 7 minutes on Top + top2. The first start also embeds any uncached profiles. The panel opens the database read-only, but DuckDB on Windows still cannot share the file with a running writer (crawl, assess, annotate). Point `--db` at a copy, or stop the writer.

## Jev teacher (16-type labels)

`tools/distill_jev.py label --db data/pdb-top.duckdb --samples top,top2` asks TypeSafe's **Jev 1.14** (`typesafe/jev-1.14`), a *decision model* served through OpenRouter's Decisions API (`POST /api/alpha/decisions`), one `choice` question per profile. The question is which of the 16 types the forum crowd would vote, and Jev returns a probability for every type. Labels are cached under `data/stage2/llm/typesafe--jev-1.14/`, each with the exact build that answered. The cost was $0.24 for 795 profiles: $0.042 per million input tokens and nothing for output.

Do not use `typesafe/jev-router`. Despite the name, it forwards each request to a different chat LLM (Gemini, Grok, GLM, Command, ...). The first round of "Jev" labels came from it by mistake; they are kept in `data/stage2/llm/typesafe--jev-router/` and `data/stage2/jev_distill_router.json`.

`distill_jev.py evaluate` runs 5-fold CV against the crowd's modal type. Results on Top + top2 (795 profiles, commonest-type baseline 11.7%):

| Model | Exact | Top 3 | Log loss |
|---|---|---|---|
| Ours (TF-IDF + e5, letters pooled with 16-type) | 17.2% | 44.4% | 2.516 |
| Jev 1.14 | 33.7% | 60.3% | 2.428 |
| Jev + ours (weight picked on other folds) | 33.6% | 60.3% | 2.115 |
| Student on 0.5 Jev + 0.5 crowd, pooled with letters (panel) | 19.1% | 46.2% | 2.468 |

Jev's accuracy rises with text length, from 17% for profiles under 400 words to 49% for those over 4,600, while ours stays flat at about 18–20%. So long profiles are not capped.

The 16-type head is fitted with TF-IDF and the scaler trained once on unique profiles, then weighted rows per type. Jev's probabilities are first rescaled to the crowd's type frequencies, which removes its ENFP/E skew.

`tools/type_ablation.py` (results in `data/stage2/type_ablation.json`) scores each choice on full profiles and on 250-word crops of pinned comments and docs, which mimic a pasted self-description:
- **Short texts are much harder.** 16-type exact drops from about 19% to 12–14%, and agent-IPIP Spearman r from 0.49 to 0.27. The panel shows these short-text numbers for inputs under 600 words.
- **Letter weight depends on length.** With the letter weight at 0.4, the 16-type head contradicts a letter head that is at least 70% sure in 40% of crops. So the weight runs from 0.8 at 300 words or fewer down to 0.4 at 2,000 words or more.
- **Small effects:** fixing the duplicated-row fit, L2-normalising profile vectors and the prior correction each change exact accuracy by under one point. Class-balancing the 16-type head raises macro-F1 (0.146 vs 0.121) but lowers exact accuracy, so it is off.
- **Self-reported type:** the creator's own profile badge, available for 498 eligible profiles. The crowd's modal type matches it 70% exactly (82–89% per letter), so the crowd is a reasonable proxy for self-perceived type. Text models predict it no better than they predict the crowd (58–59% mean balanced letter accuracy), so no separate self-type head was added.

The encoder scrubber also removes Socionics codes, SLOAN codes and function-position words (`encoder_text.EXTRA_JARGON`). Stored `scrubbed_text` is unchanged. Jev's long-profile accuracy did not drop under the stricter scrubbing: 45% before and 48% after, on 100 profiles over 2,500 words (`tools/jev_scrub_check.py`).

## Encoder comparisons

`tools/run_stage2.py --model M --samples top,top2 main` embeds and caches profiles under `data/stage2/emb/<model>/main`. Then:
- `tools/compare_text_models.py`: agent IPIP Spearman r and crowd letter balanced accuracy. `--matryoshka 64,128,...` adds sliced variants.
- `tools/compare_type_models.py --encoder SLUG [--dims K]`: 16-type models.

`Alibaba-NLP/gte-modernbert-base` (CLS pooling, 2048-token chunks) is the default since 2026-10-25. On Top + top2, stacked with TF-IDF, it beat `multilingual-e5-small` on agent-IPIP Spearman r (.562 vs .539 in `compare_text_models`; .574 vs .548 full and .346 vs .298 on 250-word excerpts in a matched check) and on letters (.593 vs .579 balanced accuracy, S/N .579 vs .555). Mean pooling scored the same. `gte-base-en-v1.5` is not usable: its remote model code fails on transformers 5.x and hung the GPU. Earlier comparisons, on the Top sample against e5-small (mean IPIP r 0.50):
- ModernBERT-base: 0.46 stacked.
- Jina v5 text-small-classification: 0.47 stacked, 14.8% vs 16.6% 16-type. It used 2048-token chunks to stay inside 8 GB of VRAM.
- Jina Matryoshka slices (32–768 dims): worse at every size.

## Stage 3: psychometric agent battery

Each complete profile's creator-authored corpus (`raw_text` by default, `--text-view scrubbed` for the jargon-free view) is placed inside `<biographical_corpus>` in a fixed persona-simulation system prompt. The model then answers through forced tool calls: five 24-item IPIP-NEO-120 blocks (1–5), two MBTI Form M blocks (A/B), and one self-typing call (MBTI, Enneagram with wing, first-person rationale). Calls are independent, so no block sees earlier answers, and they share the same prefix. Answers that are invalid or incomplete are retried up to twice. Scoring is local: `IPIP-NEO-120-compact/ipip_neo.py` and `pdb_ingest/formm.py`, a Python port of `MBTI Form M/scorer.js` (a parity test runs when Node is installed).

```powershell
rtk .venv\Scripts\python -m pdb_ingest assess --db data/pdb.duckdb --limit 5
rtk .venv\Scripts\python -m pdb_ingest assess 1912391 --runs 3 --provider <upstream>
```

`--prompt-version` picks the prompt:
- `stage3-v1` (default): first-person persona. All `top` profiles have a run, about $0.009 per profile.
- `stage3-predict-v1`: third-person "predict how this person answered".
- `stage3-predict-v2`: lean third person, 4 calls instead of 8.

Analyses group runs by prompt version, so train on one version only. `--profiles-file` restricts the run to listed profiles.

The default model is `deepseek/deepseek-v4-flash` via OpenRouter. The key comes from `OPEN_ROUTER_API_KEY` or `OPEN_ROUTER_API_KEY.txt` (git-ignored), and is never logged or stored. `--runs N` keeps adding runs until each profile has N complete ones for the same model, prompt version and text view, with seeds offset per run. `--provider` defaults to `deepinfra/fp8`: of the cheap fp8 upstreams tested, it was the only one that honoured both forced `tool_choice` and `seed`, and its prefix cache hit (second call 99% cached, 77% cheaper). `--provider any` lets OpenRouter route each call, so weights can differ between calls and the cache rarely hits. Profiles below `--min-words` (default 200) are skipped. HTTP, auth and credit failures stop the invocation.

Stored: `agent_runs` (model, prompt version, seed, corpus hash/size, cost, error), `agent_calls` (every attempt's prompt, arguments, reasoning, provider and usage), and `psychometric_evaluations` (one row per complete run: agent self-type, Form M type and thetas, IPIP domains/facets with percentiles, raw responses, and response-style diagnostics such as neutral and extreme answer shares).

## Verify

```powershell
rtk .venv\Scripts\python -m pytest tests -q
rtk .venv\Scripts\python -m unittest discover -s IPIP-NEO-120-compact -v
```

See [API contract](docs/pdb-api-contract.md), [design](docs/superpowers/specs/2026-10-23-pdb-ingestion-design.md), [implementation plan](docs/superpowers/plans/2026-10-23-pdb-ingestion.md), and [overnight results](docs/results/2026-10-24-overnight.md). Audits: `pdb_ingest audit`; Stage 2 baseline: `pdb_ingest distill`.
