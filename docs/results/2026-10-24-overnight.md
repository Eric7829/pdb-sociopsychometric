# Overnight build: 2026-10-24

All code is on git branch `span-tagging` (not merged into `master`), and the 112 tests pass. Data is in `data/pdb-top.duckdb`.

## Dataset

| | |
|---|---|
| Top crawl (`sample='top'`) | 500 usable profiles in PDB Top order (25-thread comment cap); 51 friends-only or inactive profiles replaced |
| ES top-up (`sample='es'`) | 59 assessable ESTP/ESFP/ESTJ/ESFJ profiles (by listing consensus type) in Top order; 80 processed |
| Stage 3, prompt A | 500 assessed (441 top + 59 es). The other 59 top profiles have under 200 words of corpus, 5 of them none at all |
| Stage 3, prompt C | 25 profiles (framing check) |
| Span annotation | 574 profiles, 31,175 segments, 650 quotes, 11,288 reply contexts, 0 `raw_text` mismatches |
| API spend | about $5.6 (A $4.27 + about $0.6 for the ES profiles, C $0.73) |

Creator words split as 90% prose, 5% prompts, 4% answers and 0.6% quotes. Every reply whose target comment is still visible got its context.

## Framing check: prompt A vs prompt C (25 profiles)

The two prompts rank profiles almost identically: per-domain rank agreement r = 0.92–0.97. Domain percentiles shift 4.5–9.3 points per profile on average. Neutral answers are 20% (A) vs 16% (C); extreme answers 33% vs 34%. Form M types agree 40% exactly and 83% letter by letter. A's framing changes levels a little but barely changes the ordering of profiles.

## Phase 5 audits (`data/audits/top.json`; `top-es.json` gives nearly the same numbers)

Each statistic is compared with a label-permutation null, 1,000 permutations; p=0.001 is the floor.

**1. Archetype bundling.** The dependence is real but well below the spec's 80% "pure template" bar.

| Predictor → target | n | LOO accuracy | Null | Cramér's V |
|---|---|---|---|---|
| MBTI → Enneagram (core+wing) | 381 | 0.38 | 0.08 | 0.44 |
| MBTI → Enneagram core | 381 | 0.48 | 0.15 | 0.49 |
| MBTI → AP | 250 | 0.30 | 0.05 | 0.49 |
| MBTI + Enneagram → AP | 249 | 0.31 | 0.06 | 0.68 |
| Enneagram → Tritype | 356 | 0.13 | 0.02 | 0.85 |

Knowing a profile's MBTI cuts Enneagram uncertainty by about 1.0 bit beyond what chance overlap explains (H 2.39 vs null 3.43). That is strong bundling, not determinism.

**2. Ramble vs questionnaire ambiguity.** The hypothesis that rambles lead to lower vote entropy is *not supported*. MBTI and Enneagram show the opposite, weakly: ρ = +0.14 (p = 0.005) and +0.12 (p = 0.012). The result holds after controlling for log(votes). The other systems show nothing.

**3. Self-vote vs crowd vs inventory** (prompt A, raw corpus).
- **Crowd votes track agent IPIP scores** in the expected direction:
  - share of E votes vs IPIP E: ρ = +0.65
  - share of T votes vs IPIP A: ρ = −0.53
  - share of J votes vs IPIP C: ρ = +0.49
  - share of N votes vs IPIP O: ρ = +0.34
  - all p = 0.001
- **The spec's hypothesis holds:** T-leaning votes go with low Altruism (A3, ρ = −0.55) and low Modesty (A5, ρ = −0.37).
- **Type agreement:** self-vote vs crowd matches 67% exactly (86% by letter); crowd vs Form M 29% (72%); crowd vs agent self-type 47% (80%).
- **Caveat:** the agent read raw text that can contain self-typing, and the crowd read the same text. The two are not independent sources.

## Stage 2 baseline: negative result

Frozen ModernBERT-base embeddings, mean-pooled over each creator's own words, with a multi-head simplex and entropy model under 5-fold CV (top sample, 392 profiles). Distribution fit is not better than predicting the average crowd distribution.

| View | MBTI KL model/base | top-1 model/base | Entropy ρ |
|---|---|---|---|
| main (scrubbed, prompts masked) | 1.86 / 1.78 | 0.13 / 0.09 | +0.01 |
| unmasked | 1.85 / 1.78 | 0.11 / 0.09 | +0.08 |
| raw (type labels kept) | 1.86 / 1.78 | 0.12 / 0.09 | +0.02 |

Stronger regularization (hidden 16–64, weight decay 0.3–1.0) doesn't change this. Enneagram and AP behave the same way.

**Diagnostic:** TF-IDF plus logistic regression on the same pooled text predicts the crowd's modal letters above chance. Balanced accuracy on the main view: E/I 0.63, T/F 0.66, S/N 0.50, J/P 0.49; raw is about 0.02 higher. The text carries E/I and T/F signal, and it is not mainly type labels. Mean-pooling 10k+ tokens of frozen embeddings washes it out.

**Suggested next steps for Stage 2:**
1. Per-letter targets: 4 binary marginals instead of a 16-way simplex for about 390 profiles.
2. Segment-level embeddings with attention pooling, or a lexical model as the baseline to beat.
3. Fine-tuning only if a larger sample (more ES-style top-ups) becomes available.

## Code added overnight

- `audits.py`: Phase 5 audits with permutation nulls; `pdb_ingest audit --samples top [es]`.
- `embed.py`, `distill.py`: Stage 2 (windowed, span-pooled ModernBERT-base embeddings with a cache, the spec's loss, CV against a baseline); `pdb_ingest distill --view main|unmasked|raw`, plus `tools/run_stage2.py` (read-only database).
- Span layer: `text.segment_spans`, `quotes.py`, `context.py`, `annotate.py` (`pdb_ingest annotate`), `encoder_text.encoder_view`.
- Crawl: type-filtered supplemental samples (`--sample/--type-filter/--min-words`).
- Fixes: JSONL captures split on newline only (U+2028 in comments broke annotate and replay); patient 429 backoff in the LLM client.

## Rulings I made

1. Used a branch in the main checkout, not a worktree, because data and the live processes live here. Cost if wrong: none.
2. Ran span Tasks 4–6 only after Stage 3 finished, since they write to the database. Cost if wrong: a few hours of delay.
3. `merge_quotes` gained an optional `text` argument for whitespace trimming. Cost if wrong: untrimmed edge whitespace.
4. Measured reply-context coverage over replies whose target is visible; hidden targets are excluded by design. Cost if wrong: none (both numbers reported).
5. The `raw_text` invariant is checked before quote merging. Cost if wrong: a mismatch only in quote-split segments could be missed.
6. `encoder_view` falls back to on-the-fly spans when a profile is unannotated. Cost if wrong: missing quote and context marks if `annotate` was skipped.
7. The encoder view collapses all whitespace, newlines included. Cost if wrong: the encoder doesn't see paragraph breaks.
8. Fixed the `splitlines()` capture bug found by the real backfill. Cost if wrong: none.
9. Only comments posted before the creator's can be quotes, after real data showed false quotes. Cost if wrong: rare missed quotes.
10. Scrubbing now covers context, quotes and prompts, overriding the plan per the review. Cost if wrong: less context signal.
11. No tiny-ModernBERT unit test for `hf_encoder`; the real path was smoke-tested on the GPU instead. Cost if wrong: an uncaught regression in the tokenizer wrapping.

The independent review (opus) found 0 Critical and 5 Important issues, all fixed test-first:
- stale spans after a resave
- type labels reaching the encoder through context
- the ES sample mixed into analyses
- the missing CLI and out-of-memory test
- a non-random validation split

## Deferred minor issues

- Quote offsets can shift after characters such as İ, whose lowercase is longer (0 cases in the data).
- A `QUESTIONNAIRE:` split with an empty half reports a mismatch (0 cases).
- `run_crawl` pending and dedupe aren't filtered by sample, so the top set would depend on run order if an ES crawl ran first.
- The partial-Spearman permutation isn't a strict Freedman–Lane conditional null.
- Windows can be 8,194 tokens with CLS/SEP (it runs, because ModernBERT uses RoPE); `windows()` loops forever if stride ≥ max_tokens.
- The embedding cache key ignores the model name.
- The backoff comment understates the total wait.
- Pure-Python token summing is slow (about 20 minutes per view).
- `annotate` doesn't delete rows for profiles that later become incomplete.
- About 1.4% of replies that PDB records as self-targeted lose their context.
- Prompt C itself was not A/B'd beyond the framing check, by your decision.

## Follow-up: encoder swap and stacking (same day)

ModernBERT-base was removed from the Hugging Face cache (573 MB freed) and replaced by `intfloat/multilingual-e5-small` (470 MB). E5-small is a purpose-built multilingual embedding model; it gets a "passage: " prefix, 510-token windows and unit-normalized segment vectors. Comparison script: `tools/compare_text_models.py`, run on the same profiles and folds (348 for IPIP, 386 for crowd letters):

| Agent IPIP, Spearman r | TF-IDF | ModernBERT-base | e5-small | **TF-IDF + e5 stack** |
|---|---|---|---|---|
| O | 0.42 | 0.34 | 0.43 | **0.47** |
| C | 0.46 | 0.30 | 0.45 | **0.49** |
| E | 0.41 | 0.37 | 0.41 | **0.45** |
| A | 0.61 | 0.49 | 0.54 | **0.62** |
| N | 0.46 | 0.35 | 0.47 | **0.49** |

- **IPIP:** e5-small matches TF-IDF and beats ModernBERT by about 0.09. Averaging the TF-IDF and e5 predictions gives the best result on every domain: mean r 0.50, versus 0.47 for TF-IDF alone and 0.37 for ModernBERT.
- **Crowd letters:** balanced accuracy stays modest for all models (E/I 0.57–0.62, T/F 0.61–0.64, S/N and J/P 0.51–0.57). With n≈386 the standard error is about 0.03, so the differences between models here are mostly noise.
- **Crowd vote distributions:** with e5 the Stage 2 MBTI KL is 1.779 against the baseline's 1.776. That's now level with the baseline (ModernBERT was 1.86), but still not better.
