# Stage 2 Baseline Distillation Model Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A first, honest Stage 2 baseline: predict each profile's de-contaminated crowd vote simplex and its Shannon entropy from creator text, and measure how far it beats a no-text baseline under cross-validation.

**Architecture:** Frozen `answerdotai/ModernBERT-base` embeds each segment's `encoder_view` text (span-tagging plan, Task 5), mean-pooling only the pooled ranges. Segment vectors are averaged into one profile vector, weighted by pooled word count, and cached on disk keyed by corpus hash. A small multi-head network on the cached vectors is trained with the specification's loss under 5-fold cross-validation.

**Tech Stack:** PyTorch (CUDA 12.8 build, RTX 5060 8 GB), transformers, numpy, scikit-learn (folds only), DuckDB.

**Spec:** The user's technical specification §5.1 (in chat), adapted:
- The specification asks for ModernBERT-large with an 8,192-token context. That doesn't fit fine-tuning in 8 GB, so this baseline uses frozen ModernBERT-base embeddings.
- With about 500 profiles, of which fewer are eligible per system, fine-tuning the encoder would overfit; frozen features are the honest first step.

## Global Constraints

- **Heads:** `MBTI` (16 classes), `ENNEAGRAM` (18 core-wing classes) and `AP` (24 permutations), each with a softmax over that system's full class list, plus one entropy regression head per system (bits, ≥0 via softplus). `INSTINCT`/`TRITYPE` are optional extra heads, behind a flag.
- **Loss (per batch):** `sum_s 1[N_s>=10] * log(1+N_s) * KL(p_clean_s || p_hat_s) + lam * sum_s 1[N_s>=10] * (H_s - H_hat_s)^2`, with `lam=0.1`. A head masked for a profile contributes zero gradient.
- **Targets** come from `vote_distributions`: `simplex_json` (adjusted, self-vote-subtracted), `shannon_entropy`, `training_eligible`. Labels absent from the class list are dropped and the rest renormalised, and the number dropped is logged.
- **Evaluation:** 5-fold CV with a fixed seed (0), splitting by profile.
  - Baseline: the training-fold mean simplex and mean entropy.
  - Metrics per system: mean KL(p‖p̂), top-1 agreement with the crowd's modal type, entropy MAE, and Spearman correlation of predicted versus true entropy.
- **Text views to compare:**
  - `scrub=True, mask_prompts=True` (main)
  - `scrub=True, mask_prompts=False` (the prompt-masking ablation)
  - `scrub=False` (a jargon-leak check: an expected large gain would show the model reads type labels)
- No network access except the one-time model download from Hugging Face. All outputs go under `data/stage2/`.
- Code that needs torch lives in `src/pdb_ingest/embed.py` and `src/pdb_ingest/distill.py`, importing torch lazily. Tests use `pytest.importorskip('torch')`. A new optional dependency group, `ml`, goes in `pyproject.toml`.

## Review Focus

- **A profile with no eligible heads:** it's excluded from training and evaluation and counted in the report, not crashed on (`test_profile_without_eligible_heads_is_excluded`).
- **Segments longer than 8,192 tokens:** split into windows with token overlap, so a pooled range spanning a window boundary is pooled exactly once (`test_long_segment_windows_pool_each_token_once`).
- **A crowd label outside the class list** (for example a legacy MBTI label): dropped and renormalised, with a count reported (`test_unknown_labels_are_dropped_and_counted`).
- **Embeddings cache invalidation:** a changed corpus hash recomputes the embedding, an unchanged one is reused (`test_cache_keyed_by_corpus_hash`).
- **GPU out of memory on one long segment:** fall back to CPU for that segment and record it (`test_oom_falls_back_to_cpu`, simulated).

---

### Task 1: Dataset assembly
- [ ] `distill.class_lists() -> dict[str,list[str]]`, `distill.load_targets(conn, systems) -> dict[int, dict[str, (list[float], float, int)]]` (simplex aligned to the class list, entropy, N; only eligible heads). Tests: `test_unknown_labels_are_dropped_and_counted`, `test_profile_without_eligible_heads_is_excluded`.

### Task 2: Embedding extraction with cache
- [ ] `embed.embed_segments(views: list[tuple[str,list[tuple[int,int]]]], *, model_name, device, max_tokens=8192, stride=512) -> list[tuple[ndarray, int]]` (vector, pooled token count) and `embed.profile_vector(...)`. The cache is `data/stage2/emb/<view>/<profile>_<corpus_sha>.npy`. Tests: the windowing, cache and OOM cases above, using a tiny randomly initialised ModernBERT config so tests stay offline.

### Task 3: Model, loss and cross-validation
- [ ] `distill.HeadModel(dim, class_lists, hidden=256, dropout=0.2)`, `distill.loss(...)` exactly as specified, `distill.cross_validate(X, targets, folds=5, seed=0, epochs=200, patience=20) -> dict` reporting model and baseline metrics. Tests: on synthetic data where the simplex is a function of X, the model beats the baseline on KL; masked heads get zero gradient (`test_masked_heads_have_zero_grad`).

### Task 4: CLI and real run
- [ ] `pdb_ingest distill --db data/pdb-top.duckdb --view main|unmasked|raw` writes `data/stage2/<view>/report.json`. Run all three views and summarise them in the morning report. Expected: the result for MBTI/Enneagram either beats the baseline on KL or shows the baseline is hard to beat. Both are findings.
