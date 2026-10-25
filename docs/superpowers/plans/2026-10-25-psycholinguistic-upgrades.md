# Psycholinguistic Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test two ideas from the Essays literature and ship them in the panel only if they win:
- document-level psycholinguistic features (Mairesse et al. 2007), as a third "half" next to TF-IDF and gte;
- emotional-sentence filtering before scoring (Majumder et al. 2017).

**Architecture:**
- `src/pdb_ingest/lexicons.py` loads free word norms and turns a text into a fixed vector of about 35 features, and filters a text down to its emotional sentences.
- `tools/upgrade_eval.py` scores four variants (baseline, +features, +filter, both) on every harness the project already has, using a pre-registered rule.
- Winners go into `panel.Assessor` behind constants, and their inputs join the model-cache fingerprint.

**Tech Stack:** Python, numpy, scikit-learn (existing), plain CSV/TSV lexicons. No new pip packages. No LIWC: it is commercial.

**Spec:**
- The owner's request of 2026-10-25: "plan upgrades from their ideas".
- [docs/results/2026-10-25-overnight.md](../../results/2026-10-25-overnight.md): the Essays and PANDORA sections hold the baselines this plan must beat.

## Global Constraints

- **Code style:** the project's terse style (`a=b`, no spaces after commas, short one-line bodies, few comments).
- **No OpenRouter or other API spend.**
- **External files** (lexicons, datasets) live under `data/external/<name>/`, git-ignored. Each lexicon folder keeps its source URL and licence in `SOURCE.txt`. Read them with `python -I`.
- **Assessed text is never logged or stored.** Feature vectors computed for the panel stay in memory.
- **The encoder is `embed.DEFAULT_MODEL`** (gte-modernbert-base, already GPU-verified). Any *new* model goes through a CPU smoke test first. This plan adds none.
- **Baselines** (current panel, v0.3):
  - PDB 5-fold CV: Big Five Spearman .569 full / .342 excerpt; crowd letters balanced accuracy .618 full / .576 excerpt; agent letters .669 / .626; crowd 16-type excerpt exact 15.1%, top-3 38.1%.
  - Essays: mean accuracy 56.3% (AUC .589).
  - PANDORA: mean r .06.
  - Portraits (32): crowd letter S/N .75.
- **Ship rule (pre-registered).** A variant ships only if *all three* hold:
  1. Essays mean accuracy rises by at least 0.5 points, **or** PDB excerpt Big Five r rises by at least .01.
  2. No PDB metric listed above falls by more than .005, or 0.5 points for percentages.
  3. The leak probe's mean letter shift does not grow by more than .01.

## Review Focus

- **Text with no lexicon hits** (code, a list of names, a non-English paste): the features must be finite, with zeros or neutral means, never NaN. The panel must still answer. Test in Task 1.
- **Text with no emotional sentences:** filtering must fall back to the full text, never return an empty string. Test in Task 2.
- **Lexicon files missing** (fresh checkout, `data/external` empty): the panel and the eval tool must run without the feature half and say so, not crash. Test in Task 1 and Task 4.
- **A very long profile** (a 600k-word profile exists): feature extraction must be linear-time and finish in under 2 s for 100k words. Test in Task 1.
- **The lyrics/poem case** (70% repeated lines): with the filter on, the "not a self-description" warning must still be computed on the raw input, not the filtered text. Test in Task 4.

---

### Task 0: Obtain the lexicons (owner approval needed)

Downloads outside PyPI and Hugging Face need the owner's go-ahead. Request these three:

| Folder | Source | Licence | Used for |
|---|---|---|---|
| `data/external/lexicons/brysbaert-concreteness/` | Brysbaert, Warriner & Kuperman (2014), 40k English lemmas, CSV from the authors' supplementary material | Free for research | concreteness (the S/N hypothesis) |
| `data/external/lexicons/glasgow-norms/` | Scott et al. (2019), Glasgow Norms, 5,553 words: arousal, valence, dominance, concreteness, imageability, familiarity, age of acquisition, size, gender | CC BY 4.0 | MRC-style ratings |
| `data/external/lexicons/nrc-emolex/` | Mohammad & Turney, NRC Word-Emotion Association Lexicon (EmoLex), 8 emotions plus positive/negative | Free for research, non-commercial | emotion features and the sentence filter |

- [ ] **Step 1:** With the owner's approval, download each file into its folder and write `SOURCE.txt` (URL, citation, licence, download date).
- [ ] **Step 2:** Run `python -I -c "import csv;..."` to print each file's header and row count. Expected: about 40k rows, 5,553 rows, and about 14k words.

No commit: `data/` is git-ignored.

### Task 1: Psycholinguistic document features

**Files:**
- Create: `src/pdb_ingest/lexicons.py`
- Test: `tests/test_lexicons.py`, with tiny fixture lexicons in `tests/fixtures/lexicons/` (same column layout as the real files, 10–20 words each)

**Interfaces:**
- Produces:
  - `Lexicons` (dataclass): `concreteness: dict[str,float]`, `glasgow: dict[str,dict[str,float]]`, `emotions: dict[str,frozenset[str]]`, `available: bool`.
  - `LEXICON_ROOT = Path('data/external/lexicons')` (module constant, so tests can monkeypatch it).
  - `load_lexicons(root: Path | None = None) -> Lexicons`: reads `LEXICON_ROOT` when `root` is None; missing folders give `available=False` and empty dicts.
  - `FEATURES: list[str]`: the fixed feature names, in order.
  - `doc_features(text: str, lex: Lexicons) -> np.ndarray`: shape `(len(FEATURES),)`, always finite.
- Features (word-rate features are per word; lexicon means are over words found in the lexicon, with the lexicon's midpoint when none are found):
  - **pronoun rates:** `i_rate`, `we_rate`, `you_rate`, `they_rate` (built-in word lists)
  - **function-word rates:** `negation_rate`, `article_rate`, `preposition_rate`, `auxiliary_rate`, `tentative_rate`, `certainty_rate` (built-in lists, about 10–30 words each)
  - **NRC emotion rates:** `nrc_<emotion>_rate` for the 8 emotions plus positive and negative
  - **concreteness:** `concreteness_mean`, `concrete_share` (words rated 4 or more on the 1–5 scale)
  - **Glasgow means:** `glasgow_arousal`, `glasgow_valence`, `glasgow_dominance`, `glasgow_imageability`, `glasgow_familiarity`, `glasgow_aoa`
  - **style:** `mean_word_length`, `words_per_sentence`, `ttr_500` (type-token ratio over the first 500 words), `question_rate`, `exclamation_rate`, `lexicon_coverage` (share of words found in the concreteness norms)

- [ ] **Step 1: Write failing tests** in `tests/test_lexicons.py`:
  - `test_features_are_finite_without_hits`: `doc_features('zxq vbn 123 🤖',lex)` is all finite, and `concreteness_mean` equals the midpoint 3.0.
  - `test_pronoun_and_emotion_rates`: on `'I hate my job. I am so angry.'` with the fixture NRC (hate→anger, angry→anger), `i_rate` is 2/8 and `nrc_anger_rate` is 2/8.
  - `test_concreteness_mean`: two fixture words rated 5.0 and 1.0 give `concreteness_mean` 3.0 and `concrete_share` 0.5.
  - `test_missing_lexicons`: `load_lexicons(tmp_path).available is False`, and `doc_features` still returns a finite vector of the right length.
  - `test_long_text_is_fast`: 100k words in under 2 s.
- [ ] **Step 2:** `rtk .venv\Scripts\python -m pytest tests/test_lexicons.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `lexicons.py`. Tokenize with one regex (`[a-z']+` on lower-cased text). Count with a `Counter`, then compute every rate from the counts in a single pass.
- [ ] **Step 4:** Rerun the tests. Expected: PASS. Full suite: all pass.
- [ ] **Step 5:** Commit `lexicons.py`, the tests and the fixtures: "Psycholinguistic document features from free norms (concreteness, Glasgow, NRC EmoLex, function words)".

### Task 2: Emotional-sentence filter

**Files:**
- Modify: `src/pdb_ingest/lexicons.py`
- Test: `tests/test_lexicons.py`

**Interfaces:**
- Produces: `emotional_sentences(text: str, lex: Lexicons, min_share: float = .3, min_words: int = 50) -> str`.
  - Keeps the sentences (split on `[.!?\n]+`) that contain at least one NRC emotion word (the 8 emotions, not just positive/negative).
  - Returns the full text unchanged when the kept part is under `min_share` of the words or under `min_words` words, or when no lexicons are loaded.

- [ ] **Step 1: Write failing tests:**
  - `test_filter_keeps_emotional`: a 3-sentence text with one neutral sentence comes back without it (fixture lexicon; `min_words=1`).
  - `test_filter_falls_back`: a text with no emotion words comes back identical.
  - `test_filter_never_empty`: `emotional_sentences('',lex)==''`, and any non-empty input gives a non-empty output.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: "Emotional-sentence filter (Majumder et al. 2017) with fallback to the full text".

### Task 3: Evaluation harness for the four variants

**Files:**
- Create: `tools/upgrade_eval.py`
- Reuses:
  - `type_ablation.load` (PDB profiles, excerpts, labels)
  - `short_text_experiment.vectors` (cached excerpt vectors)
  - `essays_check` / `ocean_bert_compare` loading code for Essays and PANDORA
  - `portrait_test.portrait`

**Interfaces:**
- Consumes: `load_lexicons`, `doc_features`, `FEATURES`, `emotional_sentences` (Tasks 1–2); `panel.ENCODER_LETTER_C`, `ENCODER_TYPE_C`, `soft_binary_fit`, `pool`, `letter_weight`.
- Produces: `data/stage2/upgrade_eval.json`, holding `{variant: {metric: value}}` for the variants `baseline`, `features`, `filter`, `features+filter`, plus a `ship` block that applies the pre-registered rule.
- Model per variant:
  - **Features half:** `StandardScaler` + `RidgeCV` for the Big Five; `LogisticRegression(C=.1, class_weight='balanced')` for letters; a multinomial `C=.1` for the 16-type head.
  - **Averaging:** TF-IDF, gte and the feature half each get weight 1/3 (baseline keeps the 1/2–1/2 average).
  - **Filter variants:** the filter is applied to *every* text that is scored or trained on (profiles, excerpts, essays, PANDORA, portraits) before both TF-IDF and the encoder. Filtered excerpts and profiles need fresh gte vectors; cache them in `data/stage2/excerpt_vectors_<model>.npz` under the filtered text's hash (the existing cache is keyed by text).
- Metrics:
  - **PDB 5-fold CV** (the same folds, `KFold(5, shuffle=True, random_state=0)`): Big Five Spearman, full and excerpt; crowd and agent letter balanced accuracy, full and excerpt; crowd 16-type excerpt exact and top-3.
  - **Essays:** median-split accuracy and AUC per trait, models fitted on all PDB profiles.
  - **PANDORA:** mean r, 500 test users × 1,000 words, `random.Random(0)` sample.
  - **Portraits:** crowd letter S/N accuracy.
  - **Leak probe:** mean letter shift, the `leak_probe.INTRO` logic run in-process.

- [ ] **Step 1:** Implement the tool. On a 40-profile slice (`--limit 40`), print one JSON line per variant. Run `rtk .venv\Scripts\python tools/upgrade_eval.py --db data/pdb-top.duckdb --limit 40`. Expected: 4 lines, no NaN, and a `ship` block.
- [ ] **Step 2:** Full run in the background. Expected: about 1–2 hours of GPU time for the filtered re-embeddings, then `data/stage2/upgrade_eval.json`.
- [ ] **Step 3:** Commit the tool, and add a results section (table plus ship decision) to `docs/results/2026-10-25-overnight.md`: "Upgrade eval: psycholinguistic features and emotional-sentence filtering".

### Task 4: Ship the winners in the panel (skip any variant that failed the ship rule)

**Files:**
- Modify: `src/pdb_ingest/panel.py`
  - add constants `USE_PSYCH_FEATURES`, `USE_EMOTION_FILTER`
  - add the feature half to the Big Five, letter and type heads
  - apply the filter in `Assessor.__init__` and `assess`
  - add the lexicon files to `fingerprint`
- Modify: `src/pdb_ingest/panel_static/index.html`: one line in the Big Five note naming the third half when it is on.
- Modify: `README.md`: the Personality panel section.
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: Task 1–2 functions; `load_lexicons()` is called once in `Assessor.__init__` and stored as `self.lex`. It is not pickled with the model cache: reload it in `cached_assessor`, the same way the encoder is reattached.
- Produces: unchanged `assess()` keys, plus `trained_on['psych_features']: bool`.

- [ ] **Step 1: Write failing tests:**
  - `test_panel_runs_without_lexicons`: with `LEXICON_ROOT` monkeypatched to an empty `tmp_path`, `Assessor` trains and `assess` returns `big5` and all three cards. `trained_on['psych_features']` is False.
  - `test_style_warning_uses_raw_text`: with `USE_EMOTION_FILTER=True`, the lyrics test from `test_assessor_predicts_with_cv_accuracy` still yields a "repeat" warning.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement behind the constants, set from the ship decision.
- [ ] **Step 4:** Full suite passes. Restart the panel (one ~10-minute retrain). Rerun:
  - `tools/portrait_test.py` on the first-person and third-person Medium folders
  - `tools/leak_probe.py`

  Expected: S/N and leak shift within the ship rule's bounds.
- [ ] **Step 5:** Commit, then `git tag -a v0.4 -m "Panel v0.4: <shipped upgrades>"`.
