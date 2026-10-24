# Span Tagging, Quote Detection and Reply Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before Stage 2, annotate every creator-authored text segment with character spans (prose, prompt, answer, answer label, quote) and attach the comment each creator reply answers, so an encoder can read questions and quotes as context while pooling only the creator's own words.

**Architecture:** Spans are computed from the unchanged `original_text` by pure functions in `text.py` and `quotes.py`. Reply context comes from comments already captured in `<capture-dir>/pdb.jsonl`, with no network access. An `annotate` subcommand backfills the new tables `text_spans` and `reply_context` for an existing database. `encoder_text.py` turns segments, spans and context into marked encoder input plus the character ranges to pool. Existing `raw_text`/`scrubbed_text` stay byte-identical, so Stage 3 corpus hashes and `agent_runs` remain valid.

**Tech Stack:** Python 3.11+, DuckDB, pytest (`asyncio_mode=auto`), existing `pdb_ingest` package.

**Spec:** Decided in chat on 2026-10-24, with no separate spec document. The decisions:
- Mark text instead of deleting it.
- Exclude prompts, quotes and context from pooling.
- Find quotes using the thread's other comments.
- Encode per segment.
- Ablate prompt masking in Stage 2.
- Prioritise reply context. Measured on the HOT sample (17 profiles): 918 of 985 creator comments are replies, while only 8 use `>` quote lines and 5 share a 6-word run with others' thread text.

## Global Constraints

- Run only after `data/stage3.log` ends with `stage3 done`. `annotate` writes to `data/pdb-top.duckdb`, which the crawler and `assess` hold open; DuckDB allows one writer.
- `make_segment` output (`raw_text`, `scrubbed_text`, `modality`, `word_count`, `ramble_word_count`) must not change for any input.
- Span offsets are Python `str` indices into `Segment.original_text`. Spans of one segment are non-overlapping and sorted, and together cover every non-whitespace character.
- Span kinds (exact strings): `PROSE`, `PROMPT`, `ANSWER`, `ANSWER_LABEL`, `QUOTE`.
- Quote threshold: a run of at least 6 consecutive word tokens shared with another user's comment in the same thread (tokens are lowercase `\w+`), or a line starting with `>`.
- No network access anywhere in this plan. Reply context comes only from the capture directory.
- Code style: terse, matching the repo (`a=b`, no spaces after commas, few comments). The project is not a git repository, so skip commit steps.
- Test command: `rtk .venv\Scripts\python -m pytest tests -q`.

## Review Focus

- **Non-BMP characters (emoji, combining marks) before a span:** offsets must still slice the right text in Python. Test in Task 1 (`test_offsets_survive_emoji`).
- **A quote nested inside an answer:** QUOTE takes precedence and splits the ANSWER span, with no overlap. Test in Task 2 (`test_quote_inside_answer_splits_span`).
- **A reply whose replied-to comment is hidden, deleted or uncaptured:** context is `None`, not an error and not the parent by accident. Test in Task 3 (`test_missing_reply_target_gives_no_context`).
- **A creator replying to themselves:** no context row (the creator's own words are already in the corpus). Test in Task 3 (`test_self_reply_has_no_context`).
- **Re-running `annotate`:** idempotent, with the same row counts and no duplicates. Test in Task 4 (`test_annotate_is_idempotent`).

---

### Task 1: Questionnaire spans from `original_text`

**Files:**
- Modify: `src/pdb_ingest/text.py`, `src/pdb_ingest/models.py`
- Test: `tests/test_spans.py`

**Interfaces:**
- Produces: `models.Span` dataclass `(kind: str, start: int, end: int)`; `text.segment_spans(text: str) -> list[Span]`.

- [ ] **Step 1: Write the failing tests**

```python
from pdb_ingest.text import segment_spans,make_segment
def kinds(text): return [(s.kind,text[s.start:s.end]) for s in segment_spans(text)]
def test_questionnaire_spans():
    t='Q1: Describe yourself. Answer: I love books.\nQ2: What do you fear? Isolation.'
    assert kinds(t)==[('PROMPT','Q1: Describe yourself.'),('ANSWER_LABEL','Answer:'),('ANSWER','I love books.'),
                      ('PROMPT','Q2: What do you fear?'),('ANSWER','Isolation.')]
def test_prose_is_one_span_per_paragraph():
    assert kinds('I write.\n\nI read.')==[('PROSE','I write.'),('PROSE','I read.')]
def test_offsets_survive_emoji():
    t='Q1: Why? 💀🎉 because.\n\nQ2: How? Slowly.\n\nQ3: When? Never.'
    assert ('ANSWER','Slowly.') in kinds(t)
def test_spans_match_raw_text():
    t='intro words\n\nQUESTIONNAIRE:\n\nQ1: Why? Answer'
    assert '\n\n'.join(t[s.start:s.end] for s in segment_spans(t) if s.kind in ('PROSE','ANSWER'))==make_segment('X','x',t).raw_text
```

- [ ] **Step 2: Run `rtk .venv\Scripts\python -m pytest tests/test_spans.py -q`.** Expected: FAIL, `ImportError: segment_spans`.

- [ ] **Step 3: Implement `segment_spans(text)`.** Reuse `make_segment`'s paragraph split, prompt flags, 15% threshold, `QUESTIONNAIRE:` section split and `prompt_answer` delimiter logic, but emit offsets instead of strings. Find each paragraph's offset with `text.find(p, cursor)`. Below the threshold, every paragraph is `PROSE`. Above it, a prompt paragraph splits into `PROMPT` / optional `ANSWER_LABEL` / `ANSWER`, using the same boundaries `prompt_answer` uses. An ambiguous explicit prompt (no `?`, no delimiter) becomes `PROMPT` for the label plus `ANSWER` for the rest, matching `raw_text`. A standalone answer-label paragraph (`A1: Isolation.`) becomes `ANSWER_LABEL` plus `ANSWER`. The `QUESTIONNAIRE:` section header line is a `PROMPT` span. Leave `make_segment` untouched.

- [ ] **Step 4: Run the full suite.** Expected: all pass.

### Task 2: Quote spans

**Files:**
- Create: `src/pdb_ingest/quotes.py`
- Test: `tests/test_spans.py`

**Interfaces:**
- Consumes: `models.Span`, `text.segment_spans`.
- Produces: `quotes.quote_ranges(text: str, others: list[str], n: int=6) -> list[tuple[int,int]]` and `quotes.merge_quotes(spans: list[Span], quotes: list[tuple[int,int]]) -> list[Span]`.

- [ ] **Step 1: Write the failing tests**

```python
from pdb_ingest.quotes import quote_ranges,merge_quotes
from pdb_ingest.text import segment_spans
def test_gt_lines_are_quotes():
    t='> are you sure you are an introvert\nYes, very sure.'
    assert [t[a:b] for a,b in quote_ranges(t,[])]==['> are you sure you are an introvert']
def test_shared_six_word_run_is_quote():
    other='I think you are clearly a sensor because you love details'
    t='You said "you are clearly a sensor because you love" but no.'
    (a,b),=quote_ranges(t,[other])
    assert 'you are clearly a sensor because' in t[a:b] and 'but no' not in t[a:b]
def test_five_shared_words_are_not_a_quote():
    assert quote_ranges('we both love long walks today',['i love long walks today too'])==[]
def test_quote_inside_answer_splits_span():
    t='Q1: Why? Because "you are clearly a sensor because you love details" is wrong.\n\nQ2: Ok? Yes.\n\nQ3: No? No.'
    spans=merge_quotes(segment_spans(t),quote_ranges(t,['you are clearly a sensor because you love details']))
    assert [s.kind for s in spans if s.start>=t.index('Because')][:3]==['ANSWER','QUOTE','ANSWER']
    assert all(x.end<=y.start for x,y in zip(spans,spans[1:]))
```

- [ ] **Step 2: Run the new tests.** Expected: FAIL, `ModuleNotFoundError: quotes`.

- [ ] **Step 3: Implement both functions in `quotes.py`.**
  - `quote_ranges`: tokenise with `re.finditer(r'\w+', text.lower())`, keeping offsets. Mark the token positions where any 6-gram appears in the union of the `others` 6-gram sets. Merge marked positions into character ranges, extended outward to enclosing quotation marks `"“”'` on the same line when present. Add whole lines matching `^\s*>`.
  - `merge_quotes`: split overlapping spans so `QUOTE` wins, then drop zero-length pieces.

- [ ] **Step 4: Run the full suite.** Expected: all pass.

### Task 3: Reply context

**Files:**
- Modify: `src/pdb_ingest/models.py` (`Comment` gains `reply_to_user_id: str | None = None`), `src/pdb_ingest/adapters.py` (`parse_comment` reads `reply_to_user_id`)
- Create: `src/pdb_ingest/context.py`
- Test: `tests/test_spans.py`

**Interfaces:**
- Consumes: `models.Comment`.
- Produces: `context.reply_context(reply: Comment, thread: list[Comment]) -> Comment | None`, where `thread` holds the top-level comment and all of its replies.

- [ ] **Step 1: Write the failing tests**

```python
from pdb_ingest.models import Comment
from pdb_ingest.context import reply_context
P=Comment('1','9','voter','you seem INTP',parent_id=None)
A=Comment('2','42','me','no, INFP',parent_id='1',reply_to_user_id='9')
B=Comment('3','7','other','why INFP?',parent_id='1',reply_to_user_id='42')
C=Comment('4','42','me','because Fi',parent_id='1',reply_to_user_id='7')
def test_context_is_latest_earlier_comment_by_target(): assert reply_context(C,[P,A,B,C]) is B
def test_falls_back_to_parent(): assert reply_context(A,[P,A,B,C]) is P
def test_missing_reply_target_gives_no_context():
    assert reply_context(Comment('5','42','me','hm',parent_id='1',reply_to_user_id='555'),[P]) is None
def test_self_reply_has_no_context():
    assert reply_context(Comment('6','42','me','also',parent_id='1',reply_to_user_id='42'),[P,A]) is None
def test_parse_comment_keeps_reply_target():
    from pdb_ingest.adapters import parse_comment
    assert parse_comment({'id':2,'user_id':42,'comment':'x','parent_id':1,'reply_to_user_id':9}).reply_to_user_id=='9'
```

- [ ] **Step 2: Run the new tests.** Expected: FAIL, `ModuleNotFoundError: context`.

- [ ] **Step 3: Implement.** Pick the comment in `thread` written by `reply_to_user_id` with the largest numeric id below the reply's id. Fall back to the top-level comment only when `reply_to_user_id` is `None` or equals the parent's author. Return `None` for self-replies and for blank text (hidden or deleted). Old snapshots load unchanged because the new `Comment` field has a default.

- [ ] **Step 4: Run the full suite.** Expected: all pass, including `snapshot_from_dict` round-trips in `tests/test_store.py`.

### Task 4: `annotate` backfill command and tables

**Files:**
- Modify: `src/pdb_ingest/schema.sql`, `src/pdb_ingest/store.py`, `src/pdb_ingest/cli.py`
- Create: `src/pdb_ingest/annotate.py`
- Test: `tests/test_annotate.py`

**Interfaces:**
- Consumes: `segment_spans`, `quote_ranges`, `merge_quotes`, `reply_context`, `capture.Capture`, `adapters.parse_comment`.
- Produces:
  - Tables `text_spans(profile_num BIGINT, source_type TEXT, source_ref TEXT, ordinal INTEGER, kind TEXT, start_pos INTEGER, end_pos INTEGER, PRIMARY KEY(profile_num,source_type,source_ref,ordinal))` and `reply_context(profile_num BIGINT, comment_id TEXT PRIMARY KEY, context_comment_id TEXT, context_user_id TEXT, context_text TEXT)`.
  - `annotate.annotate(store, capture_dir: Path) -> dict` returning `{'profiles','segments','spans','quotes','contexts','raw_text_mismatches'}`.
  - CLI: `pdb_ingest annotate --db D --capture-dir C`.

- [ ] **Step 1: Write the failing tests.** Build a store with `process_profile` and a fake client, as in `tests/test_pipeline.py`, plus a `Recorder` capture of the thread: the parent, a voter question and a creator reply that quotes 6+ words of it. Assert:
  - `test_annotate_writes_spans_quotes_context`: one `QUOTE` row, and a `reply_context` row whose `context_text` is the voter's question.
  - `test_annotate_is_idempotent`: running twice gives identical `count(*)` for both tables.
  - `test_raw_text_invariant`: `raw_text_mismatches==0`.

- [ ] **Step 2: Run the new tests.** Expected: FAIL, `ModuleNotFoundError: annotate`.

- [ ] **Step 3: Implement `annotate`.**
  1. Read every comment payload in `capture_dir/'pdb.jsonl'` into `id -> Comment`; on duplicates, the last one in file order wins.
  2. For each complete profile's segments, compute `segment_spans` on `original_text`.
  3. For `AUTHOR_COMMENT` and `PINNED_COMMENT` segments, `others` is the non-creator comments of the same thread. For `GOOGLE_DOC` segments, use `>` lines only (`others=[]`).
  4. Merge quotes, then insert reply context for reply segments.
  5. Recompute `raw_text` from `PROSE`/`ANSWER` spans (as in Task 1's test) and count mismatches against stored `raw_text`.
  6. Delete-then-insert per profile inside one transaction, which makes the command idempotent.

  Wire the CLI. Exit code 1 if `raw_text_mismatches>0`.

- [ ] **Step 4: Run the full suite.** Expected: all pass.

### Task 5: Encoder input view

**Files:**
- Create: `src/pdb_ingest/encoder_text.py`
- Test: `tests/test_encoder_text.py`

**Interfaces:**
- Consumes: the Task 4 tables (read via `store.connection`) and `text.JARGON`.
- Produces: `encoder_text.encoder_view(store, profile_num: int, *, scrub: bool=True, with_context: bool=True, mask_prompts: bool=True) -> list[tuple[str, list[tuple[int,int]]]]`, one `(text, pooled_ranges)` per segment in snapshot order.
  - Markers (exact): `[Q]`/`[/Q]` around `PROMPT`, `[QUOTE]`/`[/QUOTE]`, `[CTX]`/`[/CTX]` around reply context (prepended), and `ANSWER_LABEL` dropped.
  - `pooled_ranges` are offsets into the returned `text` and cover only `PROSE` and `ANSWER` content.
  - With `mask_prompts=False`, `PROMPT` content is included in `pooled_ranges` (the Stage 2 ablation).

- [ ] **Step 1: Write the failing tests.**
  - `test_markers_and_pooling`: for a reply with context and one prompt/answer pair, the output text equals `'[CTX] why INFP? [/CTX] [Q] Q1: Why? [/Q] Because books.'`, and the single pooled range slices to `'Because books.'`.
  - `test_scrub_only_inside_pooled_text`: `INFP` inside `[CTX]` is kept, `INFP` inside an answer is removed when `scrub=True`.
  - `test_mask_prompts_false_pools_prompt`: the pooled ranges also slice the prompt text.

- [ ] **Step 2: Run the new tests.** Expected: FAIL, `ModuleNotFoundError: encoder_text`.

- [ ] **Step 3: Implement `encoder_view`.** Build the text left to right, recording each pooled piece's start and end after it is appended. Apply `JARGON.sub('', piece)` to pooled pieces when `scrub`, then collapse runs of spaces as `make_segment` does. Context text is never scrubbed and never pooled.

- [ ] **Step 4: Run the full suite.** Expected: all pass.

### Task 6: Backfill the real data and report

**Files:** none new. Commands only, run after `data/stage3.log` ends with `stage3 done`.

- [ ] **Step 1:** Run `rtk .venv\Scripts\python -m pdb_ingest annotate --db data/pdb-top.duckdb --capture-dir data/run-top`. Expected: exit 0, `raw_text_mismatches: 0`, `profiles` equal to the complete-profile count.
- [ ] **Step 2:** Sanity report. Query the share of creator words by span kind, and the number of replies with context over the total number of reply segments. Print 5 random `QUOTE` spans and 5 `[CTX]` pairs for a manual check. Expected: QUOTE words under 2% of the total (consistent with the HOT sample), and context found for over 90% of replies.
- [ ] **Step 3:** Update `README.md` ("Stored data") and `CLAUDE.md` (Architecture) with the two tables, the `annotate` command and `encoder_view`, one paragraph each.
