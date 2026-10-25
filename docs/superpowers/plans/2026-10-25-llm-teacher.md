# LLM teacher, local student, write-up

**Goal:** DeepSeek (the pilot's PDB-voter prompt) labels every profile and essay; the panel's small local models learn
from those labels; the findings so far become one write-up. The panel stays local: assessed text never leaves the machine.

**Basis:** pilot `docs/results/2026-10-25-llm-voter-pilot.md` (scrubbed PDB exact 36% vs ours 15-16%; Essays 59.8% vs 56.1%).

## Global constraints
- Teacher sees scrubbed text only (the raw condition leaked 12 points). Zero-shot: no neighbour examples, so no crowd
  label from one CV fold reaches another through the teacher.
- Prompt wording is frozen at the pilot's; any change gets a new output folder and a rerun.
- PANDORA is never sent to the API (terms forbid transfer). Live spend only after the owner approves the estimate.
- Results report aggregates only; nothing from Essays or PANDORA text goes into the repo.

## Task 1: Teacher labels
`tools/llm_voter_pilot.py` gains `--all` (every top/top2/es profile with text, scrubbed only) and `--all-essays`.
3 samples per text, averaged; per-call cache under `data/stage2/llm-voter`, so the pilot's calls are reused and a
failed run resumes. Estimate from the pilot: PDB ~1,070 x 3 x $0.0014 = ~$4.6; Essays 2,467 x 3 x $0.0008 = ~$5.8.
Also report the teacher's own scores on all of PDB (vs crowd) and all of Essays (vs self-report) as the ceiling.

## Task 2: Student
`tools/teacher_student.py`, 5-fold CV with the panel's folds and halves (TF-IDF + gte):
- MBTI: 16-type head on the teacher's distribution (`soft_fit`), letter heads on its letter marginals
  (`soft_binary_fit`); scored against the crowd modal type on full profiles and 250-word excerpts.
- Big Five: ridge on the teacher's percentiles, trained on PDB + Essays training folds; scored on Essays self-report
  (median split, AUC) and against agent IPIP on PDB.
- Variants: crowd-trained (v0.3), teacher-trained, and crowd + teacher mixed 50/50.

**Ship rule (fixed now):** the teacher-trained or mixed student replaces a v0.3 card only if, on excerpts, crowd exact
or top-3 rises >= 2 points with neither dropping, and Essays mean accuracy does not fall below 56.1%. Big Five
switches only if Essays mean accuracy rises >= 1 point. Shipping retrains the panel and tags v0.4.

## Task 3: Write-up
`docs/results/2026-10-25-findings.md`: one document, aggregate numbers only, covering:
- text vs personality ceilings (small models, teacher, published Essays results),
- typology-talk leakage (scrubbed vs raw),
- crowd votes following function theory,
- S/N blindness,
- PDB vs PANDORA transfer,
- Enneagram contrasts (3 vs 1, social 5).
