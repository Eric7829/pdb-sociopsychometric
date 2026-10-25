# Pilot: DeepSeek v4 flash as a PDB voter (2026-10-25)

`tools/llm_voter_pilot.py`, Cloudflare via OpenRouter, 3 samples per text at temperature .7, averaged. 450 calls, $0.44.

| Test | n | Exact | Top-3 | E/I | S/N | T/F | J/P |
|---|---|---|---|---|---|---|---|
| PDB, scrubbed text | 25 | 36% | 60% | .76 | .68 | .80 | .64 |
| PDB, raw text (self-typing left in) | 25 | 48% | 72% | .88 | .76 | .84 | .72 |
| Our panel, crowd card, 250-word excerpts (5-fold CV) | ~1000 | 15% | 38% | | | | |

Letters are plain accuracy against the crowd's modal type. With n=25, one profile is 4 points.

Essays Big Five (n=100, median split): mean accuracy 59.8%, mean AUC .664 (O .63, C .69, E .57, A .66, N .77).
Ours on the full Essays set: 56.1% / .586. Majumder et al. 2017's best: 58.8%.

- Scrubbed text already more than doubles our exact match; leaving self-typing in adds 12 more points, so a scrubbed
  pipeline is the honest number.
- The Big Five essays result beats both our model and the published Essays models, zero-shot.
- Not done: few-shot neighbours, calibration, a larger set. PANDORA cannot be used (its terms forbid transfer).

## Full PDB labels (1 sample, scrubbed)

785 of 863 profiles with a crowd MBTI vote, $1.80 (the spending cap; 78 remain unlabelled). Calls averaged about
$0.0023, above the pilot's $0.0014, because later profiles are longer.

| | n | Exact | Top-3 | E/I | S/N | T/F | J/P |
|---|---|---|---|---|---|---|---|
| DeepSeek, 1 sample | 785 | 29.8% | 56.4% | .67 | .73 | .72 | .69 |

The pilot's 36% was on the high side, as 25 profiles would predict. It still doubles our models' exact match. Recall
on crowd S types is 49% (N 81%), far above our small models, which almost never call S.

## Local student trained on the teacher's labels: nothing ships

`tools/teacher_student.py`, 778 profiles, 5-fold CV, panel heads (TF-IDF + gte). Scored against the crowd modal type.

| Trained on | Full exact | Full top-3 | Excerpt exact | Excerpt top-3 | Excerpt S recall | Agrees with teacher (full) |
|---|---|---|---|---|---|---|
| crowd (v0.3) | 17.7% | 43.6% | 16.2% | 37.4% | 27% | 20% |
| teacher | 14.1% | 35.0% | 13.6% | 32.6% | 50% | 32% |
| 50/50 mix | 15.9% | 43.3% | 14.3% | 35.2% | 41% | 28% |

Essays Big Five (median-split accuracy / AUC): agent IPIP 56.4% / .587, teacher 56.7% / .596, both 56.7% / .596.

- The students reproduce the teacher's S calls (excerpt S recall 27% to 50%) and its ESFJ habit, but not its accuracy:
  they agree with the teacher only 22-32% of the time. The teacher's judgement uses function-level reading of a whole
  profile; 778 examples of one noisy sample each are not enough for bag-of-words and embedding heads to copy it.
- Big Five from teacher labels gains 0.4 points on Essays, short of the 1-point bar.
- By the plan's rule the panel stays at v0.3.
