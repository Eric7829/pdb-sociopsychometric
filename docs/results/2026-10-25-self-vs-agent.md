# Self-typing vs the DeepSeek agent (2026-10-25)

`tools/self_vs_agent.py`. Self = the creator's profile-specific self-vote. Chance exact agreement is 8-10% for every pair.

| Self-type vs | n | Exact | E/I | S/N | T/F | J/P |
|---|---|---|---|---|---|---|
| Crowd modal vote | 538 | 69% | .88 | .87 | .88 | .83 |
| Agent's stated type (prompt A, raw text) | 522 | 49% | .80 | .79 | .84 | .75 |
| Agent's Form M (prompt A) | 522 | 30% | .75 | .74 | .80 | .68 |
| Agent's Form M v3b (scrubbed) | 423 | 33% | .74 | .79 | .77 | .67 |
| PDB-voter teacher (scrubbed) | 493 | 31% | .67 | .75 | .71 | .68 |

- **The crowd mostly agrees with the creator (69%).** Either self-typers are accurate or voters anchor on the self-type
  shown on the profile; this data can't tell which.
- **The agent's stated type leans on self-typing in the text.** It matches the creator 55% of the time when the text
  names a type and 34% when it doesn't. The scrubbed teacher barely moves (36% vs 33%).
- **The agent's own Form M disagrees with its stated type** (41% exact). Answering the questionnaire pulls it towards
  INFP: the most common changes from the creator's type are J->P, S->N, E->I and T->F, and the reverse changes are rare.
  v3b (scrubbed) still shows J->P and E->I, but half as much N.

## Recalibrating Form M for LLM respondents (`tools/formm_recalibrate.py`)

The Form M manual (ch. 7) scores with IRT and places each division point where it best matches **best-fit type**, the
type people choose after a feedback session. Ties go to I, N, F or P. The shipped scorer carries those human
adjustments as clarity rules: weak S, T and J results (clarity <= 2, 4 and 2) become N, F and P, which moves the S-N,
T-F and J-P division points about 0.1-0.2 theta towards S, T and J. Here the PDB self-vote stands in for best-fit type, and
the respondents are the 680 Form M v3b agents (scrubbed text), 423 of them self-voted. 5-fold CV, 5 repeats:

| Scoring | E-I | S-N | T-F | J-P | Exact vs self | Exact vs crowd |
|---|---|---|---|---|---|---|
| Official (human 2PL, MLE, human midpoint adjustments) | .735 | .785 | .768 | .671 | 33.1% | 34.0% |
| Official theta, plain cut at 0 | .733 | .787 | .780 | .678 | 33.8% | 34.5% |
| Human 2PL + EAP + LLM-fitted division points | .740 | .771 | .782 | .660 | 33.8% | 34.3% |
| 2PL refitted to LLM answers + EAP + fitted points | .732 | .767 | .781 | .665 | 32.2% | 33.4% |
| Item-level logistic key (data-driven ceiling) | .739 | .788 | .757 | .656 | 31.8% | 31.0% |

Letter columns are plain agreement with the self-vote letter. Fitted division points, human parameters with EAP:
E-I -0.35 (towards E), S-N -0.07, T-F +0.06, J-P +0.01.

- Re-engineering the scoring doesn't move accuracy: every variant lands at 32-34% exact, including a fully data-driven
  item key. The limit is how the agent answers the items, not where the lines are drawn.
- The human adjustments slightly hurt LLM respondents (33.8% without them vs 33.1% with). The agents need the
  E-I line moved towards E and no N/F/P nudge, because they already lean I, N, F and P.
- For scale: people taking the real Form M agree with their own best-fit type on all four letters 72.9% of the time
  (Form M Manual Supplement, n = 8,836).
