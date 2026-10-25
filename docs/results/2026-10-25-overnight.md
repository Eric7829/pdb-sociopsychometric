# Overnight: 2026-10-25

Owner's brief before sleeping: fix the two bugs found on 2026-10-24 (typology-talk leak, overconfident encoder heads), retrain, keep version control working, and optimise the model as far as possible. No API spend (OpenRouter balance about $3.14).

State before the night is tagged **`v0.2`** (gte-modernbert encoder, three MBTI cards, cached panel models).

## 1. Typology-talk leak: fixed

The encoder scrubber removed type codes (INFP, 4w5, Ti, ILI, RCUEI) but not the *talk* around them. PDB writers who use typology vocabulary are mostly voted N/P, so the vocabulary worked as a hidden label. In one test, adding a 35-word "I'm searching for my ennea type ... sx blind" intro flipped the crowd card from ISFJ to INFP.

`encoder_text.EXTRA_JARGON` now also removes:
- ennea/enneagram, tritype, trifix, MBTI, Myers-Briggs, socionics, typology, 16personalities, sakinorva, Keirsey
- sx, sp, instinctual variants, cognitive functions, function stacks
- "type 1"–"type 9"
- partial codes such as INFx and xNTP

Ordinary words ("I type 40 wpm", "X-ray", "spy") are kept; a test covers both.

The talk was pervasive: these terms occur in **755 of about 1,000 profiles** ("enneagram" 2,032 times, "sx" 1,095, "typology" 1,060, "mbti" 1,011, "16personalities" 968). All profiles were re-embedded with gte (835 changed, 17 min, no GPU errors) by the new `tools/embed_all.py`.

## 2. Overconfident encoder heads: fixed

At C=1, the gte logistic heads were trained on *averaged* profile vectors, which vary little per dimension. A single pasted text sits far outside that range after standardising, so the logits saturated. `tools/short_text_experiment.py` ran 5-fold CV on 859 profiles, scoring on held-out full profiles and on held-out 250-word excerpts:

| Crowd letters (mean of 4) | Full bal. acc. | Excerpt bal. acc. | Excerpt log loss | Encoder predictions <.05 or >.95 on excerpts |
|---|---|---|---|---|
| M0: C=1 (v0.2) | .591 | .545 | .740 | 78% |
| M1: C=.1 | .607 | .555 | .710 | 54% |
| M1: C=.03 | .613 | .567 | .690 | 34% |
| **M1: C=.01** | **.619** | **.573** | **.675** | **14%** |
| M2: + 4 excerpts per profile in training, C=.03 | .618 | .564 | .684 | 11% |

| Agent (Form M v3b) letters | Full bal. acc. | Excerpt bal. acc. | Excerpt log loss | Extreme |
|---|---|---|---|---|
| M0: C=1 | .648 | .563 | .605 | 81% |
| **M1: C=.01** | **.671** | **.617** | **.563** | 28% |

A log loss above .693 is worse than a coin flip, so v0.2's encoder half was actively harmful on short text.

Findings:
- Strong regularisation wins on every metric, on full profiles as well as excerpts.
- Training on excerpts (M2) did not beat plain regularisation, so it was dropped.
- Big Five ridge heads were already fine: RidgeCV picks its own alpha. Excerpt r .342 at RidgeCV, .346 at alpha 1e3, .352 at 1e4, but full-profile r drops at 1e4 (.569 → .534), so RidgeCV stays.

**Status when the night's run stopped:** command execution became unavailable partway through: the tool's safety checker returned errors on every command. At that point:
- `panel.ENCODER_LETTER_C=.01` was set in the code but **not yet committed, tested or retrained** into the panel.
- The follow-up sweep (C = .003 and .001, plus the encoder half of the 16-type heads, still at C=.3) was written (`short_text_experiment.py --cs`) but **not run**.

Next steps:
1. `pytest`.
2. Run the sweep: `python tools/short_text_experiment.py --db data/pdb-top.duckdb --cs 0.01,0.003,0.001 --output data/stage2/short_text_experiment_c.json`.
3. Set the 16-type encoder C from it.
4. Commit, restart the panel (one ~10-minute retrain), tag `v0.3`.

## 3. Not done

- **Reddit "type me" thread:** Reddit returns 403 to automated requests.
- **Kaggle MBTI dataset:** needs a logged-in Kaggle account, which I can't create or use.

Both stay as owner-side options: download the Kaggle file manually into `data/` and the edge-case and genre tests can use it.
