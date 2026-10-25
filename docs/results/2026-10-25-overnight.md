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

Command execution was unavailable for part of the night (the tool's safety checker errored), so the follow-up sweep ran the next morning (`short_text_experiment.py --cs`, results in `data/stage2/short_text_experiment_c.json`):

| | C | Full | 250-word excerpts | Extreme outputs |
|---|---|---|---|---|
| Crowd letters | .01 | bal .619 | bal .573, log loss .675 | 14% |
| | **.003** | bal .618 | **bal .576, log loss .664** | 2% |
| | .001 | bal .615 | bal .574, log loss .661 | 0% |
| Agent letters | .01 | bal .671 | bal .617 | 28% |
| | **.003** | bal .669 | **bal .626** | 9% |
| Crowd 16-type (encoder half) | .3 (v0.2) | exact 16.9%, top-3 41.9% | exact 12.8%, top-3 34.7% | |
| | .01 | exact 17.6%, top-3 43.8% | exact 14.3%, top-3 35.7% | |
| | **.003** | exact 16.5%, top-3 43.5% | **exact 15.1%, top-3 38.1%** | |

The panel uses **C = .003** for both the letter heads and the 16-type heads (`ENCODER_LETTER_C`, `ENCODER_TYPE_C`). Per-letter balanced accuracy on excerpts, v0.2 → v0.3:
- crowd: E/I .53 → .59, S/N .55 → .56, T/F .59 → .63, J/P .51 → .53
- agent: E/I .56 → .61, S/N .56 → .60, T/F .59 → .67, J/P .54 → .63

The panel's short-text warning now quotes these numbers: 15% exact, 38% top-3, Big Five r .34.

## 3. Checks on the retrained panel (v0.3)

**Leak probe** (`tools/leak_probe.py`): each of the 16 first-person portraits was scored with and without a typology-talk intro that names no type. After scrubbing, what remains is generic meta-talk ("searching for my type ... to be typed in any system"). Mean letter shifts are small (.01–.09), but close calls still flip: the type changed for 5 of 16 portraits on the crowd and Jev cards and 3 of 16 on the agent card. Generic "type me" phrasing is a residual leak that word filters can't fully remove.

**Portraits: a trade-off.** v0.2's saturated encoder half was what carried Sensing on the idealized portraits:

| First-person portraits, strict | v0.2 exact / S/N | v0.3 exact / S/N | S predicted, v0.2 → v0.3 |
|---|---|---|---|
| Crowd | 5/16 / .75 | 5/16 / .50 | 6 → 0 |
| Jev | 10/16 / .75 | 5/16 / .63 | 4 → 4 |
| Agent | 5/16 / .69 | 5/16 / .81 | 3 → 7 |

The 859-profile cross-validation on real PDB excerpts favours v0.3 on every letter. The 16 portraits are idealized type descriptions rewritten by an LLM, so v0.3 ships, but the crowd card no longer predicts S on them. A tunable TF-IDF/encoder weight (now a fixed 50/50 average) is the next experiment, judged on both tests.

## 4. Not done

- **Reddit "type me" thread:** Reddit returns 403 to automated requests.
- **Kaggle MBTI dataset:** needs a logged-in Kaggle account, which I can't create or use.

Both stay as owner-side options: download the Kaggle file manually into `data/` and the edge-case and genre tests can use it.
