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

## 4. TF-IDF vs encoder weighting

`tools/weight_experiment.py` scored p = w · p_encoder + (1 − w) · p_TF-IDF for encoder C in {.003, .01, .03} and w in {.3, .5, .7, .85, 1}. It used held-out 250-word excerpts (5-fold CV, 859 profiles) and 32 portraits (16 first-person, 16 third-person, strict scrub), with models fitted on all profiles. Selected rows:

| Config | Excerpt crowd letters | Excerpt agent letters | Excerpt type exact / top-3 | Portrait crowd letters (S/N) | Portrait headline type exact / S types predicted (of 16) |
|---|---|---|---|---|---|
| C=.003 w=.3 | .565 | .604 | 15.7% / 38.4% | .789 (.72) | 22% / 0 |
| **C=.003 w=.5 (v0.3)** | **.576** | **.626** | 15.1% / 38.2% | .797 (.75) | 22% / 0 |
| C=.003 w=.7 | .581 | .628 | 13.3% / 37.5% | .781 (.75) | 19% / 1 |
| C=.01 w=.85 | .570 | .615 | 13.3% / 34.8% | .766 (.69) | 22% / 5 |
| C=.03 w=1 | .562 | .601 | 13.6% / 32.3% | .766 (.69) | 28% / 11 |

- **The 50/50 average at C=.003 is already near the best on excerpts**, so the weights stay.
- **The letter heads read Sensing fine on portraits** (S/N .72–.78 in most rows). The headline type loses S because the 16-type head carries PDB's base rates (about 70% of crowd types are N) and outvotes the letters in the pool. Configurations that let more S through (C=.03, w=1) do so by trusting the overconfident encoder again, and pay for it on real excerpts.
- So the remaining S problem is a **prior** question, not a weighting one. PDB's population is N-heavy, which helps in-sample accuracy but biases the headline for a general reader (most people are S).

## 5. Agent S/N as a teacher: no real gain

`tools/sn_teacher_experiment.py` tried two ways to use the DeepSeek agent's Form M v3b S/N for the crowd card: blending the two heads' probabilities (b = agent share), and training the crowd S/N head on soft targets a · crowd + (1 − a) · agent. Results are 5-fold CV over 795 profiles, plus 32 portraits:

| Variant | S/N vs crowd (full / excerpt) | S/N vs creator's own badge (excerpt) | S/N on portraits | Portrait headline: S types predicted |
|---|---|---|---|---|
| crowd only (v0.3) | .588 / .563 | .542 | .750 | 0 |
| blend b=.5 | .552 / .535 | .573 | .688 | 0 |
| blend b=1 (agent only) | .532 / .515 | .546 | .719 | 2 |
| soft a=.75 | .580 / .573 | .566 | .781 | 0 |
| soft a=.5 | .585 / .561 | .570 | .750 | 3 |

- The agent's S/N **disagrees with how PDB voters assign S/N** (agent-only scores .515 against the crowd on excerpts). It sits a little closer to creators' own badges (.546–.573 vs .542), but every difference here is one or two points, within the noise for 197 S-typed profiles.
- Soft targets at a=.75 are the best variant, but by about one point, so v0.3 is unchanged.
- **The headline still predicts almost no S on portraits whatever the S/N letter does**, which confirms that the 16-type head's N-heavy prior, not S/N detection, decides the headline.

## 6. Prior-neutral headline (shown next to the current one)

`tools/prior_neutral_experiment.py`: the crowd card's 16-type head divided by its training type frequencies before pooling, in 5-fold CV over 795 profiles plus 32 portraits:

| | Current (PDB prior) | Prior-neutral |
|---|---|---|
| Full profiles vs crowd: exact / top-3 | 19.0% / 44.9% | 17.9% / 43.8% |
| Full: mean recall per type (16 types weighted equally) | 12.3% | **16.4%** |
| Full: S types predicted (crowd has 197) | 52 | 214 |
| 250-word excerpts vs crowd: exact / top-3 | 15.1% / 38.4% | 14.5% / 34.8% |
| Portraits: exact / top-3 | 21.9% / 53.1% | **28.1% / 71.9%** |
| Portraits: S types predicted (16 true) | 0 | 5 |

- Prior-neutral gives up about a point of exact accuracy on PDB in exchange for predicting S about as often as the crowd does (214 vs 197, rather than 52) and treating rare types fairly (per-type recall +4 points).
- On portraits, where every type is equally common, it is clearly better.
- The panel now shows both: the PDB-prior headline, plus a "Prior-neutral" line with its own cross-validated accuracy on the crowd and Jev cards. The agent card has no 16-type head, so it has no prior to remove.

## 7. External Big Five model and PANDORA (first real self-report check)

`tools/ocean_bert_compare.py` compares `ppp57420/ocean-personality-distilbert` (Hugging Face; claims Pearson r .77) with the panel's Big Five model. Spearman r is averaged over the five traits:

| Test | External DistilBERT | Panel model |
|---|---|---|
| PDB, full profiles vs agent IPIP (752) | .13 | .57 (5-fold CV) |
| PDB, 250-word excerpts vs agent IPIP | .02 | .34 (5-fold CV) |
| **PANDORA test split, 500 users, ~1,000 words of Reddit comments each, vs self-reported percentiles** | **−.03** (O .04, C −.04, E −.08, A −.10, N .04) | **.06** (O .18, C .10, E .00, A −.12, N .12) |

- **The external model shows no signal on either test**, even on PANDORA, where it was most likely trained. Its r .77 is not reproduced at the user level; comment-level train/test splits that let the same author appear on both sides are the likely explanation. It also rates "I love wild parties, meeting strangers and being the centre of attention" at Extraversion .07 on its 0–1 scale.
- **The panel model barely transfers to real self-report on Reddit comments:** mean r .06. Openness (.18), Neuroticism (.12) and Conscientiousness (.10) carry some signal (with 500 users, r above about .09 is significant), but Extraversion is zero and Agreeableness is reversed. The r .57 it reaches against the agent's reading of PDB self-descriptions does **not** carry over to real people writing about other things.
- **Caveats:**
  - Reddit comments are general chatter, not self-descriptions, so the genre is far from what the panel is for.
  - PANDORA's scores come from several different tests that users posted themselves.
  - In this comment-level re-upload, a "user" is a distinct score profile, which can rarely merge two people.

**Essays dataset** (`tools/essays_check.py`): 2,467 student stream-of-consciousness essays (Pennebaker & King, ~660 words each), self-reported Big Five as above/below-median labels. AUC, where .5 is chance; neither model saw these essays:

| | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| Panel model | .571 | .580 | .594 | .598 | .602 | **.589** |
| External DistilBERT | .466 | .561 | .537 | .534 | .516 | .523 |

On self-reflective writing the panel's Big Five transfers to real self-report on **all five traits** (each well above chance with n = 2,467), including Agreeableness, which came out reversed on Reddit comments.

The table below compares accuracy against models *trained on the Essays data* (Table 1 of Majumder, Poria, Gelbukh & Cambria, 2017, IEEE Intelligent Systems 32(2)); the panel predicts "high" above its own median:

| Accuracy % | EXT | NEU | AGR | CON | OPN | Mean |
|---|---|---|---|---|---|---|
| Majority baseline | 51.7 | 50.0 | 53.1 | 50.8 | 51.5 | 51.4 |
| Mairesse et al. 2007 features + SVM (trained on Essays) | 55.1 | 58.1 | 55.4 | 55.3 | 59.6 | 56.7 |
| Majumder et al. 2017, best configuration per trait (trained on Essays) | 58.1 | 59.4 | 56.7 | 57.3 | 62.7 | 58.8 |
| **Panel model (never saw Essays)** | **56.8** | **58.5** | **56.7** | **54.8** | **54.8** | **56.3** |

The panel matches the trained 2007 baseline overall without seeing a single essay, and ties the best published result on Agreeableness. Openness is its weak spot. The 2017 figures pick each trait's best configuration after the fact, so they are optimistic. Genre matters more than the method: Reddit chatter gave .06, self-reflective essays give a consistent, real signal.

This is the first test against real people at scale. It argues for validating on real self-descriptions with real scores before trusting the panel's Big Five beyond PDB.

## 8. Do the models think in cognitive functions?

**Runner-up structure** among one-letter neighbours. In function theory the E/I flip shares both functions (INTP↔ENTP), the S/N flip shares the dominant, the T/F flip shares the auxiliary, and the J/P flip shares none (INTP↔INTJ):

| Source | E/I | S/N | T/F | J/P |
|---|---|---|---|---|
| Crowd: modal vs runner-up vote (420) | **30%** | 26% | 23% | **20%** |
| Jev: top-1 vs top-2 (683) | 27% | 20% | 23% | 30% |
| Panel: errors vs crowd (306) | 29% | 16% | 21% | 34% |

PDB voters follow function theory: mirror pairs are the commonest runner-up and J/P flips the rarest, at about p ≈ .02 on 420 profiles. Jev and the panel confuse along letters, mostly J/P, the weakest letter at 42% error.

**Training the crowd card on the full vote distribution** (`tools/vote_soft_experiment.py`, 5-fold CV):

| Targets | Full exact / top-3 | Excerpt exact / top-3 / log loss | J/P share of one-letter errors (full / excerpt) | Portraits exact / top-3 |
|---|---|---|---|---|
| modal vote (v0.3) | 19.0% / 44.9% | **15.1% / 38.4% / 2.593** | 34% / 29% | 22% / 53% |
| vote distribution | 19.5% / 46.3% | 14.8% / 35.8% / 2.624 | 33% / 27% | 25% / 69% |
| 50/50 mix | **20.5% / 46.2%** | 15.2% / 37.1% / 2.608 | 34% / 27% | 22% / 63% |

Not shipped. The soft targets help a little on full profiles and portraits but cost top-3 accuracy on excerpts, which is what the panel actually sees. They also barely move the J/P share of errors. Crowd votes agree with the modal type about 86% of the time, so the distributions add little, and the text models' confusions come from per-letter evidence in the writing, which vote structure doesn't change.

## 9. Not done

- **Reddit "type me" thread:** Reddit returns 403 to automated requests.
- **Kaggle MBTI dataset:** needs a logged-in Kaggle account, which I can't create or use.

Both stay as owner-side options: download the Kaggle file manually into `data/` and the edge-case and genre tests can use it.
