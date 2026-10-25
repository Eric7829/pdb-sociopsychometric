# The nine Enneagram types and 27 subtypes: Big Five and MBTI on PDB

*2026-10-25. `tools/ennea_report.py`; full numbers in `data/stage2/ennea_report.json` (local). Aggregates only.*

## Summary

- **The nine cores have clear, largely theory-consistent Big Five profiles.** Extraversion separates them most
  (epsilon-squared .28), then Agreeableness, Conscientiousness, Neuroticism and Openness (.13-.16). Big Five domains
  alone pick the right core 32% of the time (chance 11%).
- **MBTI and Enneagram are strongly tied on PDB** (Cramér's V .48): 7s are 96% E and 87% P, 4s 91% F, 5s 93% I,
  2s 93% F. PANDORA users who report both themselves show the same directions, more weakly (V .37).
- **Instincts matter less than cores.** Pooled across cores, the instinct effect on any Big Five scale is 1-2% of
  variance: so-firsts are more agreeable and conscientious, sx-firsts more extraverted, adventurous and less cautious,
  sp-firsts the most reserved.
- **Within cores, several subtypes differ sharply and in the direction subtype theory predicts**, especially the
  usual "countertypes": SX6 low agreeableness, SO7 high conscientiousness, SO9 high activity and altruism, SX4
  assertive and less self-conscious, SP3 disciplined. Cells are small (8-51 profiles), so treat these as
  hypotheses.
- **Caveat that runs through all of it:** the crowd's Enneagram votes, its Big Five votes and the agent's IPIP answers
  can all be shaped by the same Enneagram stereotypes. 70% of profiles talk about the Enneagram; on the 30% that never
  do, every pattern below except Openness survives.

## Data and method

| | Profiles |
|---|---|
| Crowd Enneagram core (training-eligible votes) | 840 |
| ... with the agent's IPIP-NEO-120 (domains + 30 facets) | 727 |
| ... with crowd SLOAN Big Five votes (>= 10) | ~650 |
| ... with a crowd MBTI type | 834 |
| ... with a dominant instinct (most first-position INSTINCT votes) | 696 |
| PANDORA users self-reporting Enneagram and MBTI | 793 |

Core counts: 1: 39, 2: 72, 3: 79, 4: 118, 5: 150, 6: 82, 7: 137, 8: 55, 9: 108.

- **Big Five sources.** (a) The Stage 3 agent's IPIP-NEO-120: DeepSeek role-playing the creator from their own text
  answers 120 items; scores are population percentiles. (b) Crowd votes on PDB's SLOAN system, as the share of votes
  for each pole.
- **Tests.** Kruskal-Wallis across cores (effect size epsilon-squared); each core vs the rest by Cohen's d and
  Mann-Whitney, Benjamini-Hochberg-corrected over all 9 x 35 tests; multinomial logistic regression from Big Five to
  core (5-fold CV, balanced accuracy); MBTI by chi-squared, Cramér's V and two-proportion z per letter.
- **Leak control.** The agent read the creator's raw text, which may name their Enneagram type. Everything is repeated
  on the 168 IPIP / 248 MBTI profiles whose text never mentions Enneagram vocabulary ("ennea", "4w5", "type 4",
  "tritype", "sx/so", "instinct").

Facet codes: N1 anxiety, N2 anger, N3 depression, N4 self-consciousness, N5 immoderation, N6 vulnerability; E1
friendliness, E2 gregariousness, E3 assertiveness, E4 activity, E5 excitement-seeking, E6 cheerfulness; O1
imagination, O2 artistic interests, O3 emotionality, O4 adventurousness, O5 intellect, O6 liberalism; A1 trust, A2
morality, A3 altruism, A4 cooperation, A5 modesty, A6 sympathy; C1 self-efficacy, C2 orderliness, C3 dutifulness, C4
achievement-striving, C5 self-discipline, C6 cautiousness.

## 1. The nine cores on the Big Five

**Agent IPIP, mean percentile** (the sample as a whole is high on O and N and low on C, as PDB writers are):

| Core | n | O | C | E | A | N |
|---|---|---|---|---|---|---|
| 1 | 35 | 70 | **60** | 33 | 34 | 64 |
| 2 | 65 | 89 | 31 | 58 | 51 | 73 |
| 3 | 67 | 77 | 43 | 61 | 23 | 68 |
| 4 | 107 | 90 | 20 | 28 | 45 | **88** |
| 5 | 128 | 80 | 39 | **22** | 28 | 73 |
| 6 | 74 | 75 | 31 | 23 | 42 | 82 |
| 7 | 108 | **91** | **15** | 61 | 36 | 74 |
| 8 | 43 | 71 | 34 | **68** | **11** | **55** |
| 9 | 100 | 78 | 29 | 30 | **61** | 74 |

| Scale | Agent IPIP epsilon-squared | Crowd SLOAN epsilon-squared |
|---|---|---|
| Extraversion | .28 | .42 |
| Agreeableness | .16 | .26 |
| Conscientiousness | .15 | .20 |
| Neuroticism | .14 | .18 |
| Openness | .13 | .09 |

All p < 1e-8. The crowd's own Big Five votes separate the cores even more, with the same ordering: crowd-voted
Extraversion is 82% for 2s and 81% for 7s against 13% for 5s; Agreeableness 85% for 9s and 82% for 2s against 20% for
8s; Conscientiousness 88% for 1s against 22% for 7s.

**Signature facets** (core vs all other cores, all q < .05; top by |d|):

| Core | Distinctive | Theory reading |
|---|---|---|
| 1 | C overall +1.13, self-discipline +.96, orderliness +.95, cautiousness +.88, achievement +.85; immoderation -1.01, liberalism -.86 | the perfectionist: controlled, principled, conventional |
| 2 | emotionality (O3) +.85, altruism +.84, friendliness +.79, gregariousness +.77, trust +.76, cheerfulness +.75, sympathy +.69 | the helper: warm, outgoing, giving |
| 3 | assertiveness +.88, self-efficacy +.83, activity +.76, achievement +.63; modesty -.76, self-consciousness -.69 | the achiever: confident, driving, image-aware |
| 4 | N overall +.71, depression +.65, vulnerability +.64, anxiety +.63, emotionality +.63, artistic interests +.59; self-efficacy -.59 | the individualist: intense, melancholic, aesthetic |
| 5 | cautiousness +.80; excitement-seeking -.75, gregariousness -.74, emotionality -.70, friendliness -.67, altruism -.65 | the investigator: detached, reserved, emotionally contained |
| 6 | self-consciousness +.59, E overall -.58, adventurousness -.57 | the loyalist: anxious in company, risk-averse |
| 7 | excitement-seeking +1.04, cheerfulness +.85, adventurousness +.85, gregariousness +.79; cautiousness -.87, orderliness -.73, dutifulness -.73 | the enthusiast: stimulation-seeking, undisciplined |
| 8 | assertiveness +1.53, self-efficacy +1.18; vulnerability -1.44, depression -1.29, self-consciousness -1.28, anxiety -1.23, modesty -1.11, morality -1.04 | the challenger: dominant, emotionally armoured, blunt |
| 9 | cooperation +.93, modesty +.91, A overall +.79, trust +.66, morality +.63; assertiveness -.75, anger -.69 | the peacemaker: accommodating, unassertive, slow to anger |

Every core's signature matches its textbook description. The largest single effects are the 8's assertiveness and
low vulnerability, the 1's conscientiousness and the 7's excitement-seeking.

**How well Big Five identifies the core.** Logistic regression from the five domains: 31.9% balanced accuracy; from
all 30 facets: 33.0% (chance 11.1%). The Big Five carries real information about the core, but most of the Enneagram
is not reducible to it.

### Leak control: profiles that never mention the Enneagram (n = 168)

| Scale | All profiles epsilon-squared | No Enneagram talk |
|---|---|---|
| E | .28 | .18 |
| C | .15 | .15 |
| A | .16 | .10 |
| N | .14 | .09 |
| O | .13 | **.00** |

Classification falls from 32% to 22% (still twice chance; the sample is a quarter the size). The core means keep their
shape: 8s lowest in N (52) and A (16), 7s lowest in C (14), 4s highest in N (84), 2s and 3s the most extraverted. Two
signatures survive with large effects even here: 8s (vulnerability -1.78, assertiveness +1.48) and 7s
(excitement-seeking +.95, C -.82). **Openness differences disappear completely**, so the high-O profiles of 4s and 7s
above are probably the agent (or the typing) echoing the stereotype rather than the text.

## 2. Enneagram and MBTI

**Crowd types, PDB** (834 profiles, Cramér's V .48, p < 1e-200). Share of each pole and z against the whole sample:

| Core | E | S | T | J | Commonest types | Most over-represented (lift) |
|---|---|---|---|---|---|---|
| 1 | .34 | .45 (+2.0) | **.90 (+5.0)** | **.87 (+5.6)** | INTJ, ISTJ, ENTJ | ISTJ 9.4x, ESTJ 6.3x |
| 2 | **.85 (+6.5)** | .47 (+3.2) | **.07 (-7.2)** | .67 (+4.2) | ENFJ, ESFJ, ESFP | ENFJ 5.8x, ESFJ 5.0x |
| 3 | **.78 (+5.6)** | .33 | .63 (+2.4) | **.80 (+6.6)** | ENTJ, ESFJ, INTJ | ENTJ 4.7x, ESTJ 4.1x |
| 4 | .18 (-6.2) | .19 (-2.7) | **.09 (-8.8)** | .23 (-4.3) | INFP, INFJ, ISFP | INFP 4.1x, ISFP 2.8x |
| 5 | **.07 (-9.6)** | .11 (-5.0) | **.81 (+7.9)** | .45 | INTP, INTJ, INFJ | INTJ 3.6x, INTP 3.5x |
| 6 | .17 (-5.3) | .44 (+2.7) | .48 | .42 | INFP, INFJ, ISFP | ISTJ 4.8x, ISFP 2.7x |
| 7 | **.96 (+11.6)** | .10 (-5.0) | .52 | **.13 (-7.0)** | ENTP, ENFP, ENTJ | ENTP 4.0x, ENFP 3.9x |
| 8 | .85 (+5.8) | .53 (+3.7) | .80 (+4.6) | .27 (-2.3) | ESTP, ENTP, ENTJ | ESTP 7.1x, ESFP 4.2x |
| 9 | .27 (-3.9) | **.53 (+5.1)** | .34 (-3.1) | .46 | ISFJ, ISTP, INFP | ISFJ 5.7x, ISTP 2.9x |

Whole-sample shares: E .41, S .26, T .51, J .43. The pattern is the familiar PDB correlation table: 7 = ExxP, 5 = IxTx,
4 = IxFx, 2 = ExFJ, 3 and 1 = xxTJ, 8 = ExTx with a strong ESTP pull, 9 = ISxx. 6 is the only core without a clear
MBTI lean.

**Without Enneagram talk** (248 profiles) the association is just as strong (V .52) and every direction holds.

**PANDORA self-report** (793 Reddit users who state both, V .37). Same directions on the strong links:

| Core | PDB crowd | PANDORA self-report |
|---|---|---|
| 1 | T .90, J .87 | J .90; T only .42 (INFJ 54% of 1s) |
| 2 | E .85, F .93, J .67 | F .91, J .84; mostly I (INFJ 61%) |
| 3 | E .78, J .80 | E .70, T .77; J/P even |
| 4 | I, F .91 | I, F .89 |
| 5 | I .93, T .81 | I .90, T .77 |
| 7 | E .96, P .87 | E .84, P .96, T .85 (ENTP 66%) |
| 8 | E .85, T .80 | E .71, T .85 |
| 9 | I, S .53, F | I, F; S not over-represented |

The F/T and E/I links replicate in people's own reports. The S leanings of 1, 2, 6, 8 and 9 on PDB don't: PANDORA
users almost never report S (about 13%), and its INFJ-heavy population (INFJ is the commonest type for 1, 2, 4 and 6)
pulls everything towards N and J.

**Wings.** Within each core the crowd favours one wing: 7w6 over 7w8 (91 vs 46), 8w7 over 8w9 (47 vs 8), 9w1 over 9w8
(70 vs 38), 2w3 over 2w1 (56 vs 16), 6w5 over 6w7 (58 vs 24). 5w4 / 5w6 (77 vs 73) and 4w5 / 4w3 (63 vs 55) are
even.

## 3. Instincts

**Which cores lean which instinct** (696 profiles; chi-squared p = 2e-6, V .20):

| Core | sp | so | sx |
|---|---|---|---|
| 1 | **.47** | .27 | .27 |
| 2 | .32 | .31 | .37 |
| 3 | .35 | **.42** | .23 |
| 4 | .27 | .38 | .35 |
| 5 | **.50** | .27 | .23 |
| 6 | **.65** | .18 | .17 |
| 7 | .22 | .38 | **.40** |
| 8 | .40 | .23 | .38 |
| 9 | .44 | .36 | .20 |

6s are typed self-preservation-first two times in three, 5s and 1s about half the time, 7s rarely.

**Instinct effects across all cores** (Big Five after removing each core's mean, so this is the instinct's own
effect; mean percentile-point difference from the core mean, q < .05):

| Scale | sp | so | sx | epsilon-squared |
|---|---|---|---|---|
| E overall | -5.7 | +2.0 | +5.3 | .023 |
| Excitement-seeking | -3.1 | -2.2 | +6.6 | .020 |
| Cheerfulness | -5.4 | +1.9 | +5.0 | .024 |
| Adventurousness | -4.9 | +1.3 | +5.1 | .017 |
| Emotionality | -4.4 | +1.2 | +4.5 | .011 |
| C overall | -0.3 | +4.9 | -5.1 | .021 |
| Achievement-striving | -2.7 | +6.0 | -3.2 | .018 |
| Self-discipline | +0.2 | +4.4 | -5.3 | .014 |
| Cautiousness | +0.7 | +3.0 | -4.4 | .019 |
| A overall | -3.0 | +4.8 | -1.4 | .010 |
| Sympathy | -5.0 | +5.6 | +0.3 | .016 |
| Cooperation | -1.0 | +4.6 | -3.9 | .021 |
| Immoderation | -0.3 | -3.7 | +4.7 | .021 |
| Self-consciousness | +3.8 | -2.3 | -2.5 | .014 |

- **sp-first:** the most inward and reserved (lower E, sympathy, emotionality, artistic interest; more self-conscious).
- **so-first:** the most dutiful and cooperative (higher C, achievement, cooperation, sympathy, altruism; least
  impulsive).
- **sx-first:** the most intense (excitement-seeking, adventurousness, emotionality, immoderation), least cautious and
  self-disciplined.

The effects are real but small: about 2% of variance, against 13-28% for the core.

**Instincts and MBTI.** E share: sp .31, so .59, sx .54; T share: sp .58, so .47, sx .35. Self-preservation types
are typed more I and T, sexual types more F.

## 4. The 27 subtypes

Agent IPIP mean percentiles for each core x dominant-instinct cell with at least 8 profiles, the facets that most
separate it from the other two subtypes of its core (Cohen's d, uncorrected: exploratory), and its commonest crowd
MBTI types. Naranjo and Chestnut's subtype names are given for orientation; "countertype" marks the subtype usually
described as running against its core's type.

| Subtype | n | O | C | E | A | N | Vs the other two of its core | MBTI | Theory |
|---|---|---|---|---|---|---|---|---|---|
| **SP1** | 10 | 71 | 58 | 30 | 43 | 68 | trust +.92, morality +.86, vulnerability +.73 | ISTJ, INTJ | "worry": anxious, scrupulous |
| **SO1** | 8 | 61 | **69** | **15** | 36 | 64 | E -1.04, gregariousness -.92, excitement -.90 | mixed | "non-adaptability": the rigid model |
| **SX1** | 8 | 69 | 47 | 48 | **18** | 69 | excitement +1.72, cautiousness -1.51, adventurousness +1.32, morality -1.08 | INTJ | "zeal", countertype: the reformer who acts on others |
| **SP2** | 17 | 86 | 27 | 42 | 67 | 79 | assertiveness -1.34, activity -1.12, modesty +.82 | ESFJ | "privilege", countertype: more shy and childlike |
| **SO2** | 18 | 87 | 43 | **74** | 57 | 63 | gregariousness +.79, assertiveness +.72, achievement +.69, self-consciousness -.74 | ENFJ | "ambition": the leader-helper |
| **SX2** | 20 | 91 | 25 | 62 | **28** | 77 | A -1.01, trust -.99, altruism -.94, sympathy -.91 | ESFP, ENFJ | "seduction/aggression": less selfless |
| **SP3** | 18 | 67 | **60** | 53 | 23 | 66 | adventurousness -1.07, gregariousness -1.04, excitement -1.04, self-discipline +.82 | ENTJ, INTJ | "security", countertype: the quiet workhorse |
| **SO3** | 25 | 80 | 39 | 69 | 22 | 70 | adventurousness +.55, morality -.52, gregariousness +.50 | ENTJ, ESTJ | "prestige": the public performer |
| **SX3** | 13 | 89 | 29 | 73 | 31 | 70 | self-efficacy -.71, liberalism +.62, sympathy +.62 | ESFJ, ENFJ | "charisma": achieving for others |
| **SP4** | 23 | 88 | 29 | **18** | 49 | 88 | orderliness +.85, C +.65, gregariousness -.57 | INFP, ISFP | "tenacity", countertype: endures quietly |
| **SO4** | 32 | 95 | **15** | 20 | 53 | **93** | anxiety +.70, vulnerability +.69, self-consciousness +.63, self-efficacy -.65 | INFP | "shame": the sufferer |
| **SX4** | 32 | 92 | 17 | **47** | 32 | 83 | self-consciousness -1.11, E +1.00, anxiety -.99, assertiveness +.91 | ENFP, ISFP | "competition": the outward, angry 4 |
| **SP5** | 51 | 78 | 36 | 22 | 27 | 74 | intellect -.64 (otherwise close to the core mean) | INTP, INTJ | "castle": the default 5 |
| **SO5** | 32 | 80 | 50 | 25 | 30 | **65** | vulnerability -.61, N -.60, depression -.54, self-discipline +.53 | INTP, INTJ | "totem": steady, ideal-driven |
| **SX5** | 27 | 84 | 35 | **15** | 28 | 81 | anger +.55, N +.54, friendliness -.48, intellect +.47 | INTP, INTJ, INFJ | "confidence", countertype: the emotional 5 |
| **SP6** | 42 | 76 | 30 | 21 | **51** | 84 | sympathy +.77, modesty +.75, altruism +.73, cooperation +.63 | INFP, INFJ | "warmth": seeks safety in alliances |
| **SO6** | 12 | 84 | 35 | 15 | 40 | 81 | artistic interests +.70, intellect +.60, gregariousness -.50 | INTP, ISTJ | "duty": rule- and idea-bound |
| **SX6** | 11 | 57 | 24 | 32 | **14** | 84 | sympathy -1.32, cooperation -1.10, morality -1.03, altruism -1.03 | ISTJ, ISFP | "strength", counterphobic countertype |
| **SP7** | 20 | 79 | 13 | 56 | **9** | 69 | sympathy -1.45, altruism -1.21, morality -1.21, O -1.21 | ENTP | "keeper of the castle": the pragmatic opportunist |
| **SO7** | 37 | 92 | 22 | 61 | 37 | 69 | achievement +.71, C +.70, immoderation -.67, self-discipline +.57 | ENTP, ENFP | "sacrifice", countertype: restrains own appetite |
| **SX7** | 37 | **96** | **8** | 67 | 55 | 77 | sympathy +.92, altruism +.91, cheerfulness +.89, A +.87, trust +.77 | ENFP | "fascination": the idealistic dreamer |
| **SP8** | 15 | 57 | 38 | 63 | **5** | 51 | emotionality -.99, altruism -.82, sympathy -.81, O -.78 | ESTP | "satisfaction": the blunt pragmatist |
| **SX8** | 12 | 78 | 39 | **74** | 14 | 50 | friendliness +.59, altruism +.58, A +.57, cheerfulness +.53 | ESFP, ENTJ | "possession": the passionate rebel |
| **SP9** | 40 | 70 | 21 | 22 | 43 | 79 | sympathy -1.36, A -1.35, altruism -1.34, trust -1.34, cooperation -1.14 | ISTP, INTP | "appetite": comfort-seeking, withdrawn |
| **SO9** | 31 | 85 | 42 | 44 | **84** | 65 | altruism +1.10, sympathy +1.05, dutifulness +1.01, activity +.99, friendliness +.98 | ESFJ, ISFJ, INFJ | "participation", countertype: works hard for the group |
| **SX9** | 17 | 85 | 28 | 27 | 78 | 74 | trust +.78, anger -.67, artistic interests +.64, emotionality +.57 | ISFJ, INFP | "fusion": merges with one other |

SO8 (9 profiles, 7 with IPIP) is below the threshold.

**What stands out.**
- **Countertypes behave like countertypes.** SX6 is the least agreeable 6 by more than a standard deviation (the
  counterphobic 6); SO7 is the most conscientious and least impulsive 7; SO9 the most active, dutiful and altruistic 9;
  SX4 the most assertive and least self-conscious 4; SP3 the most disciplined and least sociable 3; SX1 the most
  excitement-seeking and least cautious 1; SP2 the least assertive 2. Each matches the subtype's textbook description
  in direction.
- **Some cores split almost in two.** 9s: SP9 (agreeableness 43, MBTI ISTP/INTP) against SO9 (84, ESFJ/ISFJ). 7s: SP7
  (A 9, ENTP) against SX7 (A 55, ENFP). 4s: SO4 (N 93, E 20, INFP) against SX4 (N 83, E 47, ENFP/ISFP). In each case
  the crowd's MBTI choice follows the same split.
- **Agreeableness carries most subtype differences.** It is the top separating scale for SX2, SX6, SP7, SX7, SP8, SP9,
  SO9 and SP6; the instinct often decides whether a core's need is pursued warmly or bluntly.
- **The default subtype is rarely the striking one.** SP5 is almost exactly the 5 average; the distinctive 5s are SO5
  (calmer) and SX5 (more volatile).

## Caveats

- **Circularity.** Crowd votes are shaped by stereotypes; voters who know a profile is "SX6" may vote its Big Five and
  MBTI to fit. The agent read raw text that may state the subtype. The leak-control subset protects the core-level
  results (except Openness) but is too small to check subtypes.
- **Small cells.** Subtype cells hold 8-51 profiles and the within-core contrasts are uncorrected; with 35 scales per
  cell, a few |d| around .6 are expected by chance. The large, theory-matching effects (|d| > 1) are the ones worth
  taking seriously.
- **Agent Big Five is a reading of text, not a self-report.** It agrees with real self-report on essays (AUC .59 via
  the panel model), but PDB profiles of typology enthusiasts are not a general population: the whole sample is high in
  O and N and low in C.
- **PANDORA** confirms the MBTI links with real self-report, but only 64 of its users report both Enneagram and Big
  Five, too few to test the Big Five profiles independently.

## Possible follow-ups

- Repeat the subtype table on crowd SLOAN votes, a Big Five source independent of the agent.
- Tritype: does the second and third fix add Big Five information beyond the core?
- Subtype-level leak control once more profiles without Enneagram talk are crawled.
