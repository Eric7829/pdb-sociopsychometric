# Brief for an independent reviewer

You are being asked, as a neutral third party, to settle a methodological disagreement in a personal research project and decide how it should proceed. The project owner (a university engineering student, "the owner") and their coding assistant (an AI, "the assistant") disagree. Read everything below, then give your own reasoned decision. You are free to disagree with both sides.

## The project in one paragraph

A scraper collects self-written profiles from the Personality Database (PDB) "Pdb Profiles" community: users who post long self-descriptions, questionnaires and Google Docs about themselves, and whom other users then vote on with MBTI, Enneagram and other typology labels. Each profile therefore has (a) the person's own text and (b) a crowd vote distribution. In addition, an LLM "agent" (DeepSeek v4 Flash, via OpenRouter) reads each person's text, role-plays them (a persona prompt inspired by Park et al. 2024, "Generative Agent Simulations of 1,000 People"), and fills in two real instruments: IPIP-NEO-120 (Big Five) and MBTI Form M; both are scored locally. Small text models (TF-IDF + multilingual-e5-small, linear heads) are then trained to predict these targets from text, and served in a local web panel where someone can paste a self-description and get Big Five percentiles and an MBTI type. A third label source, Jev 1.14 (TypeSafe's "decision model" on OpenRouter: give it text and a multiple-choice question, it returns a probability per option), was asked which of the 16 types the PDB crowd would vote for each profile; its probabilities are used as soft targets to distil a better 16-type head.

## The disagreement

The owner pasted a ~180-word self-description into the panel. Summary of its content (not verbatim): an engineering student who keeps late hours; self-described as lazy, an introvert, quiet, with few friends; daydreams a lot; used to be very good at chess; describes themself as spiteful and grudge-holding, defensive about feelings, impulsive but "balanced out by Machiavellianism"; thinks in first principles to find loopholes others miss, which feeds their ego. (The owner jokingly called it an "Ayanokoji monologue", after an anime character PDB commonly types as INTP or ENTP-adjacent schemer.)

The panel returned:

| Output | Result |
|---|---|
| Crowd card, headline type (letters pooled with Jev-distilled 16-type head, letter weight 0.4) | **ENTP** 37%, INTP 14%, INFP 10% |
| Crowd card, letter heads (class-balanced logistic, TF-IDF + e5) | I 75%, N 76%, T 80%, P 84%, i.e. INTP |
| Agent card (predicting DeepSeek's Form M result) | INTP (I 79%, N 82%, T 71%, P 74%) |

The owner considers themself INTP (an introvert) and thought the E/I was "backwards". After investigation the assistant found it is not a display bug: the 16-type head and the letter heads disagree, and the 16-type head outweighs the letters in the pool.

**The owner's position:** use the DeepSeek agent's MBTI (Form M) result as the ground truth for type, because it matched their self-perception here and because the Stanford paper shows LLM agents can reproduce people's own survey answers.

**The assistant's position:** do not. Reasons given:
1. Park et al. validated agents against each participant's own self-report (GSS ~85% normalised accuracy, Big Five ~0.8 normalised correlation) after 2-hour structured interviews. Here there are no self-reports to validate against (except the owner's own IPIP-300 result, see below), and the input is whatever people chose to post, not an interview.
2. The agent's type distribution is implausible as a population truth: on 342 profiles with all labels it calls 28% E (crowd 45%), 6% S (crowd 30%), 38% T (crowd 49%), 18% J (crowd 46%). Top agent types: INFP 123, ENFP 72, INTP 60, INTJ 34, ENTP 17.
3. A text model trained to predict the agent's type got 37.7% exact vs 36% for always guessing INFP, so there is almost no learnable signal beyond the collapse.
4. Against the crowd modal type, agent Form M is 28.7% exact vs Jev 1.14 36.0% (same 342 profiles). Per letter vs crowd: agent E/I 74%, S/N 73%, T/F 78%, J/P 64%; Jev 76%, 77%, 77%, 68%.
5. Neither the crowd nor the agent is the person's "actual type"; the only real ground truth is self-report on a validated instrument, and the assistant suggested 20-30 volunteers. The owner says they cannot find more than about 3 people willing to take part.

The assistant proposed instead: correct Jev's label skew toward crowd base rates before distillation, class-balance the 16-type head, flag in the UI when the headline type contradicts a letter head at >=70%, and relabel the crowd card as "how PDB voters would type this text".

## Key numbers (all 5-fold cross-validated unless stated)

Dataset: PDB "Top" listing, 500 usable profiles (sample `top`) + next 500 (`top2`) + 59 ES-filtered top-up (`es`, excluded from type training). 795 Top+top2 profiles have an eligible crowd MBTI distribution (>=10 votes after removing the creator's self-vote). Crowd modal type counts (795): ENTP 93, INFP 92, INTP 88, INTJ 79, ENFP 78, INFJ 72, ENTJ 52, ENFJ 44, others smaller. Jev 1.14 top-1 counts on the same 795: ENFP 222, INFP 164, INTP 91, ENTP 84, INTJ 63, ESFJ 29, INFJ 27, ENTJ 19 (E share 52% vs crowd 42%).

Crowd 16-type (795 profiles, commonest-type baseline 11.7%):

| Model | Exact | Top-3 | Log loss |
|---|---|---|---|
| Ours (TF-IDF + e5, letters x 16-type pool) | 17.2% | 44.4% | 2.516 |
| Jev 1.14 alone | 33.7% | 60.3% | 2.428 |
| Jev + ours ensemble | 33.6% | 60.3% | 2.115 |
| Student on 0.5 Jev + 0.5 crowd one-hot, pooled with letters (shipped in panel) | 19.1-19.7% | 45.5-46.2% | 2.468 |

Jev accuracy by text length quartile: <400 words 16.6%, 400-1.7k 29.3%, 1.7-4.6k 39.7%, >4.6k 49.2%; ours flat 14-20%. (Possible residual typology jargon in long texts has not been ruled out; scrubbing removes explicit type codes, not cognitive-function talk.) An earlier mistake routed "Jev" requests through `typesafe/jev-router`, which forwards to frontier chat LLMs (Gemini, Grok, GLM...); those labels scored 42.7% on 386 profiles and were discarded. A memorisation test on profiles created after Jev's release was abandoned: only 2 usable recent profiles.

Agent Big Five: text models predict the agent's IPIP domains with mean Spearman r 0.50 (TF-IDF + e5). Agent IPIP correlates with crowd typing in expected directions (crowd E share vs agent IPIP Extraversion rho +0.65; crowd T share vs agent Agreeableness rho -0.53). Framing check: first-person persona prompt vs third-person "predict this person's answers" prompt give rank correlations 0.92-0.97 across domains. The owner's own validation: the panel's Big Five from a 174-word intro vs the owner's real IPIP-300 result: mean absolute error 7.8 percentile points, vs 22.4 for a "typical PDB profile" baseline. That is the only real self-report ground truth in the project (n = 1).

Encoders tried for the text models: e5-small is the default; ModernBERT-base, Jina v5 small (2048-token chunks on an 8 GB GPU) and Matryoshka slices of Jina were all worse. Long-context encoders (gte-base 8k, nomic 8k) and fine-tuning an encoder on Jev's soft labels have not been tried.

## Constraints

- Budget: personal, a few dollars at a time on OpenRouter. Agent runs cost ~$0.009/profile; Jev ~$0.0003/profile.
- Hardware: one laptop RTX 5060, 8 GB VRAM.
- Recruiting: the owner can find about 3 willing participants, not 20-30.
- Ethics: the scraped data stays local and is not redistributed; the panel only listens on localhost and never stores assessed text. The project should not be used for hiring or for typing people without consent.
- The Stage 3 agent must not be a Claude model (owner's rule).

## What we need from you

1. **Settle the dispute.** Should DeepSeek's Form M result be treated as ground truth for MBTI type? If not, what should the panel's headline type represent, and how should conflicts between heads be shown? Engage with the Park et al. argument directly.
2. **Define a defensible operational notion of "actual type"** for this project, given that only ~3 volunteers are available. Consider options such as: public datasets where people's own validated test results are paired with their writing (name specific ones you are confident exist and say how they could be used); PDB profiles that include the person's own official test result in their text (self-report, extracted as a label); test-retest or multi-rater consistency; latent-class or IRT-style models that treat crowd votes, agent answers and Jev as noisy raters of one latent type; using the Big Five (which has real validation) and mapping to MBTI via known correlations; or dropping "actual type" in favour of explicitly labelled proxies. Weigh each for cost, validity and feasibility here.
3. **Decide the next steps**, in priority order, with a concrete success criterion for each and what result would make you change course. Be specific to this codebase and these numbers.
4. **Flag anything else** in the setup that you think is a serious methodological error or a wasted effort.

Be direct; give a clear recommendation rather than a survey. The relevant README, the results report and two code files follow.
