# What does your writing say about your personality? 🧠

**I scraped 1,000+ self-descriptions from a personality-typing forum, ran an AI "psych lab" on them, and distilled it into a small model that reads your Big Five and MBTI from text, running 100% on your own machine.** No API calls, no LLM wrapper: TF-IDF and a 149M-parameter embedding model, trained on a consumer GPU. Then I tested it against real people, published research and commercial personality tests.

The idea comes from Stanford's [Generative Agent Simulations of 1,000 People](https://arxiv.org/abs/2411.10109) (Park et al., 2024): let an AI agent become a person and take their personality tests. I pointed that method at a forum full of people describing themselves, then taught a local model to do the agent's job.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-data%20store-FFF000?logo=duckdb&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-F7931E?logo=scikitlearn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HF-gte--modernbert-FFD21E?logo=huggingface&logoColor=black)
![Playwright](https://img.shields.io/badge/Playwright-scraper-2EAD33?logo=playwright&logoColor=white)
![tests](https://img.shields.io/badge/tests-144%20passing-brightgreen)

---

## ⚡ Highlights

- 🖥️ **A local Big Five model with ρ = 0.57:** on held-out profiles, the small model's Big Five matches the full 120-item IPIP-NEO personality test the AI agent took for each writer, at Spearman ρ = **0.57** (5-fold cross-validated, averaged over all five traits). No LLM at inference time.
- 📝 **Transfers to real people it never saw:** on the classic Essays benchmark (2,467 students who took a personality questionnaire), the local model never saw an essay and still scores **56.3%**, within half a point of a 2007 model trained on that exact dataset (56.7%), with all five traits above chance.
- 🎯 **The AI teacher types you like a forum veteran:** the DeepSeek agent the local model learns from hits **30% exact MBTI type** (random guessing: 8%) across 785 profiles, and **59.8%** on Essays zero-shot, above the best published model trained on it (58.8%).
- 🕵️ **Caught the shortcut most "AI typed me" demos take:** leave people's own "I'm an INFJ" in the text and accuracy jumps **12 points**. Scrub it and you see what the AI can actually read.
- 🌀 **The forum thinks in Jungian functions, AI thinks in letters:** voters treat INTP and ENTP as close cousins; models confuse INTP with INTJ.
- 🔢 **Enneagram from a Big Five test:** a model I built from forum data names your core Enneagram type **38%** of the time and has it in its top three **72%** of the time (chance: 11% and 33%), on par with dedicated Enneagram tests in the only independent comparison I found.
- 🧬 **Every Enneagram type has a Big Five fingerprint,** and the infamous "countertype" subtypes show up in the data exactly where the theory says they should.

---

## 🎓 Inspired by Stanford: from generative agents to a local model

[Park et al. (2024)](https://arxiv.org/abs/2411.10109) built AI agents from two-hour interviews with 1,052 real people and showed the agents could answer those people's surveys and personality tests about **85% as accurately as the people replicate their own answers** two weeks later. This project borrows the core of that method, then takes it somewhere new:

| | Stanford (Park et al. 2024) | This project |
|---|---|---|
| Source of each "person" | a 2-hour recorded interview | the person's own public self-description |
| Agent's job | answer surveys, personality tests and experiments as that person | take the IPIP-NEO-120 Big Five test and MBTI Form M as that person |
| Scale | 1,052 people | 1,000+ forum profiles |
| What comes out | a simulated person (an LLM call per question) | **a small local model** trained on the agents' answers: paste text, get Big Five + MBTI in about a second, offline |

The agent is the expensive teacher; the local model is the product.

## 📊 The results

### Reading your MBTI type from how you write

![MBTI accuracy chart](docs/img/mbti-accuracy.svg)

Exact 16-type matching is the hardest test there is: one wrong letter counts as a miss. Per letter, the local model gets S/N right 73% of the time and E/I 62% on full profiles, and the agent reads a profile the way an experienced forum voter would, working out the dominant and auxiliary cognitive functions. Leave self-typing in the text and the agent jumps to 48%, which is why every model here only sees scrubbed text.

### Big Five that holds up against real questionnaires

![Big Five essays chart](docs/img/essays-big5.svg)

The Essays dataset (Pennebaker & King) pairs 2,467 student essays with each writer's own personality questionnaire. Published models were **trained on it**. Mine was trained only on forum self-descriptions and was never shown a single essay.

### Each Enneagram type has a Big Five fingerprint

![Enneagram Big Five heatmap](docs/img/enneagram-big5.svg)

Every type lands where its textbook description says it should: 8s are the most assertive and least vulnerable (assertiveness d = +1.5), 7s chase excitement and skip the discipline, 4s run highest in Neuroticism, 1s are the most conscientious, 9s the most agreeable. The full breakdown, including all **27 instinctual subtypes**, is in the [Enneagram report](docs/results/2026-10-25-enneagram-report.md).

### The crowd uses cognitive functions. The models don't.

![Function flip chart](docs/img/function-flips.svg)

When forum voters split on a type, the runner-up is usually the E/I mirror (same functions, different order). That's Jungian function theory showing up in thousands of votes. Text models make the opposite mistake and confuse J/P, the one letter that changes every function.

---

## 🔬 More findings

| Finding | Number |
|---|---|
| Profiles that talk about typology (the hidden label) | **755 of ~1,000** |
| MBTI ↔ Enneagram link on the forum (Cramér's V) | **.48**, replicated on Reddit self-reports (.37) |
| Big Five → Enneagram core, top-3 hit rate | **72%** (chance 33%) |
| AI agents answering MBTI Form M "in character" drift toward | **INFP** (J→P, S→N, E→I) |
| Short self-introductions vs 4x as much venting text, on Reddit | introductions read **better** |
| Self-type vs crowd agreement on the forum | **69%** exact |

All of it is written up with methods and tables in [`docs/results/`](docs/results/), starting with the [findings summary](docs/results/2026-10-25-findings.md).

---

## 🛠️ How it works

```mermaid
flowchart LR
    A[Personality forum<br/>JSON API] -->|Playwright scraper<br/>rate-limited, replayable| B[(DuckDB)]
    B --> C[Text cleanup<br/>scrub type labels]
    C --> D[AI agent battery<br/>IPIP-NEO-120 + MBTI Form M]
    C --> E[Local models<br/>TF-IDF + gte-modernbert]
    D -->|labels| E
    E --> F[Personality panel<br/>runs 100% locally]
    C --> G[PDB-voter agent<br/>DeepSeek]
    E & G --> H[Validation<br/>Essays · PANDORA · portraits]
```

1. **Scraper.** A Playwright + JSON API crawler with an allowlisted transport, rate limiting, retry/backoff and full capture logs. Every run can be **replayed offline** from its recorded evidence, byte-checked by SHA-256.
2. **Leak-proof text.** Character-span annotations rebuild each profile exactly; a scrubber strips MBTI codes, Enneagram talk, cognitive functions and instincts, so no model can just read the answer.
3. **AI psych lab.** Following Stanford's generative-agent method ([Park et al., 2024](https://arxiv.org/abs/2411.10109)), an LLM role-plays each writer and takes real instruments: the IPIP-NEO-120 Big Five (120 items, 30 facets) and MBTI Form M. Answers come back as forced, structured tool calls in 8 independent blocks, and scoring happens locally, item by item.
4. **Local models (the product).** TF-IDF and gte-modernbert-base (149M parameters, 2,048-token chunks) feed ridge and logistic heads; Big Five reaches ρ = 0.57 against the agents' test results on held-out profiles. Tuned with 5-fold cross-validation and pre-registered ship rules: an upgrade ships only if it improves the target benchmark without hurting any other.
5. **Personality panel.** A local web app: paste any text and get Big Five percentiles plus three MBTI readings (trained on forum votes, on the Jev typing model, and on the AI agent), with confidence and a short-text warning. Pasted text never leaves your machine.
6. **Validation.** The Essays benchmark, PANDORA (10,000 Reddit users with self-reported types), LLM-written type portraits, and a leak probe.

---

## 🚀 Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e '.[dev]'
.venv\Scripts\python -m playwright install chromium
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m pdb_ingest panel --db data/pdb-top.duckdb --samples top top2
```

The panel opens at `http://127.0.0.1:8765`. Crawling, the agent battery and every analysis tool are documented in the [technical reference](docs/technical-reference.md).

## 🔒 Data & ethics

- Only public forum profiles were collected, through the site's own API, at one request per second.
- Raw data, scraped text and research datasets stay local (git-ignored). This repo publishes **code and aggregate results only**, and no individual is named or quoted.
- The panel never logs or stores what you paste.

## 📚 Built on

**Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative Agent Simulations of 1,000 People.* arXiv:2411.10109.** The core idea and agent method behind this project.

Also: Pennebaker & King 1999 (Essays) · Mairesse et al. 2007 · Majumder et al. 2017 · Gjurković et al. 2021 (PANDORA) · Goldberg's IPIP · Naranjo and Ichazo (Enneagram subtypes)

<sub>Numbers are cross-validated on held-out profiles unless noted. The 59.8% Essays result and the agent's raw-text 48% come from 100-essay and 25-profile pilots; the full methods and every caveat are in the reports.</sub>
