# IPIP-NEO-120 in Python

Offline questionnaire, raw-score calculation, and empirical percentile scoring for
all five domains and 30 facets. Requires Python 3.10 or later. No packages,
spreadsheet software, accounts, or internet connection are required.

From this folder:

```powershell
python ipip_neo.py take
```

Answer each item from 1 (Very Inaccurate) to 5 (Very Accurate). Describe yourself
as you generally are now. Progress is saved after every answer in `answers.json`.
Enter `back` to revise the previous question, or `q` to pause. Run the same command
to resume. Finishing saves `results.json` and a readable `results.txt`.
The original XLS contains demonstration responses; this Python flow starts blank.

Separate people or attempts can use separate paths:

```powershell
python ipip_neo.py take --answers sessions/alex.json --output sessions/alex-results.json
```

## Score existing answers

```powershell
python ipip_neo.py score answers.json --output results.json
```

The input is either a JSON array of exactly 120 integer answers or an object with
a `responses` array. Answers must be in original question order and must **not**
already be reverse-scored. Incomplete or invalid answers are rejected. Saved
partial sessions use `null` for unanswered questions and can be resumed with `take`.
Reports include responses, raw sums, percentiles, and the reference-data provenance.
The program saves data locally; take care when sharing answer or report files.

## Inspect the scoring curves

```powershell
python ipip_neo.py curve N
python ipip_neo.py curve N --raw 72
python ipip_neo.py curve E1 --raw 12
```

Domain codes: N Neuroticism, E Extraversion, O Openness, A Agreeableness,
C Conscientiousness. Each has six facets, numbered 1 through 6.
`percentile_curve()` also returns the entire curve as Python data for plotting.

## Use in another Python program

```python
from ipip_neo import load_items, score, percentile_curve

questions = load_items()
report = score([3] * 120)  # Example only: replace with real original responses.
print(report["scores"]["N"])  # Raw sum, percentile, scale name and range.
curve = percentile_curve("N1")
```

Keep `ipip_neo.py`, `items.json`, and `empirical-norms.json` together. They total
about 32 KB; the original Office documents and text lookups are not needed to run
Python. Data files are located relative to the script, so it works from another
working directory. Answer/report paths are relative to your working directory.
Existing files at the chosen answer/report paths are updated by the command.

## Scoring and reference sample

Reverse-keyed answers become `6 - answer`. Four items sum to a facet (4–20),
and six facets sum to a domain (24–120). Domain percentiles are calculated from
domain sums, never from averaged facet percentiles.

The histogram reference is **410,376 complete protocols** from Johnson's
619,150-row `IPIP120.dat`. Every response must be 1–5; there are no additional age,
sex, country or duplicate filters. Percentiles use actual empirical midranks:

`100 * (number below your score + 0.5 * number tied) / reference sample size`

The reference is a pooled, self-selected internet sample, not a representative
national population. These are locally derived norms rather than an
author-published norm table. They differ from the age/sex-specific T-score norms
in the original XLS. The Python flow implements the empirical scoring method;
it does not reproduce that spreadsheet's separate norm system.
Scores retain full precision in JSON and display two decimal places in text;
display precision does not imply that percentile differences that small are meaningful.

The 120 item texts and key were converted from the included source text, which
was extracted from the original scoring workbook. The original files are preserved.

- Author's repository: https://osf.io/tbmh5/
- Original questionnaire/key/scoring tool: https://osf.io/ycvdk/
- Dataset and format documentation: https://osf.io/wxvth/
- Official IPIP item scales: https://ipip.ori.org/30FacetNEO-PI-RItems.htm
- Johnson (2014), *Journal of Research in Personality*, 51, 78–89.
  https://doi.org/10.1016/j.jrp.2014.05.003

IPIP items are public domain. No raw reference-person records are bundled.

## Verification

```powershell
python -m unittest -v
```

Tests cover reverse scoring, midpoint raw sums, midrank arithmetic, input
validation, monotonic percentile curves, saved-file scoring, and quitting,
revising, resuming and completing the interactive questionnaire.
