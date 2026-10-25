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
