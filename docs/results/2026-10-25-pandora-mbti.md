# PANDORA: the panel's MBTI cards on self-reported Reddit types (2026-10-25)

`tools/pandora_mbti_test.py`, panel models v0.3 (top + top2), unchanged. PANDORA (Gjurković et al. 2021) users with a
4-letter self-reported type; text from 23 vent/advice/mental-health/personal-question subreddits (main) or 5
introduction subreddits (short). Comments naming a type are dropped, then the panel's scrub runs. Aggregates only.

| Slice | Users | Words | Card | Letters (mean balanced acc.) | E/I | S/N | T/F | J/P | Exact | Top-3 | Majority-type exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 4,264 | 761 | crowd | .548 | .536 | .514 | .611 | .530 | 13.0% | 35.2% | 22.7% (INTP) |
| | | | Jev | .557 | .551 | .542 | .610 | .524 | 18.1% | 45.5% | |
| | | | agent | .538 | .516 | .519 | .610 | .509 | 15.4% | 39.5% | |
| short | 957 | 216 | crowd | .578 | .591 | .541 | .633 | .545 | 19.1% | 45.6% | 27.3% (INTJ) |
| | | | Jev | .588 | .645 | .540 | .642 | .525 | 21.2% | 49.8% | |
| | | | agent | .563 | .567 | .531 | .635 | .518 | 19.2% | 49.7% | |
| short, no r/introvert | 332 | 197 | crowd | .576 | .547 | .516 | .658 | .582 | 21.1% | 47.3% | 19.0% (INTP) |
| | | | Jev | .586 | .574 | .544 | .684 | .543 | 20.5% | 46.4% | |
| | | | agent | .565 | .528 | .515 | .690 | .526 | 20.8% | 46.4% | |

- **The letters transfer weakly.** T/F holds best (balanced accuracy .61 to .69, AUC up to .74). E/I, S/N and J/P sit
  at .51 to .58, close to chance. On PDB the same heads score about .58 to .67 under cross-validation.
- **Short self-introductions read better than long vent text**, even at a quarter of the length. They describe the
  person, while advice and venting threads mostly describe someone else or a situation. That matches what the panel is
  for.
- **r/introvert inflates E/I.** Without it, E/I falls from .59 to .55 (crowd) and .65 to .57 (Jev). "I'm an introvert"
  does the work there, not the model.
- **Exact type doesn't beat always guessing the most common type** on main and short. PANDORA is 23% INTP and 20% INTJ,
  while the cards spread their guesses across INFP/ISFP/ISTP. The Jev card does best on exact and top-3.
- **S/N is the weakest letter everywhere** (.51 to .54). S types are 12 to 14% of PANDORA users, the same S blindness
  seen on PDB.

Caveats: self-reported types often come from online tests, a different label from PDB crowd votes; Reddit comments are
not self-descriptions; the paper's own baselines were trained on all of a user's comments and were deleted with the
rest of the archive, so there is no like-for-like comparison here.
