# Observed PDB API contract — October 23, 2026

Verified using Chromium network interception and browser-native fetch, with a user-established PDB session. No PDB HTML parsing is used for ingestion. Raw local evidence is in `data/discovery/` and `data/smoke-captures/`; those directories are intentionally ignored.

## Read-only GET endpoints

Base: `https://api.personality-database.com`.

| Endpoint | Request | Response/termination |
|---|---|---|
| `/api/v1/profiles` | `offset`, `limit=100`, `cid=15`, `pid=1`, `sub_cat_id=4065`, `cat_id=15`, `property_id=1` | `profiles[]`, `subcategory.profile_count`; advance by limit, stop at a short/empty valid `profiles` page. The first two pages were observed. A very large out-of-range offset returned an error envelope, which must never count as end-of-list. |
| `/api/v1/profile/{id}` | numeric profile ID | Profile, `systems[]`, `breakdown_systems`; unauthenticated HTTP 200 can carry `{code:401,message:...}`. |
| `/api/v1/comments/{id}` | `sort=NEW`, `offset`, `range=all`, `limit=100`, `version=W3` | `comments[]`, boolean `has_more`, integer `next_offset`. Pinned comments occur before ordinary comments within this same paginated stream. Respect `has_more`, not total count or deduplicated page length. |
| `/api/v1/comment_replies` | `comment_id`, `limit=100`, `offset` | `comments[]`, `has_more`, `next_offset`; a comment's `reply_count` tells whether a reply stream needs fetching. Replies reference parents using `parent_id` and contain author identity and badge fields. |

The frontend originally requests comment pages of 20 and replies of 10; browser fetch verified that limits of 100 work, reducing the number of requests while retaining the two-second minimum request interval. Empty terminal replies and a terminal top-level page were observed. `comment_count` is not a reliable top-level record count: the example produced 497 top-level records while reporting 1331 comments.

## Field mappings

- Profile `user_id` → creator account ID; `contributor` → creator username; `mbti_profile` → display name (slug derived locally); `wiki_description` → bio quote; `personality_type` → header summary; **`total_vote_counts`** → all-system vote banner. Do not use legacy `vote_count` or `vote_count_mbti` as the banner.
- `breakdown_systems[system_id][]`: `personality_type` identifies the class and `theCount` its observed count. Compare against `systems[].system_vote_count` when present; missing/inconsistent positive breakdowns are schema errors.
- Comment `user_id`, `username`, `comment`, `is_pinned`, `is_hidden`, `is_active`, `parent_id`, `reply_count` establish ownership and content. `vote_types` / `theVote` are the profile-specific active badge, distinct from `user_mbti` account flair. `user_vote_id` and `vote_type` refer to the viewing user's reaction to the comment and are not the author's typology vote.
- System IDs: 1 MBTI, 2 ENNEAGRAM, 3 SOCIONICS, 4 ALIGNMENT, 5 INSTINCT, 6 TRITYPE, 7 TEMPERAMENT, 8 AP, 9 BIG5_SLOAN, 11 JUNGIAN, 12 BIG5_E, 13 BIG5_N, 14 BIG5_C, 15 BIG5_A, 16 BIG5_O. Unknown IDs are retained as `PDB_{id}` without training eligibility.

The example's owner ID is 4957070. Their active badge contains Classic Jungian, temperament and some Big Five votes; it contains no MBTI or Enneagram self-vote. The observed core counts are MBTI `{INFP:25, INTP:5, INTJ:1}` and ENNEAGRAM `{4w5:28,5w4:5,6w5:3,5w6:1}`. Four Google Doc keys were observed in a creator-pinned comment (the original specification mentioned two).

The reply request construction was also checked in the frontend's `basic.61625d18.async.js` bundle: `/v1/comment_replies`, `comment_id`, `offset`, `limit`. REA's static application analysis was partial due to bundle size; the subsequent live JSON response is the validation of that request shape. No challenge code was analyzed or bypassed.
