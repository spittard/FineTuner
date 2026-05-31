# Plugging report — automated assessment summary

Source: `plugging_matches.json`  
Records: **3217**  
CSV: `plugging_report_assessment.csv` (one row per plugging record; use filters / pivot in Excel)

## Severity counts

| severity | count |
|----------|------:|
| high | 67 |
| info | 2335 |
| med | 815 |

## Issue code counts

| code | count |
|------|------:|
| `DUPLICATE_ROUNDED_SCORES_IN_TOP5` | 1505 |
| `NO_QUERY_GEO_RANK1_HAS_LOCATION` | 511 |
| `STRONG_SCORE_NON_EXACT_NAME` | 96 |
| `HYBRID_SEMANTIC_LED_AT_CEILING` | 96 |
| `SUSPECT_GEO_RANK4_CITY_TOKEN_EQUALS_STATE_TOKEN` | 17 |
| `SUSPECT_GEO_RANK2_CITY_TOKEN_EQUALS_STATE_TOKEN` | 16 |
| `SUSPECT_GEO_RANK5_CITY_TOKEN_EQUALS_STATE_TOKEN` | 15 |
| `SUSPECT_GEO_RANK1_CITY_TOKEN_EQUALS_STATE_TOKEN` | 15 |
| `SUSPECT_GEO_RANK3_CITY_TOKEN_EQUALS_STATE_TOKEN` | 11 |
| `GATE_A_TOP5_TIE_CLUSTER` | 3 |
| `SUSPECT_GEO_RANK3_NORMALIZED_CITY_EQUALS_STATE` | 1 |
| `SUSPECT_GEO_RANK1_NORMALIZED_CITY_EQUALS_STATE` | 1 |

## How to use with SMEs

1. Sort CSV by `severity` (high → med → info) then `issue_codes`.
2. For each row, read `narrative` as the same style of critique as a manual screenshot pass.
3. Map `issue_codes` to fix tracks: matcher (rank/score), data (bad city/state), UI (display tiers), rationale (copy).
4. Re-run after `match_plugging_records.py` + matcher changes so scores reflect new logic.

## Rubric (manual pass — same depth as single-record review)

- **Geo–query alignment**: If query has no locale, bare candidates should not rank below geo-only duplicates at the same score.
- **Score vs name truth**: Near-100% implies exact legal string (or documented exception); hybrid semantic-led at ceiling needs rationale.
- **Ties**: Identical scores with different specificity (geo vs bare) need visible tie-breakers or ordering rules.
- **Data hygiene**: City/state duplicates, `UT, UT`, city==state token, etc.
- **Trade-style drift**: Different suffixes (`Group` vs `IT Solutions`) at high confidence need explicit justification.

