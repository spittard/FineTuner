# Plugging report — SME-nonsensical findings (code 3× + report 3×)

Scope: reviewed the report-producing chain three times and assessed the live report three times for things an SME doing this **manually** would reject. Numbers below are from `plugging_report.csv` (16,085 match rows ≈ 3,217 records × 5) and `plugging_matches.json` unless noted.

**Honesty / freshness:** `plugging_report.md` (May 1) and `plugging_report_sample.md` (Apr 29) were generated **before** the recent matcher edits (token-reorder demote, corp/enterprise demote, top-5 spread, no-geo tie-break flip) and **no rematch has been run since**. So scoring findings below still appear in the on-disk report; they need a **full rematch** to clear. Two findings are pure **report-generator** bugs and were fixed in this pass (no rematch needed).

---

## A. Fixed in this pass (report generator only — `tests/generate_plugging_report.py`)

### A1. "Type" column is meaningless — 85% of rows say "Hybrid (semantic-led)"
- **Evidence:** 13,631 / 16,085 rows = **84.7%** labeled `Hybrid (semantic-led)`.
- **Cause:** `match_type_label()` compared the **raw** `semantic_score` (a FAISS inner-product sum, often ~6.0) against `string_score` (≤1.0). The `semantic-led` branch (`semantic_score > string_score and >= 0.4`) therefore fired for almost every hybrid row.
- **SME view:** the Type column carries no information; "semantic-led" on an obvious lexical match looks wrong.
- **Fix:** use `normalized_semantic_score` (0–1) for the comparison. Also `acronym_expansion_low_conf` now labels as "Acronym".

### A2. 66 rows render as broken markdown table rows
- **Evidence:** 66 `match_name` values contain a literal `|` (e.g. `Prestige | Global Meeting Source`). The rationale HTML was pipe-escaped, but the **Company cell** was not, so each splits the row into extra columns.
- **SME view:** garbled table lines; looks like a data/export defect.
- **Fix:** `format_company_cell()` now escapes `|` → `\|` (valid in GFM cells, including inside code spans).

---

## B. Need a full rematch to clear (matcher scoring; edits already in `matcher.py`, unverified until rematch)

### B1. Token-reorder false twins ranked High Confidence
- **Example (live report):** `Corporation of Hamilton` (a municipal corporation) → top-5 all `Hamilton Corporation` (private cos in NJ/NV) at **99.9%**, bucketed **High Confidence**.
- **SME view:** a city government is not "Hamilton Corporation"; 99.9% is indefensible.
- **Status:** matcher now demotes reordered short names (×0.85) and `Corporation of {X}` vs `{X} Enterprises`; audit Gate B targets row 7880903. **Requires rematch.**

### B2. Tie mush — identical percentages across top-5
- **Evidence:** `DUPLICATE_ROUNDED_SCORES_IN_TOP5` = 1,937 records; `GATE_A_TOP5_TIE_CLUSTER` = 324 records within 0.001.
- **Example:** Hamilton rows all 99.9%; Prestige rows three at 96.0%.
- **SME view:** "why are these all the same / how do I pick?" Ordering looks arbitrary.
- **Status:** `_ensure_top5_score_spread` added; audit Gate A targets ≤5. **Requires rematch.**

### B3. No-query-geo: geo-tagged office ranked above the bare/national row
- **Evidence:** `NO_QUERY_GEO_RANK1_HAS_LOCATION` = 715; `NO_QUERY_GEO_BARE_ROW_NOT_FIRST` = 386.
- **Example:** `Xylix Secure Group` (no locale) → rank-1 a Salt Lake City row while the bare `Xylix Secure` sits at rank 3, same/near score.
- **SME view:** if I didn't give a city, don't prefer one office over the umbrella record.
- **Status:** tie-break flipped to prefer bare rows when query has no geo. **Requires rematch.**

### B4. Malformed candidate geography
- **Evidence:** `SUSPECT_GEO_*` ≈ 84 rows (city token == state token, `UT, UT`, etc.).
- **SME view:** obvious bad data erodes trust in the whole block.
- **Status:** likely **source-data** issue; matcher can't always repair. Flag in assessment CSV; consider a data-cleaning pass.

### B5. Whitespace-variant duplicate offices
- **Example:** `Hamilton Corporation` (Mount Laurel, NJ) and `Hamilton Corporation` (Mount  Laurel, NJ — double space) both shown at 99.9%.
- **Status:** matcher dedup now normalizes whitespace in the legal key. **Requires rematch to confirm collapse.**

---

## C. Wording / display polish (lower severity)

- **C1. Verdict banner vs Score column:** older rationale rendered `EXCELLENT MATCH (100%)` while the Score column showed `99.9%`. Current code rounds the banner to 1 decimal; confirm on regeneration that banner % == score column %.
- **C2. Boilerplate concept text:** "Reflects highly aligned industries" appears even for a city-gov vs private-corp pair; consider suppressing concept line when name relationship is weak.
- **C3. "EXCELLENT/STRONG MATCH" green banner on 80–88% wrong-office rows** reads more confident than the tier; align banner thresholds with tier bands.

---

## Distributions (rank-1, live report)

| Tier | rank-1 records |
|------|---------------:|
| High | 662 |
| Good | 522 |
| Medium | 975 |
| Low | 1,058 |

| match_type (all 5 ranks) | rows |
|---------------------------|-----:|
| Hybrid (semantic-led) | 13,631 |
| Acronym | 616 |
| Exact + partial geo | 577 |
| Exact name, wrong office | 529 |
| Exact (name only) | 424 |
| Exact + location | 272 |
| Exact name, verify address | 36 |

(Type distribution will shift once A1's normalized-semantic fix is regenerated.)

---

## Next mechanical step (closed loop)

1. Regenerate report from existing `plugging_matches.json` → A1/A2 land immediately (no RPC).
2. For B1–B5: restart RPC, `match_plugging_records.py --no-resume`, regenerate, `audit_scoring.py`, `verify_control_set.py`, re-run `scripts/assess_plugging_report_full.py`. Compare before/after `high`-severity counts.
