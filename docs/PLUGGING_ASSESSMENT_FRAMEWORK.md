# Plugging report — assessment framework

This captures how each plugging record should be judged (same bar as a detailed screenshot review) and where outputs live.

## Automated pass (every rematch)

```powershell
python scripts/assess_plugging_report_full.py
```

Produces:

- `plugging_report_assessment.csv` — one row per `row_id` with `severity`, `issue_codes`, and a short `narrative` in the same critique style as a manual SME pass.
- `plugging_report_assessment_summary.md` — counts and the rubric snapshot.

Re-run after any change to `plugging_matches.json` (full rematch) so narratives match current scores.

## Manual rubric — check every record for

1. **Geo–query alignment**  
   If the query has **no** city/state, **bare** candidates (no city/state on file) should not sit **below** geo-tagged rows at the **same** score. Prefer the national / undifferentiated row first when scores tie. (Example: *Xylix Secure Group* — bare *Xylix Secure* should outrank Salt Lake City variants when the query verified no geography.)

2. **Score vs name truth**  
   Near-ceiling scores must align with SME copy: non-identical legal strings (missing tokens such as “Group”, different trade styles such as “IT Solutions”) should not read as “almost exact” unless rationale explains why.

3. **Ties and tie-breakers**  
   Identical or rounded-identical percentages with **different** geo specificity look arbitrary unless the UI or export shows tie-breakers (count, legal key, location_score, etc.).

4. **Data hygiene**  
   Flag `city`/`state` anomalies: duplicate tokens (`UT`, `UT`), city token equals state, obvious import errors.

5. **Hybrid at ceiling**  
   “Hybrid (semantic-led)” at top-tier scores with lexical gaps needs defensible rationale, not only a green dot.

6. **Gate A-style collapse**  
   Top-5 scores within ~0.001: ordering and SME trust both suffer; treat as matcher/audit concern until spread is healthy.

## Fix tracks (map from `issue_codes`)

| Track | Examples |
|--------|-----------|
| **matcher** | Tie order when query omits geo; score floors; top-5 spread |
| **data** | Bad city/state on source rows |
| **UI** | Percent tier labels vs actual score semantics |
| **rationale** | Explain token/suffix/geo choices for hybrid rows |

## Related scripts

- `scripts/analyze_plugging_report_scenarios.py` — aggregate buckets (includes “no query geo but bare later than rank 1”).
- `audit_scoring.py` — automated gates on `plugging_matches.json`.
