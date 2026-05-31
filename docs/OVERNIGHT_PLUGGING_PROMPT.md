# Overnight autonomous prompt — plugging match quality & SME-defensible reports

**See also:** [SME_VALIDATION_ITERATION_PROMPT.md](SME_VALIDATION_ITERATION_PROMPT.md) — weighted ≥95% criteria, SME confusion rubric, and mandatory report rotation before regeneration.

Paste this into a new agent session when you will not be available. **Do not ask the user questions**; operate until blocked or caps are hit.

## North star

1. **Best match** for `(company name, optional city/state)` on the plugging corpus.
2. **Explainability**: every top pick should be narratable to a non-technical SME (“same legal name”, “same state, closer city”, “no address on file — we prefer the row that has a verifiable office”, etc.).
3. **Sanity**: an **exact same legal name** with **no** candidate geography must **not** rank **above or tie ahead of** an exact same name that **does** have city/state when the query omitted location — **completeness and transparency beat anonymous index rows**.

## Already shipped (read before changing)

- `CompanyMatcher._rank_and_user_facing_score`: **100% gate** = exact name **and** full **query+candidate** city+state match only (no more “both sides blank” bypass that inflated bare rows).
- **No query geo + exact name:** user-facing score is **capped at ~94%** (`EXACT_WHEN_QUERY_HAS_NO_GEO_CAP`) so one office line is not shown near **99%** when the same legal name may exist in multiple places; tie-break still prefers rows with addresses over bare rows.
- **Frequency**: when the query has **no** geo and the candidate is **exact** but **location-less**, frequency boost is **zero** (parity with “don’t juice mystery rows”).
- **Sort tie-break** (after score, `location_score` when applicable): **`_candidate_has_geo`** then exact then concept — so offices with addresses sort before same-score bare rows.

Files: [`src/finetuner/core/matcher.py`](../src/finetuner/core/matcher.py), [`src/finetuner/web/services/rationale_service.py`](../src/finetuner/web/services/rationale_service.py).

## Hard constraints (do not violate without explicit human decision)

- Do **not** change **0.5 / 0.25 / 0.25** `name_score` blend or **0.8 / 0.2** hybrid location blend.
- Do **not** change [`tier_config.json`](../tier_config.json) thresholds in autonomous runs.
- Do **not** bump `CACHE_VERSION` unless index-time behavior changes.
- After matcher/rationale scoring changes: run **`python tests/verify_control_set.py`** when RPC/cache is up; refresh rationales if you change explanation text.

## Validation loop

1. **Spot-check** high-stakes patterns in [`plugging_report.md`](../plugging_report.md) or UI: exact-name multi-site, query **no** geo, acronym junk, wrong-state (Groups360 / SANCC class).
2. **Regenerate** plugging artifacts when RPC matches project (see [`run_pipeline.py`](../run_pipeline.py) / `match_plugging_records.py` per your env).
3. **Audit**: [`audit_scoring.py`](../audit_scoring.py) — gates A–F; treat failures as a ordered backlog (Sanity → E → D → B → C → A → F).
4. **SME checklist** (mental): “If I only saw the company name on the spreadsheet, would this top pick still feel fair?”

## Backlog ideas (only if gates + SMEs still complain)

- **Gate B / Hamilton class**: `string_score` 1.0 on shared token but different entity tail — tighten `calculate_string_similarity` for entity-type word mismatch (`corporation` vs `enterprises`) **or** narrow the audit gate.
- **Gate A / ties**: micro-jitter from `concept_alignment` in the **0.85–0.90** band (see scoring audit plan) to break false ties without touching declared weights.
- **Rationale copy rename**: “Name Similarity” line is **`string_score`** — consider labeling “Lexical / string similarity” to match [`docs` plan](../.cursor/plans) if SMEs misread it.

## Environment (PowerShell)

```powershell
Set-Location E:\projects\FineTuner\FineTuner
$env:PYTHONPATH = "E:\projects\FineTuner\FineTuner\src"
# Terminal 1: cache RPC (after matcher edits)
python -m finetuner.core.cache_rpc --serve
# Then load cache key your project uses, e.g.:
python -m finetuner.core.cache_rpc --load <key>_loc
# Terminal 2: batch + report + optional audit
python match_plugging_records.py --no-resume --checkpoint-every 200
python tests/generate_plugging_report.py
python audit_scoring.py
```

## Stop condition

Append a short summary to [`pipeline_run.log`](../pipeline_run.log) (or new dated note): what changed, control-set / audit outcome, and **one paragraph** an SME can read before a review meeting.
