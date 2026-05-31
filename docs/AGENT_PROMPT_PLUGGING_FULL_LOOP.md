# Agent prompt: full plugging assessment → fix → rotate → recompute (closed loop)

**Canonical / extended version (includes Phase B: manufacture egregious cases + test before full rematch):**  
→ **`docs/CLAUDE_PLUGGING_CLOSED_LOOP.md`** — use that for Claude or any agent; persist it in rules if needed.

This file is the **short** loop. **Do not** end with “you should run X next.” **You** run the pipeline, read outputs, patch code, and repeat until gates pass or you document a hard blocker.

## Non‑negotiables

1. **Closed loop**: assess → triage → minimal code/data fix → restart RPC if matcher changed → full rematch → regenerate report → re‑audit → repeat. No handoff homework to the user.
2. **Honesty**: scanning JSON/heuristics is **not** “SME reviewed all rows.” Say “automated assessment” vs “human sign‑off.”
3. **SME‑puke bar**: if a row would make a careful SME lose trust (score/label lies about exactness, wrong geo order when query has no geo, `UT`/`UT` style junk, top‑5 tie mush, hybrid at ceiling with no rationale), it **must** be flagged and either fixed or explicitly waived with evidence.

## Preconditions

- Repo root: FineTuner project with `plugging_records.json`, `src/finetuner/`.
- RPC cache server running with the **same** cache key the plugging pipeline uses; after **any** `matcher.py` (or scoring) change: **restart RPC** and reload cache before rematch.
- Python env has project deps (`requirements.txt`).

## Artifacts to rotate (before destructive regen)

Timestamp suffix: `YYYYMMDD_HHMMSS` or use `run_pipeline.py` / a one-liner to copy:

- `plugging_matches.json` → `archive/plugging_matches_<ts>.json` (or project `archive_*` convention)
- `plugging_report.md`, `plugging_report.csv` → same pattern if they exist

Do **not** delete the rotated copies.

## Full recompute (you run these)

```powershell
Set-Location <REPO_ROOT>
# 1) Ensure RPC is up and matcher code is loaded (restart server if matcher changed).

# 2) Full rematch (no resume — fresh JSON)
python match_plugging_records.py --no-resume

# 3) SME markdown + CSV from matches
python tests/generate_plugging_report.py

# 4) Automated per‑record assessment (narrative + issue codes)
python scripts/assess_plugging_report_full.py

# 5) Audit gates on matches JSON
python audit_scoring.py

# 6) After matcher scoring changes — always
python tests/verify_control_set.py
```

Exit codes must be **0** for `verify_control_set.py` and `audit_scoring.py` before you claim “green.”

## Assess **each** of the ~3k matching records

“Each” means **machine coverage + explicit rubric**, not pretending you read 3k screenshots:

1. **Run** `python scripts/assess_plugging_report_full.py` so **every** `row_id` in `plugging_matches.json` gets a CSV row: `severity`, `issue_codes`, `narrative`.
2. **Sort** `plugging_report_assessment.csv` by `severity` (high → med → info), then by `issue_codes`.
3. **Summarize** in your reply: counts by code, top 10 recurring narratives, and **how many** `high` rows (not “I glanced at a few”).
4. **Spot‑validate** a **stratified** sample (e.g. 20 rows): all `HIGH_SCORE_NON_EXACT_NAME`, random 10 from `NO_QUERY_GEO_BARE_ROW_NOT_FIRST`, all `GATE_A_TOP5_TIE_CLUSTER` if count ≤ 30 else random 15 — confirm the narrative matches the JSON; if the script misses a pattern, **extend** `scripts/assess_plugging_report_full.py` and re‑run.

## SME‑puke scenarios (must fix or waive in writing)

| Symptom | Likely track |
|--------|----------------|
| Query **no** geo but bare HQ row ranks **below** geo office at same/near score | `matcher.py` tie‑break / location blend when `use_location` is false |
| Top‑5 scores within **0.001** (Gate A) | `matcher.py` spread / scoring differentiation |
| Top‑1 ≥ 0.95 on known false twin (e.g. token reorder / corp vs enterprises) | `matcher.py` guards |
| Near‑100% UI but name not exact / hybrid semantic‑led with no explanation | `rationale_service.py` + tier/display rules in report generator |
| `city`/`state` garbage (duplicate state, city==state token) | **data** fix or import; if matcher can’t fix source, file `DATA_ISSUE` in assessment CSV narrative |

## Refactor rules

- **Minimal diffs**: one logical fix per iteration; no drive‑by refactors.
- **Matcher**: follow `CLAUDE.md` — update `rationale_service.py` when score components change; bump `CACHE_VERSION` only when index/cache semantics change (not pure post‑score tie tweaks).
- **Verify**: `python tests/verify_control_set.py` after any scoring/ranking change.

## Loop until done

1. Run pipeline above.
2. If `audit_scoring.py` or SME‑critical codes fail: pick the **largest** failure bucket, implement the **smallest** fix, rotate if needed, **recompute everything**, update `plugging_report_assessment_summary.md` (by re‑running the script).
3. Stop only when: audit + control set green, **or** you hit an external blocker (e.g. RPC down, missing `plugging_records.json`) — state the blocker and what **you** already tried.

## What you deliver in chat

- Rotated paths (if any).
- Command results (pass/fail, key numbers).
- Patch summary (files + why).
- Assessment summary from **full** CSV (counts + sample deep dives).
- No false claims of line‑by‑line human review of 3k rows.
