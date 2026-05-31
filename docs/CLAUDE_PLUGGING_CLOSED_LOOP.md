# Claude (any agent): plugging closed loop — assess every row, capture issues, manufacture worst cases, test, refactor, full regen

**Persist this file.** Paste its workflow into a task when touching plugging quality. Do not tell the human to run the long jobs; **you** run them (or block on RPC with a clear reason).

---

## What “assess every result in the full plugging report” means here

1. **Every** plugging record = every `row_id` in `plugging_matches.json` (same cardinality as the generated `plugging_report.md` / `.csv`).
2. You **must** run the automated assessor so **each** row gets a machine row: issue codes + narrative + severity → **`plugging_report_assessment.csv`** (and the summary `.md`).
3. That is **not** the same as a human reading 3k screenshots. Say honestly: **full automated coverage** + **stratified deep dives** + **regression tests on manufactured egregious cases**.

---

## Non‑negotiables

| Rule | Detail |
|------|--------|
| Closed loop | You assess → capture to files → fix code/data → **test egregious set first** → only then rotate + full rematch + full report + audit + control set. |
| No homework dump | Never end with “the user should run rematch.” |
| SME‑puke bar | Wrong geo order with no query geo, score lying about exactness, tie mush, `UT`/`UT`, hybrid at ceiling with no story — **fix or waive in writing** with evidence. |
| Proof | Chat reply lists **paths**, **exit codes**, **counts** from CSV/summary — not vibes. |

---

## Phase A — Full corpus assessment (every row) + capture to disk

**Inputs:** `plugging_matches.json` (and optionally prior rotation).

**You run:**

```powershell
Set-Location <REPO_ROOT>
python scripts/assess_plugging_report_full.py
```

**Outputs (must exist after this phase):**

- `plugging_report_assessment.csv` — **one line per plugging row**; columns include `issue_codes`, `narrative`, `severity`.
- `plugging_report_assessment_summary.md` — counts by flag + rubric.

**You also:** read the summary; in chat, report **total rows**, **`high` / `med` / `info` counts**, **top issue codes by frequency**, and **one paragraph** on the dominant failure themes.

---

## Phase B — **Intermediate: manufacture the most egregious fuckups and test before full regeneration**

**Goal:** Do not burn an hour on a full rematch until the worst failures are **reproduced in a small, fast loop** (pytest or a smoke script over RPC).

**Steps:**

1. **Export** a bounded regression bundle from the assessment + source records (query/city/state per `row_id`):

   ```powershell
   python scripts/export_plugging_egregious_cases.py --max-cases 120
   ```

   **Writes:** `tests/fixtures/plugging_egregious_cases.json`  
   (Prioritize `severity=high`, then codes like `NO_QUERY_GEO_BARE_ROW_NOT_FIRST`, `GATE_A_TOP5_TIE_CLUSTER`, `HIGH_SCORE_NON_EXACT_NAME`, `SUSPECT_GEO_*`, Gate B row_ids from `audit_scoring.py` if documented.)

2. **Add or tighten tests** that hit those cases (RPC required — skip in CI unless env flag set):

   - Example env gate: `RUN_PLUGGING_REGRESSION=1` and `finetuner.core.cache_rpc.is_server_running()`.
   - For each case: `search(company, city, state, top_k=5)` and assert properties implied by `issue_codes` (e.g. if query has no geo and case includes `NO_QUERY_GEO_BARE_ROW_NOT_FIRST`, after your matcher fix the **first** rank at equal score band should be **bare**, or top‑1 name should match documented expectation for that `row_id`).

3. **Run only** those tests until green:

   ```powershell
   $env:RUN_PLUGGING_REGRESSION=1
   python -m pytest tests/test_plugging_egregious_regression.py -q --tb=short
   ```

4. **Only when** egregious regression tests pass (or you’ve shrunk the failure set to known waivers with comments), proceed to Phase C.

**Why this step exists:** Full `match_plugging_records.py --no-resume` is expensive. Manufacturing egregious cases turns “fix matcher” into a **tight feedback loop** so you don’t ship another broken full JSON.

---

## Phase C — Rotate artifacts, full rematch, full report, audit, control set

**Rotate** (timestamped copies; do not delete):

- `plugging_matches.json`
- `plugging_report.md`, `plugging_report.csv` (if present)

**Full recompute:**

```powershell
python match_plugging_records.py --no-resume
python tests/generate_plugging_report.py
python scripts/assess_plugging_report_full.py
python audit_scoring.py
python tests/verify_control_set.py
```

**Success:** `audit_scoring.py` exit **0**, `verify_control_set.py` exit **0**, assessment summary shows **reduced** `high` / critical codes vs prior rotation (paste before/after counts in chat).

---

## Refactor discipline

- Smallest patch that fixes a **bucket** of assessment codes or a **regression case**.
- Matcher changes: update `rationale_service.py` if user-visible scoring story changes; `CACHE_VERSION` only when index/cache contract changes (see `CLAUDE.md`).
- After scoring/ranking change: always **`verify_control_set.py`**.

---

## “Done” in chat must include

1. Paths to rotated files + new `plugging_matches.json` / reports / assessment CSV.  
2. Egregious bundle path + test command + pass/fail.  
3. Audit + control set results.  
4. Honest gap list (what still needs human SME on flagged rows).

---

## Related files

| File | Role |
|------|------|
| `scripts/assess_plugging_report_full.py` | Per-row machine assessment → CSV |
| `scripts/export_plugging_egregious_cases.py` | Phase B: build `tests/fixtures/plugging_egregious_cases.json` |
| `tests/test_plugging_egregious_regression.py` | Phase B: RPC regression (optional env) |
| `docs/PLUGGING_ASSESSMENT_FRAMEWORK.md` | Rubric detail |
| `docs/AGENT_PROMPT_PLUGGING_FULL_LOOP.md` | Shorter loop (superseded for ordering by this doc) |
