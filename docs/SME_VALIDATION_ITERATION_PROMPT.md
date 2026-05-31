# SME validation iteration prompt (agent handoff)

Paste this entire block (from **ROLE** through **STOP**) into a new agent session for autonomous work. **Do not wait on the user.** Read supporting docs first: [`OVERNIGHT_PLUGGING_PROMPT.md`](OVERNIGHT_PLUGGING_PROMPT.md), [`audit_scoring.py`](../audit_scoring.py) (gate definitions), and plan notes under `.cursor/plans/` (e.g. scoring audit, name similarity).

---

## ROLE

You are improving **company matching for the plugging corpus** so results are **defensible to non-technical SMEs**. You iterate until **weighted validation ≥ 95%** (see scoring below) or you hit **max full cycles** / a **hard blocker** (unsafe change, ambiguous requirement).

---

## NORTH STAR

1. **Best match** for `(company name, optional city/state)`.
2. **Explainability**: the top match and score must be narratable without jargon (“same legal name and same state”, “exact name but we prefer the row that lists an office”, etc.).
3. **SME sniff test** (mandatory each cycle): *If I showed only the plugging row and the #1 match line, would a reasonable SME call it nonsense?* If **yes**, record **row_id**, **failure mode**, and treat as **P0** for the next fix.

### Calibration: WPBF Media (`row_id` **7919849**)

Use this row to sanity-check **no-location** queries before filing a bug:

- **Query:** `WPBF Media` (no city/state).
- **#1 match** is **exact** legal name **WPBF Media** with candidate metadata **Memphis, TN** — this is **correct** ranking.
- **Ranks 2–5** are **different** companies (`W Media` / `W media`, acronym-style noise at ~81%), **not** a second “exact WPBF” row wrongly beating #1.
- **~94% vs 100%:** With no geography on the query, an **exact** legal-name match is **capped around 94%** so one office row is not shown at ~99% when the same name may exist elsewhere. **100%** is reserved for exact name **and** verified matching city/state on **both** sides (`exact_geo_full_match`). Among tied exacts, rows **with** an address still sort before bare rows.

---

## ROTATE BEFORE RECREATE (mandatory)

Before overwriting SME artifacts:

- **`plugging_matches.json`**, **`plugging_report.md`**, **`plugging_report.csv`**

you MUST **rotate backups** (timestamped `*.bak_<utc>`) OR run **`python run_pipeline.py`**, which rotates and prunes per `--keep` (default 3).  

**Do not** manually delete the newest retained `*.bak_*` files. If project policy says to use only `run_pipeline.py` for batch match + audit, follow that policy.

---

## WEIGHTED VALIDATION (95% rule)

Compute **one score in [0, 1]** each iteration using the table below. **Done when weighted sum of passing checks ≥ 0.95.**

| ID | Check | Weight | Pass condition |
|----|--------|--------|----------------|
| G0 | Score sanity (`audit_scoring`) | 5% | PASS (no NaN, in range, sorted top-K) |
| G1 | Gate A — tie clusters | 5% | PASS |
| G2 | Gate B — token reorder rows | 5% | PASS |
| G3 | Gate C — acronym junk | 5% | PASS |
| G4 | Gate D — wrong-state rows | 5% | PASS |
| G5 | Gate E — dupes | 5% | PASS |
| G6 | Gate F — 0.880 cliff | 5% | PASS |
| CS | Control set | 20% | `tests/verify_control_set.py` exit 0 (0 assertion failures). If policy allows partial credit: **(111 − n_fail) / 111 × 20%** capped at 20% |
| S1 | SME: exact + query no geo | 5% | Spot-check: same legal name; row **with** city/state not ranked **after** bare row when scores tie / story makes sense |
| S2 | SME: wrong-state class | 5% | Spot-check Groups360 / SANCC-style: top-1 state matches query intent where gates require it |
| S3 | SME: token–tail ambiguity | 5% | e.g. “Corporation of Hamilton” vs “Hamilton Enterprises” — top-1 not absurdly “EXCELLENT” without caveat OR gate/doc accepts rule |
| S4 | SME: acronym junk | 5% | Short query / acronym path: top-1 not arbitrary index noise without low-confidence signal |
| S5 | SME: tie cluster readability | 5% | Sample rows with all top-5 within 0.001: SME can see *any* ordering rationale (report or rationale text) |
| R1 | Rationale / labels | 5% | “Name Similarity” ambiguity: prefer honest labeling (**lexical / string** vs full `name_score`) or doc note in UI |
| R2 | Tier vs score | 5% | Tier bands in report do not **blatantly** contradict numeric score story for sampled rows |
| P1 | RPC + code parity | 5% | After `matcher.py` / `text_preprocessor.py` changes, RPC restarted; matching process uses new code |
| P2 | Full corpus size | 5% | `len(plugging_matches.json)` matches `plugging_records.json` (e.g. 3217); no accidental `--limit` run |

**Scoring:** For each row, if pass → add its weight; **total = sum of weights passed / 1.0** (weights sum to 100%). **Target: total ≥ 0.95.**

Adjust **S*** spot-check rows to match current `audit_scoring.py` row IDs and project samples; keep at least **5** distinct SME checks.

---

## FAILURE MODES (log when SME says “nonsense”)

Use one tag per issue: **wrong_entity**, **wrong_geo**, **false_twin**, **acronym_junk**, **tie_cluster**, **label_mismatch**, **freq_inversion**, **other**.

---

## HARD CONSTRAINTS (unless user explicitly overrides)

- Do **not** change **0.5 / 0.25 / 0.25** (`name_score`) or **0.8 / 0.2** (hybrid location blend).
- Do **not** change **`tier_config.json`** thresholds in unattended runs.
- Do **not** bump **`CACHE_VERSION`** unless index-time / embedding behavior changes.
- Prefer **one focused change** per iteration; re-run validation after RPC restart when needed.

---

## ITERATION LOOP

1. Read this prompt + [`OVERNIGHT_PLUGGING_PROMPT.md`](OVERNIGHT_PLUGGING_PROMPT.md).
2. **Rotate** artifacts or run **`python run_pipeline.py`** (full cycle as appropriate).
3. After matcher changes: **restart** `cache_rpc`, **load** cache; then rematch / regen reports.
4. Run **`audit_scoring.py`** (or pipeline step that runs it) and **`python tests/verify_control_set.py`**.
5. Open **`plugging_report.md`** (and/or UI) — run **SME spot-checks** S1–S5; check R1–R2 on samples.
6. Compute **weighted score**. If **≥ 0.95** → **STOP** (write summary). Else pick **highest-weight failing** item (prefer **G4/D**, **CS**, then SME **wrong_geo** / **wrong_entity**), implement one fix, goto 2.
7. **Max full cycles:** e.g. **5** — if still &lt; 0.95, **STOP** with remaining failures table and recommended human actions.

---

## SEED: KNOWN UNSOLVED / VERIFY (as of last manual pass)

Re-validate after every code change:

- **Audit:** Gates **A**, **B**, **F** historically flaky; confirm current `plugging_matches.json`.
- **Control set:** **CU Cooperative (Durham, CT)** → top match must be **CT**, not **CA** (`companies_control_set_results.md` once showed failure).
- **UX:** “Name Similarity” in rationales reflects **`string_score`**, not blended **`name_score`** — see name_similarity plan in `.cursor/plans/`.

---

## STOP (deliverable)

Append to **`pipeline_run.log`** or a dated **`docs/SME_VALIDATION_RUN_<YYYYMMDD>.md`**:

- Weighted percentage **before / after**
- List **failing check IDs** and weights left on table
- **Files changed**
- **One short paragraph** an SME can read before a review meeting

---

## ENVIRONMENT (PowerShell)

```powershell
Set-Location E:\projects\FineTuner\FineTuner
$env:PYTHONPATH = "E:\projects\FineTuner\FineTuner\src"
```

Then RPC serve/load, `run_pipeline.py` or discrete match + `tests/generate_plugging_report.py`, audit, `verify_control_set.py` per project norms.
