# Accuracy review — FineTuner matching (3-pass source + output)

Goal: can match **accuracy** improve? This is an evidence-backed assessment of the scoring pipeline (`matcher.py`, `text_preprocessor.py`) and outputs (`plugging_report.*`, `companies_control_set_results.md`). Each item lists: **what**, **evidence**, **expected accuracy effect**, **risk**, and whether it needs a **rematch + control-set verify**.

Two kinds of "accuracy":
- **Ranking accuracy** — is the right record top-1 / in top-5?
- **Calibration** — does the score *mean* what it says (so SMEs can trust the % and tiers)?

Current baseline: `verify_control_set.py` last run — *"All location assertions passed (where specified)"* over 111 control entries; exact-name top-1 = 100%. The weaknesses below mostly hurt **rank 2–5 ordering** and **calibration**, which is where SME trust breaks.

---

## P0 — High value, low risk

### 1. Hard lexical floors flatten the top-5 (tie mush) — biggest ranking issue
- **Where:** `matcher.py` Phase-2 — `if string_score >= 0.92 or is_full_overlap: name_score = 0.95` / `elif string_score >= 0.80: name_score = 0.90`.
- **Evidence:** Prestige Global Meeting Source — 3 distinct candidates all land at name_score **0.9500** → 96.0% ties. Hamilton — 5 rows at **0.999**. Audit: 1,937 records with duplicate rounded top-5 scores; 324 within 0.001.
- **Effect:** the floor **erases** the ordering signal exactly among near-matches, so the SME can't tell rank 2 from rank 4. De-flattening restores correct intra-cluster order.
- **Proposal:** keep the *visibility* floor but preserve order inside it: `name_score = floor + (1 - floor) * raw_blend` (monotonic) instead of a flat clamp; or add a small string/semantic-derived epsilon before the round. (The `_ensure_top5_score_spread` helper is a band-aid downstream; fixing the floor is the root cause.)
- **Risk:** low — doesn't change which item is #1 for exact matches; only differentiates ties. Needs rematch + control verify to confirm no regressions.

### 2. "Semantic 100%" everywhere — relative normalization
- **Where:** `matcher.py` `sem_score_norm = original_semantic_score / max_sem_score`.
- **Evidence:** rank-1's semantic is **always** 1.0 (it is the max); rationale shows "Semantic Link: EXCELLENT (100%)" on nearly every row. Note stored `semantic_score` ≈ 6.0 while the index is `IndexFlatIP` (cosine ≤ 1 expected) — embeddings likely **not unit-normalized at add time**, so the raw number is not a usable absolute cosine and the code masks it by dividing by the max.
- **Effect:** semantic is non-comparable across queries and inflates weak top hits; calibration suffers and the 25% semantic weight is partially meaningless.
- **Proposal:** verify embeddings are L2-normalized before `index.add` (so IP = true cosine); then use **absolute** cosine for `sem_score_norm` (clamp [0,1]). Recalibrate tiers afterward.
- **Risk:** medium — shifts all scores; **must** rematch + re-verify control set + re-tune `tier_config.json`.

---

## P1 — High value, needs A/B on control set

### 3. String similarity has no Jaro-Winkler / token-set ratio
- **Where:** `text_preprocessor.calculate_string_similarity` = weighted Jaccard ⊕ `difflib.SequenceMatcher`. No edit-distance, no token_set/token_sort.
- **Evidence:** reordering is handled by a special-case guard; abbreviation/typo handling leans on coverage heuristics + magic caps (0.50/0.65 short-string caps; `# BIAS FIX` length penalty).
- **Proposal:** add `rapidfuzz` (`token_set_ratio`, `token_sort_ratio`, `WRatio`, `JaroWinkler`) and blend with the existing score (take max or weighted). Fast (C++), well-tested. Likely lifts top-1 on messy/reordered/abbreviated names.
- **Risk:** medium — new dependency + rematch + control verify.

### 4. `clean_company_name` discards the distinguishing entity-type word
- **Where:** suffix set strips `group, holdings, enterprises, associates, corporation, company, …`.
- **Evidence:** this is *why* "Corporation of Hamilton" ↔ "Hamilton Corporation" and "{X}" ↔ "{X} Enterprises" collide at ~99% — the token that separates a government corporation from a private enterprise is deleted before comparison.
- **Proposal:** don't fully strip entity-type words; keep them as **low-weight** tokens (already supported via `GENERIC_TERMS` weighting) so they still nudge ranking and break false twins, without dominating. Alternatively add an "entity-type mismatch" penalty.
- **Risk:** medium — affects many rows; rematch + verify.

### 5. Concept alignment (25%) rarely discriminates
- **Where:** name blend = `0.50*string + 0.25*semantic + 0.25*concept`.
- **Evidence:** concept ≈ 0.95–1.00 on almost every row ("Reflects highly aligned industries" is near-universal), including unrelated entities (city-gov vs private corp = 0.997). A signal that's high for everything adds little ranking power but consumes 25% of the name score and inflates scores.
- **Proposal:** A/B lower concept to 10–15% and move weight to string; measure top-1 on the control set. Keep concept for tie-breaking / display only if it doesn't help ranking.
- **Risk:** medium — rematch + verify; easy to revert (weights only).

---

## P2 — Calibration / data hygiene

### 6. State-only location = 1.0
- **Where:** `calculate_location_score` returns `state_score (=1.0)` when neither side has a city.
- **Effect:** "same state, no city" scores like "exact city+state," weakening the 0.88 cliff that separates verified-office from same-state-only.
- **Proposal:** cap state-only at ~0.6–0.7 so exact city+state stays distinctly higher.
- **Risk:** medium — shifts geo rows; verify control set (some assertions are state-level).

### 7. Malformed candidate geography in source data
- **Evidence:** ~84 rows with `UT, UT`, city token == state token, etc.
- **Proposal:** a data-cleaning pass on the master file; matcher can't reliably repair source errors.

### 8. Doc/code mismatches (no accuracy change, fix for correctness)
- `generate_acronym` docstring says "capitalized words only"; code uses all words.
- `calculate_location_score` docstring says "Levenshtein ≤ 2"; code uses `SequenceMatcher ≥ 0.8`.
- 2-letter acronyms are intentionally ignored in Phase 0 (e.g. `GE`) — acceptable but document the precision/recall trade-off.

---

## Recommended order (with measurement)

1. **#1 lexical-floor de-flattening** (root-cause for tie mush) → rematch → `audit_scoring.py` (Gate A should drop) → `verify_control_set.py`.
2. **#5 concept weight A/B** (weights-only, cheap to revert) → control-set top-1 delta.
3. **#3 rapidfuzz blend** + **#4 keep entity-type tokens** together → control-set top-1 delta + egregious regression set.
4. **#2 absolute cosine** (highest calibration value, highest blast radius) → full rematch + tier re-tune.

Every scoring change is gated by: **restart RPC → rematch → `audit_scoring.py` (exit 0) → `verify_control_set.py` (assertions pass) → `scripts/assess_plugging_report_full.py` (high-severity count must not rise).** Measure top-1 agreement against the 111-entry control set before/after each change; do not ship a change that lowers control-set top-1.
