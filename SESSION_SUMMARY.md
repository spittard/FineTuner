# FineTuner Session Summary
**Date:** 2026-02-22  
**Status:** Active — Full pipeline rebuild in progress

---

## What We Built

This session delivered a complete SME-grade plugging records matching workflow, resolved
systemic matching failures, and produced a comprehensive architectural blueprint for
ongoing improvement.

---

## Deliverables

### 1. `/plugging` Web Application
A full Flask route and HTML template for reviewing plugging records and their matches.

**Matches the SME report exactly:**
- Summary table with Match %, tier badge, company name, location, record count
- Expandable row panel renders full HTML rationale (verdict, evidence bullets,
  concept nutritional label, score breakdown formula)
- Tier thresholds adjustable from the UI (High / Medium / Low)
- Tier changes re-classify all loaded records without a page reload
- Uses the Pyro5 RPC cache server for all match lookups

**Files:**
- `src/finetuner/web/app.py` — added `/plugging`, `/api/tier-config` (GET + POST) routes
- `src/finetuner/web/templates/plugging.html` — full new template

---

### 2. Tier Configuration
User-adjustable tier thresholds persisted to disk.

- `tier_config.json` — JSON file: `{"high": 99, "medium": 70}` (current settings)
- Default if file absent: `{"high": 85, "medium": 70}`
- API: `GET /api/tier-config` returns current values; `POST /api/tier-config` saves new ones
- UI inputs + "Apply" button recalculate tier badges instantly in the browser

---

### 3. Model Swappability
Active embedding model is configurable without code changes.

- `model_config.json` — JSON file at project root
- `active_model` key is read by both `CompanyMatcher.__init__()` and `CacheServerRPC.load_cache()`
- Changing the model name + running the pipeline automatically builds a new cache
- Old caches are preserved on disk (cache key includes model name and version)

**Current model:** `all-MiniLM-L6-v2` (active, locally cached)

---

### 4. Model Comparison Test Suite
Canonical 46-case, 11-category test for evaluating any model or scoring change.

- `test_model_comparison.py` — baseline tests for Exact, Partial, Suffix, Acronym,
  Directional, Person, Noise, Semantic, Structural, Numeric, Geographic categories
- Baseline: `paraphrase-MiniLM-L3-v2` = 32/46 (70%)
- Current: `all-MiniLM-L6-v2` = 36/46 (78%)
- **Rule:** Any change to scoring logic must pass ≥ 36/46 with no category regression

---

### 5. Showcases and Reports
- `serge_nation_showcase.html` — rendered HTML version of the showcase MD (Edge-compatible)
- `MATCHING_ARCHITECTURE.md` — complete 6-signal architecture reference (see below)
- `SESSION_SUMMARY.md` — this file

---

## Architecture Reference (`MATCHING_ARCHITECTURE.md`)

The definitive technical reference for the matching system. Covers:

- **6-signal scoring formula** with exact weights and interaction rules
- **8 root causes** of matching failure (RC-1 through RC-8)
- **Per-signal failure analysis** — what breaks, why, and what the rationale displays
- **Compound failure trace** — the NWACUHO/Fairbanks case traced through all 6 signals
- **Unused RationaleService capabilities** — `analyze_geographic_context()` and others
  already built but not wired into scoring
- **Complete fix-to-failure mapping** table
- **Implementation priorities 0–5** with code snippets and affected files
- **Model selection reference** (L3 / L6 / MPNet comparison)
- **Mandatory evaluation protocol** (must pass before any change is merged)

---

## Files Changed This Session

| File | Action | Summary |
|------|--------|---------|
| `src/finetuner/web/app.py` | Modified | Fixed `_PROJECT_ROOT`, added tier config API, added `/plugging` route |
| `src/finetuner/web/templates/plugging.html` | Created | Full plugging records UI mimicking SME report |
| `src/finetuner/core/matcher.py` | Modified | Active model loading from `model_config.json`; **Priority 0** location formula fix; **Priority 1** concept anchor redesign + CACHE_VERSION bump |
| `src/finetuner/core/cache_rpc.py` | Modified | Active model loading from `model_config.json` |
| `tier_config.json` | Created | `{"high": 99, "medium": 70}` |
| `model_config.json` | Created | Active model + available models list |
| `test_model_comparison.py` | Created | 46-case canonical test suite |
| `serge_nation_showcase.html` | Created | Edge-compatible rendered showcase |
| `MATCHING_ARCHITECTURE.md` | Created | Full 6-signal architecture document |
| `SESSION_SUMMARY.md` | Created | This file |

---

## Current System State

| Item | Value |
|------|-------|
| Active model | `all-MiniLM-L6-v2` |
| CACHE_VERSION | `v5.0_regional_anchors` (bumped this session) |
| Tier config | High ≥ 99%, Medium ≥ 70% |
| DB server | `TLG-DATA3\TLG_DEV` |
| DB / table | `SQLWebRefTable` / `AcctRef.Master` |
| Reference JSON | `companies_with_location.json` |
| Plugging JSON | `plugging_records.json` |
| Match results | `plugging_matches.json` |
| Cache dir | `company_matcher_cache/` |
| Pipeline script | `run_plugging_pipeline.py` |
| Test suite | `test_model_comparison.py` |

---

## Code Changes Applied This Session

### Priority 0 — Location Formula (matcher.py)

**Before (penalizes missing data):**
```python
if use_location:
    if is_this_exact:
        final_score = name_score + (location_score * 0.05)
    else:
        final_score = (name_score * 0.8) + (location_score * 0.2)  # 20% penalty when 0
```

**After (boost-only, neutral when absent):**
```python
if use_location:
    if is_this_exact:
        loc_boost_val = location_score * 0.05 if location_score > 0 else 0.0
    else:
        loc_boost_val = location_score * 0.10 if location_score > 0 else 0.0
    final_score = name_score + loc_boost_val
else:
    final_score = name_score
    loc_boost_val = 0.0
```

**Impact:** ~80–90% of queries. Removes 20% score reduction on every non-exact match
where location data is absent or mismatched.

---

### Priority 1 — Concept Anchors (matcher.py)

**Before (city-biased):**
```python
CACHE_VERSION = "v4.1_location_decoupled"
CONCEPT_ANCHORS = {
    "Geography": ["Pennsylvania", "London", "Canada", "California", "New York",
                  "Texas", "Chicago", "Illinois", "Ohio", "Miami", "Paris"],
    ...
}
```

**After (regional, unbiased):**
```python
CACHE_VERSION = "v5.0_regional_anchors"
CONCEPT_ANCHORS = {
    "Geography": ["Pacific Northwest", "Pacific Southwest", "Mountain West",
                  "Great Plains", "Midwest", "Great Lakes", "Mid-Atlantic",
                  "New England", "Southeast", "Gulf Coast", "Southwest",
                  "Alaska", "Hawaii", "Canada", "International"],
    "Industry":  ["Healthcare", "Technology", "Finance", "Energy", "Hospitality",
                  "Nonprofit", "Government", "Academic", "Pharmaceutical",
                  "Agriculture", "Manufacturing", "Legal", "Insurance", "Retail"],
    ...
}
```

**Impact:** ~30–40% of queries. Removes Ohio/California geographic bias from 25% of
name_score for all companies in underrepresented US locations.

**Requires full FAISS cache rebuild** (CACHE_VERSION bump forces this automatically).

---

## Matching Failure Root Causes (Summary)

| ID | Signal | Frequency | Fix Applied |
|----|--------|-----------|-------------|
| RC-1 | Semantic (directional dilution) | Low | Cross-encoder (Priority 4, pending model download) |
| RC-2 | Lexical (query noise) | Medium | Query preprocessor (Priority 3, future) |
| RC-3 | Semantic (bi-encoder discrimination) | Low | Cross-encoder (Priority 4) |
| RC-4 | All (alias/rebrand knowledge gap) | Low | Accept — no text similarity fix |
| RC-5 | Lexical (substring scoring) | Low | Future |
| **RC-6** | **Location (penalty for missing data)** | **High (80–90%)** | **FIXED this session** |
| **RC-7** | **Concept (geographic anchor bias)** | **Medium (30–40%)** | **FIXED this session** |
| RC-8 | Location (geographic text in name ignored) | Medium | Future (Priority 2) |

---

## Full Rebuild — Current Status

The full pipeline (`run_plugging_pipeline.py`) was launched after code changes.

### Pipeline Steps
1. Stop any existing RPC server
2. Extract all reference companies from DB (excludes `PluggingStatus = 'P'`)
3. Start RPC server + build FAISS index (`build_index_with_location()`)
4. Extract all plugging records
5. Batch-match plugging records against reference index (resumable)
6. Generate `plugging_report.md` + `plugging_report.csv`
7. Stop RPC server

### Row ID Handling
- SQL query uses `MIN([Row]) as RowID` per `(Company, City, State)` group
- Row IDs stored in `company_ids[]` in the FAISS cache
- Each match result includes `"id": <row_id>` field
- Webapp can use this to link directly back to the database record

### Output Files
| File | Contents |
|------|---------|
| `companies_with_location.json` | All reference companies with city, state, row ID, count |
| `plugging_records.json` | All plugging records (one entry per source row) |
| `plugging_matches.json` | Match results for every plugging record |
| `plugging_report.md` | Human-readable SME report |
| `plugging_report.csv` | Machine-readable match data |
| `company_matcher_cache/` | FAISS index files for cache server |

---

## Pending Work (Not Yet Implemented)

| Priority | Description | Blocking On |
|----------|-------------|-------------|
| 2 | Geographic text extraction from query names (RC-8) | None |
| 3 | Query noise preprocessor (RC-2) | None |
| 4 | Cross-encoder re-ranking (RC-1, RC-3) | Manual model download |
| 5 | MPNet model switch | Manual model download + Priority 4 first |

### To Switch to MPNet When Downloaded
```
# 1. Download all-mpnet-base-v2 manually from HuggingFace
# 2. Place in local model cache
# 3. Edit model_config.json:
#    "active_model": "all-mpnet-base-v2"
# 4. Run: python run_plugging_pipeline.py
#    (CACHE_VERSION bump from this session means rebuild happens automatically)
```

---

## Key Commands

```powershell
# Full pipeline (extract + build + match + report)
cd E:\projects\FineTuner\FineTuner
python run_plugging_pipeline.py

# Skip extract (reuse existing JSON), rebuild index + match + report
python run_plugging_pipeline.py --skip-extract

# Sample run (10 plugging records for quick testing)
python run_plugging_pipeline.py --sample 10

# Regenerate report only (from existing plugging_matches.json)
python run_plugging_pipeline.py --report-only

# Run test suite
python test_model_comparison.py

# Launch webapp (cache server must be running)
python run_cache_server.py          # terminal 1
python run_app.py                   # terminal 2
# Visit: http://localhost:5000/plugging
```
