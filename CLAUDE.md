# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent accountability (non-optional)

Paid work requires **honest scope** and **reproducible proof**. Violations waste client time (e.g. hour-long runs followed by overclaimed “SME review”).

### You may not claim without proof in this session

- Full **SME review** of every plugging / report row, “validated every result,” or “checked each match,” unless you **actually** did that here and can show **how** (e.g. per-`row_id` log + rubric), not heuristics-only.
- That a **script** “reviewed” the corpus in a human sense. Say: **which script**, **inputs**, **counts**, **flags** — and that rules **≠** ground truth.

### Before starting long jobs (RPC load, full rematch, full report gen)

1. State **what will and will not** be verified by the run.
2. If the user needs **human SME sign-off**, say so **up front**; offer **bounded** follow-up (sample strata, flagged rows only, Gate A list).
3. After the run: **deliverables (paths)** + **what was verified (how)** + **what was left unverified**.

### “Done” answer must include

1. Files/commands touched.  
2. Checks run (e.g. `verify_control_set`, audit exit code) with **result**.  
3. **Explicit gaps** (what a human or client still must do).

Heuristic triage (`scripts/sme_assess_plugging_matches.py`, `scripts/analyze_plugging_report_scenarios.py`) is **triage**, not proof of SME approval on all rows.

### Plugging quality: full closed loop (assess every row → file → fix → test worst → full regen)

For plugging / `plugging_matches.json` / SME report work, follow **`docs/CLAUDE_PLUGGING_CLOSED_LOOP.md`**: machine-assess **every** row to `plugging_report_assessment.csv`, **export** egregious cases (`scripts/export_plugging_egregious_cases.py`), **run** `tests/test_plugging_egregious_regression.py` with RPC before a full rematch, then rotate artifacts and recompute. Do not delegate long runs to the user without executing them when the environment allows.

### Cursor ↔ local Claude Code (same machine)

**File + CLI bridge:** write **`bridge/TO_CLAUDE.md`**, run **`scripts/bridge_run_claude.ps1`**, read **`bridge/FROM_CLAUDE.last.txt`**. Details: **`docs/CURSOR_CLAUDE_LOCAL_BRIDGE.md`** and **`bridge/README.md`**.

---

## Project Overview

FineTuner is a high-precision company name matching system for a 2.9M+ company dataset. It combines:
- **Semantic search** via FAISS with sentence-transformer embeddings
- **Hybrid scoring** (string similarity, acronym fidelity, location awareness)
- **Multiple interfaces**: CLI, Flask web app, and Pyro5 RPC server

## Common Commands

### Running the System

```powershell
# CLI search (interactive mode)
python company_search.py --data companies.json --interactive

# CLI with location filtering
python company_search.py "Bank of America" --city "Charlotte" --state "NC" --top 10

# Web application (requires RPC server running first)
python run_cache_server.py   # Terminal 1: Start RPC cache server
python run_app.py            # Terminal 2: Start Flask app at localhost:5000
```

### Testing & Verification

```powershell
# Verify changes against control set (ALWAYS run after scoring logic changes)
python tests/verify_control_set.py

# Generate detailed report after scoring changes
python tests/generate_ultra_report.py

# Debug a specific match interactively
python company_matcher_interactive.py companies.json
```

### Cache Management

```powershell
# Clear all cached embeddings and indexes (triggers ~45-60 min rebuild)
Remove-Item -Recurse -Force company_matcher_cache\*
```

## Architecture

### Core Package Structure (`src/finetuner/`)

- **`core/matcher.py`** - `CompanyMatcher` class: the main matching engine with three-phase matching (acronym expansion, FAISS retrieval, hybrid re-ranking). Contains `CACHE_VERSION` - increment when making breaking changes to indexing logic.
- **`core/vector_store.py`** - FAISS wrapper with lazy loading and memory-mapping for large indexes
- **`core/cache_server.py`** / `cache_rpc.py` - Pyro5 RPC server for shared multi-process cache access
- **`utils/text_preprocessor.py`** - String metrics, acronym generation/fidelity, location scoring
- **`web/app.py`** - Flask routes (`/search`, `/plugging`, `/status`, etc.)
- **`web/services/rationale_service.py`** - Generates human-readable match explanations

### Entry Points (root directory)

- `company_search.py` - CLI entry point
- `run_app.py` - Flask web app entry point
- `run_cache_server.py` - RPC cache server entry point
- `run_plugging_pipeline.py` - Batch processing pipeline

### Matching Logic Overview

1. **Phase 0**: Acronym expansion (e.g., "ABA" → "American Bar Association")
2. **Phase 1**: FAISS semantic retrieval (top-1000 candidates)
3. **Phase 2**: Hybrid re-ranking combining:
   - String similarity (50%) + Semantic score (25%) + Concept alignment (25%)
   - Lexical boosts for high string matches
   - Location score when city/state provided
4. **Phase 3**: Exact match override

## Configuration Files

| File | Purpose |
|------|---------|
| `model_config.json` | Active embedding model (`all-MiniLM-L6-v2` default) |
| `tier_config.json` | Score thresholds: Great (98+), High (93+), Medium (88+) |

## Key Development Rules

1. **Always use absolute imports**: `from finetuner.core.matcher import CompanyMatcher`
2. **Verify after scoring changes**: Run `python tests/verify_control_set.py`
3. **Update RationaleService**: When adding new score components, update `rationale_service.py` to explain them
4. **Increment CACHE_VERSION**: In `matcher.py` when making breaking changes to indexing
5. **Performance awareness**: Dataset is 2.9M+ companies. Use FAISS and precomputed caches; avoid loops over entire dataset

## Data Files

- `companies.json` / `companies_with_location.json` - Main company datasets
- `companies_control_set.json` - Validation pairs for algorithm verification
- `company_matcher_cache/` - Cached FAISS indexes and embeddings
