# FineTuner - Agent Context

## Project Overview
FineTuner is a high-performance company matching system. It uses a hybrid approach:
1. **Semantic Search**: FAISS index with `paraphrase-MiniLM-L3-v2` embeddings for initial candidate retrieval (top 50).
2. **Deterministic Re-ranking**: A sophisticated scoring algorithm that considers:
   - Weighted Jaccard Similarity.
   - Acronym Fidelity Score.
   - Category Mismatch Penalties.
   - Proper Noun Penalties.
   - Short String Caps.

## Core Structure
- `src/finetuner/core/`:
  - `matcher.py`: The `CompanyMatcher` class - core logic.
  - `model.py`: Embedding generation and sentence transformer integration.
  - `vector_store.py`: FAISS index management.
- `src/finetuner/web/`: Flask-based web interface for testing matches.
- `tests/`:
  - `verify_control_set.py`: Core verification script.
  - `generate_ultra_report.py`: Generates the comprehensive `control_set_report_ULTRA.md`.
- `company_matcher_cache/`: Stores pre-computed embeddings and FAISS indexes.

## Key Concepts for Agents

### 1. Scoring Logic
The score is a mix of Semantic (30%) and String (70%) similarity. 
String similarity is heavily penalized for mismatches in:
- **Acronyms**: Must have high fidelity if one exists.
- **Categories**: e.g., "Hospital" vs "School".
- **Proper Nouns**: e.g., "Hartford" vs "Jefferson".

### 2. The Control Set
The project relies on a "Control Set" (`companies_control_set.json`) to verify algorithm changes. 
Always run `python tests/verify_control_set.py` after modifying the scoring logic.

### 3. Report Generation
The source of truth for current performance is `control_set_report_ULTRA.md`. 
It provides a detailed breakdown of scores, penalties, and rationales for every match in the control set.

## Common Tasks
- **Updating Scoring**: Modify `src/finetuner/core/matcher.py`.
- **Adding Test Cases**: Add to `companies_control_set.json`.
- **Fixing Rationale**: Modify `src/finetuner/core/matcher.py` (look for rationale generation logic).

## Environment
- **OS**: Windows (PowerShell/Cmd)
- **Python**: 3.x
- **Dependencies**: `sentence-transformers`, `faiss-cpu`, `numpy`, `torch`.
