# FineTuner Project Analysis

**Document Version:** 1.0
**Analysis Date:** 2026-04-22
**Author:** Claude Code Analysis

---

## Executive Summary

FineTuner is a sophisticated company name matching system designed to handle large-scale datasets (2.9M+ companies) with sub-second query response times. The system employs a hybrid matching approach combining:

1. **Semantic Search** via FAISS with sentence-transformer embeddings
2. **Lexical Analysis** via string similarity, weighted Jaccard, and acronym fidelity
3. **Concept Probing** via "nutritional label" anchor vectors for industry/geographic disambiguation
4. **Location-Aware Ranking** with city/state matching and popularity boosting

The architecture supports multiple access patterns: CLI, Flask web application, and Pyro5 RPC for multi-process coordination.

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   CLI Interface  │   Flask Web App  │    External Integrations      │
│ company_search.py│    run_app.py    │      (Future APIs)            │
└────────┬─────────┴────────┬─────────┴───────────────┬───────────────┘
         │                  │                         │
         │                  ▼                         │
         │     ┌─────────────────────────┐           │
         │     │    SearchService        │           │
         │     │  (RPC Client Wrapper)   │           │
         │     └───────────┬─────────────┘           │
         │                 │ Pyro5 RPC               │
         │                 ▼                         │
┌────────┴─────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              CacheIndexServer (cache_server.py)              │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                 │   │
│  │  │  Cache Manager   │  │  File Watcher    │                 │   │
│  │  │  (multi-cache)   │  │  (hot-reload)    │                 │   │
│  │  └────────┬─────────┘  └──────────────────┘                 │   │
│  │           │                                                  │   │
│  │           ▼                                                  │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │         CompanyMatcher (matcher.py)                   │   │   │
│  │  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐ │   │   │
│  │  │  │ Build/Load │  │  3-Phase   │  │  Explanation    │ │   │   │
│  │  │  │   Index    │  │  Matching  │  │  Generation     │ │   │   │
│  │  │  └──────┬─────┘  └─────┬──────┘  └─────────────────┘ │   │   │
│  │  │         │              │                              │   │   │
│  │  │         ▼              ▼                              │   │   │
│  │  │  ┌─────────────────────────────────────────────────┐ │   │   │
│  │  │  │           VectorStore (FAISS Wrapper)           │ │   │   │
│  │  │  │  - IndexFlatIP (cosine similarity)              │ │   │   │
│  │  │  │  - Memory-mapped loading for large indexes      │ │   │   │
│  │  │  │  - Lazy embedding proxy (avoid double load)     │ │   │   │
│  │  │  └─────────────────────────────────────────────────┘ │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐  │
│  │  Source JSON Files  │  │       company_matcher_cache/        │  │
│  │  - companies.json   │  │  - *_embeddings.npy (6+ GB)         │  │
│  │  - companies_with_  │  │  - *_index.faiss                    │  │
│  │    location.json    │  │  - *_names.pkl                      │  │
│  │  - control_set.json │  │  - *_metadata.pkl                   │  │
│  └─────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | File | Purpose | Lines |
|-----------|------|---------|-------|
| `CompanyMatcher` | `src/finetuner/core/matcher.py` | Main matching engine | ~1,850 |
| `TextPreprocessor` | `src/finetuner/utils/text_preprocessor.py` | String metrics, acronyms, location scoring | ~490 |
| `VectorStore` | `src/finetuner/core/vector_store.py` | FAISS wrapper with lazy loading | ~210 |
| `CacheIndexServer` | `src/finetuner/core/cache_server.py` | Multi-cache management, hot-reload | ~540 |
| `CacheServerRPC` | `src/finetuner/core/cache_rpc.py` | Pyro5 RPC interface | ~280 |
| `SearchService` | `src/finetuner/web/services/search_service.py` | RPC client wrapper for web | ~270 |
| `RationaleService` | `src/finetuner/web/services/rationale_service.py` | Human-readable match explanations | ~940 |
| Flask App | `src/finetuner/web/app.py` | Web routes and API endpoints | ~280 |

---

## 2. Matching Algorithm Deep Dive

### 2.1 Three-Phase Matching Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 0: ACRONYM EXPANSION                       │
│  Query: "ABA"                                                       │
│  ↓                                                                  │
│  1. Check acronym_index for direct matches                          │
│  2. Calculate fidelity score (0.0–1.0) for each expansion           │
│  3. Score = 0.70 + (fidelity × 0.20) + (semantic × 0.10)           │
│  4. Also check reverse: "American Bar Association" → finds "ABA"    │
│                                                                     │
│  Note: 2-letter acronyms IGNORED (too much noise from states)       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: SEMANTIC RETRIEVAL                      │
│  1. Encode query via sentence-transformer (all-MiniLM-L6-v2)        │
│  2. FAISS IndexFlatIP search → top 1,000 candidates                 │
│  3. Generate "concept signature" for query (anchor similarities)    │
│                                                                     │
│  Note: candidate_k = 1000 is a tuneable retrieval funnel           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: HYBRID RE-RANKING                       │
│                                                                     │
│  For each candidate in 1,000:                                       │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  BASE SCORE CALCULATION                                    │    │
│  │                                                            │    │
│  │  String Score (50%):                                       │    │
│  │    - Weighted Jaccard (term importance)                    │    │
│  │    - SequenceMatcher (character alignment)                 │    │
│  │    - Distinctive word penalty                              │    │
│  │    - Category mismatch penalty (hospital vs school)        │    │
│  │    - Proper noun mismatch penalty                          │    │
│  │    - Short string cap (< 8 chars matched → max 0.65)       │    │
│  │                                                            │    │
│  │  Semantic Score (25%):                                     │    │
│  │    - Normalized: raw_score / max_score_in_batch            │    │
│  │                                                            │    │
│  │  Concept Alignment (25%):                                  │    │
│  │    - Signature correlation against 31 concept anchors:     │    │
│  │      • Geography: 11 anchors (PA, London, NYC, etc.)       │    │
│  │      • Industry: 11 anchors (Medical, Tech, Finance, etc.) │    │
│  │      • Structure: 4 anchors (Corporate, Non-Profit, etc.)  │    │
│  │      • Nature: 5 anchors (Global, Local, Industrial, etc.) │    │
│  │                                                            │    │
│  │  base_score = (string × 0.50) + (semantic × 0.25)          │    │
│  │             + (concept_alignment × 0.25)                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  BOOSTS & OVERRIDES                                        │    │
│  │                                                            │    │
│  │  Acronym Fidelity Boost:                                   │    │
│  │    IF fidelity > 0.8 AND acronym.len > 2:                  │    │
│  │       name_score += fidelity × 0.15                        │    │
│  │                                                            │    │
│  │  Tiered Lexical Boost:                                     │    │
│  │    IF string_score >= 0.92 OR full_token_overlap:          │    │
│  │       name_score = max(name_score, 0.95)                   │    │
│  │    ELIF string_score >= 0.80:                              │    │
│  │       name_score = max(name_score, 0.90)                   │    │
│  │                                                            │    │
│  │  Location Score (when city/state provided):                │    │
│  │    - City match: 60% weight                                │    │
│  │    - State match: 40% weight                               │    │
│  │    - Handles variations (NYC→New York, LA→Los Angeles)     │    │
│  │                                                            │    │
│  │  Final Score Blend:                                        │    │
│  │    IF has_location:                                        │    │
│  │       final = (name_score × 0.70) + (location × 0.30)     │    │
│  │    ELSE:                                                   │    │
│  │       final = name_score                                   │    │
│  │                                                            │    │
│  │  Popularity Boost (location queries only):                 │    │
│  │    IF count > 1:                                           │    │
│  │       final += log(count) / log(max_count) × 0.05          │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: EXACT MATCH OVERRIDE                    │
│  IF query.lower() in company_names_set:                             │
│    - Find ALL matching entries (multiple locations possible)        │
│    - Set score = 1.0 (with location blend if applicable)            │
│    - Mark match_type = "exact"                                      │
│    - Apply frequency boost for popular exact matches                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Score Weight Summary

| Component | Weight | Notes |
|-----------|--------|-------|
| String Similarity | 50% | Character/token overlap |
| Semantic Similarity | 25% | FAISS cosine score (normalized) |
| Concept Alignment | 25% | Anchor correlation |
| Location Score | 30% (of final) | Only when city/state provided |
| Name Score | 70% (of final) | Only when city/state provided |
| Acronym Boost | up to +15% | Only for high-fidelity (>0.8) matches |
| Popularity Boost | up to +5% | Log-scaled, location queries only |

### 2.3 Concept Anchors ("Nutritional Label")

The system computes a 31-dimensional "concept signature" for each company name by measuring cosine similarity against predefined anchor terms:

```python
CONCEPT_ANCHORS = {
    "Geography": ["Pennsylvania", "London", "Canada", "California",
                  "New York", "Texas", "Chicago", "Illinois", "Ohio",
                  "Miami", "Paris"],  # 11 anchors
    "Industry": ["Automotive", "Medical", "Technology", "Construction",
                 "Legal", "Food", "Finance", "Education", "Insurance",
                 "Retail", "Manufacturing"],  # 11 anchors
    "Structure": ["Corporate", "Non-Profit", "Government", "Small Business"],  # 4
    "Nature": ["Global", "Local", "Industrial", "Consumer", "Professional"]  # 5
}
```

This helps disambiguate semantically similar names that differ in domain (e.g., "Northwest Medical" vs "Northwest Construction").

---

## 3. Caching System

### 3.1 Cache Structure

```
company_matcher_cache/
├── <hash>_embeddings.npy    # 6+ GB: float32 vectors (2.9M × 384)
├── <hash>_index.faiss       # FAISS IndexFlatIP
├── <hash>_names.pkl         # Pickled dictionary containing:
│   │                        #   - company_names (preprocessed)
│   │                        #   - original_company_names
│   │                        #   - company_locations (optional)
│   │                        #   - company_counts (optional)
│   │                        #   - company_ids (optional)
│   │                        #   - acronym_index
│   │                        #   - similarity_cache (precomputed pairs)
│   │                        #   - acronym_cache
│   │                        #   - fast lookup structures
│   └── <hash>_metadata.pkl  # model_name, cache_version, counts
```

### 3.2 Cache Key Generation

Two strategies for cache key generation:

1. **File-based** (fast, no file loading):
   ```python
   key = MD5(filepath + file_size + file_mtime + model_name + CACHE_VERSION)
   ```

2. **Content-based** (slower, verifies content):
   ```python
   key = MD5(sorted_company_names + model_name + CACHE_VERSION)
   ```

### 3.3 Memory Optimization

- **Lazy Embedding Proxy**: `_LazyEmbeddingProxy` avoids loading the 6+ GB embeddings.npy since FAISS index already contains vectors internally
- **Memory-Mapped FAISS**: Uses `faiss.IO_FLAG_MMAP` to page-load index on demand
- **Result**: ~600 MB RAM usage vs 13+ GB without optimization

---

## 4. Data Flow

### 4.1 Index Building Pipeline

```
1. Load JSON (companies_with_location.json)
   ↓
2. Extract fields: Company Name, City, State, Count, ID
   ↓
3. Preprocess names (lowercase, strip)
   ↓
4. [Optional] Checkpointed encoding (--work-dir for resume)
   ↓
5. Multi-process encoding via sentence-transformers
   - Chunks of 5,000 for progress visibility
   - Pool distributed across CPU cores
   ↓
6. Build FAISS IndexFlatIP
   ↓
7. Create fast lookup structures:
   - _company_names_lower_set (O(1) exact match check)
   - _company_names_lower_to_index (lowercase → index list)
   - _company_words_dict (word → indices containing word)
   ↓
8. Create acronym index:
   - Generate acronym for each name
   - Map: "ABA" → [105, 2099, 5001]
   ↓
9. [Disabled for large datasets] Precompute similarity matrix
   - O(N²) too expensive for 2.9M entries
   ↓
10. Save all to cache files
```

### 4.2 Query Flow (RPC Mode)

```
Web Client
    │ HTTP POST /search
    ▼
Flask App (app.py)
    │ SearchService.search()
    ▼
SearchService (search_service.py)
    │ Pyro5 RPC call
    ▼
CacheServerRPC (cache_rpc.py)
    │ Get matcher from loaded_caches
    ▼
CompanyMatcher.match_with_location()
    │ 3-phase matching pipeline
    ▼
Return results
    │ SearchService formats + adds rationales
    ▼
JSON Response to client
```

---

## 5. Strengths

1. **Hybrid Matching**: Combines lexical precision with semantic understanding
2. **Concept Probing**: Novel approach to disambiguate similar names by domain
3. **Memory Efficiency**: Lazy loading and memory-mapping for large indexes
4. **RPC Architecture**: Multi-process support with hot-reload capability
5. **Explainability**: Detailed rationales for each match
6. **Location Awareness**: City/state boosting with variation handling
7. **Control Set Verification**: Built-in regression testing framework
8. **Checkpointed Building**: Resume capability for long index builds

---

## 6. Current Limitations

### 6.1 Performance Bottlenecks

| Issue | Impact | Location |
|-------|--------|----------|
| O(N) re-ranking loop | 1,000 candidates × expensive calculations | `match_with_location()` |
| Per-query concept signature | Anchor vector dot products on every query | `_get_concept_signature()` |
| TextPreprocessor calls in loop | String similarity not vectorized | Phase 2 re-ranking |
| Single-threaded RPC | One query at a time per server | `CacheServerRPC` |
| No batch query optimization | Each RPC call creates new proxy | `SearchService._get_rpc_client()` |

### 6.2 Algorithmic Limitations

| Issue | Impact |
|-------|--------|
| Fixed weight scheme | 50/25/25 may not be optimal for all domains |
| No learning from feedback | No way to improve from user corrections |
| Acronym 2-letter filter | May miss valid acronyms (NY, DC as companies) |
| Location baking removed | Was decoupled but may lose geographic context |
| No fuzzy location matching | "Philadelphia, PA" won't match "Phila" in source |

### 6.3 Architectural Limitations

| Issue | Impact |
|-------|--------|
| No horizontal scaling | Single RPC server handles all requests |
| No query logging | Can't analyze search patterns or failures |
| No A/B testing framework | Hard to evaluate algorithm changes |
| Tight coupling | Matcher does too much (scoring + caching + matching) |
| No rate limiting | Web app vulnerable to overload |

---

## 7. Configuration Summary

### 7.1 Key Configuration Files

| File | Purpose | Key Values |
|------|---------|------------|
| `model_config.json` | Embedding model selection | `all-MiniLM-L6-v2` (default) |
| `tier_config.json` | Score tier thresholds | Great: 98%, High: 93%, Medium: 88% |
| `matcher.py:CACHE_VERSION` | Cache invalidation | `v4.1_location_decoupled` |

### 7.2 Tuneable Parameters

| Parameter | Current Value | Location |
|-----------|---------------|----------|
| `candidate_k` | 1,000 | `match_with_location()` |
| String weight | 0.50 | `match_with_location()` |
| Semantic weight | 0.25 | `match_with_location()` |
| Concept weight | 0.25 | `match_with_location()` |
| Name/Location blend | 70/30 | `match_with_location()` |
| Acronym boost factor | 0.15 | `match_with_location()` |
| Lexical boost thresholds | 0.92, 0.80 | `match_with_location()` |
| RPC timeout | 120s | `SearchService.search()` |

---

## 8. Test Coverage

### 8.1 Existing Tests

- `tests/verify_control_set.py` - Control set validation
- `tests/generate_plugging_report.py` - Batch matching reports
- `tests/test_control_set.py` - Basic control set tests
- `tests/batch_test.py` - Batch processing tests

### 8.2 Missing Test Coverage

- Unit tests for `TextPreprocessor` methods
- Unit tests for scoring component calculations
- Integration tests for RPC server
- Load testing / stress testing
- Edge case handling (empty strings, special characters)

---

## 9. Dependencies

```
Flask==2.3.3               # Web framework
sentence-transformers>=2.2.2  # Embedding generation
faiss-cpu==1.7.4           # Vector similarity search
numpy>=1.24.3              # Numerical operations
tqdm==4.66.1               # Progress bars
pyodbc                     # SQL Server integration
Pyro5                      # RPC framework
watchdog                   # File monitoring for hot-reload
```

---

## 10. Conclusion

FineTuner is a well-architected system with strong foundations for company name matching at scale. The hybrid approach balances precision (lexical) with recall (semantic), while the concept probing adds valuable disambiguation.

Key opportunities for improvement:
1. **Performance**: Vectorize re-ranking operations, add batch processing
2. **Accuracy**: Implement feedback loops, tune weights per domain
3. **Architecture**: Add horizontal scaling, observability, and A/B testing

See the accompanying proposal documents for detailed improvement plans.
