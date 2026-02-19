# Company Matcher Algorithm Documentation

## Overview

The `CompanyMatcher` is a hybrid semantic + lexical matching system designed to find similar company names in a large database (2.9M+ entities). It combines the power of neural embeddings with traditional string matching techniques to provide accurate, fast, and explainable results.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY INPUT                               │
│                  "Hartford Hospital School of Nursing"           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: RETRIEVAL                            │
│              (Semantic Search via FAISS Index)                   │
│                                                                  │
│  • Encode query using SentenceTransformer                       │
│  • Search FAISS index for top 50 semantic candidates            │
│  • Fast approximate nearest neighbor search                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: RE-RANKING                           │
│              (String Similarity + Penalties)                     │
│                                                                  │
│  For each of 50 candidates, calculate:                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Weighted Jaccard Similarity                              │ │
│  │ 2. Sequence Matcher Score                                   │ │
│  │ 3. Apply: Discriminating Word Penalty                       │ │
│  │ 4. Apply: Short String Cap                                  │ │
│  │ 5. Apply: Category Mismatch Penalty                         │ │
│  │ 6. Apply: Proper Noun Mismatch Penalty                      │ │
│  │ 7. Apply: Length/Coverage Adjustments                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Final Score = (String Score × 0.7) + (Semantic Score × 0.3)    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: OUTPUT                               │
│                                                                  │
│  • Sort by final score                                          │
│  • Return top K matches with metadata                           │
│  • Mark exact matches as "exact", others as "hybrid"            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Sentence Transformer Model

**Model:** `paraphrase-MiniLM-L3-v2` (ultra-fast variant)

Converts company names into 384-dimensional dense vectors that capture semantic meaning. This enables finding companies that are conceptually similar even if they don't share exact words.

```python
# Example: These names have similar embeddings
"Volkswagen Group China"  ←→  "Volkswagen China"
"Travel Leaders - Dube"   ←→  "Dube Travel / Travel Leaders"
```

**Vector Storage:**

```
┌─────────────────────────────────────────────────────────────────┐
│                       VECTOR STORAGE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IN MEMORY (Runtime):                                           │
│    self.embeddings → NumPy array (2.9M × 384) ≈ 4.2 GB         │
│    self.index      → FAISS IndexFlatIP (same vectors indexed)  │
│                                                                 │
│  ON DISK (company_matcher_cache/):                              │
│    {hash}_embeddings.npy  → Serialized NumPy array             │
│    {hash}_index.faiss     → Serialized FAISS index             │
│    {hash}_names.pkl       → Company name lists (original+clean)│
│    {hash}_metadata.pkl    → Model name, count, cache key       │
│                                                                 │
│  Cache Key: MD5 hash of filepath + file_size + mtime + model   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. FAISS Index

**Type:** `IndexFlatIP` (Inner Product / Cosine Similarity)

A high-performance vector similarity search library from Facebook. Enables searching 2.9M+ company names in milliseconds.

- **Build Time:** ~45 minutes for full index
- **Search Time:** <100ms per query
- **Memory:** ~4GB for 2.9M companies

### 3. Generic Term Weights

Words are weighted by their distinctiveness. Common organizational terms receive lower weights:

| Category | Examples | Weight |
|----------|----------|--------|
| Organization Suffixes | group, association, foundation | 0.2 |
| Facility Types | center, school, hospital | 0.3 |
| Event Types | meeting, breakfast, conference | 0.3 |
| Common Modifiers | national, international, global | 0.4 |
| Location Modifiers | north, south, shore, bay | 0.5 |
| Distinctive Words | (any other word) | 1.0 |

**Purpose:** Prevents matches based solely on shared generic terms.

### 4. Category Words

Defines the TYPE of entity for mismatch detection:

```python
CATEGORY_WORDS = {
    'facility_type': {'center', 'school', 'hospital', 'church', 'synagogue'},
    'event_type': {'meeting', 'conference', 'wedding', 'tournament'},
    'service_type': {'senior', 'medical', 'financial', 'legal', 'nursing'},
}
```

**Purpose:** Penalizes matches where entity types differ (e.g., "Senior Center" vs "Financial Center").

### 5. Common Words (for Proper Noun Detection)

A comprehensive list of words that should NOT be treated as proper nouns even if capitalized:

- Articles: the, a, an, of, and
- Business terms: inc, corp, llc, company
- Facility types: center, school, hospital
- Descriptors: national, international, first, second

**Purpose:** Helps identify true proper nouns (Hartford, Jefferson, Volkswagen) vs. generic terms.

---

## Scoring Algorithm

### Step 1: Weighted Jaccard Similarity

Traditional Jaccard treats all words equally. We weight by term importance:

```
Standard Jaccard:
  "North Shore Senior Center" ∩ "North Shore Financial Center"
  = {north, shore, center} / {north, shore, senior, center, financial}
  = 3/5 = 60%

Weighted Jaccard:
  = (0.5 + 0.5 + 0.3) / (0.5 + 0.5 + 1.0 + 0.3 + 1.0)
  = 1.3 / 3.3 = 39%
```

**Product:** `base_score` (0.0 - 1.0) — The higher of Weighted Jaccard or Sequence Matcher score.

### Step 2: Discriminating Word Penalty

If the query contains distinctive words (weight ≥ 0.8) that are missing from the target, apply a penalty:

```
Query: "Internal J&J Meeting and Breakfast"
Target: "AEP Breakfast Meeting"

Distinctive words in query: ["internal", "j&j"]
Missing in target: ["internal", "j&j"]
Missing ratio: 2/2 = 100%

Penalty = 1.0 - (1.0 × 0.4) = 0.6 (40% reduction)
```

**Product:** `distinctive_penalty` (0.6 - 1.0) — Multiplier applied to base_score.

### Step 3: Short String Cap

Prevents very short matches from getting artificially high scores:

| Matched Characters | Maximum Score |
|-------------------|---------------|
| < 5 chars | 50% |
| 5-7 chars | 65% |
| ≥ 8 chars | No cap |

**Example:** "BOD" matching "Bod Pro" is capped at 50% despite high character overlap ratio.

**Product:** `capped_score` (0.0 - 0.65) — Uses `min()` to cap, not multiply. Applied before other penalties.

### Step 4: Category Mismatch Penalty

When both names have category words but they differ:

```
Query: "North Shore Senior Center"
Target: "North Shore Financial Center"

Query service_type: {senior}
Target service_type: {financial}

Mismatch detected → Penalty = 0.75 (25% reduction)
```

**Product:** `category_penalty` (0.75 - 1.0) — Multiplier applied to running score.

### Step 5: Proper Noun Mismatch Penalty

When query has identifying proper nouns missing from target:

```
Query: "Hartford Hospital School of Nursing"
Target: "Jefferson Hospital Nursing School"

Query proper nouns: {hartford}
Target proper nouns: {jefferson}
Missing: {hartford}

Penalty = 1.0 - (1/1 × 0.25) = 0.75 (25% reduction)
```

**Product:** `proper_noun_penalty` (0.75 - 1.0) — Multiplier applied to running score.

### Step Summary: How Products Combine

```
String Score = base_score × distinctive_penalty × category_penalty × proper_noun_penalty
             = Step1      × Step2               × Step4            × Step5

(Step 3 applies as min() cap before multipliers if matched chars < 8)
```

### Final Score Calculation

```
Final Score = (String Score × 0.7) + (Normalized Semantic Score × 0.3)
```

The 70/30 weighting favors lexical accuracy while still benefiting from semantic understanding.

---

## Score Component Deep Dive

### Understanding the Score Types

| Score Type | Range | Description |
|------------|-------|-------------|
| **Raw Semantic Score** | 0.0 - 10.0+ | Dot product from FAISS (unnormalized) |
| **Normalized Semantic Score** | 0.0 - 1.0 | Raw score ÷ max score in candidate set |
| **String Score** | 0.0 - 1.0 | Weighted lexical similarity after penalties |
| **Final Score** | 0.0 - 1.0 | Combined: (String × 0.7) + (Semantic × 0.3) |

### Example 1: Exact Match Breakdown

**Query:** `"PDMA Association"`

```
┌─────────────────────────────────────────────────────────────────┐
│ CANDIDATE: "PDMA Association" (exact match in database)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ SEMANTIC SCORES:                                                │
│   Raw Semantic Score:        6.9332  (from FAISS dot product)   │
│   Max Score in Candidates:   6.9332  (this IS the top result)   │
│   Normalized Semantic:       6.9332 / 6.9332 = 1.0000           │
│                                                                 │
│ STRING SCORE:                                                   │
│   Exact match detected → String Score = 1.0000                  │
│                                                                 │
│ FINAL CALCULATION:                                              │
│   Final = (1.0 × 0.7) + (1.0 × 0.3) = 1.0000 (100%)            │
│   Match Type: "exact"                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Example 2: High-Confidence Hybrid Match

**Query:** `"Nicolas/Sanchez Wedding"`
**Candidate:** `"Sanchez Wedding"` (Rank 2)

```
┌─────────────────────────────────────────────────────────────────┐
│ CANDIDATE: "Sanchez Wedding"                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ PHASE 1 - SEMANTIC RETRIEVAL:                                   │
│   Raw Semantic Score:        5.1842                             │
│   Max Score (exact match):   5.3808                             │
│   Normalized Semantic:       5.1842 / 5.3808 = 0.9634           │
│                                                                 │
│ PHASE 2 - STRING SIMILARITY:                                    │
│   Query tokens:    {nicolas, sanchez, wedding}                  │
│   Target tokens:   {sanchez, wedding}                           │
│   Intersection:    {sanchez, wedding}                           │
│                                                                 │
│   Step 1 - Weighted Jaccard:                                    │
│     Intersection weights: sanchez(1.0) + wedding(0.3) = 1.3     │
│     Union weights: nicolas(1.0) + sanchez(1.0) + wedding(0.3)   │
│                  = 2.3                                          │
│     Weighted Jaccard: 1.3 / 2.3 = 0.565                         │
│                                                                 │
│   Step 2 - Discriminating Word Penalty:                         │
│     Distinctive words: [nicolas, sanchez]  (weight >= 0.8)      │
│     Missing from target: [nicolas]                              │
│     Missing ratio: 1/2 = 0.5                                    │
│     Penalty: 1.0 - (0.5 × 0.4) = 0.80                          │
│                                                                 │
│   Step 3 - Short String Cap: N/A (>8 chars matched)             │
│   Step 4 - Category Mismatch: None (both "wedding")             │
│   Step 5 - Proper Noun Penalty:                                 │
│     Query proper nouns: {nicolas, sanchez}                      │
│     Target proper nouns: {sanchez}                              │
│     Missing: {nicolas}                                          │
│     Penalty: 1.0 - (0.5 × 0.25) = 0.875                        │
│                                                                 │
│   Combined String Score:                                        │
│     Base (0.565) × Discrim(0.80) × PropNoun(0.875) ≈ 0.78      │
│                                                                 │
│ FINAL CALCULATION:                                              │
│   Final = (0.78 × 0.7) + (0.9634 × 0.3)                        │
│         = 0.546 + 0.289 = 0.8352 ≈ 85.12%                      │
│   Match Type: "hybrid"                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Example 3: Category Mismatch Penalty in Action

**Query:** `"North Shore Senior Center"`
**Candidate:** `"North Shore Medical Center"` (NOT in top 20 after fix)

```
┌─────────────────────────────────────────────────────────────────┐
│ CANDIDATE: "North Shore Medical Center"                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ BEFORE IMPROVEMENTS (v1.1):                                     │
│   String Score (unweighted): ~0.80                              │
│   Semantic Score (norm):     ~0.95                              │
│   Final: (0.80 × 0.7) + (0.95 × 0.3) = 0.845 = 84.5%           │
│                                                                 │
│ AFTER IMPROVEMENTS (v2.0):                                      │
│   Step 1 - Weighted Jaccard:                                    │
│     Query:  north(0.5) shore(0.5) senior(1.0) center(0.3)      │
│     Target: north(0.5) shore(0.5) medical(1.0) center(0.3)     │
│     Intersection: {north, shore, center} = 0.5+0.5+0.3 = 1.3   │
│     Union: = 0.5+0.5+1.0+0.3+1.0 = 3.3                         │
│     Weighted Jaccard: 1.3 / 3.3 = 0.394                        │
│                                                                 │
│   Step 4 - CATEGORY MISMATCH:                                   │
│     Query service_type:  {senior}                               │
│     Target service_type: {medical}                              │
│     MISMATCH DETECTED → Penalty = 0.75                         │
│                                                                 │
│   Step 5 - Proper Noun Penalty: None (no proper nouns)          │
│                                                                 │
│   String Score: 0.394 × 0.75 = 0.296                           │
│   Final: (0.296 × 0.7) + (0.95 × 0.3) = 0.492 ≈ 49%            │
│                                                                 │
│   RESULT: Dropped from 84.5% to ~49%, out of top 20!           │
└─────────────────────────────────────────────────────────────────┘
```

### Example 4: Short String Cap in Action

**Query:** `"SEMMOA BOD"`
**Candidate:** `"SE Production"` (Rank 13 after fix)

```
┌─────────────────────────────────────────────────────────────────┐
│ CANDIDATE: "SE Production"                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ TOKEN ANALYSIS:                                                 │
│   Query tokens:  {semmoa, bod}                                  │
│   Target tokens: {se, production}                               │
│   Intersection:  {} (empty - no exact token matches)            │
│                                                                 │
│ BEFORE SHORT STRING CAP:                                        │
│   Sequence matcher finds partial overlap "SE" ↔ "SEMMOA"       │
│   Character-level similarity: ~0.45                             │
│   Semantic similarity (normalized): ~0.60                       │
│   Uncapped Final: (0.45 × 0.7) + (0.60 × 0.3) = 0.495 ≈ 49%    │
│                                                                 │
│ AFTER SHORT STRING CAP:                                         │
│   Matched characters: "SE" = 2 chars                            │
│   2 chars < 5 chars threshold → CAP AT 50%                     │
│   String Score capped: min(0.45, 0.50) = 0.45                  │
│                                                                 │
│   But discriminating word penalty also applies:                 │
│   Missing distinctive: [semmoa, bod] = 100%                    │
│   Penalty: 1.0 - (1.0 × 0.4) = 0.60                           │
│   Adjusted: 0.45 × 0.60 = 0.27                                 │
│                                                                 │
│   Final: (0.27 × 0.7) + (0.60 × 0.3) = 0.369 ≈ 36.8%          │
│                                                                 │
│   RESULT: Dropped from 49% to 36.8%                            │
└─────────────────────────────────────────────────────────────────┘
```

### Example 5: Proper Noun Penalty in Action

**Query:** `"Hartford Hospital School of Nursing"`
**Candidate:** `"Jefferson Hospital Nursing School"`

```
┌─────────────────────────────────────────────────────────────────┐
│ CANDIDATE: "Jefferson Hospital Nursing School"                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ PROPER NOUN EXTRACTION:                                         │
│   Query original:  "Hartford Hospital School of Nursing"        │
│   Capitalized words: [Hartford, Hospital, School, Nursing]      │
│   After filtering COMMON_WORDS: {hartford}                      │
│   (Hospital, School, Nursing are in COMMON_WORDS list)          │
│                                                                 │
│   Target original: "Jefferson Hospital Nursing School"          │
│   Capitalized words: [Jefferson, Hospital, Nursing, School]     │
│   After filtering: {jefferson}                                  │
│                                                                 │
│ PROPER NOUN COMPARISON:                                         │
│   Query proper nouns:  {hartford}                               │
│   Target proper nouns: {jefferson}                              │
│   Missing from target: {hartford}                               │
│   Missing ratio: 1/1 = 100%                                    │
│                                                                 │
│ PENALTY CALCULATION:                                            │
│   Penalty = 1.0 - (1.0 × 0.25) = 0.75 (25% reduction)          │
│                                                                 │
│ BEFORE (v1.1): ~85% (based on shared structure)                │
│ AFTER (v2.0):  ~74% (with proper noun penalty)                 │
│                                                                 │
│   RESULT: Different hospitals correctly separated!              │
└─────────────────────────────────────────────────────────────────┘
```

### Score Normalization Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                 SEMANTIC SCORE NORMALIZATION                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Raw FAISS scores vary based on:                                 │
│   • Query length                                                │
│   • Embedding magnitude                                         │
│   • Model characteristics                                       │
│                                                                 │
│ Typical raw score ranges:                                       │
│   • Exact match:     5.0 - 8.0                                 │
│   • Strong match:    4.0 - 6.0                                 │
│   • Weak match:      2.0 - 4.0                                 │
│   • Poor match:      0.0 - 2.0                                 │
│                                                                 │
│ Normalization formula:                                          │
│   normalized = raw_score / max_raw_score_in_batch              │
│                                                                 │
│ This ensures:                                                   │
│   • Best semantic match always gets 1.0                        │
│   • All others scaled proportionally                           │
│   • Comparable across different queries                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Control Set Examples

### Example 1: Exact Match
**Query:** "PDMA Association"

| Rank | Match | Score | Type |
|------|-------|-------|------|
| 1 | PDMA Association | 100.00% | exact |
| 2 | Association Headquarters-PDMA | 87.67% | hybrid |
| 3 | PDMA Alliance | 81.50% | hybrid |

**Analysis:** Exact match correctly identified. Related entities score appropriately lower.

---

### Example 2: Word Reordering (Strength)
**Query:** "Travel Leaders - Dube Travel"

| Rank | Match | Score | Type |
|------|-------|-------|------|
| 1 | Dube Travel Leaders | 100.00% | hybrid |
| 2 | Travel Leaders - Dube Travel | 100.00% | exact |
| 3 | Dube Travel / Travel Leaders | 93.38% | hybrid |

**Analysis:** Algorithm correctly handles word reordering. "Dube Travel / Travel Leaders" is recognized as essentially the same entity despite different word order and punctuation.

---

### Example 3: Subset Matching (Strength)
**Query:** "Volkswagen Group China"

| Rank | Match | Score | Type |
|------|-------|-------|------|
| 1 | Volkswagen Group China | 100.00% | exact |
| 2 | Volkswagen China | 99.04% | hybrid |
| 3 | Volkswagen Group Japan | 80.84% | hybrid |

**Analysis:** "Volkswagen China" correctly scores very high as it's the same entity without the "Group" suffix. "Volkswagen Group Japan" scores lower due to proper noun mismatch ("China" vs "Japan").

---

### Example 4: Category Mismatch Penalty (Improvement)
**Query:** "North Shore Senior Center"

| Rank | Match | Score | Notes |
|------|-------|-------|-------|
| 1 | North Shore Senior Center | 100.00% | exact |
| 2 | North Shore Cancer Center | 84.81% | "Cancer" not in service_type |
| 3 | Northshore Senior Center | 84.46% | Valid variant |
| 6 | North Shore Elder Services | 82.08% | Related service |

**What's NOT in top 20:**
- ~~"North Shore Medical Center"~~ (was 86.40%, penalized for service_type mismatch)
- ~~"North Shore Financial Center"~~ (was 86.10%, penalized for service_type mismatch)

**Analysis:** The category mismatch penalty successfully pushed "Medical Center" and "Financial Center" out of the top results because they have different service types than "Senior Center".

---

### Example 5: Proper Noun Penalty (Improvement)
**Query:** "Hartford Hospital School of Nursing"

| Rank | Match | Score | Notes |
|------|-------|-------|-------|
| 1 | Hartford Hospital School of Nursing | 100.00% | exact |
| 2 | Hartford Hospital Offices USA | 77.58% | Same proper noun |
| 3 | Hartford School District | 74.96% | Same proper noun |
| 4 | JEFFERSON HOSPITAL NURSING SCHOOL | 74.06% | Different proper noun |

**Before Improvement:** "Jefferson Hospital Nursing School" scored 85.47%
**After Improvement:** Score reduced to 74.06%

**Analysis:** The proper noun penalty correctly identifies that "Hartford" and "Jefferson" are different identifying names, reducing the false positive score.

---

### Example 6: Short String Cap (Improvement)
**Query:** "SEMMOA BOD"

| Rank | Match | Score | Notes |
|------|-------|-------|-------|
| 1 | SEMMOA BOD | 100.00% | exact |
| 2 | SEMMOA AACM | 81.42% | Same org prefix |
| 7 | Semmoa | 71.27% | Partial match |
| 13 | SE Production | 36.84% | Short string capped |

**Before Improvement:** "SE Production" scored 49.01%
**After Improvement:** Score reduced to 36.84%

**Analysis:** Short, noisy partial matches are now pushed to lower scores.

---

## Recommended Thresholds

Based on control set validation:

| Score Range | Recommendation | Examples |
|-------------|----------------|----------|
| **≥ 95%** | Auto-merge safe | Exact matches, minor punctuation differences |
| **90-94%** | High confidence | Word reordering, abbreviation variants |
| **80-89%** | Review recommended | Corporate siblings, structural matches |
| **70-79%** | Manual review required | May include false positives |
| **< 70%** | Low confidence | Likely unrelated entities |

---

## Performance

| Metric | Value |
|--------|-------|
| Index Build Time | ~45 min (2.9M companies) |
| Cache Load Time | ~55 sec |
| Query Time (single) | ~800ms |
| Query Time (batch) | ~12ms/query |
| Memory Usage | ~4GB |

---

## Files

| File | Purpose |
|------|---------|
| `CompanyMatcher.py` | Core matching algorithm |
| `companies.json` | Source dataset (2.9M companies) |
| `companies_control_set.json` | 100-company test set |
| `companies_control_set_results.md` | Latest test results |
| `test_control_set.py` | Control set testing script |
| `company_matcher_cache/` | Cached embeddings and FAISS index |

---

## Usage

```python
from CompanyMatcher import CompanyMatcher

# Initialize matcher
matcher = CompanyMatcher()

# OPTION 1: Standard Build (Name Only)
matcher.build_index(filepath='companies.json')

# OPTION 2: Location-Aware Build (Name + City + State + ID)
# Use this when your data source includes location info
data = [
    {"ID": 101, "Company Name": "Acme Corp", "City": "New York", "State": "NY", "Count": 100},
    {"ID": 102, "Company Name": "Acme Inc", "City": "Chicago", "State": "IL", "Count": 50}
]
matcher.build_index_with_location(data=data)  # Or filepath='companies_with_location.json'

# --- QUERYING ---

# 1. Standard Search
results = matcher.match("Acme Corp", top_k=5)

# 2. Location-Aware Search
results = matcher.match_with_location("Acme", city="NYC", state="NY", top_k=5)

for r in results:
    print(f"{r['name']} ({r['score']*100:.1f}%)")
    if 'city' in r:
        print(f"  Location: {r['city']}, {r['state']}")
        print(f"  DB ID: {r.get('id')}")
```

---

## Location-Aware Matching (New in v3.0)

When location data is available, the matcher can use City and State to disambiguate between similar companies.

### How It Works
1. **Fuzzy Normalization**: "NYC" → "New York", "Calif" → "CA", "Chi-Town" → "Chicago"
2. **Hybrid Scoring**: `Final Score = (Name Score × 0.8) + (Location Score × 0.2)`
3. **Exact Match Handling**: If names match exactly, location acts as a 5% tie-breaker boost.

### Example
Query: **"First National Bank"** in **"Houston, TX"**

| Candidate | Location | Name Match | Location Match | Final Score |
|-----------|----------|------------|----------------|-------------|
| First National Bank | Houston, TX | 100% | 100% | **105.0%** (Rank #1) |
| First National Bank | New York, NY | 100% | 0% | 100.0% |
| First National Bank | Chicago, IL | 100% | 0% | 100.0% |

Without location data, all three would be tied at 100%.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial implementation with semantic search |
| 1.1 | Dec 2025 | Added hybrid re-ranking (70% string / 30% semantic) |
| 2.0 | Dec 2025 | Added generic term weighting, category mismatch penalty, short string cap, proper noun detection |
| 3.0 | Jan 2026 | Added Location-Aware Matching (City/State), fuzzy location normalization, ID tracking |

