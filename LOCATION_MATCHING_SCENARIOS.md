# Company Matching with Location: Demo Scenarios

This document describes the fuzzy company name + location matching scenarios demonstrated in `demo_fuzzy_location_match.py`. These scenarios illustrate how the `CompanyMatcher` handles real-world matching challenges where company names don't exactly match and location data helps disambiguate results.

---

## Overview

The matching system uses a **hybrid approach**:
1. **Semantic Matching** - Uses ML embeddings to find companies with similar meaning
2. **Lexical Matching** - Uses string similarity algorithms (Weighted Jaccard)
3. **Location Scoring** - Uses fuzzy city/state matching to boost relevant results

### Scoring Formula

| Match Type | Formula |
|------------|---------|
| **Non-exact name match** | `Final Score = (Name Score × 0.8) + (Location Score × 0.2)` |
| **Exact name match** | `Final Score = 1.0 + (Location Score × 0.05)` |

For exact name matches, location provides a **5% tie-breaker boost** rather than a full 20% weight, since the name is already a perfect match.

---

## Matching Process Workflow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPANY MATCHER SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   FAISS     │    │  String     │    │  Location   │    │   Cache     │  │
│  │   Index     │    │  Similarity │    │  Normalizer │    │   Layer     │  │
│  │ (Semantic)  │    │  (Lexical)  │    │  (Fuzzy)    │    │ (Speed)     │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └─────────────┘  │
│         │                  │                  │                             │
│         └──────────────────┼──────────────────┘                             │
│                            ▼                                                │
│                   ┌─────────────────┐                                       │
│                   │  Score Combiner │                                       │
│                   │  & Re-Ranker    │                                       │
│                   └─────────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Matching Flow

```
                              ┌──────────────────────┐
                              │       INPUT          │
                              │  Company: "Acme"     │
                              │  City: "Chi-Town"    │
                              │  State: "Illinois"   │
                              └──────────┬───────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │         PHASE 0: PREPROCESSING         │
                    ├────────────────────────────────────────┤
                    │  • Normalize query text (lowercase)    │
                    │  • Normalize city: Chi-Town → Chicago  │
                    │  • Normalize state: Illinois → IL      │
                    │  • Check: Is location provided?        │
                    └────────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │      PHASE 1: SEMANTIC SEARCH          │
                    │           (FAISS Index)                │
                    ├────────────────────────────────────────┤
                    │  1. Generate embedding for query       │
                    │  2. Search FAISS for nearest neighbors │
                    │  3. Return top 100 candidates          │
                    │                                        │
                    │  Candidates: [Acme Corp, ACME Inc,     │
                    │               Acme Corporation, ...]   │
                    └────────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │       PHASE 2: RE-RANKING              │
                    │    (String Similarity + Location)      │
                    ├────────────────────────────────────────┤
                    │  For each candidate:                   │
                    │                                        │
                    │  ┌──────────────────────────────────┐  │
                    │  │     NAME SCORE (0-100%)          │  │
                    │  │  • Weighted Jaccard Similarity   │  │
                    │  │  • Penalty for category mismatch │  │
                    │  │  • Proper noun handling          │  │
                    │  └──────────────────────────────────┘  │
                    │                                        │
                    │  ┌──────────────────────────────────┐  │
                    │  │   LOCATION SCORE (0-100%)        │  │
                    │  │  • City match (60% weight)       │  │
                    │  │  • State match (40% weight)      │  │
                    │  │  • Fuzzy matching for both       │  │
                    │  └──────────────────────────────────┘  │
                    │                                        │
                    │  Combined Score:                       │
                    │  = (Name × 0.8) + (Location × 0.2)     │
                    └────────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │      PHASE 3: EXACT MATCH CHECK        │
                    ├────────────────────────────────────────┤
                    │  Is query an exact match?              │
                    │                                        │
                    │  ┌─────────┐         ┌─────────────┐   │
                    │  │   NO    │         │    YES      │   │
                    │  │         │         │             │   │
                    │  │ Keep    │         │ Score = 1.0 │   │
                    │  │ hybrid  │         │ + Location  │   │
                    │  │ scores  │         │ × 0.05      │   │
                    │  └─────────┘         └─────────────┘   │
                    │                                        │
                    └────────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │        PHASE 4: FINAL RANKING          │
                    ├────────────────────────────────────────┤
                    │  • Sort by final score (descending)    │
                    │  • Return top K results                │
                    │  • Include metadata (ID, city, state,  │
                    │    count, location_score, name_score)  │
                    └────────────────────┬───────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │       OUTPUT         │
                              ├──────────────────────┤
                              │ 1. Acme Corp         │
                              │    Chicago, IL       │
                              │    Score: 94.4%      │
                              │                      │
                              │ 2. ACME Inc          │
                              │    Houston, TX       │
                              │    Score: 74.2%      │
                              │    ...               │
                              └──────────────────────┘
```

### Location Score Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LOCATION SCORING BREAKDOWN                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: City="NYC", State="NY"                                              │
│  Target: City="New York", State="NY"                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 1: CITY MATCHING (60% of location score)                      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Query City: "NYC"                                                  │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │ Check Variation │──▶ "NYC" found in CITY_VARIATIONS              │   │
│  │  │ Dictionary      │    Maps to: "new york"                         │   │
│  │  └─────────────────┘                                                │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  Target City: "New York" → normalized: "new york"                   │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  Compare: "new york" == "new york" → EXACT MATCH                    │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  City Score: 1.0 (100%)                                             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 2: STATE MATCHING (40% of location score)                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Query State: "NY"                                                  │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │ 2-letter code?  │──▶ YES → Keep as "ny"                          │   │
│  │  └─────────────────┘                                                │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  Target State: "NY" → normalized: "ny"                              │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  Compare: "ny" == "ny" → EXACT MATCH                                │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  State Score: 1.0 (100%)                                            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 3: COMBINE                                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Location Score = (City Score × 0.6) + (State Score × 0.4)          │   │
│  │                 = (1.0 × 0.6) + (1.0 × 0.4)                          │   │
│  │                 = 0.6 + 0.4                                         │   │
│  │                 = 1.0 (100%)                                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Decision Tree: Which Scoring Formula?

```
                         ┌─────────────────────┐
                         │  Is location data   │
                         │  provided (city or  │
                         │  state)?            │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
             ┌──────────┐                    ┌──────────┐
             │    NO    │                    │   YES    │
             └────┬─────┘                    └────┬─────┘
                  │                               │
                  ▼                               ▼
      ┌───────────────────────┐      ┌───────────────────────┐
      │  Score = Name Score   │      │  Is this an EXACT     │
      │  (Pure name matching) │      │  name match?          │
      └───────────────────────┘      └───────────┬───────────┘
                                                 │
                                 ┌───────────────┴───────────────┐
                                 │                               │
                                 ▼                               ▼
                          ┌──────────┐                    ┌──────────┐
                          │    NO    │                    │   YES    │
                          └────┬─────┘                    └────┬─────┘
                               │                               │
                               ▼                               ▼
               ┌───────────────────────────┐   ┌───────────────────────────┐
               │  HYBRID SCORING           │   │  TIE-BREAKER SCORING      │
               │                           │   │                           │
               │  Score = (Name × 0.8)     │   │  Score = 1.0              │
               │        + (Location × 0.2) │   │        + (Location × 0.05)│
               │                           │   │                           │
               │  Location has 20% weight  │   │  Location adds 5% boost   │
               │  to help find best match  │   │  to break ties only       │
               └───────────────────────────┘   └───────────────────────────┘
```

---

## Deep Dive: What Re-Ranking Does

### Why Re-Ranking is Necessary

**The Problem with Pure Semantic Search:**

FAISS semantic search is fast and finds conceptually similar companies, but it has limitations:

| Limitation | Example |
|------------|---------|
| **Ignores spelling** | "Acme" and "ACME" have different embeddings even though they're the same |
| **Misses word order** | "Bank First National" might rank as high as "First National Bank" |
| **No location awareness** | Can't distinguish "Acme Corp (NYC)" from "Acme Corp (LA)" |
| **Overly semantic** | "Apple Inc" might match "Orange Corp" (both fruits) |

**Re-ranking fixes these issues** by applying additional scoring layers to the semantic candidates.

### Re-Ranking Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RE-RANKING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: 100 semantic candidates from FAISS                                  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: STRING SIMILARITY (Weighted Jaccard)                         │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Purpose: Measure actual text overlap, not just semantic meaning      │  │
│  │                                                                       │  │
│  │  How it works:                                                        │  │
│  │  1. Tokenize both strings into words                                  │  │
│  │  2. Weight each word (common words like "Inc" get lower weight)       │  │
│  │  3. Calculate: weighted_intersection / weighted_union                 │  │
│  │                                                                       │  │
│  │  Example:                                                             │  │
│  │  Query: "Acme Corporation"                                            │  │
│  │  Candidate: "Acme Corp"                                               │  │
│  │                                                                       │  │
│  │  Tokens:  ["acme", "corporation"] vs ["acme", "corp"]                 │  │
│  │  Weights: [1.0,    0.3]            vs [1.0,   0.3]                    │  │
│  │           ↑ unique word             ↑ common suffix                   │  │
│  │                                                                       │  │
│  │  Intersection: "acme" (weight 1.0)                                    │  │
│  │  Union: "acme", "corporation", "corp" (weights: 1.0 + 0.3 + 0.3)      │  │
│  │  Score: 1.0 / 1.6 = 0.625 (62.5%)                                     │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: PENALTY SYSTEM                                               │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Purpose: Reduce false positives from overly generous matching        │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  CATEGORY MISMATCH PENALTY                                      │  │  │
│  │  │                                                                 │  │  │
│  │  │  Detects when company types don't match:                        │  │  │
│  │  │  • "Bank" vs "Insurance" → Penalty applied                      │  │  │
│  │  │  • "Hospital" vs "Clinic" → OK (both healthcare)                │  │  │
│  │  │  • "LLC" vs "Inc" → OK (both legal suffixes)                    │  │  │
│  │  │                                                                 │  │  │
│  │  │  Categories: Bank, Insurance, Hospital, University,             │  │  │
│  │  │              Restaurant, Hotel, Law Firm, etc.                  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  PROPER NOUN PENALTY                                            │  │  │
│  │  │                                                                 │  │  │
│  │  │  Detects when key identifying words are different:              │  │  │
│  │  │  • "First National Bank" vs "Second National Bank" → Penalty    │  │  │
│  │  │  • "John Smith LLC" vs "Jane Smith LLC" → Penalty               │  │  │
│  │  │  • "Acme Corp" vs "Acme Inc" → OK (same proper noun)            │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  SHORT STRING PENALTY                                           │  │  │
│  │  │                                                                 │  │  │
│  │  │  Short company names are prone to false matches:                │  │  │
│  │  │  • "ABC" could match "ABC Corp", "ABC Inc", "ABC LLC"           │  │  │
│  │  │  • Extra scrutiny applied to queries < 3 words                  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: NAME SCORE COMBINATION                                       │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Name Score = (Semantic Score × 0.4) + (String Score × 0.6)           │  │
│  │             - Category Penalty                                        │  │
│  │             - Proper Noun Penalty                                     │  │
│  │             - Short String Penalty                                    │  │
│  │                                                                       │  │
│  │  Why 60% string / 40% semantic?                                       │  │
│  │  • String similarity catches exact/near-exact matches                 │  │
│  │  • Semantic similarity catches conceptual matches                     │  │
│  │  • String is weighted higher because company names should match       │  │
│  │    literally, not just conceptually                                   │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: LOCATION SCORING (if provided)                               │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Location Score = (City Score × 0.6) + (State Score × 0.4)            │  │
│  │                                                                       │  │
│  │  City Matching:                                                       │  │
│  │  1. Normalize variations (NYC → New York, SF → San Francisco)         │  │
│  │  2. Exact match → 100%                                                │  │
│  │  3. Fuzzy match (>70% similar) → Similarity %                         │  │
│  │  4. No match → 0%                                                     │  │
│  │                                                                       │  │
│  │  State Matching:                                                      │  │
│  │  1. Normalize (California → CA, texas → TX)                           │  │
│  │  2. Exact match → 100%                                                │  │
│  │  3. Abbreviation match (CA = California) → 100%                       │  │
│  │  4. Fuzzy match (>70% similar) → Similarity %                         │  │
│  │  5. No match → 0%                                                     │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 5: FINAL SCORE COMBINATION                                      │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  If NOT exact name match:                                             │  │
│  │      Final = (Name Score × 0.8) + (Location Score × 0.2)              │  │
│  │                                                                       │  │
│  │  If exact name match:                                                 │  │
│  │      Final = 1.0 + (Location Score × 0.05)                            │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Output: Candidates sorted by Final Score                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Re-Ranking Example: "1st Natinal Bank" in Houston, TX

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BEFORE RE-RANKING (FAISS Semantic Results)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Semantic search returns candidates based on embedding similarity:          │
│                                                                             │
│  Rank  Company                         Semantic Score                       │
│  ────  ───────────────────────────     ──────────────                       │
│  1     1st National Bank (Phoenix)     0.92  ← "1st" matches exactly        │
│  2     First National Bank (NYC)       0.85  ← "National Bank" matches      │
│  3     First National Bank (Houston)   0.85  ← Same semantic similarity     │
│  4     First National Bank (Chicago)   0.85  ← Can't distinguish locations! │
│  5     First Natl Bank of Texas        0.78  ← "Natl" abbreviation          │
│                                                                             │
│  Problem: Houston is ranked #3, not #1!                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AFTER RE-RANKING                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each candidate, calculate:                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  First National Bank (Houston, TX)                                  │   │
│  │                                                                     │   │
│  │  String Similarity: "1st natinal bank" vs "first national bank"     │   │
│  │  • Tokens overlap: "bank" matches                                   │   │
│  │  • "1st" ≈ "first" (recognized as equivalent)                       │   │
│  │  • "natinal" ≈ "national" (fuzzy match ~90%)                        │   │
│  │  • String Score: 0.66 (66%)                                         │   │
│  │                                                                     │   │
│  │  Location Match:                                                    │   │
│  │  • City: "Houston" == "Houston" → 100%                              │   │
│  │  • State: "TX" == "TX" → 100%                                       │   │
│  │  • Location Score: (1.0 × 0.6) + (1.0 × 0.4) = 1.0 (100%)           │   │
│  │                                                                     │   │
│  │  Final Score: (0.66 × 0.8) + (1.0 × 0.2) = 0.728 (72.8%)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1st National Bank (Phoenix, AZ)                                    │   │
│  │                                                                     │   │
│  │  String Similarity: "1st natinal bank" vs "1st national bank"       │   │
│  │  • "1st" matches exactly                                            │   │
│  │  • "natinal" ≈ "national" (fuzzy match)                             │   │
│  │  • String Score: 0.84 (84%) ← Higher than Houston!                  │   │
│  │                                                                     │   │
│  │  Location Match:                                                    │   │
│  │  • City: "Houston" != "Phoenix" → 0%                                │   │
│  │  • State: "TX" != "AZ" → 0%                                         │   │
│  │  • Location Score: 0.0 (0%)                                         │   │
│  │                                                                     │   │
│  │  Final Score: (0.84 × 0.8) + (0.0 × 0.2) = 0.673 (67.3%)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  NEW RANKING:                                                               │
│                                                                             │
│  Rank  Company                         Final Score                          │
│  ────  ───────────────────────────     ───────────                          │
│  1     First National Bank (Houston)   72.8%  ← NOW #1! Location boosted    │
│  2     1st National Bank (Phoenix)     67.3%  ← Better name, wrong location │
│  3     First Natl Bank of Texas        60.2%  ← TX state helps              │
│  4     First National Bank (Chicago)   52.8%  ← No location match           │
│  5     First National Bank (NYC)       52.8%  ← No location match           │
│                                                                             │
│  ✓ Houston correctly ranked #1 despite slightly lower name similarity!     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Insights: Why Re-Ranking Works

| Issue | How Re-Ranking Solves It |
|-------|--------------------------|
| **FAISS can't see typos** | String similarity catches "Natinal" ≈ "National" |
| **FAISS ignores location** | Location score boosts the right geographic match |
| **Semantic over-matches** | Penalties reduce scores for category/noun mismatches |
| **Multiple same-name companies** | Location breaks ties between identical names |
| **Abbreviations confuse embeddings** | String matching handles "1st" = "First", "Corp" = "Corporation" |

### The 80/20 Name/Location Split

Why is location only 20% of the score?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RATIONALE FOR 80% NAME / 20% LOCATION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Scenario: Query "Acme Corp" in "Houston, TX"                               │
│                                                                             │
│  Candidate A: "Acme Corporation" in Chicago, IL                             │
│  Candidate B: "XYZ Industries" in Houston, TX                               │
│                                                                             │
│  If location were 50%:                                                      │
│  • A: (0.9 × 0.5) + (0.0 × 0.5) = 0.45                                      │
│  • B: (0.1 × 0.5) + (1.0 × 0.5) = 0.55  ← WRONG! XYZ wins on location      │
│                                                                             │
│  With 80/20 split:                                                          │
│  • A: (0.9 × 0.8) + (0.0 × 0.2) = 0.72  ← CORRECT! Acme wins on name       │
│  • B: (0.1 × 0.8) + (1.0 × 0.2) = 0.28                                      │
│                                                                             │
│  PRINCIPLE: Company name is the PRIMARY identifier.                         │
│             Location is a DISAMBIGUATION tool, not a filter.                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Example: Scenario 4 Walkthrough

```
Query: "First National Bank" in "LA, CA"

┌─────────────────────────────────────────────────────────────────────────────┐
│  CANDIDATE PROCESSING                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Candidate 1: First National Bank (Los Angeles, CA)                         │
│  ├─ Name match: "first national bank" == "first national bank" → EXACT     │
│  ├─ Name Score: 1.0 (100%)                                                  │
│  ├─ City: "LA" → "los angeles" == "los angeles" → Match (1.0)               │
│  ├─ State: "CA" → "ca" == "ca" → Match (1.0)                                │
│  ├─ Location Score: (1.0 × 0.6) + (1.0 × 0.4) = 1.0 (100%)                  │
│  └─ Final Score: 1.0 + (1.0 × 0.05) = 1.05 (105%) ◄── WINNER                │
│                                                                             │
│  Candidate 2: First National Bank (Houston, TX)                             │
│  ├─ Name match: "first national bank" == "first national bank" → EXACT     │
│  ├─ Name Score: 1.0 (100%)                                                  │
│  ├─ City: "LA" → "los angeles" != "houston" → No match (0.0)                │
│  ├─ State: "CA" → "ca" != "tx" → No match (0.0)                             │
│  ├─ Location Score: (0.0 × 0.6) + (0.0 × 0.4) = 0.0 (0%)                    │
│  └─ Final Score: 1.0 + (0.0 × 0.05) = 1.0 (100%)                            │
│                                                                             │
│  [Similar for Chicago, IL and New York, NY - all get 100%]                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  FINAL RANKING:                                                             │
│  1. First National Bank (Los Angeles, CA) - 105%                            │
│  2. First National Bank (Chicago, IL) - 100%                                │
│  3. First National Bank (Houston, TX) - 100%                                │
│  4. First National Bank (New York, NY) - 100%                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Test Data

The demo uses 17 companies across multiple locations:

| ID | Company Name | City | State | Count |
|----|--------------|------|-------|-------|
| 1001 | First National Bank | New York | NY | 500 |
| 1002 | First National Bank | Houston | TX | 350 |
| 1003 | First National Bank | Los Angeles | CA | 275 |
| 1004 | First National Bank | Chicago | IL | 400 |
| 2001 | First Natl Bank of Texas | Dallas | TX | 200 |
| 2002 | First National Bank & Trust | Miami | FL | 150 |
| 2003 | 1st National Bank | Phoenix | AZ | 100 |
| 3001 | Global Tech Solutions | San Francisco | CA | 600 |
| 3002 | Global Tech Solutions | Seattle | WA | 450 |
| 3003 | Global Tech Solutions | Austin | TX | 300 |
| 3004 | Global Technology Solutions Inc | Boston | MA | 250 |
| 3005 | GlobalTech Solutions LLC | Denver | CO | 175 |
| 4001 | Acme Corporation | New York | NY | 800 |
| 4002 | Acme Corporation | Los Angeles | CA | 650 |
| 4003 | Acme Corp | Chicago | IL | 400 |
| 4004 | ACME Inc | Houston | TX | 300 |
| 4005 | Acme Industries | Detroit | MI | 225 |

---

## Scenario 1: Typo in Company Name

### Query
- **Company:** `"1st Natinal Bank"` (note the typo: "Natinal" instead of "National")
- **City:** `"Houston"`
- **State:** `"TX"`

### Expected Behavior
The system should find "First National Bank" in Houston, TX despite the typo in the query.

### Results

| Rank | ID | Company Name | Score | City | State | Location Score | Name Score |
|------|-----|--------------|-------|------|-------|----------------|------------|
| 1 | 1002 | First National Bank | 72.8% | Houston | TX | 100.0% | 66.0% |
| 2 | 2003 | 1st National Bank | 67.3% | Phoenix | AZ | 0.0% | 84.1% |
| 3 | 2001 | First Natl Bank of Texas | 60.2% | Dallas | TX | 40.0% | 65.3% |
| 4 | 1004 | First National Bank | 52.8% | Chicago | IL | 0.0% | 66.0% |
| 5 | 1003 | First National Bank | 52.8% | Los Angeles | CA | 0.0% | 66.0% |

### How It Works

1. **Fuzzy Name Matching:** The typo "Natinal" still has high similarity to "National" through the semantic embeddings and string similarity algorithms. "1st" is also recognized as similar to "First".

2. **Location Boost:** The Houston, TX entry gets a 100% location score because the city and state match exactly.

3. **Score Calculation:** 
   - Name Score: 66.0% (due to typo reducing similarity)
   - Location Score: 100.0% (exact match)
   - Final: `(0.66 × 0.8) + (1.0 × 0.2) = 0.528 + 0.2 = 72.8%`

4. **Why "1st National Bank" (Phoenix) is #2:** It has a higher name score (84.1%) because "1st" matches exactly, but with 0% location score, its final score is lower: `0.841 × 0.8 = 67.3%`

### Key Insight
**Location can compensate for imperfect name matches.** Even though Phoenix had a better name match, Houston won because the location data correctly identified the target.

---

## Scenario 2: Abbreviated Company Name

### Query
- **Company:** `"First Natl Bank"` (abbreviated "Natl" instead of "National")
- **City:** (not provided)
- **State:** `"TX"`

### Expected Behavior
Texas locations should rank higher than non-Texas locations.

### Results

| Rank | ID | Company Name | Score | City | State | Location Score | Name Score |
|------|-----|--------------|-------|------|-------|----------------|------------|
| 1 | 2001 | First Natl Bank of Texas | 82.4% | Dallas | TX | 40.0% | 93.0% |
| 2 | 1002 | First National Bank | 73.0% | Houston | TX | 40.0% | 81.3% |
| 3 | 1004 | First National Bank | 65.0% | Chicago | IL | 0.0% | 81.3% |
| 4 | 1003 | First National Bank | 65.0% | Los Angeles | CA | 0.0% | 81.3% |
| 5 | 1001 | First National Bank | 65.0% | New York | NY | 0.0% | 81.3% |

### How It Works

1. **Abbreviation Recognition:** "Natl" is recognized as similar to both "Natl" (exact) and "National" (semantic similarity).

2. **State-Only Matching:** When only state is provided (no city), location score is calculated on state alone:
   - Texas entries get 40% location score (state weight is 40% of total location score)
   - Non-Texas entries get 0% location score

3. **Score Calculation for Dallas (TX):**
   - Name Score: 93.0% ("First Natl Bank" closely matches "First Natl Bank of Texas")
   - Location Score: 40.0% (state matches, no city provided)
   - Final: `(0.93 × 0.8) + (0.4 × 0.2) = 0.744 + 0.08 = 82.4%`

### Key Insight
**State-only queries still provide useful disambiguation.** Texas entries ranked 1st and 2nd, while identical "First National Bank" entries in other states ranked lower.

---

## Scenario 3: City Abbreviation + Full State Name

### Query
- **Company:** `"Global Tech"`
- **City:** `"SF"` (abbreviation for San Francisco)
- **State:** `"California"` (full state name instead of "CA")

### Expected Behavior
San Francisco, CA should rank #1 because the system normalizes city abbreviations and state names.

### Results

| Rank | ID | Company Name | Score | City | State | Location Score | Name Score |
|------|-----|--------------|-------|------|-------|----------------|------------|
| 1 | 3001 | Global Tech Solutions | 94.4% | San Francisco | CA | 100.0% | 93.0% |
| 2 | 3003 | Global Tech Solutions | 74.4% | Austin | TX | 0.0% | 93.0% |
| 3 | 3002 | Global Tech Solutions | 74.4% | Seattle | WA | 0.0% | 93.0% |
| 4 | 3004 | Global Technology Solutions Inc | 33.1% | Boston | MA | 0.0% | 41.4% |
| 5 | 3005 | GlobalTech Solutions LLC | 30.6% | Denver | CO | 0.0% | 38.2% |

### How It Works

1. **City Normalization:** The system maintains a city variation dictionary:
   ```python
   CITY_VARIATIONS = {
       "sf": "san francisco",
       "nyc": "new york",
       "la": "los angeles",
       "philly": "philadelphia",
       # ... more variations
   }
   ```
   "SF" is normalized to "San Francisco" before comparison.

2. **State Normalization:** The system maintains a state abbreviation map:
   ```python
   STATE_ABBREV = {
       "california": "ca",
       "texas": "tx",
       # ... all 50 states
   }
   ```
   "California" is normalized to "CA" before comparison.

3. **Perfect Location Match:** After normalization, "SF, California" matches "San Francisco, CA" with 100% location score.

### Key Insight
**The system handles common abbreviations and variations gracefully.** Users don't need to know the exact format used in the database.

---

## Scenario 4: Multiple Exact Name Matches - Location as Tie-Breaker

### Query
- **Company:** `"First National Bank"` (exact match exists 4 times)
- **City:** `"LA"` (abbreviation for Los Angeles)
- **State:** `"CA"`

### Expected Behavior
Los Angeles, CA should rank #1 even though all four "First National Bank" entries have identical 100% name scores.

### Results

| Rank | ID | Company Name | Score | City | State | Location Score | Name Score |
|------|-----|--------------|-------|------|-------|----------------|------------|
| 1 | 1003 | First National Bank | **105.0%** | Los Angeles | CA | 100.0% | 100.0% |
| 2 | 1004 | First National Bank | 100.0% | Chicago | IL | 0.0% | 100.0% |
| 3 | 1002 | First National Bank | 100.0% | Houston | TX | 0.0% | 100.0% |
| 4 | 1001 | First National Bank | 100.0% | New York | NY | 0.0% | 100.0% |
| 5 | 2003 | 1st National Bank | 72.0% | Phoenix | AZ | 0.0% | 90.0% |

### How It Works

1. **Exact Match Detection:** All four "First National Bank" entries are recognized as exact name matches (100% name score).

2. **City Abbreviation:** "LA" is normalized to "Los Angeles" via the city variations dictionary.

3. **Location Tie-Breaker:** For exact name matches, location provides a 5% boost:
   - LA entry: `1.0 + (1.0 × 0.05) = 105%`
   - Other entries: `1.0 + (0.0 × 0.05) = 100%`

### Key Insight
**Location is a tie-breaker, not a primary filter.** When company names match exactly, location adds a small boost to differentiate between locations. This prevents location from incorrectly penalizing an exact name match.

---

## Scenario 5: City Slang + Full State Name

### Query
- **Company:** `"Acme"` (partial company name)
- **City:** `"Chi-Town"` (slang for Chicago)
- **State:** `"Illinois"` (full state name)

### Expected Behavior
Chicago, IL "Acme Corp" should rank highest because the system recognizes city slang.

### Results

| Rank | ID | Company Name | Score | City | State | Location Score | Name Score |
|------|-----|--------------|-------|------|-------|----------------|------------|
| 1 | 4003 | Acme Corp | 94.4% | Chicago | IL | 100.0% | 93.0% |
| 2 | 4004 | ACME Inc | 74.2% | Houston | TX | 0.0% | 92.8% |
| 3 | 4002 | Acme Corporation | 72.5% | Los Angeles | CA | 0.0% | 90.7% |
| 4 | 4001 | Acme Corporation | 72.5% | New York | NY | 0.0% | 90.7% |
| 5 | 4005 | Acme Industries | 72.1% | Detroit | MI | 0.0% | 90.1% |

### How It Works

1. **Partial Name Matching:** "Acme" matches well against all Acme variations through semantic similarity:
   - "Acme Corp" → 93.0%
   - "ACME Inc" → 92.8%
   - "Acme Corporation" → 90.7%

2. **City Slang Recognition:**
   ```python
   CITY_VARIATIONS = {
       "chi-town": "chicago",
       # ... other variations
   }
   ```
   "Chi-Town" is normalized to "Chicago".

3. **State Normalization:** "Illinois" is normalized to "IL".

4. **Location Boost:** Chicago, IL gets 100% location score, boosting it to the top.

### Key Insight
**The system handles colloquial city names and slang.** This is important for user-friendly search where people might type informal location names.

---

## Scenario 6: No Location Provided (Control)

### Query
- **Company:** `"First National Bank"` (exact match exists 4 times)
- **City:** (not provided)
- **State:** (not provided)

### Expected Behavior
All four exact matches should be tied with the same score - there's no way to disambiguate without location.

### Results

| Rank | ID | Company Name | Score | City | State | Location Score | Name Score |
|------|-----|--------------|-------|------|-------|----------------|------------|
| 1 | 1004 | First National Bank | 100.0% | Chicago | IL | 0.0% | 100.0% |
| 2 | 1003 | First National Bank | 100.0% | Los Angeles | CA | 0.0% | 100.0% |
| 3 | 1002 | First National Bank | 100.0% | Houston | TX | 0.0% | 100.0% |
| 4 | 1001 | First National Bank | 100.0% | New York | NY | 0.0% | 100.0% |
| 5 | 2003 | 1st National Bank | 90.0% | Phoenix | AZ | 0.0% | 90.0% |

### How It Works

1. **No Location Scoring:** When city and state are not provided, `use_location` is False, so no location scoring is applied.

2. **Pure Name Matching:** Results are ranked purely by name similarity.

3. **Tie Situation:** All four "First National Bank" entries have identical 100% scores. The ordering within the tie is arbitrary (based on database order).

### Key Insight
**This is the control scenario** showing why location data is valuable. Without it, you cannot distinguish between companies with the same name in different locations.

---

## Scenario 7: Same Query WITH Location

### Query
- **Company:** `"First National Bank"` (same as Scenario 6)
- **City:** `"NYC"` (abbreviation for New York City)
- **State:** `"NY"`

### Expected Behavior
New York should clearly rank #1 now that location is provided.

### Results

| Rank | ID | Company Name | Score | City | State | Location Score | Name Score |
|------|-----|--------------|-------|------|-------|----------------|------------|
| 1 | 1001 | First National Bank | **105.0%** | New York | NY | 100.0% | 100.0% |
| 2 | 1004 | First National Bank | 100.0% | Chicago | IL | 0.0% | 100.0% |
| 3 | 1003 | First National Bank | 100.0% | Los Angeles | CA | 0.0% | 100.0% |
| 4 | 1002 | First National Bank | 100.0% | Houston | TX | 0.0% | 100.0% |
| 5 | 2003 | 1st National Bank | 72.0% | Phoenix | AZ | 0.0% | 90.0% |

### How It Works

1. **NYC Normalization:** "NYC" is normalized to "New York" via the city variations dictionary.

2. **Perfect Location Match:** New York, NY matches the query location with 100% score.

3. **Location Tie-Breaker Applied:** 
   - New York: `1.0 + (1.0 × 0.05) = 105%`
   - Others: `1.0 + (0.0 × 0.05) = 100%`

### Key Insight
**Comparing Scenario 6 vs 7 demonstrates the value of location data.** The same query with location added clearly identifies the correct target company.

---

## Summary: Location Normalization Features

### City Variations Supported
| Input | Normalized To |
|-------|---------------|
| NYC | New York |
| LA | Los Angeles |
| SF | San Francisco |
| Philly | Philadelphia |
| Chi-Town | Chicago |
| DC | Washington |

### State Formats Supported
| Input | Normalized To |
|-------|---------------|
| California | CA |
| california | CA |
| CA | CA |
| ca | CA |

### Fuzzy Matching
For cities and states not in the variation dictionaries, the system uses `difflib.SequenceMatcher` with a 70% similarity threshold:
- "Sant Francisco" → matches "San Francisco" (typo tolerance)
- "Houstan" → matches "Houston" (typo tolerance)

---

## Running the Demo

```bash
cd E:\projects\FineTuner\FineTuner
python demo_fuzzy_location_match.py
```

This will display all seven scenarios with full results and scoring breakdowns.

