# Company Matching Control Set - Ultra Detailed Report

**Generated:** 2025-12-28 06:50:00

---

## Project Objectives & Scenarios

This project provides high-precision company matching across diverse scenarios:

### Exact Match
Perfect character-for-character matching.
- **Example:** `"IBM" → "IBM"`

### Acronym Expansion
Literal mapping of initials to full name.
- **Example:** `"IBM" → "International Business Machines"`

### Acronym Reverse
Mapping full name back to its acronym.
- **Example:** `"International Business Machines" → "IBM"`

### Subsequence Acronym
Handling initials found within word starts.
- **Example:** `"IBM" → "International Bureau of Management"`

### Typo Handling
Tolerance for missing/extra characters.
- **Example:** `"Microsft" → "Microsoft"`

### Abbreviation Sync
Equating variations like Corp/Corporation.
- **Example:** `"Acme Corp" → "Acme Corporation"`

### Plural Handling
Handling singular/plural variations.
- **Example:** `"Machine" ↔ "Machines"`

### Partial Match
Finding the entity within a longer string.
- **Example:** `"Acme" → "Acme Logistics LLC"`

### Word Order
Matches despite rearranged words.
- **Example:** `"First National Bank" → "Bank First National"`

### Noise Handling
Filtering prefixes like 'X DO NOT USE'.
- **Example:** `"X DO NOT USE - FORD" → "Ford Motor Company"`

### Semantic Logic
Related business concepts and synonyms.
- **Example:** `"Software" → "Systems"`

### Suffix Variation
Handling LLC, Inc, Corp, Ltd variations.
- **Example:** `"Acme Inc" → "Acme LLC"`

### Multi-Word Overlap
Handling companies with shared names.
- **Example:** `"The Coca Cola Co" → "Coca-Cola Enterprises"`

---

## Acronym Fidelity Score (Thorough Explanation)

The **Acronym Fidelity Score** is an algorithmic measure of how precisely a name expands an acronym. Unlike general semantic matching, it strictly validates the letter pattern against word starts.

### Scoring Patterns

| Fidelity | Relationship | Example Match Pattern |
|----------|--------------|-----------------------|
| **1.00** | Perfect Expansion | **I**nternational **B**usiness **M**achines → **IBM** |
| **0.95** | Prefix Expansion | **I**nternational **B**usiness **M**achines **C**orp → **IBM** |
| **0.90** | Subsequence | **I**nternational **B**ureau of **M**anagement → **IBM** |
| **0.70** | Word Collision | **I**B**M** Solutions → **IBM** (Internal letters matching sequence) |
| **0.65** | Partial Word | **I**ntercontinental **B**anking **M**etrics → **IBM** |
| **0.40** | Fuzzy/Broken | Some initials match, but order or significant words are skipped. |

### Detailed Pattern Logic (Example: IBM)
```
Query: "IBM"
Target: "International Business Machines"

1. Extract Words: [International, Business, Machines]
2. Extract Initials: [I, B, M]
3. Pattern Verification:
   - Initials match "IBM" exactly? YES
   - Words contain internal acronym letters? NO
4. Result: 1.00 Fidelity (Highest Confidence)
```

---

## 1. Query: `PDMA Association`

✅ **Exact Match Found:** `PDMA Association` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | PDMA Association | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Association Headquarters-PDMA | 93.00% | 0.950 | 0.706 | 0.00 | High word-for-word overlap. |
| 3 | PA | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'PA'. |
| 4 | PDMA Alliance | 81.50% | 0.850 | 0.733 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | PDMA | 71.30% | 0.630 | 0.905 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | PDMA inc | 68.90% | 0.630 | 0.827 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | PDMA Corporation | 68.50% | 0.630 | 0.813 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | PDS User Group Association | 65.40% | 0.637 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | PDS Users Group Association | 65.30% | 0.637 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | PDA PARTNERS | 36.10% | 0.233 | 0.659 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | PDA ALC | 35.80% | 0.235 | 0.646 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | PD Properties | 35.10% | 0.200 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | PDC Affiliates | 35.00% | 0.225 | 0.642 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | PDC Group Services | 31.50% | 0.173 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | PD Symposium | 30.70% | 0.161 | 0.650 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: PDMA Association**
> None

**Rank #2: Association Headquarters-PDMA**
> None

**Rank #3: PA**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: PDMA Association (100.00%)                                      │
│  Match #2: Association Headquarters-PDMA (93.00%)                          │
│                                                                            │
│  Score Difference: 7.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 2. Query: `Nicolas/Sanchez Wedding`

✅ **Exact Match Found:** `Nicolas/Sanchez Wedding` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Nicolas/Sanchez Wedding | 100.00% | 1.000 | 0.879 | 0.00 | Perfect character-for-character match. |
| 2 | Sanchez Wedding | 85.10% | 0.787 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 3 | Garcia Sanchez Wedding | 80.80% | 0.744 | 0.956 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Castillo Sanchez Wedding | 79.90% | 0.744 | 0.928 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | Sanchez Wedding Reception | 79.00% | 0.744 | 0.899 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Rodriguez/ Sanchez Wedding | 77.30% | 0.744 | 0.842 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Sanchez/Flores Wedding | 76.30% | 0.744 | 0.806 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Gibson Sanchez Wedding | 76.20% | 0.744 | 0.805 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Sanchez and Alfonso Wedding | 76.20% | 0.744 | 0.804 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Sanchez/Ramirez Wedding | 76.00% | 0.744 | 0.800 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Sanchez/Puerto Wedding | 75.80% | 0.744 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Datzman Sanchez Wedding | 75.40% | 0.744 | 0.779 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Sanchez / Bryan Wedding | 75.40% | 0.744 | 0.777 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Sanchez/Fuentes Wedding | 75.10% | 0.744 | 0.768 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | SANCHEZ-RAMIREZ WEDDING RECEPTION | 75.00% | 0.744 | 0.764 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Nicolas/Sanchez Wedding**
> None

**Rank #2: Sanchez Wedding**
> None

**Rank #3: Garcia Sanchez Wedding**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Nicolas/Sanchez Wedding (100.00%)                               │
│  Match #2: Sanchez Wedding (85.10%)                                        │
│                                                                            │
│  Score Difference: 14.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 3. Query: `Kehilat Ariel Synagogue`

✅ **Exact Match Found:** `Kehilat Ariel Synagogue` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Kehilat Ariel Synagogue | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Kehilat Ariel Messianic Synagogue | 93.00% | 0.950 | 0.817 | 0.00 | High word-for-word overlap. |
| 3 | KAS | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'KAS'. |
| 4 | Kehilat Ariel | 73.00% | 0.742 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Kehilath Israel Synagogue | 59.60% | 0.475 | 0.878 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Beth Israel Synagogue | 55.50% | 0.475 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Synagogue 3000 Organization | 54.90% | 0.475 | 0.722 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Beth El Synagogue | 54.70% | 0.475 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Bet Torah Synagogue | 54.20% | 0.475 | 0.697 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Temple Sinai Synagogue | 53.90% | 0.475 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Bet Aviv Synagogue | 53.90% | 0.475 | 0.688 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Brooklyn Heights Synagogue | 53.90% | 0.475 | 0.687 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Beth Shalom Synagogue | 53.50% | 0.475 | 0.676 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | North Shore Synagogue | 53.30% | 0.475 | 0.669 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Beth Sholom Synagogue | 53.20% | 0.475 | 0.666 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Kehilat Ariel Synagogue**
> None

**Rank #2: Kehilat Ariel Messianic Synagogue**
> None

**Rank #3: KAS**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Kehilat Ariel Synagogue (100.00%)                               │
│  Match #2: Kehilat Ariel Messianic Synagogue (93.00%)                      │
│                                                                            │
│  Score Difference: 7.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 4. Query: `Next Level Events`

✅ **Exact Match Found:** `Next Level Events` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Next Level Events | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Next Level Events Inc | 95.10% | 1.000 | 0.837 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | Next Level Plus Events | 93.00% | 0.950 | 0.868 | 0.00 | High word-for-word overlap. |
| 4 | Next Level Now | 84.30% | 0.883 | 0.748 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Next Level Games | 83.80% | 0.883 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Next Level Event Design | 83.80% | 0.883 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Next Level Fairs | 83.50% | 0.883 | 0.723 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Next Level Promotions | 83.40% | 0.883 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Next Level Performance | 83.20% | 0.883 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Level Up Events | 80.50% | 0.773 | 0.881 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Higher Level Events | 78.80% | 0.773 | 0.824 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Level 10 Events | 77.90% | 0.773 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Level 9 Events | 77.30% | 0.773 | 0.774 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Pro Level Events | 77.10% | 0.773 | 0.767 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Next View Events | 76.80% | 0.773 | 0.757 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Next Level Events**
> None

**Rank #2: Next Level Events Inc**
> None

**Rank #3: Next Level Plus Events**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Next Level Events (100.00%)                                     │
│  Match #2: Next Level Events Inc (95.10%)                                  │
│                                                                            │
│  Score Difference: 4.90%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 5. Query: `Site Foundation Golf Tournament`

✅ **Exact Match Found:** `Site Foundation Golf Tournament` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Site Foundation Golf Tournament | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Hearth Foundation Golf Tournament | 80.10% | 0.787 | 0.831 | 0.00 | High word-for-word overlap. |
| 3 | Hi Kid Foundation Golf Tournament | 77.70% | 0.787 | 0.753 | 0.00 | High word-for-word overlap. |
| 4 | WSU Athletic Foundation Golf Tournament | 77.70% | 0.787 | 0.753 | 0.00 | High word-for-word overlap. |
| 5 | Natl Football Foundation Golf Tournament | 77.40% | 0.787 | 0.744 | 0.00 | High word-for-word overlap. |
| 6 | Tournament Golf Foundation Incorporated | 74.60% | 0.689 | 0.879 | 0.00 | High word-for-word overlap. |
| 7 | Golf League Amateur Golf Tournament | 74.20% | 0.744 | 0.739 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Midwest Classic Golf Tournament | 74.20% | 0.744 | 0.737 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | World Amateur Golf Tournament | 74.20% | 0.744 | 0.737 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Desert Invitational Golf Tournament | 74.00% | 0.744 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Western States Invitational Golf Tournament | 74.00% | 0.744 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | College Prep Golf Tournament | 73.90% | 0.744 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Caribbean Golf Invitational Foundation | 73.90% | 0.744 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | National Christian Foundation Golf | 73.70% | 0.744 | 0.720 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Stanford Intercollegiate Golf Tournament | 73.50% | 0.744 | 0.716 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Site Foundation Golf Tournament**
> None

**Rank #2: Hearth Foundation Golf Tournament**
> None

**Rank #3: Hi Kid Foundation Golf Tournament**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Site Foundation Golf Tournament (100.00%)                       │
│  Match #2: Hearth Foundation Golf Tournament (80.10%)                      │
│                                                                            │
│  Score Difference: 19.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 6. Query: `Interim WG Meeting - BIER`

✅ **Exact Match Found:** `Interim WG Meeting - BIER` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Interim WG Meeting - BIER | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Bi Annual Meeting | 52.30% | 0.394 | 0.825 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | BI Meeting | 49.50% | 0.315 | 0.916 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Bim Object Meeting | 49.40% | 0.394 | 0.726 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | American Biz Meeting | 49.10% | 0.394 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | BIA OIEP Meeting | 48.70% | 0.394 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | ETW Meeting Management | 48.60% | 0.394 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | BIE ELO MEETING | 48.60% | 0.394 | 0.700 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | CMG Board meeting | 48.30% | 0.394 | 0.692 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | RFW Meeting Planners | 48.20% | 0.394 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | HRW Meeting Services | 48.20% | 0.394 | 0.688 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Biz meeting | 46.50% | 0.315 | 0.816 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | WMS Meeting | 46.30% | 0.315 | 0.809 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | HRW Meeting | 45.10% | 0.315 | 0.767 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | AB Meeting | 44.10% | 0.315 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Interim WG Meeting - BIER**
> None

**Rank #2: Bi Annual Meeting**
> None

**Rank #3: BI Meeting**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Interim WG Meeting - BIER (100.00%)                             │
│  Match #2: Bi Annual Meeting (52.30%)                                      │
│                                                                            │
│  Score Difference: 47.70%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 7. Query: `DermaQuest Inc`

✅ **Exact Match Found:** `DermaQuest Inc` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | DermaQuest Inc | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Dermaquest, Incorporated | 94.20% | 1.000 | 0.808 | 0.00 | High word-for-word overlap. |
| 3 | DI | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'DI'. |
| 4 | Dermaquest Skin Care | 85.00% | 0.900 | 0.734 | 0.00 | High word-for-word overlap. |
| 5 | Dermaquest Skin Therapy | 83.60% | 0.900 | 0.688 | 0.00 | High word-for-word overlap. |
| 6 | Derma E | 43.60% | 0.318 | 0.712 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Perquest Inc | 42.90% | 0.350 | 0.614 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | EQuest | 42.80% | 0.337 | 0.641 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | DermaPure | 42.80% | 0.332 | 0.653 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Mapquest | 42.00% | 0.350 | 0.583 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Dermapen | 41.90% | 0.300 | 0.695 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | TrialQuest | 41.70% | 0.315 | 0.656 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | AgQuest | 41.60% | 0.318 | 0.644 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Derma USA | 41.30% | 0.332 | 0.602 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Alquest | 40.80% | 0.318 | 0.619 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: DermaQuest Inc**
> None

**Rank #2: Dermaquest, Incorporated**
> None

**Rank #3: DI**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: DermaQuest Inc (100.00%)                                        │
│  Match #2: Dermaquest, Incorporated (94.20%)                               │
│                                                                            │
│  Score Difference: 5.80%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 8. Query: `Ellwood Group Inc`

✅ **Exact Match Found:** `Ellwood Group Inc` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Ellwood Group Inc | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | EGI | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'EGI'. |
| 3 | Ellwood Associates | 86.20% | 0.900 | 0.774 | 0.00 | High word-for-word overlap. |
| 4 | Ellwood Community Church | 84.60% | 0.900 | 0.719 | 0.00 | High word-for-word overlap. |
| 5 | Delwood | 48.50% | 0.375 | 0.742 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Sellwood | 47.40% | 0.375 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Linwood Group | 46.70% | 0.321 | 0.808 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Wildwood Group | 46.60% | 0.300 | 0.853 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | Millwood Inc | 46.20% | 0.360 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Kenwood Group | 45.00% | 0.321 | 0.749 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Reignwood Group | 44.70% | 0.281 | 0.834 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Lockwood Group | 43.80% | 0.300 | 0.760 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Marwood Group | 43.40% | 0.257 | 0.846 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | WILDWOOD | 43.20% | 0.300 | 0.741 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Oakwood Group | 42.50% | 0.257 | 0.816 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Ellwood Group Inc**
> None

**Rank #2: EGI**
> None

**Rank #3: Ellwood Associates**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Ellwood Group Inc (100.00%)                                     │
│  Match #2: EGI (90.00%)                                                    │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 9. Query: `American Miniature Horse Registry`

✅ **Exact Match Found:** `American Miniature Horse Registry` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | American Miniature Horse Registry | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | American Miniature Horse Association | 83.50% | 0.844 | 0.815 | 0.00 | High word-for-word overlap. |
| 3 | American Miniature Horse Association Headquarters | 81.90% | 0.844 | 0.761 | 0.00 | High word-for-word overlap. |
| 4 | Miniature Horse & Pony Show | 73.10% | 0.744 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | American Horse Show Association | 70.70% | 0.744 | 0.620 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | American Youth Horse Council | 70.10% | 0.744 | 0.601 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | American Hackney Horse Society | 70.10% | 0.744 | 0.600 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | American Horse Council | 64.50% | 0.651 | 0.632 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | AMERICAN HORSE SCHOOL | 64.50% | 0.651 | 0.631 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | American Horse Publication | 63.80% | 0.651 | 0.610 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | PPHRNA Horse Registry | 63.80% | 0.651 | 0.607 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | American Horse Publications | 63.60% | 0.651 | 0.600 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Charity Fair Horse Show | 54.80% | 0.487 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Thoroughbred Horse Show Association | 52.90% | 0.487 | 0.627 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | INDIANAPOLIS CHARITY HORSE SHOW | 52.30% | 0.487 | 0.607 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: American Miniature Horse Registry**
> None

**Rank #2: American Miniature Horse Association**
> None

**Rank #3: American Miniature Horse Association Headquarters**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: American Miniature Horse Registry (100.00%)                     │
│  Match #2: American Miniature Horse Association (83.50%)                   │
│                                                                            │
│  Score Difference: 16.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 10. Query: `YADA ENTERPRISES, INC`

✅ **Exact Match Found:** `YADA ENTERPRISES, INC` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | YADA ENTERPRISES, INC | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Yada Yada | 90.50% | 0.900 | 0.916 | 0.00 | High word-for-word overlap. |
| 3 | Yama Enterprises | 52.90% | 0.337 | 0.975 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Yacada | 47.60% | 0.360 | 0.748 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Yadea Group | 46.80% | 0.375 | 0.686 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Yasuda Corporation Limited | 46.60% | 0.360 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Yama | 45.60% | 0.337 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Yamagada Corporation | 45.40% | 0.300 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Yata LLC | 45.10% | 0.337 | 0.716 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Yapa | 45.10% | 0.337 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | YaMa Group | 44.90% | 0.337 | 0.709 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Yara | 44.40% | 0.337 | 0.692 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | YABA | 44.20% | 0.337 | 0.687 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Ya | 44.10% | 0.300 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Yamwa Corporation | 43.20% | 0.300 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: YADA ENTERPRISES, INC**
> None

**Rank #2: Yada Yada**
> None

**Rank #3: Yama Enterprises**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: YADA ENTERPRISES, INC (100.00%)                                 │
│  Match #2: Yada Yada (90.50%)                                              │
│                                                                            │
│  Score Difference: 9.50%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 11. Query: `Seafood Nutrition Partnership`

✅ **Exact Match Found:** `Seafood Nutrition Partnership` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Seafood Nutrition Partnership | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Sustainable Seafood Partnership | 81.30% | 0.810 | 0.821 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | SEAFOOD NUTRITION | 76.50% | 0.681 | 0.964 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Seafood Products Association | 59.50% | 0.528 | 0.752 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Seafood Choices Alliance | 59.50% | 0.528 | 0.751 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Food Industries Seafood Association | 59.20% | 0.528 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Seafood Choices Alliances | 58.80% | 0.528 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Seafood Business Solutions | 57.10% | 0.528 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | 21st Century Seafood | 56.90% | 0.528 | 0.666 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Global Sustainable Seafood Initiative | 56.90% | 0.528 | 0.664 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Solutions for Seafood | 56.80% | 0.528 | 0.663 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | SeaFood Industry Research Fund | 56.70% | 0.528 | 0.659 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Global Seafood Alliance | 56.60% | 0.528 | 0.655 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Global  Seafood Alliance | 56.60% | 0.528 | 0.655 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | BC Seafood Alliance | 56.40% | 0.528 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Seafood Nutrition Partnership**
> None

**Rank #2: Sustainable Seafood Partnership**
> None

**Rank #3: SEAFOOD NUTRITION**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Seafood Nutrition Partnership (100.00%)                         │
│  Match #2: Sustainable Seafood Partnership (81.30%)                        │
│                                                                            │
│  Score Difference: 18.70%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 12. Query: `AVIAKOMPANIYA SIBIR, PAO`

✅ **Exact Match Found:** `AVIAKOMPANIYA SIBIR, PAO` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | AVIAKOMPANIYA SIBIR, PAO | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | ASP | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'ASP'. |
| 3 | AVIAKOMPANIYA MIZHNARODNI AVIA | 59.00% | 0.528 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Faizan Kabir | 33.90% | 0.170 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Ebira Vonya International | 33.90% | 0.188 | 0.691 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Shibir  Desai | 33.40% | 0.148 | 0.768 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Kabira | 33.30% | 0.093 | 0.893 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Salaha Kabir | 32.30% | 0.148 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Avyaya Integrated | 32.20% | 0.167 | 0.682 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Kabir Capital | 31.60% | 0.144 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Avyaya | 31.50% | 0.093 | 0.832 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Camp Kanya | 31.50% | 0.157 | 0.682 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Darya Varia | 31.50% | 0.131 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Albireo AB | 31.20% | 0.135 | 0.725 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Kabira Technology | 30.70% | 0.111 | 0.764 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: AVIAKOMPANIYA SIBIR, PAO**
> None

**Rank #2: ASP**
> None

**Rank #3: AVIAKOMPANIYA MIZHNARODNI AVIA**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: AVIAKOMPANIYA SIBIR, PAO (100.00%)                              │
│  Match #2: ASP (85.50%)                                                    │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 13. Query: `Hartford Hospital School of Nursing`

✅ **Exact Match Found:** `Hartford Hospital School of Nursing` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Hartford Hospital School of Nursing | 100.00% | 1.000 | 0.942 | 0.00 | Perfect character-for-character match. |
| 2 | Hartford Hospital Offices USA | 83.40% | 0.850 | 0.798 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Hartford School District | 75.00% | 0.744 | 0.763 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | New Hartford School | 74.20% | 0.744 | 0.739 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Hartford Hospital | 74.10% | 0.630 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Hartford Elementary School | 73.90% | 0.744 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Hartford High School | 73.70% | 0.744 | 0.722 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Hartford Art School | 72.70% | 0.744 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | JEFFERSON HOSPITAL NURSING SCHOOL | 69.70% | 0.675 | 0.749 | 0.00 | High word-for-word overlap. |
| 10 | Philadelphia General Hospital School of Nursing | 68.00% | 0.675 | 0.692 | 0.00 | High word-for-word overlap. |
| 11 | Rhode Island Hospital School of Nursing | 67.70% | 0.675 | 0.682 | 0.00 | High word-for-word overlap. |
| 12 | Metropolitan Hospital School of Nursing | 67.70% | 0.675 | 0.680 | 0.00 | High word-for-word overlap. |
| 13 | Philadelphia General Hospital Nursing | 66.50% | 0.637 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Hartford Hospital Women's Health Service | 65.50% | 0.637 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Hartford Public Schools | 59.00% | 0.525 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Hartford Hospital School of Nursing**
> None

**Rank #2: Hartford Hospital Offices USA**
> None

**Rank #3: Hartford School District**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Hartford Hospital School of Nursing (100.00%)                   │
│  Match #2: Hartford Hospital Offices USA (83.40%)                          │
│                                                                            │
│  Score Difference: 16.60%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 14. Query: `Internal J&J Meeting and Breakfast`

✅ **Exact Match Found:** `Internal J&J Meeting and Breakfast` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Internal J&J Meeting and Breakfast | 100.00% | 1.000 | 0.969 | 0.00 | Perfect character-for-character match. |
| 2 | AEP Breakfast Meeting | 62.80% | 0.558 | 0.793 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Breakfast Meeting NYC | 62.70% | 0.558 | 0.788 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Bisnow Breakfast Meeting | 61.80% | 0.558 | 0.759 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | GMCVB Breakfast & Meeting | 61.20% | 0.558 | 0.738 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Breakfast Meeting Oct2018 | 60.90% | 0.558 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Optos Breakfast Meeting | 60.70% | 0.558 | 0.722 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Clime - Breakfast meeting | 60.70% | 0.558 | 0.720 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Breakfast Meeting | 59.40% | 0.420 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Breakfast Before Business | 52.80% | 0.394 | 0.840 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Executive Breakfast Session | 52.20% | 0.394 | 0.821 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Morning after breakfast | 50.80% | 0.394 | 0.776 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Breakfast for Learning | 50.30% | 0.394 | 0.757 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Annual Members Breakfast | 50.20% | 0.394 | 0.756 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Local 1 Breakfast | 49.50% | 0.394 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Internal J&J Meeting and Breakfast**
> None

**Rank #2: AEP Breakfast Meeting**
> None

**Rank #3: Breakfast Meeting NYC**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Internal J&J Meeting and Breakfast (100.00%)                    │
│  Match #2: AEP Breakfast Meeting (62.80%)                                  │
│                                                                            │
│  Score Difference: 37.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 15. Query: `Spina Bifida Coalition of Cincinnati`

✅ **Exact Match Found:** `Spina Bifida Coalition of Cincinnati` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Spina Bifida Coalition of Cincinnati | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | SBCC | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'SBCC'. |
| 3 | Spina Bifida Association of Cincinnati, Inc. | 81.70% | 0.844 | 0.754 | 0.00 | High word-for-word overlap. |
| 4 | Spina Bifida Association of Michigan | 75.00% | 0.744 | 0.763 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Illinois Spina Bifida Association | 74.70% | 0.744 | 0.755 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Spina Bifida Association of Kentucky | 74.30% | 0.744 | 0.741 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Spina Bifida Resource Network | 74.20% | 0.744 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | SPINA BIFIDA ASSOCIATION OF ALABAMA | 74.20% | 0.744 | 0.739 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Colorado Spina Bifida Association | 74.10% | 0.744 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | SPINA BIFIDA ASSN AM | 73.70% | 0.744 | 0.723 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Spina Bifida Association of Massachusetts | 73.60% | 0.744 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Spina Bifida Association of America Headquarters | 73.40% | 0.744 | 0.711 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Spina Bifida Association of Southeast Florida | 73.20% | 0.744 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Spina Bifida Association of Central Florida | 73.00% | 0.744 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Spina Bifida Association of SE Michigan | 72.20% | 0.744 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Spina Bifida Coalition of Cincinnati**
> None

**Rank #2: SBCC**
> None

**Rank #3: Spina Bifida Association of Cincinnati, Inc.**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Spina Bifida Coalition of Cincinnati (100.00%)                  │
│  Match #2: SBCC (90.00%)                                                   │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 16. Query: `THE SOCA GROUP ORGANIZATION`

✅ **Exact Match Found:** `THE SOCA GROUP ORGANIZATION` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | THE SOCA GROUP ORGANIZATION | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | SGO | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'SGO'. |
| 3 | Team SOCA | 78.90% | 0.744 | 0.894 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Soca Society | 78.00% | 0.744 | 0.864 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | System Organization Group | 75.90% | 0.744 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | SOCA Convention | 75.80% | 0.744 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Soca Takeover | 75.40% | 0.744 | 0.778 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Soca Passion | 74.70% | 0.744 | 0.754 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Organization Management Group | 74.00% | 0.744 | 0.733 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Four Organization | 73.20% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Membership Organization | 73.20% | 0.744 | 0.705 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | SOCA | 65.00% | 0.551 | 0.881 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | SoCal United | 38.40% | 0.248 | 0.700 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Socia Team | 38.20% | 0.200 | 0.807 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Socca Sa | 36.90% | 0.216 | 0.728 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: THE SOCA GROUP ORGANIZATION**
> None

**Rank #2: SGO**
> None

**Rank #3: Team SOCA**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: THE SOCA GROUP ORGANIZATION (100.00%)                           │
│  Match #2: SGO (85.50%)                                                    │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 17. Query: `Shiroyama Junior High School`

✅ **Exact Match Found:** `Shiroyama Junior High School` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Shiroyama Junior High School | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | SJHS | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'SJHS'. |
| 3 | Heights Christian Junior High School | 80.40% | 0.825 | 0.754 | 0.00 | High word-for-word overlap. |
| 4 | Junior High School 45 | 79.70% | 0.825 | 0.732 | 0.00 | High word-for-word overlap. |
| 5 | Haga Junior High School | 79.60% | 0.825 | 0.729 | 0.00 | High word-for-word overlap. |
| 6 | Jubail Junior High School | 79.30% | 0.825 | 0.718 | 0.00 | High word-for-word overlap. |
| 7 | CARROLL JUNIOR HIGH SCHOOL | 79.00% | 0.825 | 0.709 | 0.00 | High word-for-word overlap. |
| 8 | Junior High School 22 | 79.00% | 0.825 | 0.708 | 0.00 | High word-for-word overlap. |
| 9 | South Junior High School | 78.90% | 0.825 | 0.707 | 0.00 | High word-for-word overlap. |
| 10 | Hardin Junior High School | 78.80% | 0.825 | 0.702 | 0.00 | High word-for-word overlap. |
| 11 | Durand Junior High School | 78.50% | 0.825 | 0.693 | 0.00 | High word-for-word overlap. |
| 12 | Kirby Junior High School | 78.30% | 0.825 | 0.686 | 0.00 | High word-for-word overlap. |
| 13 | Clara Junior High School | 78.30% | 0.825 | 0.685 | 0.00 | High word-for-word overlap. |
| 14 | Seminole Junior High School | 78.10% | 0.825 | 0.680 | 0.00 | High word-for-word overlap. |
| 15 | George WA High School | 70.40% | 0.708 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Shiroyama Junior High School**
> None

**Rank #2: SJHS**
> None

**Rank #3: Heights Christian Junior High School**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Shiroyama Junior High School (100.00%)                          │
│  Match #2: SJHS (85.50%)                                                   │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 18. Query: `National Home Health`

✅ **Exact Match Found:** `National Home Health` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | National Home Health | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | National Home Health Care | 90.10% | 0.905 | 0.892 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | National Home Healthcare | 90.00% | 0.900 | 0.900 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 4 | Community Home Health | 87.80% | 0.883 | 0.865 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | RESIDENTIAL HOME HEALTH | 87.60% | 0.883 | 0.858 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Comprehensive Home Health | 87.50% | 0.883 | 0.855 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | American Home Health | 87.20% | 0.883 | 0.845 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Assisted Home Health | 86.90% | 0.883 | 0.834 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | One Home Health | 86.80% | 0.883 | 0.834 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Alternative Home Health | 86.80% | 0.883 | 0.832 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Family Home Health | 86.70% | 0.883 | 0.829 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Associated Home Health | 86.70% | 0.883 | 0.827 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Vital Care Home Health | 86.00% | 0.883 | 0.805 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | progressive home health | 85.90% | 0.883 | 0.802 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | National Home Health  Care Conference | 85.90% | 0.900 | 0.762 | 0.00 | Direct prefix match (target contains extra trailing words). |

### Match Narratives

**Rank #1: National Home Health**
> None

**Rank #2: National Home Health Care**
> None

**Rank #3: National Home Healthcare**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: National Home Health (100.00%)                                  │
│  Match #2: National Home Health Care (90.10%)                              │
│                                                                            │
│  Score Difference: 9.90%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 19. Query: `American News Women's Club`

✅ **Exact Match Found:** `American News Women's Club` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | American News Women's Club | 100.00% | 1.000 | 0.953 | 0.00 | Perfect character-for-character match. |
| 2 | American Women's Club | 77.80% | 0.738 | 0.871 | 0.00 | High word-for-word overlap. |
| 3 | American Women Club | 75.60% | 0.651 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | DC Democratic Women's Club | 73.40% | 0.744 | 0.710 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | UM Women's Club | 68.70% | 0.651 | 0.770 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Women's International Club | 68.10% | 0.651 | 0.752 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | International Women's Club | 68.10% | 0.651 | 0.751 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Women's Business Club | 68.00% | 0.651 | 0.748 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Women's Democratic Club | 67.80% | 0.651 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | American Girl Club | 67.60% | 0.651 | 0.735 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Democratic Women's Club | 67.50% | 0.651 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | American Business Club | 67.00% | 0.651 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Women's Sporting Club | 66.90% | 0.651 | 0.711 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | American Club | 58.20% | 0.490 | 0.797 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Women in Management Club | 56.70% | 0.487 | 0.754 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: American News Women's Club**
> None

**Rank #2: American Women's Club**
> None

**Rank #3: American Women Club**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: American News Women's Club (100.00%)                            │
│  Match #2: American Women's Club (77.80%)                                  │
│                                                                            │
│  Score Difference: 22.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 20. Query: `Denise Roberge`

✅ **Exact Match Found:** `Denise Roberge` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Denise Roberge | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Tamara Denise | 73.10% | 0.744 | 0.702 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Denise Long | 72.60% | 0.744 | 0.683 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Tasha Denise | 72.40% | 0.744 | 0.679 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Sandra Denise | 72.40% | 0.744 | 0.677 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Denise O | 71.50% | 0.744 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Denise Michelle | 71.30% | 0.744 | 0.641 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Tamra Denise | 71.10% | 0.744 | 0.634 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | DENISE ACCOUNTS | 71.00% | 0.744 | 0.630 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Denise Abril | 70.90% | 0.744 | 0.627 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Denise Beard | 70.80% | 0.744 | 0.625 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Denise Gour | 70.80% | 0.744 | 0.624 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Denise White | 70.80% | 0.744 | 0.623 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Denise Reed | 70.60% | 0.744 | 0.618 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Charmaine Denise | 70.50% | 0.744 | 0.615 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Denise Roberge**
> None

**Rank #2: Tamara Denise**
> None

**Rank #3: Denise Long**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Denise Roberge (100.00%)                                        │
│  Match #2: Tamara Denise (73.10%)                                          │
│                                                                            │
│  Score Difference: 26.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 21. Query: `Synergy Soccer Club`

✅ **Exact Match Found:** `Synergy Soccer Club` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Synergy Soccer Club | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Synergy Football Club | 81.80% | 0.810 | 0.836 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Synergy Volleyball Club | 80.60% | 0.810 | 0.797 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Alliance Soccer Club | 78.60% | 0.810 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | FC Alliance Soccer Club | 78.30% | 0.810 | 0.721 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Advantage Soccer Club | 78.30% | 0.810 | 0.720 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Kitsap Soccer Club | 78.30% | 0.810 | 0.720 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Synergy Chain Investors Club | 77.90% | 0.810 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Polish Soccer Club | 77.80% | 0.810 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Synergy Gymnastics Booster Club | 77.50% | 0.810 | 0.692 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | AC Alliance Soccer Club | 77.20% | 0.810 | 0.684 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Nordic Soccer Club | 77.10% | 0.810 | 0.682 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Soccer Club | 69.90% | 0.681 | 0.742 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Club CO Soccer | 68.50% | 0.668 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Synergy Sports Worldwide | 59.10% | 0.528 | 0.738 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Synergy Soccer Club**
> None

**Rank #2: Synergy Football Club**
> None

**Rank #3: Synergy Volleyball Club**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Synergy Soccer Club (100.00%)                                   │
│  Match #2: Synergy Football Club (81.80%)                                  │
│                                                                            │
│  Score Difference: 18.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 22. Query: `NFC Forum`

✅ **Exact Match Found:** `NFC Forum` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | NFC Forum | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | NFC Forum Members | 100.00% | 0.900 | 0.900 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | NFC Forum         . | 96.60% | 1.000 | 0.887 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 4 | Near Field Communications Forum (NFC Forum) | 89.70% | 0.787 | 0.677 | 0.00 | Substring match (target contains query text). |
| 5 | NF | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'NF'. |
| 6 | Nexus Forum | 84.60% | 0.744 | 0.585 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | NFC Fighting | 73.80% | 0.744 | 0.726 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | NFC Orientation | 73.40% | 0.744 | 0.713 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | NFC Consulting | 73.20% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | NFC MARKETING | 73.00% | 0.744 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | NFC Life | 72.40% | 0.744 | 0.677 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | NFC Amenity | 72.20% | 0.744 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | NFC Insurance | 72.20% | 0.744 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Eagles Forum | 71.90% | 0.744 | 0.661 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | NFC Tabagators | 70.90% | 0.744 | 0.629 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: NFC Forum**
> None

**Rank #2: NFC Forum Members**
> None

**Rank #3: NFC Forum         .**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: NFC Forum (100.00%)                                             │
│  Match #2: NFC Forum Members (100.00%)                                     │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 23. Query: `A Better Choice Limousine & Concierge`

✅ **Exact Match Found:** `A Better Choice Limousine & Concierge` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | A Better Choice Limousine & Concierge | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | First Choice Limousine Services | 76.70% | 0.744 | 0.822 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Capital Travel Limousine | 55.40% | 0.427 | 0.853 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Chicago Limousine Transportation | 54.30% | 0.427 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | 1st Class Limousine | 54.30% | 0.427 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | A Formal Limousine Services | 54.00% | 0.427 | 0.806 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Luxury Limousine and Entertainment | 54.00% | 0.427 | 0.804 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | National Limousine Association | 54.00% | 0.427 | 0.804 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | District Executive Limousine | 53.90% | 0.427 | 0.801 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Greater Atlanta Limousine | 53.90% | 0.427 | 0.801 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | VIP Limousine Service | 53.90% | 0.427 | 0.800 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | My Limousine Service | 53.80% | 0.427 | 0.799 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | VA Limousine Association | 53.70% | 0.427 | 0.794 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Capital Travel & Limousine | 53.60% | 0.427 | 0.790 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Mint Life Limousine | 53.50% | 0.427 | 0.787 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: A Better Choice Limousine & Concierge**
> None

**Rank #2: First Choice Limousine Services**
> None

**Rank #3: Capital Travel Limousine**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: A Better Choice Limousine & Concierge (100.00%)                 │
│  Match #2: First Choice Limousine Services (76.70%)                        │
│                                                                            │
│  Score Difference: 23.30%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 24. Query: `Danish Sisterhood of America`

✅ **Exact Match Found:** `Danish Sisterhood of America` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Danish Sisterhood of America | 100.00% | 1.000 | 0.974 | 0.00 | Perfect character-for-character match. |
| 2 | The Danish Sisterhood of America | 97.40% | 1.000 | 0.913 | 0.00 | Substring match (target contains query text). |
| 3 | Danish Sisterhood and Brotherhood of America | 93.00% | 0.950 | 0.791 | 0.00 | High word-for-word overlap. |
| 4 | Danish Brotherhood & Danish Sisterhood of America | 86.10% | 0.900 | 0.769 | 0.00 | Substring match (target contains query text). |
| 5 | DANISH BROTHERHOOD AND DANISH SISTERHOOD OF AMERICA | 85.70% | 0.900 | 0.758 | 0.00 | Substring match (target contains query text). |
| 6 | DSA | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'DSA'. |
| 7 | Danish Sisterhood of the Americas | 82.70% | 0.825 | 0.831 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | DANISH SISTERHOOD | 77.60% | 0.681 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | The Dansih Sisterhood of America | 77.40% | 0.810 | 0.690 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Danish Sisterhood of Amercia | 75.80% | 0.810 | 0.637 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Inspire Global Sisterhood | 58.00% | 0.528 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | 21st Century Sisterhood | 57.90% | 0.528 | 0.700 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Sisters Across America | 57.30% | 0.528 | 0.679 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Baltimore Hebrew Sisterhood | 56.50% | 0.528 | 0.651 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | PEO Sisterhood International | 56.50% | 0.528 | 0.651 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Danish Sisterhood of America**
> None

**Rank #2: The Danish Sisterhood of America**
> None

**Rank #3: Danish Sisterhood and Brotherhood of America**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Danish Sisterhood of America (100.00%)                          │
│  Match #2: The Danish Sisterhood of America (97.40%)                       │
│                                                                            │
│  Score Difference: 2.60%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 25. Query: `Brooklyn Comics Club`

✅ **Exact Match Found:** `Brooklyn Comics Club` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Brooklyn Comics Club | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | BCC | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'BCC'. |
| 3 | Brooklyn Baseball Club | 80.00% | 0.810 | 0.776 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Brooklyn NY Film Club | 79.80% | 0.810 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | North Brooklyn Comic Book Club | 79.50% | 0.810 | 0.761 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Brooklyn Ski Club | 79.10% | 0.810 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Brooklyn Conversation Club | 78.90% | 0.810 | 0.741 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Brooklyn Football Club | 78.80% | 0.810 | 0.739 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | New Brooklyn Book Club | 78.50% | 0.810 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Brooklyn College Diversity Club | 76.80% | 0.810 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Brooklyn Barbell Club | 76.80% | 0.810 | 0.671 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Municipal Club of Brooklyn | 76.60% | 0.810 | 0.664 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Brooklyn Creative Artists | 59.10% | 0.528 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Brooklyn Comic Con by | 58.20% | 0.528 | 0.709 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | BROOKLYN COMMUNITY CENTER | 57.70% | 0.528 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Brooklyn Comics Club**
> None

**Rank #2: BCC**
> None

**Rank #3: Brooklyn Baseball Club**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Brooklyn Comics Club (100.00%)                                  │
│  Match #2: BCC (85.50%)                                                    │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 26. Query: `Global Interagency Security Forum`

✅ **Exact Match Found:** `Global Interagency Security Forum` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Global Interagency Security Forum | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | GISF | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'GISF'. |
| 3 | Cyber Security Collaboration Forum | 76.70% | 0.779 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Infrastructure Security and Resilience Forum | 76.40% | 0.779 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | The Cyber Security Forum Initiative | 76.30% | 0.779 | 0.726 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | NY Cyber Security Forum | 75.90% | 0.779 | 0.712 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Executive Security Action Forum | 75.50% | 0.779 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Security Network Forum | 74.60% | 0.682 | 0.897 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | International Security Forum | 74.10% | 0.682 | 0.879 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | BITS Security Forum | 72.30% | 0.682 | 0.818 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Info Security Forum | 71.40% | 0.682 | 0.788 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Information Security Forum | 71.20% | 0.682 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Westcon Security Forum | 71.00% | 0.682 | 0.776 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Cargo Security Forum | 70.50% | 0.682 | 0.759 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Security Forum | 68.80% | 0.577 | 0.945 | 0.00 | Matched via strong semantic/conceptual similarity. |

### Match Narratives

**Rank #1: Global Interagency Security Forum**
> None

**Rank #2: GISF**
> None

**Rank #3: Cyber Security Collaboration Forum**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Global Interagency Security Forum (100.00%)                     │
│  Match #2: GISF (90.00%)                                                   │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 27. Query: `Lancet Software`

✅ **Exact Match Found:** `Lancet Software` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Lancet Software | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Lancet Technology | 76.80% | 0.744 | 0.825 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Lancet Technology, Incorporated | 72.40% | 0.744 | 0.678 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | JAT Software | 71.40% | 0.744 | 0.645 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | CAS Software | 71.30% | 0.744 | 0.640 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Agile Software | 70.30% | 0.744 | 0.607 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Med Software | 70.00% | 0.744 | 0.599 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | BRICT Software | 69.70% | 0.744 | 0.587 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Quest Software | 69.60% | 0.744 | 0.583 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Griffin Software | 69.60% | 0.744 | 0.583 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | BST Software | 69.50% | 0.744 | 0.583 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | PROFESSIONAL SOFTWARE | 69.50% | 0.744 | 0.582 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Tech Software | 69.50% | 0.744 | 0.582 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | SIS Software | 69.50% | 0.744 | 0.581 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Software Technology | 69.50% | 0.744 | 0.580 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Lancet Software**
> None

**Rank #2: Lancet Technology**
> None

**Rank #3: Lancet Technology, Incorporated**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Lancet Software (100.00%)                                       │
│  Match #2: Lancet Technology (76.80%)                                      │
│                                                                            │
│  Score Difference: 23.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 28. Query: `Our Lady of the Lakes Catholic Church and School`

✅ **Exact Match Found:** `Our Lady of the Lakes Catholic Church and School` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Our Lady of the Lakes Catholic Church and School | 100.00% | 1.000 | 0.917 | 0.00 | Perfect character-for-character match. |
| 2 | OUR LADY OF THE LAKES CATHOLIC CHURCH | 88.80% | 0.857 | 0.962 | 0.00 | High word-for-word overlap. |
| 3 | Our Lady of the Lakes Catholic School | 87.90% | 0.857 | 0.931 | 0.00 | High word-for-word overlap. |
| 4 | Our Lady of the Lake Roman Catholic Church | 83.60% | 0.828 | 0.855 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | Our Lady of the Lake Catholic Church | 80.30% | 0.759 | 0.905 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Our Lady Guadalupe Catholic Church | 77.60% | 0.759 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Our Lady of Health Catholic Church | 77.60% | 0.759 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Our Lady of the Lake Catholic Parish | 75.50% | 0.730 | 0.812 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Our Lady of The Lakes Church | 75.40% | 0.683 | 0.919 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Our Lady Lake Church | 72.20% | 0.614 | 0.975 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Our Lady of the Lake Church | 68.70% | 0.614 | 0.858 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Lakes Catholic | 63.10% | 0.472 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | Lake Catholic High School | 56.00% | 0.425 | 0.874 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 14 | Lakeshore Catholic High School | 55.70% | 0.425 | 0.867 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 15 | River Lakes Community Church | 55.10% | 0.425 | 0.847 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Our Lady of the Lakes Catholic Church and School**
> None

**Rank #2: OUR LADY OF THE LAKES CATHOLIC CHURCH**
> None

**Rank #3: Our Lady of the Lakes Catholic School**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Our Lady of the Lakes Catholic Church and School (100.00%)      │
│  Match #2: OUR LADY OF THE LAKES CATHOLIC CHURCH (88.80%)                  │
│                                                                            │
│  Score Difference: 11.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 29. Query: `Broadway Bound International`

✅ **Exact Match Found:** `Broadway Bound International` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Broadway Bound International | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | BBI | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'BBI'. |
| 3 | Broadway Bound West | 87.20% | 0.883 | 0.845 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Broadway Bound Kids | 85.80% | 0.883 | 0.798 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Bound Four Broadway | 85.70% | 0.883 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Broadway Bound Studio | 85.60% | 0.883 | 0.791 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Broadway Bound Dance | 84.70% | 0.883 | 0.761 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Broadway Bound Childrens Theatre | 82.80% | 0.883 | 0.700 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Broadway Bound Kidz | 82.70% | 0.883 | 0.694 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Broadway Bound | 80.90% | 0.742 | 0.966 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Broadway Asia International | 77.60% | 0.773 | 0.785 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Broadway Across American | 62.70% | 0.554 | 0.797 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Broadway Across Am | 61.60% | 0.554 | 0.759 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | BROADWAY NATIONAL TOUR | 60.20% | 0.554 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Broadway Meets Country | 60.10% | 0.554 | 0.712 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Broadway Bound International**
> None

**Rank #2: BBI**
> None

**Rank #3: Broadway Bound West**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Broadway Bound International (100.00%)                          │
│  Match #2: BBI (90.00%)                                                    │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 30. Query: `E. H. Wachs`

✅ **Exact Match Found:** `E. H. Wachs` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | E. H. Wachs | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | EHW | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'EHW'. |
| 3 | E.H. Wachs | 61.30% | 0.447 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Wachs Water Services | 58.20% | 0.528 | 0.708 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Wachs / Russell Wedding | 57.40% | 0.528 | 0.683 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Wachs Services | 57.10% | 0.435 | 0.889 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Elen Wachs | 57.00% | 0.435 | 0.885 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Aecon-Wachs | 51.60% | 0.435 | 0.705 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | WACHSA | 40.80% | 0.180 | 0.939 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Wachsman | 37.80% | 0.159 | 0.890 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | AACHS | 37.30% | 0.154 | 0.884 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Wachsman PR | 36.60% | 0.186 | 0.787 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | GCHS | 35.70% | 0.125 | 0.899 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 14 | Steven Fuchs | 35.70% | 0.177 | 0.776 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Fuchs | 35.30% | 0.116 | 0.907 | 0.00 | Matched via strong semantic/conceptual similarity. |

### Match Narratives

**Rank #1: E. H. Wachs**
> None

**Rank #2: EHW**
> None

**Rank #3: E.H. Wachs**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: E. H. Wachs (100.00%)                                           │
│  Match #2: EHW (90.00%)                                                    │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 31. Query: `Marine Corps Fox 2/5`

✅ **Exact Match Found:** `Marine Corps Fox 2/5` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Marine Corps Fox 2/5 | 100.00% | 1.000 | 0.961 | 0.00 | Perfect character-for-character match. |
| 2 | United State Marine Corps | 77.00% | 0.779 | 0.750 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Marine Corps Personnel Support | 76.90% | 0.779 | 0.746 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | American US Marine Corps | 76.90% | 0.779 | 0.745 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | MARINE CORPS BASE CAMP | 76.90% | 0.779 | 0.745 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Marine Corps Community Service | 76.80% | 0.779 | 0.742 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Marine Corps Support Facility | 76.70% | 0.779 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | US Marine Corps Intelligence | 76.60% | 0.779 | 0.736 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | United States Marine Corps | 76.60% | 0.779 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | US Marine Corps HMM 161 | 76.60% | 0.779 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Marine Corps Air Station | 76.50% | 0.779 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Young US Marine Corps | 76.50% | 0.779 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Marine Corps League National | 76.50% | 0.779 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Navy Marine Corps Reserve | 76.50% | 0.779 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Navy Marine Corps Ball | 76.40% | 0.779 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Marine Corps Fox 2/5**
> None

**Rank #2: United State Marine Corps**
> None

**Rank #3: Marine Corps Personnel Support**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Marine Corps Fox 2/5 (100.00%)                                  │
│  Match #2: United State Marine Corps (77.00%)                              │
│                                                                            │
│  Score Difference: 23.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 32. Query: `Fantasia Turistica`

✅ **Exact Match Found:** `Fantasia Turistica` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Fantasia Turistica | 100.00% | 1.000 | 0.979 | 0.00 | Perfect character-for-character match. |
| 2 | FT | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'FT'. |
| 3 | Fantasia Travels | 77.50% | 0.744 | 0.850 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Fantasia Travel | 76.70% | 0.744 | 0.820 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Fantasia Accessry | 75.40% | 0.744 | 0.778 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Ferrari Fantasia | 75.10% | 0.744 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Fantasia Home Parties | 72.90% | 0.744 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Franquia Fantasia | 72.40% | 0.744 | 0.679 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Fantasia by Stohler | 72.20% | 0.744 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Fantasia Veneziana | 72.00% | 0.744 | 0.666 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Noreen Fantasia | 72.00% | 0.744 | 0.664 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Bobo Entertainment Fantasia Tour | 70.00% | 0.744 | 0.599 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Fantasia Travel PreCruise Group | 69.20% | 0.744 | 0.571 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Fantasia | 68.60% | 0.551 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 15 | Grupo Fantasia Eventos y Producciones | 68.50% | 0.744 | 0.548 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Fantasia Turistica**
> None

**Rank #2: FT**
> None

**Rank #3: Fantasia Travels**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Fantasia Turistica (100.00%)                                    │
│  Match #2: FT (85.50%)                                                     │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 33. Query: `Esoterix`

✅ **Exact Match Found:** `Esoterix` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Esoterix | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Esoterix Headquarters | 85.90% | 0.900 | 0.763 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | Esoterix Integrated Genetics | 83.70% | 0.900 | 0.692 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 4 | CENTRIX | 42.60% | 0.300 | 0.719 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Metrix | 42.50% | 0.321 | 0.668 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Verix | 42.00% | 0.277 | 0.753 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Netrix | 41.90% | 0.321 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Lectrix | 41.70% | 0.300 | 0.688 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Isometrix | 41.50% | 0.318 | 0.642 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Corix | 41.40% | 0.277 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Vmetrix | 41.00% | 0.300 | 0.666 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Metametrix | 40.90% | 0.300 | 0.663 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | TYMETRIX | 40.70% | 0.281 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Vectrix | 40.60% | 0.300 | 0.652 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Cimetrix | 40.50% | 0.281 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Esoterix**
> None

**Rank #2: Esoterix Headquarters**
> None

**Rank #3: Esoterix Integrated Genetics**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Esoterix (100.00%)                                              │
│  Match #2: Esoterix Headquarters (85.90%)                                  │
│                                                                            │
│  Score Difference: 14.10%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 34. Query: `Coker Group`

✅ **Exact Match Found:** `Coker Group` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Coker Group | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | The Coker Group | 100.00% | 0.900 | 0.903 | 0.00 | Substring match (target contains query text). |
| 3 | Coker Grp | 98.50% | 0.900 | 0.684 | 0.00 | High word-for-word overlap. |
| 4 | Coker Cheerleading Group | 97.50% | 0.900 | 0.701 | 0.00 | High word-for-word overlap. |
| 5 | CG | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'CG'. |
| 6 | Coker College | 85.80% | 0.900 | 0.758 | 0.00 | High word-for-word overlap. |
| 7 | Coker Family Reunion | 85.70% | 0.900 | 0.756 | 0.00 | High word-for-word overlap. |
| 8 | Coker Consultants | 84.90% | 0.900 | 0.729 | 0.00 | High word-for-word overlap. |
| 9 | Coker University | 84.70% | 0.900 | 0.724 | 0.00 | High word-for-word overlap. |
| 10 | Coker Law | 84.60% | 0.900 | 0.719 | 0.00 | High word-for-word overlap. |
| 11 | Coker Coaters | 84.50% | 0.900 | 0.718 | 0.00 | High word-for-word overlap. |
| 12 | Coker Legal | 84.50% | 0.900 | 0.718 | 0.00 | High word-for-word overlap. |
| 13 | Coker Capital | 84.20% | 0.900 | 0.707 | 0.00 | High word-for-word overlap. |
| 14 | Betty Coker | 83.30% | 0.900 | 0.678 | 0.00 | High word-for-word overlap. |
| 15 | Friends of Leslie Coker | 83.10% | 0.900 | 0.671 | 0.00 | High word-for-word overlap. |

### Match Narratives

**Rank #1: Coker Group**
> None

**Rank #2: The Coker Group**
> None

**Rank #3: Coker Grp**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Coker Group (100.00%)                                           │
│  Match #2: The Coker Group (100.00%)                                       │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 35. Query: `GILEAD IT`

✅ **Exact Match Found:** `GILEAD IT` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | GILEAD IT | 100.00% | 1.000 | 0.928 | 0.00 | Perfect character-for-character match. |
| 2 | GI | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'GI'. |
| 3 | Gilead Productions | 77.30% | 0.744 | 0.841 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Gilead Science | 77.30% | 0.744 | 0.840 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Gilead Sciences | 77.00% | 0.744 | 0.831 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Gilead 1N | 76.80% | 0.744 | 0.823 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Gilead Services | 76.20% | 0.744 | 0.804 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | GILEAD MEDICAL | 76.10% | 0.744 | 0.800 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | GILEAD PHARMACEUTICAL | 75.50% | 0.744 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Gilead Media | 75.30% | 0.744 | 0.774 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Gilead Canada | 75.20% | 0.744 | 0.771 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Gilead Finance | 75.10% | 0.744 | 0.767 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Gilead Pharmaceuticals | 74.70% | 0.744 | 0.754 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Baltimore Gilead | 74.60% | 0.744 | 0.750 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Gilead 3 Productions | 74.10% | 0.744 | 0.735 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: GILEAD IT**
> None

**Rank #2: GI**
> None

**Rank #3: Gilead Productions**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: GILEAD IT (100.00%)                                             │
│  Match #2: GI (85.50%)                                                     │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 36. Query: `4143 Affiliate INDA 2016`

✅ **Exact Match Found:** `4143 Affiliate INDA 2016` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | 4143 Affiliate INDA 2016 | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | 1528 Affiliate INDA 2016 | 84.10% | 0.900 | 0.704 | 0.00 | High word-for-word overlap. |
| 3 | 4143 Affiliate Aan 2017 | 75.60% | 0.744 | 0.784 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | 430 Affiliate ALA 2016 | 72.20% | 0.744 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | 1035 Affiliate CASE 2016 | 71.30% | 0.744 | 0.641 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | 1035 AFSA Affiliate 2016 | 70.30% | 0.744 | 0.608 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | 00430 Affiliate 2016 Cardiometabolic | 70.20% | 0.744 | 0.605 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | 1035 Affiliate AFSA 2016 | 70.10% | 0.744 | 0.602 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | 430 Affiliate ASM 2016 | 70.10% | 0.744 | 0.602 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | 430 NAVBO Affiliate 2016 | 70.10% | 0.744 | 0.600 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | 430 Affiliate NALP 2016 | 70.00% | 0.744 | 0.599 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | ASA AFFILIATE ACCOUNT 2016 | 69.80% | 0.744 | 0.591 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | SEPA - Affiliate Account 2016 | 69.70% | 0.744 | 0.589 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | 4143 Affiliates ASTRO 2016 | 67.70% | 0.637 | 0.770 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | 2016 Google Affiliate | 65.80% | 0.651 | 0.676 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: 4143 Affiliate INDA 2016**
> None

**Rank #2: 1528 Affiliate INDA 2016**
> None

**Rank #3: 4143 Affiliate Aan 2017**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: 4143 Affiliate INDA 2016 (100.00%)                              │
│  Match #2: 1528 Affiliate INDA 2016 (84.10%)                               │
│                                                                            │
│  Score Difference: 15.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 37. Query: `Pipe and Plant Solutions`

✅ **Exact Match Found:** `Pipe and Plant Solutions` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Pipe and Plant Solutions | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | PPS | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'PPS'. |
| 3 | Pipe & Plant | 79.50% | 0.750 | 0.900 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Advanced Pipe Solutions | 78.90% | 0.773 | 0.827 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | TV Pipe Solutions | 73.80% | 0.773 | 0.655 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Infra Pipe Solutions | 73.70% | 0.773 | 0.653 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Plant Solutions Limited | 68.10% | 0.650 | 0.754 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Pipe | 59.50% | 0.472 | 0.880 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | Pipe Line Contractors | 59.50% | 0.554 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Texas Pipe Works | 59.40% | 0.554 | 0.688 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Specialized Pipe Technologies | 58.30% | 0.554 | 0.652 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Plumbers & Pipe Fitters | 58.30% | 0.554 | 0.649 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Cal Pipe Industries | 58.10% | 0.554 | 0.643 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Pipe Products | 56.70% | 0.457 | 0.824 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | PIPE TOOLS | 55.90% | 0.457 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Pipe and Plant Solutions**
> None

**Rank #2: PPS**
> None

**Rank #3: Pipe & Plant**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Pipe and Plant Solutions (100.00%)                              │
│  Match #2: PPS (85.50%)                                                    │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 38. Query: `Stephen Rourke`

✅ **Exact Match Found:** `Stephen Rourke` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Stephen Rourke | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Rourke Publishing | 73.40% | 0.744 | 0.710 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Damon Rourke | 72.70% | 0.744 | 0.686 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Rourke Manufacturing | 71.50% | 0.744 | 0.649 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Rourke Rooms | 68.90% | 0.744 | 0.562 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Rourke Educational Media | 68.80% | 0.744 | 0.559 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Rourke | 67.80% | 0.551 | 0.975 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Stephen Scott | 67.80% | 0.744 | 0.524 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Stephen Shea | 67.50% | 0.744 | 0.515 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Rourke and Ashley | 67.50% | 0.744 | 0.514 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Laing O' Rourke | 67.40% | 0.744 | 0.511 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Stephen Young | 67.00% | 0.744 | 0.498 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Stephen Oh | 66.90% | 0.744 | 0.494 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | O Rourke & Assoc | 66.70% | 0.744 | 0.488 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Stephen Lacy | 66.70% | 0.744 | 0.487 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Stephen Rourke**
> None

**Rank #2: Rourke Publishing**
> None

**Rank #3: Damon Rourke**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Stephen Rourke (100.00%)                                        │
│  Match #2: Rourke Publishing (73.40%)                                      │
│                                                                            │
│  Score Difference: 26.60%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 39. Query: `MIT Initiative on the Digital Economy`

✅ **Exact Match Found:** `MIT Initiative on the Digital Economy` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | MIT Initiative on the Digital Economy | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | MIT Energy Initiative | 55.70% | 0.448 | 0.813 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | MIT Information Services and Technology | 44.20% | 0.291 | 0.794 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | MIT INFORMATION SERVICES | 42.00% | 0.222 | 0.883 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | MIT Information Systems | 40.60% | 0.238 | 0.798 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | MIT Global Initiatives | 40.50% | 0.218 | 0.840 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | MIT Dept of Economic | 38.00% | 0.210 | 0.778 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | MIT Technology Review | 36.40% | 0.162 | 0.836 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | MIT Development Services | 36.10% | 0.135 | 0.888 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | MIT Corporate Development | 36.10% | 0.162 | 0.823 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | MIT Tech | 35.80% | 0.124 | 0.904 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | MIT Resource Development | 35.00% | 0.135 | 0.852 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | MIT Tech Conference | 34.80% | 0.165 | 0.777 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | MIT Distributors | 34.70% | 0.135 | 0.841 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | MIT Info Systems | 34.60% | 0.164 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: MIT Initiative on the Digital Economy**
> None

**Rank #2: MIT Energy Initiative**
> None

**Rank #3: MIT Information Services and Technology**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: MIT Initiative on the Digital Economy (100.00%)                 │
│  Match #2: MIT Energy Initiative (55.70%)                                  │
│                                                                            │
│  Score Difference: 44.30%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 40. Query: `Urx Community USA`

✅ **Exact Match Found:** `Urx Community USA` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Urx Community USA | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | USA Community Service Commission | 72.10% | 0.773 | 0.599 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Florida Urological Society USA | 59.60% | 0.554 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Uratta National Association USA | 56.80% | 0.554 | 0.600 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | URX Conference | 51.90% | 0.457 | 0.663 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Sites USA | 51.40% | 0.457 | 0.646 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | URENCO USA | 51.00% | 0.457 | 0.633 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | EDX USA | 50.80% | 0.457 | 0.628 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | US Community | 47.30% | 0.392 | 0.664 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Community Link | 45.90% | 0.392 | 0.617 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Community Path | 45.40% | 0.392 | 0.601 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | US Communities | 38.70% | 0.263 | 0.675 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | United Communities | 36.60% | 0.233 | 0.674 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Communities United | 36.40% | 0.225 | 0.688 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | York Communities | 35.30% | 0.247 | 0.601 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Urx Community USA**
> None

**Rank #2: USA Community Service Commission**
> None

**Rank #3: Florida Urological Society USA**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Urx Community USA (100.00%)                                     │
│  Match #2: USA Community Service Commission (72.10%)                       │
│                                                                            │
│  Score Difference: 27.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 41. Query: `Spredfast Engage`

✅ **Exact Match Found:** `Spredfast Engage` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Spredfast Engage | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | SE | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'SE'. |
| 3 | Spredfast Events | 77.70% | 0.744 | 0.853 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Spredfast Product | 73.90% | 0.744 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Spredfast | 67.50% | 0.551 | 0.964 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Shredfast | 39.50% | 0.210 | 0.827 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Steadfast PR | 39.30% | 0.267 | 0.686 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Redfast | 38.50% | 0.200 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | HyperFast Agent | 38.30% | 0.290 | 0.598 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Superfast Business | 38.20% | 0.270 | 0.644 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Steadfast Management | 38.20% | 0.300 | 0.574 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Wordfast Training | 37.90% | 0.253 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Steadfast Living | 37.90% | 0.290 | 0.585 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Steadfast REIT | 37.20% | 0.279 | 0.587 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | ProcessFast | 37.00% | 0.170 | 0.838 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Spredfast Engage**
> None

**Rank #2: SE**
> None

**Rank #3: Spredfast Events**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Spredfast Engage (100.00%)                                      │
│  Match #2: SE (85.50%)                                                     │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 42. Query: `City of Dallas-Parks & Recreation`

✅ **Exact Match Found:** `City of Dallas-Parks & Recreation` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | City of Dallas-Parks & Recreation | 100.00% | 1.000 | 0.928 | 0.00 | Perfect character-for-character match. |
| 2 | City of Dallas Park & Recreation | 87.00% | 0.844 | 0.933 | 0.00 | High word-for-word overlap. |
| 3 | Dallas Parks and Recreation Department | 86.60% | 0.844 | 0.916 | 0.00 | High word-for-word overlap. |
| 4 | Dallas Parks and Recreation Dept | 86.50% | 0.844 | 0.914 | 0.00 | High word-for-word overlap. |
| 5 | City of Dallas Park & Recreation Department | 84.30% | 0.844 | 0.842 | 0.00 | High word-for-word overlap. |
| 6 | Baltimore City Recreation and Parks | 83.70% | 0.844 | 0.822 | 0.00 | High word-for-word overlap. |
| 7 | Dallas Parks and Recreation | 81.70% | 0.738 | 1.000 | 0.00 | High word-for-word overlap. |
| 8 | Baltimore City Parks and Recreation Department | 81.50% | 0.844 | 0.747 | 0.00 | High word-for-word overlap. |
| 9 | City of Miami Parks & Recreation | 81.40% | 0.844 | 0.746 | 0.00 | High word-for-word overlap. |
| 10 | PARKS FOR DOWNTOWN DALLAS | 79.60% | 0.744 | 0.919 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | DALLAS PARK & RECREATION DEPARTMENT | 77.60% | 0.744 | 0.851 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Texas Recreation & Parks Society | 75.70% | 0.744 | 0.787 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | PALM SPRINGS PARKS RECREATION | 75.60% | 0.744 | 0.786 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Texas Parks and Recreation Society | 74.90% | 0.744 | 0.760 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Philadelphia Parks & Recreation Outdoor Experience Program | 74.50% | 0.744 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: City of Dallas-Parks & Recreation**
> None

**Rank #2: City of Dallas Park & Recreation**
> None

**Rank #3: Dallas Parks and Recreation Department**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: City of Dallas-Parks & Recreation (100.00%)                     │
│  Match #2: City of Dallas Park & Recreation (87.00%)                       │
│                                                                            │
│  Score Difference: 13.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 43. Query: `Kai Pono Builders, Inc.`

✅ **Exact Match Found:** `Kai Pono Builders, Inc.` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Kai Pono Builders, Inc. | 100.00% | 1.000 | 0.835 | 0.00 | Perfect character-for-character match. |
| 2 | Kai Pono Builders | 95.60% | 0.938 | 1.000 | 0.00 | High word-for-word overlap. |
| 3 | Pono Kai Resort | 73.30% | 0.773 | 0.640 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Pono Kai | 67.70% | 0.638 | 0.768 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Kai Partners | 53.10% | 0.425 | 0.779 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | S Kai | 52.30% | 0.425 | 0.752 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Kai Kai Lam | 52.00% | 0.425 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Kaipona Builders, Incorporated | 51.70% | 0.447 | 0.681 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | KAI Partners, Inc. | 51.10% | 0.457 | 0.638 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Kai Kai Communications | 50.70% | 0.425 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | KAI Research | 50.60% | 0.425 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Kai USA Ltd | 50.10% | 0.425 | 0.679 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Kai Hawaii Company | 49.90% | 0.425 | 0.673 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Kai Kani | 49.90% | 0.425 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Kai Trade | 49.90% | 0.425 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Kai Pono Builders, Inc.**
> None

**Rank #2: Kai Pono Builders**
> None

**Rank #3: Pono Kai Resort**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Kai Pono Builders, Inc. (100.00%)                               │
│  Match #2: Kai Pono Builders (95.60%)                                      │
│                                                                            │
│  Score Difference: 4.40%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 44. Query: `MUSICFIRST COALITION`

✅ **Exact Match Found:** `MUSICFIRST COALITION` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | MUSICFIRST COALITION | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | MC | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'MC'. |
| 3 | music FIRST Coalition | 76.00% | 0.744 | 0.798 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Future of Music Coalition | 74.30% | 0.744 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | District New Music Coalition | 73.60% | 0.744 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Coalition BEATS Music Program | 72.90% | 0.744 | 0.695 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Black Music Action Coalition | 72.20% | 0.744 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Black Music Coalition Action | 72.10% | 0.744 | 0.669 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | The Coalition for Music Education | 71.30% | 0.744 | 0.640 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Coalition of Music Stores | 70.60% | 0.744 | 0.619 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | CDFI Coalition | 70.40% | 0.744 | 0.613 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Musicfirst | 64.10% | 0.551 | 0.849 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Music Allies | 39.30% | 0.270 | 0.681 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Music Promotions | 38.60% | 0.283 | 0.625 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Music Movement Organization | 38.50% | 0.268 | 0.657 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: MUSICFIRST COALITION**
> None

**Rank #2: MC**
> None

**Rank #3: music FIRST Coalition**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: MUSICFIRST COALITION (100.00%)                                  │
│  Match #2: MC (85.50%)                                                     │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 45. Query: `Frontier Power Products`

✅ **Exact Match Found:** `Frontier Power Products` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Frontier Power Products | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | FPP | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'FPP'. |
| 3 | Frontier Business Products | 79.00% | 0.810 | 0.744 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Frontier Natural Products | 76.70% | 0.810 | 0.669 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Advanced Power Products | 76.40% | 0.810 | 0.658 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Worldwide Power Products | 76.40% | 0.810 | 0.656 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Power Service Products | 76.20% | 0.810 | 0.651 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Frontier Products | 72.60% | 0.668 | 0.860 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | Power Products | 69.50% | 0.681 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Frontier Building Supply | 57.50% | 0.528 | 0.684 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Frontier Supply Chain Solutions | 57.40% | 0.528 | 0.683 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Frontier Pro Services | 56.80% | 0.528 | 0.662 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Frontier Energy | 53.10% | 0.435 | 0.754 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Frontier Utilities | 52.80% | 0.435 | 0.744 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Frontier Line | 52.50% | 0.435 | 0.735 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Frontier Power Products**
> None

**Rank #2: FPP**
> None

**Rank #3: Frontier Business Products**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Frontier Power Products (100.00%)                               │
│  Match #2: FPP (90.00%)                                                    │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 46. Query: `1960`

✅ **Exact Match Found:** `1960` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | 1960 | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | District 1960 | 85.40% | 0.900 | 0.745 | 0.00 | Substring match (target contains query text). |
| 3 | 60 | 81.10% | 0.900 | 0.603 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Playhouse 1960 | 80.30% | 0.900 | 0.577 | 0.00 | Substring match (target contains query text). |
| 5 | 1960 Family Practice | 80.20% | 0.900 | 0.572 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 6 | PAGE CLASS OF 1960 | 79.40% | 0.900 | 0.548 | 0.00 | Substring match (target contains query text). |
| 7 | NFA Class of 1960 | 78.60% | 0.900 | 0.520 | 0.00 | Substring match (target contains query text). |
| 8 | 1961 | 55.10% | 0.450 | 0.788 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | 1966 | 54.10% | 0.450 | 0.753 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | 1963 | 54.10% | 0.450 | 0.752 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | 1950 | 53.50% | 0.450 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | 1962 | 52.90% | 0.450 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | 1964 | 52.60% | 0.450 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | 1970 | 52.20% | 0.450 | 0.690 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | 1965 | 51.40% | 0.450 | 0.664 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: 1960**
> None

**Rank #2: District 1960**
> None

**Rank #3: 60**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: 1960 (100.00%)                                                  │
│  Match #2: District 1960 (85.40%)                                          │
│                                                                            │
│  Score Difference: 14.60%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 47. Query: `Pacific Northwest Diabetes Research Inst`

✅ **Exact Match Found:** `Pacific Northwest Diabetes Research Inst` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Pacific Northwest Diabetes Research Inst | 100.00% | 1.000 | 0.950 | 0.00 | Perfect character-for-character match. |
| 2 | Pacific Northwest Diabetes Research | 83.90% | 0.769 | 1.000 | 0.00 | High word-for-word overlap. |
| 3 | Diabetes Research | 59.00% | 0.490 | 0.825 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Diabetes Research Wellness Foundation | 57.30% | 0.505 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Junior Diabetes Research Foundation | 57.10% | 0.505 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Diabetes Research Institute Foundation | 56.80% | 0.505 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Diabetes Research Institute | 53.80% | 0.435 | 0.779 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Diabetes Research Association | 52.50% | 0.435 | 0.737 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Diabetes Research & Wellness | 50.90% | 0.435 | 0.683 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | The Diabetes Research Foundation | 50.90% | 0.435 | 0.682 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Northern Diabetes Health Network | 42.70% | 0.299 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Presbyterian Diabetes Resource Center | 41.60% | 0.292 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Northern Diabetes Health | 41.30% | 0.250 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Bayer Diabetes Care Northeast | 39.30% | 0.255 | 0.712 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | California Diabetes Program | 38.20% | 0.214 | 0.775 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Pacific Northwest Diabetes Research Inst**
> None

**Rank #2: Pacific Northwest Diabetes Research**
> None

**Rank #3: Diabetes Research**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Pacific Northwest Diabetes Research Inst (100.00%)              │
│  Match #2: Pacific Northwest Diabetes Research (83.90%)                    │
│                                                                            │
│  Score Difference: 16.10%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 48. Query: `Mentors & Mentees`

✅ **Exact Match Found:** `Mentors & Mentees` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Mentors & Mentees | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | RE Mentors | 75.70% | 0.744 | 0.786 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | TRUE Mentors | 75.60% | 0.744 | 0.785 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Master Mentors | 75.50% | 0.744 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | 3 Mentors | 75.20% | 0.744 | 0.771 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Mentors Inc. | 61.20% | 0.551 | 0.755 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Mentor Mate | 52.10% | 0.375 | 0.862 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Mentor Event | 51.80% | 0.375 | 0.852 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | Mentor Meetings | 50.30% | 0.360 | 0.837 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Mentor Meeting | 49.80% | 0.360 | 0.819 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Mentor 2 Mentor | 49.00% | 0.375 | 0.757 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Mentor Masters | 48.40% | 0.341 | 0.816 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Mentor 4 | 47.00% | 0.332 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Mentor X | 46.50% | 0.332 | 0.775 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Mentor Lumber | 46.40% | 0.337 | 0.758 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Mentors & Mentees**
> None

**Rank #2: RE Mentors**
> None

**Rank #3: TRUE Mentors**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Mentors & Mentees (100.00%)                                     │
│  Match #2: RE Mentors (75.70%)                                             │
│                                                                            │
│  Score Difference: 24.30%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 49. Query: `NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46`

✅ **Exact Match Found:** `NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46 | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06 | 69.40% | 0.593 | 0.930 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 3 | NaLA 2023 fall conference M01674661235470 01-25-23 10:40:54 | 67.20% | 0.554 | 0.946 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | NALSC 2024 Annual Conference M01680804350265 04-06-23 14:05:56 | 44.70% | 0.314 | 0.757 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | HBMA 2024 Fall Conference | 40.60% | 0.253 | 0.763 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Ascend Annual Conference 2023 M01689096187969 07-11-23 13:23:11 | 39.30% | 0.243 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Fall Series 2025 M01738617336927 02-03-25 16:15:38 | 38.80% | 0.237 | 0.741 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | CWG US Conference 2022 M01639062306301 12-09-21 10:05:10 | 38.70% | 0.235 | 0.742 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Large Conference M01652296916362 05-11-22 15:22:01 | 37.60% | 0.213 | 0.756 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | OPSC Fall Conference 2025 | 35.20% | 0.155 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | NACA Conference 2024 | 35.10% | 0.141 | 0.843 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | 2022 PLSO Fall Conference | 33.80% | 0.147 | 0.785 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | 2025 NAAP Conference | 33.40% | 0.109 | 0.859 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 14 | NNU Conference 2025 | 33.40% | 0.111 | 0.854 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 15 | NABG Conference 2022 | 33.40% | 0.116 | 0.841 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46**
> None

**Rank #2: NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06**
> None

**Rank #3: NaLA 2023 fall conference M01674661235470 01-25-23 10:40:54**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46 (100.00%)│
│  Match #2: NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06 (69.40%)│
│                                                                            │
│  Score Difference: 30.60%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 50. Query: `Donnelley Work Session`

✅ **Exact Match Found:** `Donnelley Work Session` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Donnelley Work Session | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | DWS | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'DWS'. |
| 3 | SMDS Work Session | 72.70% | 0.810 | 0.533 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Donnelley Financial Services | 56.50% | 0.528 | 0.651 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Experience Session 5 | 55.60% | 0.528 | 0.622 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Donnelley Financial Solutions | 54.60% | 0.528 | 0.588 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Donnelley for Congress | 54.50% | 0.528 | 0.585 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Professional Development Session | 53.90% | 0.528 | 0.565 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | RH Donnelley Headquarters | 53.40% | 0.528 | 0.548 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | DONNELLEY MARKETING 1 | 53.30% | 0.528 | 0.545 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | KinderCare Working Session | 53.10% | 0.528 | 0.538 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Executive Breakfast Session | 52.80% | 0.528 | 0.529 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | SBH Strategy Session | 52.80% | 0.528 | 0.529 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | RR DONNELLEY FINANCIAL SERVICE | 52.70% | 0.528 | 0.524 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | R R Donnelley Logistics | 52.60% | 0.528 | 0.522 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Donnelley Work Session**
> None

**Rank #2: DWS**
> None

**Rank #3: SMDS Work Session**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Donnelley Work Session (100.00%)                                │
│  Match #2: DWS (90.00%)                                                    │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 51. Query: `North Shore Senior Center`

✅ **Exact Match Found:** `North Shore Senior Center` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | North Shore Senior Center | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | NSSC | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'NSSC'. |
| 3 | North Shore Cancer Center | 84.80% | 0.900 | 0.727 | 0.00 | High word-for-word overlap. |
| 4 | Northshore Senior Center | 84.50% | 0.857 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Coastal North Town Center | 82.30% | 0.850 | 0.761 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | North Shore Elder Services | 82.10% | 0.850 | 0.753 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | North Shore Community College | 81.10% | 0.850 | 0.719 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | North Shore Community Bank | 80.50% | 0.850 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | North Bay Regional Center | 80.50% | 0.850 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Glen Cove Senior Center | 80.40% | 0.850 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | North Coast Center | 77.80% | 0.744 | 0.858 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | East Shore Center | 74.90% | 0.744 | 0.762 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | North Shore Bank | 73.80% | 0.744 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Shore Christian Center | 73.70% | 0.744 | 0.721 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | North Shore Supply | 73.70% | 0.744 | 0.721 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: North Shore Senior Center**
> None

**Rank #2: NSSC**
> None

**Rank #3: North Shore Cancer Center**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: North Shore Senior Center (100.00%)                             │
│  Match #2: NSSC (90.00%)                                                   │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 52. Query: `Singles Who Like Food & Fun`

✅ **Exact Match Found:** `Singles Who Like Food & Fun` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Singles Who Like Food & Fun | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Fun Asian Singles | 55.20% | 0.435 | 0.826 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Christian Singles Fun Events | 55.10% | 0.505 | 0.659 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Fun Social Singles 35+ | 55.00% | 0.505 | 0.654 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Food Fun & Fellowship | 50.80% | 0.435 | 0.677 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Singles Who Dance | 50.60% | 0.435 | 0.674 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Hot Singles | 36.10% | 0.135 | 0.888 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Singles Imagine | 35.60% | 0.174 | 0.780 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Singles Adventures | 34.90% | 0.162 | 0.787 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Singles in Paradise | 34.70% | 0.211 | 0.666 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Singles Supper Club | 34.50% | 0.211 | 0.659 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Enjoy Life Foods | 34.40% | 0.194 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | For Singles By Singles | 33.90% | 0.197 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Food N Friends | 33.30% | 0.169 | 0.716 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Social Singles Adventures | 33.10% | 0.169 | 0.710 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Singles Who Like Food & Fun**
> None

**Rank #2: Fun Asian Singles**
> None

**Rank #3: Christian Singles Fun Events**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Singles Who Like Food & Fun (100.00%)                           │
│  Match #2: Fun Asian Singles (55.20%)                                      │
│                                                                            │
│  Score Difference: 44.80%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 53. Query: `Zen Meetings & Events`

✅ **Exact Match Found:** `Zen Meetings & Events` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Zen Meetings & Events | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Zen Events México | 77.50% | 0.773 | 0.781 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Zen Events Group | 72.50% | 0.638 | 0.930 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Zen Events, LLC | 66.80% | 0.638 | 0.739 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Zen at Work | 62.00% | 0.554 | 0.775 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | EVENT ZEN | 61.40% | 0.457 | 0.979 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Zen Wellness Classes | 60.60% | 0.554 | 0.728 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Zen | 59.80% | 0.472 | 0.889 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | SERVICE ZEN | 58.70% | 0.457 | 0.891 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Zen Group | 57.80% | 0.472 | 0.823 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Zen Associates | 56.70% | 0.472 | 0.788 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | PROJECT ZEN | 56.60% | 0.457 | 0.818 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Zen Business | 56.40% | 0.457 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Zen Consulting | 56.20% | 0.457 | 0.808 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Zen Planner | 55.90% | 0.457 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Zen Meetings & Events**
> None

**Rank #2: Zen Events México**
> None

**Rank #3: Zen Events Group**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Zen Meetings & Events (100.00%)                                 │
│  Match #2: Zen Events México (77.50%)                                      │
│                                                                            │
│  Score Difference: 22.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 54. Query: `Chicago South Swim Club`

✅ **Exact Match Found:** `Chicago South Swim Club` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Chicago South Swim Club | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | CSSC | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'CSSC'. |
| 3 | South Carolina Swim Club | 82.40% | 0.825 | 0.823 | 0.00 | High word-for-word overlap. |
| 4 | South West Florida Swim Club | 81.30% | 0.825 | 0.783 | 0.00 | High word-for-word overlap. |
| 5 | South Metro Storm Swim Club | 78.50% | 0.825 | 0.692 | 0.00 | High word-for-word overlap. |
| 6 | Baltimore City Swim Club | 78.00% | 0.779 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Detroit Recreation Swim Club | 77.30% | 0.779 | 0.759 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Ohio State Swim Club | 77.00% | 0.779 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Southern Kentucky Swim Club | 76.10% | 0.779 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Carolina Aquatics Swim Club | 75.90% | 0.779 | 0.711 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Team Carolina Swim Club | 75.70% | 0.779 | 0.707 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Westside YMCA Swim Club | 75.60% | 0.779 | 0.702 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | University of Michigan Swim Club | 75.60% | 0.779 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Palo Alto Swim Club | 75.40% | 0.779 | 0.695 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Los Angeles Swim Club | 75.40% | 0.779 | 0.694 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Chicago South Swim Club**
> None

**Rank #2: CSSC**
> None

**Rank #3: South Carolina Swim Club**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Chicago South Swim Club (100.00%)                               │
│  Match #2: CSSC (85.50%)                                                   │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 55. Query: `Edna, Dabra@SAP.IO`

✅ **Exact Match Found:** `Edna, Dabra@SAP.IO` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Edna, Dabra@SAP.IO | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | ED | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'ED'. |
| 3 | Edna Owusu | 75.70% | 0.744 | 0.787 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Edna Rose | 75.00% | 0.744 | 0.766 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Edna Travel | 74.50% | 0.744 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Edna ISD | 74.10% | 0.744 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | City of Edna | 73.90% | 0.744 | 0.728 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | EDNA LUMBER COMPANY | 72.50% | 0.744 | 0.683 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Cena SAP | 39.10% | 0.263 | 0.692 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | SAP Brasil | 37.20% | 0.242 | 0.675 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Nadia Saputo | 36.60% | 0.225 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | SAP Brazil | 35.20% | 0.208 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Sapna Creations | 34.80% | 0.180 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Saphis Poa | 32.90% | 0.180 | 0.675 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Sapna Creation | 32.80% | 0.180 | 0.674 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Edna, Dabra@SAP.IO**
> None

**Rank #2: ED**
> None

**Rank #3: Edna Owusu**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Edna, Dabra@SAP.IO (100.00%)                                    │
│  Match #2: ED (85.50%)                                                     │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 56. Query: `Boys and Girls Club of Dawson Community Centre`

✅ **Exact Match Found:** `Boys and Girls Club of Dawson Community Centre` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Boys and Girls Club of Dawson Community Centre | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | GIRLS CLUB | 54.00% | 0.459 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Dawson Community College | 51.20% | 0.355 | 0.880 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | DAWSON COMMUNITY BLUES | 50.70% | 0.355 | 0.862 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | GIRLS Bridge Club | 48.10% | 0.377 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Dawson County School District | 38.40% | 0.235 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Dawson County Schools | 37.40% | 0.204 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | DAWSON COUNTY EDUCATION COOPERATIVE | 37.00% | 0.218 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Dawson Reunion | 36.00% | 0.144 | 0.864 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Dawson Dance Team | 35.90% | 0.163 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | DAWSON FAMILY REUNION | 35.60% | 0.160 | 0.812 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Dawson Family | 35.20% | 0.133 | 0.862 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | Dawson Carter Family Reunion | 34.50% | 0.182 | 0.725 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Jackson Dawson Reunion | 34.40% | 0.171 | 0.750 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | DAWSON COLLEGE | 34.30% | 0.135 | 0.827 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Boys and Girls Club of Dawson Community Centre**
> None

**Rank #2: GIRLS CLUB**
> None

**Rank #3: Dawson Community College**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Boys and Girls Club of Dawson Community Centre (100.00%)        │
│  Match #2: GIRLS CLUB (54.00%)                                             │
│                                                                            │
│  Score Difference: 46.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 57. Query: `Beissbarth`

✅ **Exact Match Found:** `Beissbarth` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Beissbarth | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Beissbarth GmbH | 88.10% | 0.900 | 0.838 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | Breitbart | 42.40% | 0.332 | 0.639 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Bitbar | 42.30% | 0.281 | 0.755 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Brietbart | 40.30% | 0.284 | 0.682 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Ziebart | 39.50% | 0.265 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Versabar | 38.10% | 0.250 | 0.687 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Rebar | 38.00% | 0.240 | 0.705 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | ABBARCH | 37.90% | 0.265 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Ubar | 37.70% | 0.193 | 0.807 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | NettBar | 37.30% | 0.212 | 0.750 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | SideBar | 36.40% | 0.212 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | HandleBar | 36.30% | 0.189 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | MakerBar | 36.00% | 0.200 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | backbar | 35.90% | 0.212 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Beissbarth**
> None

**Rank #2: Beissbarth GmbH**
> None

**Rank #3: Breitbart**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Beissbarth (100.00%)                                            │
│  Match #2: Beissbarth GmbH (88.10%)                                        │
│                                                                            │
│  Score Difference: 11.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 58. Query: `US Night Vision`

✅ **Exact Match Found:** `US Night Vision` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | US Night Vision | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | PM Night Vision | 81.50% | 0.810 | 0.827 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | US Vision Care | 79.60% | 0.810 | 0.763 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Night Vision Entertainment | 79.50% | 0.810 | 0.760 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Night Vision Manufacturers | 79.40% | 0.810 | 0.756 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | WORLD VISION US | 79.20% | 0.810 | 0.751 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | NIGHT VISION SYSTEMS INC | 78.80% | 0.810 | 0.739 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | ITT Night Vision | 78.60% | 0.810 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Night Vision | 75.80% | 0.681 | 0.937 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | US Vision | 72.40% | 0.668 | 0.854 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | US Vision Inc | 68.50% | 0.668 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Big Night America | 58.90% | 0.528 | 0.733 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | World Vision USA | 58.20% | 0.528 | 0.709 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Smart Vision Lights | 58.00% | 0.528 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | A Night For Sight | 57.80% | 0.528 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: US Night Vision**
> None

**Rank #2: PM Night Vision**
> None

**Rank #3: US Vision Care**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: US Night Vision (100.00%)                                       │
│  Match #2: PM Night Vision (81.50%)                                        │
│                                                                            │
│  Score Difference: 18.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 59. Query: `Amedysis, Incorporated`

✅ **Exact Match Found:** `Amedysis, Incorporated` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Amedysis, Incorporated | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Amedysis, Inc. | 95.80% | 1.000 | 0.861 | 0.00 | High word-for-word overlap. |
| 3 | Amedysis Home Health | 89.50% | 0.900 | 0.882 | 0.00 | High word-for-word overlap. |
| 4 | AI | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'AI'. |
| 5 | Medysis | 55.20% | 0.375 | 0.964 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Avysis | 46.00% | 0.321 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Lysis | 44.50% | 0.277 | 0.838 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Unysis Corporation | 43.70% | 0.257 | 0.858 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | Dialysis Corporation | 43.10% | 0.281 | 0.781 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | INVENYSIS | 43.10% | 0.265 | 0.819 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Cardialysis | 42.90% | 0.284 | 0.766 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Affiliated Dialysis | 41.80% | 0.233 | 0.849 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Intelysis | 41.40% | 0.265 | 0.762 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Vysis, Inc. | 41.30% | 0.277 | 0.730 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Personalysis Corporation | 40.60% | 0.225 | 0.829 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Amedysis, Incorporated**
> None

**Rank #2: Amedysis, Inc.**
> None

**Rank #3: Amedysis Home Health**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Amedysis, Incorporated (100.00%)                                │
│  Match #2: Amedysis, Inc. (95.80%)                                         │
│                                                                            │
│  Score Difference: 4.20%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 60. Query: `Taiyo Air Service Co.,Ltd`

✅ **Exact Match Found:** `Taiyo Air Service Co.,Ltd` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Taiyo Air Service Co.,Ltd | 100.00% | 1.000 | 0.961 | 0.00 | Perfect character-for-character match. |
| 2 | TASC | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'TASC'. |
| 3 | Taiyo Air Services Co. | 78.80% | 0.697 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Fuyo Air Service Co. Ltd. | 73.70% | 0.697 | 0.831 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | CITS Taikoo Air Service Ltd | 72.80% | 0.744 | 0.690 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Air Service Development | 65.20% | 0.651 | 0.656 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Tec Air Service | 65.10% | 0.651 | 0.652 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | World Air Service | 64.80% | 0.651 | 0.640 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Air Service | 60.90% | 0.551 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | AIR SERVICE CORP | 60.30% | 0.551 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Air Service Corporation | 58.70% | 0.551 | 0.671 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Service Air | 56.00% | 0.490 | 0.723 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Nichiyo Air Services | 52.00% | 0.427 | 0.737 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Taiyo Pacific Partners | 51.30% | 0.427 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Air Craft Services | 49.70% | 0.427 | 0.661 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Taiyo Air Service Co.,Ltd**
> None

**Rank #2: TASC**
> None

**Rank #3: Taiyo Air Services Co.**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Taiyo Air Service Co.,Ltd (100.00%)                             │
│  Match #2: TASC (85.50%)                                                   │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 61. Query: `National Conference on Race & Ethnicity in American Higher E`

✅ **Exact Match Found:** `National Conference on Race & Ethnicity in American Higher E` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | National Conference on Race & Ethnicity in American Higher E | 100.00% | 1.000 | 0.971 | 0.00 | Perfect character-for-character match. |
| 2 | NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER | 88.80% | 0.875 | 0.918 | 0.00 | High word-for-word overlap. |
| 3 | NCORE NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER EDUCATION | 86.40% | 0.900 | 0.778 | 0.00 | High word-for-word overlap. |
| 4 | National Conference On Race & Ethnicity In America Higher Ed | 86.30% | 0.849 | 0.897 | 0.00 | High word-for-word overlap. |
| 5 | National Conference on Race & Ethnicity in AM Higher Education | 82.80% | 0.849 | 0.779 | 0.00 | High word-for-word overlap. |
| 6 | National Conference on Race & Ethnicity in Am. Higher Educ | 82.70% | 0.849 | 0.777 | 0.00 | High word-for-word overlap. |
| 7 | National Conference on Race & Ethnicity in Higher Education | 79.90% | 0.802 | 0.792 | 0.00 | High word-for-word overlap. |
| 8 | National Conference on Race & Ethnicity in AmericanHigher Ed | 74.60% | 0.730 | 0.783 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | National Conference on Race Ethnicity | 70.90% | 0.584 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | National Conference on Race & Ethnicity | 68.40% | 0.584 | 0.916 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | National Conference on Race and Ethnicity | 68.00% | 0.584 | 0.902 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | NATIONAL CONFERENCE ON RACE & ETHNICITY - NCORE | 67.20% | 0.622 | 0.790 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | NAT CONFERENCE ON RACE & ETHNICITY | 55.30% | 0.440 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | National Conference on African Americans | 47.20% | 0.352 | 0.752 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Trans American Race Co | 35.40% | 0.154 | 0.821 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: National Conference on Race & Ethnicity in American Higher E**
> None

**Rank #2: NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER**
> None

**Rank #3: NCORE NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER EDUCATION**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: National Conference on Race & Ethnicity in American Higher E (100.00%)│
│  Match #2: NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER (88.80%)│
│                                                                            │
│  Score Difference: 11.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 62. Query: `Reminger Law Firm`

✅ **Exact Match Found:** `Reminger Law Firm` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Reminger Law Firm | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Reminger & Reminger Law Firm | 97.00% | 1.000 | 0.901 | 0.00 | Substring match (target contains query text). |
| 3 | RLF | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'RLF'. |
| 4 | Withers Law Firm | 82.30% | 0.810 | 0.854 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | S Law Firm | 81.90% | 0.810 | 0.841 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Speer Law Firm | 81.80% | 0.810 | 0.838 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Didier Law Firm | 81.30% | 0.810 | 0.820 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Marr Law Firm | 81.20% | 0.810 | 0.818 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Gremminger Law Firm | 81.20% | 0.810 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Arzinger Law Firm | 81.10% | 0.810 | 0.813 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | RA Law Firm | 80.80% | 0.810 | 0.804 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Alters Law Firm | 80.70% | 0.810 | 0.802 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Revo Law Firm | 80.70% | 0.810 | 0.800 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | DC Law Firm | 80.70% | 0.810 | 0.800 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | B Law Firm | 80.60% | 0.810 | 0.799 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Reminger Law Firm**
> None

**Rank #2: Reminger & Reminger Law Firm**
> None

**Rank #3: RLF**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Reminger Law Firm (100.00%)                                     │
│  Match #2: Reminger & Reminger Law Firm (97.00%)                           │
│                                                                            │
│  Score Difference: 3.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 63. Query: `SEMMOA BOD`

✅ **Exact Match Found:** `SEMMOA BOD` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | SEMMOA BOD | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | SEMMOA AACM | 74.00% | 0.744 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Bod Pro | 73.50% | 0.744 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | SEMMOA Coop | 73.20% | 0.744 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | CSA BOD | 72.20% | 0.744 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | World Bod | 70.30% | 0.744 | 0.610 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Semmoa | 65.80% | 0.551 | 0.906 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | BoD | 62.30% | 0.551 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | SEMM | 34.00% | 0.180 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | SE Production | 32.30% | 0.196 | 0.619 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Sem Jacar | 32.20% | 0.189 | 0.631 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | SED | 32.00% | 0.145 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Sewbo | 31.20% | 0.168 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | BODE | 30.60% | 0.145 | 0.681 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Bode Pro | 30.30% | 0.159 | 0.641 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: SEMMOA BOD**
> None

**Rank #2: SEMMOA AACM**
> None

**Rank #3: Bod Pro**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: SEMMOA BOD (100.00%)                                            │
│  Match #2: SEMMOA AACM (74.00%)                                            │
│                                                                            │
│  Score Difference: 26.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 64. Query: `Telefonica Global Solutions`

✅ **Exact Match Found:** `Telefonica Global Solutions` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Telefonica Global Solutions | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Telefonica Multinational Solutions | 90.10% | 0.883 | 0.942 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 3 | Telefonica Global Solutions USA Inc. | 90.10% | 0.931 | 0.830 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 4 | TGS | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'TGS'. |
| 5 | TELEFONICA INTERNATIONAL USA | 71.70% | 0.633 | 0.912 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Telefonica International USA Inc | 70.00% | 0.633 | 0.856 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Telefonica International Wholesale Services | 69.30% | 0.633 | 0.833 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Telefonica Internacional USA | 68.20% | 0.633 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Telefonica | 66.80% | 0.540 | 0.965 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Telefonica International | 66.80% | 0.528 | 0.993 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Telefonica USA | 63.50% | 0.522 | 0.899 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | 02 Telefonica | 63.00% | 0.522 | 0.880 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | Telefonica España | 61.80% | 0.522 | 0.842 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Telefonica Chile | 61.30% | 0.522 | 0.823 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Grupo Telefonica | 61.20% | 0.522 | 0.821 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Telefonica Global Solutions**
> None

**Rank #2: Telefonica Multinational Solutions**
> None

**Rank #3: Telefonica Global Solutions USA Inc.**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Telefonica Global Solutions (100.00%)                           │
│  Match #2: Telefonica Multinational Solutions (90.10%)                     │
│                                                                            │
│  Score Difference: 9.90%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 65. Query: `Travel Leaders - Dube Travel`

✅ **Exact Match Found:** `Travel Leaders - Dube Travel` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Travel Leaders - Dube Travel | 100.00% | 1.000 | 0.980 | 0.00 | Perfect character-for-character match. |
| 2 | Dube Travel Leaders | 100.00% | 1.000 | 1.000 | 0.00 | High word-for-word overlap. |
| 3 | Dube Travel / Travel Leaders | 93.40% | 0.950 | 0.896 | 0.00 | High word-for-word overlap. |
| 4 | Dube / Travel Leaders | 93.00% | 0.950 | 0.855 | 0.00 | High word-for-word overlap. |
| 5 | Dube Travel/Travel Leaders | 88.70% | 0.883 | 0.896 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Travel Leaders Go | 85.10% | 0.810 | 0.949 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Travel Leaders Travel Quest | 84.20% | 0.810 | 0.918 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | TRAVEL LEADERS INTERNATIONAL | 83.70% | 0.810 | 0.902 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | Travel Leaders Travel Agency | 83.70% | 0.810 | 0.899 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | TRAVEL LEADERS WORLDWIDE | 83.40% | 0.810 | 0.889 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Travel Leaders Network | 83.30% | 0.810 | 0.889 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Travel Leaders Travel Now | 83.10% | 0.810 | 0.881 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | Travel Leaders WI | 83.10% | 0.810 | 0.880 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 14 | LEADERS IN TRAVEL | 82.90% | 0.810 | 0.873 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 15 | Travel Leaders Travel More | 82.70% | 0.810 | 0.868 | 0.00 | Matched via strong semantic/conceptual similarity. |

### Match Narratives

**Rank #1: Travel Leaders - Dube Travel**
> None

**Rank #2: Dube Travel Leaders**
> None

**Rank #3: Dube Travel / Travel Leaders**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Travel Leaders - Dube Travel (100.00%)                          │
│  Match #2: Dube Travel Leaders (100.00%)                                   │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 66. Query: `Hi- Tours`

✅ **Exact Match Found:** `Hi- Tours` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Hi- Tours | 100.00% | 1.000 | 0.882 | 0.00 | Perfect character-for-character match. |
| 2 | Hi Tours | 100.00% | 0.900 | 1.000 | 0.00 | High word-for-word overlap. |
| 3 | Hi-Tours | 100.00% | 0.900 | 0.882 | 0.00 | High word-for-word overlap. |
| 4 | Hi Tour | 96.70% | 0.787 | 0.884 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | Hi Tours2 | 96.30% | 0.787 | 0.874 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Hi Life Tours | 93.00% | 0.950 | 0.837 | 0.00 | High word-for-word overlap. |
| 7 | HI LITE TOURS | 93.00% | 0.950 | 0.780 | 0.00 | High word-for-word overlap. |
| 8 | HiFiveLive Tours | 90.60% | 0.744 | 0.786 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | HT | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'HT'. |
| 10 | Journey Tours | 77.10% | 0.744 | 0.836 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Excursion Tours | 76.80% | 0.744 | 0.826 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Nice Tours | 76.70% | 0.744 | 0.821 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Sky Tours | 76.50% | 0.744 | 0.816 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | DESTINATION TOURS | 76.40% | 0.744 | 0.812 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Journeys Tours | 76.40% | 0.744 | 0.811 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Hi- Tours**
> None

**Rank #2: Hi Tours**
> None

**Rank #3: Hi-Tours**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Hi- Tours (100.00%)                                             │
│  Match #2: Hi Tours (100.00%)                                              │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 67. Query: `Volkswagen Group China`

✅ **Exact Match Found:** `Volkswagen Group China` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Volkswagen Group China | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Volkswagen China | 99.00% | 1.000 | 0.968 | 0.00 | High word-for-word overlap. |
| 3 | VOLKSWAGEN CHINA INVESTMENT COMPANY LTD | 84.50% | 0.900 | 0.716 | 0.00 | High word-for-word overlap. |
| 4 | Volkswagen Group Japan | 76.60% | 0.744 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Volkswagen Group Australia | 75.70% | 0.744 | 0.788 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | VOLKSWAGEN KOREA | 75.20% | 0.744 | 0.771 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Grupo Volkswagen | 75.10% | 0.744 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Group Volkswagen Spain | 75.00% | 0.744 | 0.766 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | VOLKSWAGEN Group Rus | 74.70% | 0.744 | 0.754 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Volkswagen Singapore | 74.60% | 0.744 | 0.752 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Volvo Group China | 74.50% | 0.744 | 0.749 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Volkswagen Australia | 74.30% | 0.744 | 0.741 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Volkswagen Group UK | 74.20% | 0.744 | 0.738 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | VOLKSWAGEN GERMANY | 74.20% | 0.744 | 0.736 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Volkswagen AG | 74.10% | 0.744 | 0.735 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Volkswagen Group China**
> None

**Rank #2: Volkswagen China**
> None

**Rank #3: VOLKSWAGEN CHINA INVESTMENT COMPANY LTD**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Volkswagen Group China (100.00%)                                │
│  Match #2: Volkswagen China (99.00%)                                       │
│                                                                            │
│  Score Difference: 1.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 68. Query: `Sun Tx`

✅ **Exact Match Found:** `Sun Tx` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Sun Tx | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | SUN TRAN | 87.70% | 0.744 | 0.687 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Sun Trans | 86.90% | 0.744 | 0.662 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Sun Tan City | 86.30% | 0.744 | 0.666 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Sun City Texas | 77.10% | 0.744 | 0.835 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Standard Sun | 73.70% | 0.744 | 0.722 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Sun City | 73.50% | 0.744 | 0.713 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Sun Am | 73.40% | 0.744 | 0.711 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Sun Outdoors | 73.40% | 0.744 | 0.710 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | sun coast | 73.30% | 0.744 | 0.707 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Rising Sun | 73.20% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Sun Com | 73.20% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Sun Vista | 73.20% | 0.744 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Sun Space | 72.90% | 0.744 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | New Sun | 72.80% | 0.744 | 0.691 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Sun Tx**
> None

**Rank #2: SUN TRAN**
> None

**Rank #3: Sun Trans**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Sun Tx (100.00%)                                                │
│  Match #2: SUN TRAN (87.70%)                                               │
│                                                                            │
│  Score Difference: 12.30%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 69. Query: `Southern Vermont Deerfield Valley Chamber of commerce`

✅ **Exact Match Found:** `Southern Vermont Deerfield Valley Chamber of commerce` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Southern Vermont Deerfield Valley Chamber of commerce | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Deerfield Beach Chamber of Commerce | 66.80% | 0.614 | 0.793 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Chamber of Commerce Mid-Ohio Valley | 65.80% | 0.633 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Northwest Valley Chamber of Commerce | 62.30% | 0.570 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | South Valley Chamber of Commerce | 62.00% | 0.570 | 0.738 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Greater Valley Chamber of Commerce | 61.80% | 0.570 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Utah Valley Chamber of Commerce | 61.70% | 0.570 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Eagle Valley Chamber of Commerce | 61.20% | 0.570 | 0.712 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Deerfield Chamber of Commerce | 60.90% | 0.490 | 0.888 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Vermont Chamber of Commerce | 59.40% | 0.490 | 0.835 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Deer Park Chamber of Commerce | 56.10% | 0.425 | 0.880 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Northern Virginia Chamber of Commerce | 52.60% | 0.425 | 0.764 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Virginia Peninsula Chamber of Commerce | 52.10% | 0.425 | 0.746 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Wyoming State Chamber of Commerce | 52.00% | 0.425 | 0.742 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Midwest City Chamber of Commerce | 51.70% | 0.425 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Southern Vermont Deerfield Valley Chamber of commerce**
> None

**Rank #2: Deerfield Beach Chamber of Commerce**
> None

**Rank #3: Chamber of Commerce Mid-Ohio Valley**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Southern Vermont Deerfield Valley Chamber of commerce (100.00%) │
│  Match #2: Deerfield Beach Chamber of Commerce (66.80%)                    │
│                                                                            │
│  Score Difference: 33.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 70. Query: `DGR Ministries`

✅ **Exact Match Found:** `DGR Ministries` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | DGR Ministries | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | DM | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'DM'. |
| 3 | DG Ministries | 76.30% | 0.744 | 0.808 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Power Ministries | 74.50% | 0.744 | 0.748 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Impact Ministries | 73.50% | 0.744 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Progressive Ministries | 73.20% | 0.744 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Empowered Ministries | 73.10% | 0.744 | 0.700 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Total Ministries | 73.00% | 0.744 | 0.697 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Customized Ministries | 72.80% | 0.744 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Chosen Generation Ministries | 72.80% | 0.744 | 0.690 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Legacy Ministries | 72.80% | 0.744 | 0.690 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Chosen Ministries | 72.80% | 0.744 | 0.690 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Caring Ministries | 72.70% | 0.744 | 0.689 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | CHIEF MINISTRIES | 72.70% | 0.744 | 0.687 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Regional Ministries | 72.60% | 0.744 | 0.685 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: DGR Ministries**
> None

**Rank #2: DM**
> None

**Rank #3: DG Ministries**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: DGR Ministries (100.00%)                                        │
│  Match #2: DM (90.00%)                                                     │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 71. Query: `Impacto 6`

✅ **Exact Match Found:** `Impacto 6` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Impacto 6 | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Impacto 52 | 83.80% | 0.850 | 0.809 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Impacto Strategies | 83.40% | 0.850 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Impacto Vital | 81.20% | 0.850 | 0.724 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Triple Impacto | 81.00% | 0.850 | 0.717 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Impacto Tactico | 80.70% | 0.850 | 0.707 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Impacto de Fe | 80.50% | 0.850 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Impacto EDL | 80.10% | 0.850 | 0.686 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Primer Impacto | 78.90% | 0.850 | 0.648 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Impacto Productive Products | 78.60% | 0.850 | 0.638 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Impacto YOUTH | 77.50% | 0.850 | 0.600 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Impacto Inc. | 64.30% | 0.630 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Impact 365 | 45.60% | 0.375 | 0.646 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Impact 360 | 44.50% | 0.375 | 0.608 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Impact ON | 43.50% | 0.350 | 0.632 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Impacto 6**
> None

**Rank #2: Impacto 52**
> None

**Rank #3: Impacto Strategies**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Impacto 6 (100.00%)                                             │
│  Match #2: Impacto 52 (83.80%)                                             │
│                                                                            │
│  Score Difference: 16.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 72. Query: `Neos Therapeutics, Inc.`

✅ **Exact Match Found:** `Neos Therapeutics, Inc.` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Neos Therapeutics, Inc. | 100.00% | 1.000 | 0.874 | 0.00 | Perfect character-for-character match. |
| 2 | Neos Therapeutics | 93.70% | 0.917 | 0.984 | 0.00 | High word-for-word overlap. |
| 3 | NTI | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'NTI'. |
| 4 | Neos Therapeutics LP | 85.20% | 0.842 | 0.876 | 0.00 | High word-for-word overlap. |
| 5 | Neogene Therapeutics, Inc. | 76.90% | 0.779 | 0.745 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Neogene Therapeutics | 74.00% | 0.708 | 0.813 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Neos Partners | 73.10% | 0.708 | 0.785 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | NEOS Central | 73.10% | 0.708 | 0.784 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Neos Spa | 72.30% | 0.708 | 0.759 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Neos Therapeutic | 59.20% | 0.417 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Neo Medical Inc | 44.30% | 0.242 | 0.910 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | NEOS | 43.40% | 0.233 | 0.904 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | Neo Medical | 42.70% | 0.242 | 0.856 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 14 | Neo Medic | 42.30% | 0.263 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Neo Health Services | 42.10% | 0.281 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Neos Therapeutics, Inc.**
> None

**Rank #2: Neos Therapeutics**
> None

**Rank #3: NTI**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Neos Therapeutics, Inc. (100.00%)                               │
│  Match #2: Neos Therapeutics (93.70%)                                      │
│                                                                            │
│  Score Difference: 6.30%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 73. Query: `International Tax Institute`

✅ **Exact Match Found:** `International Tax Institute` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | International Tax Institute | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | INTERNATIONAL TAX INSTITUTE INC | 97.70% | 1.000 | 0.924 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | International Property Tax Institute | 93.00% | 0.950 | 0.858 | 0.00 | High word-for-word overlap. |
| 4 | International Tax & Auditing Institute | 93.00% | 0.950 | 0.711 | 0.00 | High word-for-word overlap. |
| 5 | National Tax Institute | 90.40% | 0.900 | 0.914 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Professional Tax Institute | 87.40% | 0.883 | 0.852 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Tax Research Institute | 87.10% | 0.883 | 0.843 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Federal Tax Institute | 87.10% | 0.883 | 0.841 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | International Tax Form | 86.90% | 0.883 | 0.837 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Tax Executive Institute | 86.80% | 0.883 | 0.833 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | International Tax Advisors | 86.60% | 0.883 | 0.824 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | International Tax Review | 86.10% | 0.883 | 0.809 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | ITI | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'ITI'. |
| 14 | Tax Executives Institute | 85.40% | 0.883 | 0.785 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Bank Tax Institute | 85.20% | 0.883 | 0.777 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: International Tax Institute**
> None

**Rank #2: INTERNATIONAL TAX INSTITUTE INC**
> None

**Rank #3: International Property Tax Institute**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: International Tax Institute (100.00%)                           │
│  Match #2: INTERNATIONAL TAX INSTITUTE INC (97.70%)                        │
│                                                                            │
│  Score Difference: 2.30%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 74. Query: `Mitsubishi M501G`

✅ **Exact Match Found:** `Mitsubishi M501G` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Mitsubishi M501G | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | NEC Mitsubishi | 78.10% | 0.744 | 0.867 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 3 | Mitsubishi Power | 77.90% | 0.744 | 0.861 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Mitsubishi Motors | 77.90% | 0.744 | 0.860 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | MITSUBISHI MOTOR | 77.80% | 0.744 | 0.859 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Mitsubishi Electronics | 77.70% | 0.744 | 0.856 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Mitsubishi Electric | 77.40% | 0.744 | 0.844 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Mitsubishi Japan | 77.20% | 0.744 | 0.838 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Mitsubishi Canada | 76.50% | 0.744 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Mitsubishi Ingredients | 76.50% | 0.744 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Mitsubishi USA | 76.40% | 0.744 | 0.810 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Mitsubishi Engine | 76.40% | 0.744 | 0.810 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Mitsubishi Securities | 76.20% | 0.744 | 0.804 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Mitsubishi Chemical | 76.00% | 0.744 | 0.799 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Mitsubishi Germany | 75.90% | 0.744 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Mitsubishi M501G**
> None

**Rank #2: NEC Mitsubishi**
> None

**Rank #3: Mitsubishi Power**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Mitsubishi M501G (100.00%)                                      │
│  Match #2: NEC Mitsubishi (78.10%)                                         │
│                                                                            │
│  Score Difference: 21.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 75. Query: `Huskies Sports`

✅ **Exact Match Found:** `Huskies Sports` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Huskies Sports | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Huskies Basketball | 77.10% | 0.744 | 0.835 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Miami Huskies | 75.90% | 0.744 | 0.794 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Mass Huskies | 75.60% | 0.744 | 0.785 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Huskies Hockey League | 75.40% | 0.744 | 0.778 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Huskies Hockey Club | 74.90% | 0.744 | 0.762 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Dual State Huskies | 73.70% | 0.744 | 0.721 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | NH Huskies | 73.70% | 0.744 | 0.720 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Huskies Elite Cheer | 73.40% | 0.744 | 0.711 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Huskies Baseball State Tournament | 73.20% | 0.744 | 0.705 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Howard Huskies | 72.70% | 0.744 | 0.687 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | UConn Huskies Athletics | 72.30% | 0.744 | 0.674 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Horizon Huskies | 72.20% | 0.744 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | HRS Huskies | 72.10% | 0.744 | 0.668 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Empire State Huskies | 72.00% | 0.744 | 0.664 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Huskies Sports**
> None

**Rank #2: Huskies Basketball**
> None

**Rank #3: Miami Huskies**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Huskies Sports (100.00%)                                        │
│  Match #2: Huskies Basketball (77.10%)                                     │
│                                                                            │
│  Score Difference: 22.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 76. Query: `Acacia Pharma Group Inc.`

✅ **Exact Match Found:** `Acacia Pharma Group Inc.` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Acacia Pharma Group Inc. | 100.00% | 1.000 | 0.929 | 0.00 | Perfect character-for-character match. |
| 2 | ACACIA PHARMA, Inc. | 94.10% | 1.000 | 0.804 | 0.00 | High word-for-word overlap. |
| 3 | Acacia Pharma | 92.20% | 0.917 | 0.935 | 0.00 | High word-for-word overlap. |
| 4 | Acacia Pharma Ltd | 91.70% | 0.917 | 0.918 | 0.00 | High word-for-word overlap. |
| 5 | Acacia Research Group | 77.40% | 0.708 | 0.926 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Acacia Network Inc. | 77.20% | 0.779 | 0.756 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Acacia Partners | 75.80% | 0.708 | 0.874 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Acacia Travel Inc. | 75.80% | 0.779 | 0.708 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Acacia Financial Group | 75.50% | 0.708 | 0.863 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Acacia Marketing  Group | 75.00% | 0.708 | 0.847 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Acacia Marketing Group | 75.00% | 0.708 | 0.847 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Acacia Research Corp | 74.90% | 0.708 | 0.844 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Acacia Institute | 74.70% | 0.708 | 0.836 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Acacia Systems | 73.70% | 0.708 | 0.805 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | ACACIA NETWORK | 73.60% | 0.708 | 0.802 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Acacia Pharma Group Inc.**
> None

**Rank #2: ACACIA PHARMA, Inc.**
> None

**Rank #3: Acacia Pharma**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Acacia Pharma Group Inc. (100.00%)                              │
│  Match #2: ACACIA PHARMA, Inc. (94.10%)                                    │
│                                                                            │
│  Score Difference: 5.90%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 77. Query: `Acumatica Summit 2017 Z7NWPDKS625`

✅ **Exact Match Found:** `Acumatica Summit 2017 Z7NWPDKS625` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Acumatica Summit 2017 Z7NWPDKS625 | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Acumatica User Group Southeast | 52.80% | 0.438 | 0.740 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Contact Center Compliance Summit | 51.70% | 0.500 | 0.557 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Acumatica Asia | 50.30% | 0.350 | 0.859 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | ACUMATICA PRESIDENTS CLUB | 49.60% | 0.438 | 0.631 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Acumatica - Sales Office | 49.50% | 0.438 | 0.628 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Acumatica The Cloud ERP | 49.20% | 0.438 | 0.619 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Medical Device Summit | 47.50% | 0.438 | 0.561 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | ACMG Icws 2017 | 44.40% | 0.394 | 0.561 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | ACA Summit | 41.90% | 0.350 | 0.581 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Zillow Summit | 41.90% | 0.350 | 0.580 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | POWER SUMMIT | 41.50% | 0.350 | 0.567 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Summit Tire | 41.40% | 0.350 | 0.563 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | MNG Summit | 41.40% | 0.350 | 0.562 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Medical Summit | 41.30% | 0.350 | 0.558 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Acumatica Summit 2017 Z7NWPDKS625**
> None

**Rank #2: Acumatica User Group Southeast**
> None

**Rank #3: Contact Center Compliance Summit**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Acumatica Summit 2017 Z7NWPDKS625 (100.00%)                     │
│  Match #2: Acumatica User Group Southeast (52.80%)                         │
│                                                                            │
│  Score Difference: 47.20%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 78. Query: `Linklaters CIS`

✅ **Exact Match Found:** `Linklaters CIS` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Linklaters CIS | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | LC | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'LC'. |
| 3 | CIS Partners | 76.20% | 0.744 | 0.805 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Cis GmbH | 75.80% | 0.744 | 0.791 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | One Cis | 75.70% | 0.744 | 0.788 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | CIS Method | 75.60% | 0.744 | 0.786 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Cis 22 | 75.20% | 0.744 | 0.773 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | CIS Groups | 74.90% | 0.744 | 0.761 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Cis Global | 74.60% | 0.744 | 0.753 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Cis Technologies | 74.10% | 0.744 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Cis Secure | 73.70% | 0.744 | 0.722 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Groupe CIS | 73.40% | 0.744 | 0.713 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Affiliates of Cis | 73.30% | 0.744 | 0.709 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | CIS Conf | 73.30% | 0.744 | 0.708 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Trio Cis | 73.30% | 0.744 | 0.707 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Linklaters CIS**
> None

**Rank #2: LC**
> None

**Rank #3: CIS Partners**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Linklaters CIS (100.00%)                                        │
│  Match #2: LC (90.00%)                                                     │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 79. Query: `Christian Girls Family Ministry`

✅ **Exact Match Found:** `Christian Girls Family Ministry` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Christian Girls Family Ministry | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Christian Girls Family Ministry Training | 88.40% | 0.900 | 0.846 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | Christian Faith Fellowship Couples Ministry | 74.50% | 0.744 | 0.748 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Christian Family Fellowship Church | 74.50% | 0.744 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | FRIENDSHIP CHRISTIAN CHURCH MINISTRY | 74.40% | 0.744 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Christian Bible Church Marriage Ministry | 73.90% | 0.744 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Community Outreach Christian Ministry | 73.80% | 0.744 | 0.725 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Christian Growth Family Ministries | 73.30% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Christian Family Bible Fellowship | 73.30% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Christian Friends Ministry | 70.20% | 0.651 | 0.822 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Christian Womens Ministry | 70.10% | 0.651 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Christian Youth Ministry | 69.70% | 0.651 | 0.805 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Christian Marriage Ministry | 69.50% | 0.651 | 0.797 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Christian Community Ministry | 69.40% | 0.651 | 0.794 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Family Christian Church | 68.60% | 0.651 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Christian Girls Family Ministry**
> None

**Rank #2: Christian Girls Family Ministry Training**
> None

**Rank #3: Christian Faith Fellowship Couples Ministry**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Christian Girls Family Ministry (100.00%)                       │
│  Match #2: Christian Girls Family Ministry Training (88.40%)               │
│                                                                            │
│  Score Difference: 11.60%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 80. Query: `Alosa Foundation`

✅ **Exact Match Found:** `Alosa Foundation` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Alosa Foundation | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | AF | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'AF'. |
| 3 | Formosa Foundation | 70.50% | 0.637 | 0.863 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | CL Foundation | 66.40% | 0.637 | 0.725 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | CAP Foundation | 66.10% | 0.637 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | FIRST FOUNDATION | 66.00% | 0.637 | 0.712 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | UH FOUNDATION | 65.80% | 0.637 | 0.705 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | New Foundation | 65.80% | 0.637 | 0.705 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Foundation Workshop | 65.70% | 0.637 | 0.702 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Robina Foundation | 65.60% | 0.637 | 0.700 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Alwan Foundation | 65.60% | 0.637 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | UM Foundation | 65.40% | 0.637 | 0.693 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | NEST Foundation | 65.40% | 0.637 | 0.692 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Llosa's foundation | 65.40% | 0.637 | 0.692 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Foundation Building | 65.30% | 0.637 | 0.690 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Alosa Foundation**
> None

**Rank #2: AF**
> None

**Rank #3: Formosa Foundation**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Alosa Foundation (100.00%)                                      │
│  Match #2: AF (90.00%)                                                     │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 81. Query: `La Chaine des Rotisseurs Wine Club of Newport Beach`

✅ **Exact Match Found:** `La Chaine des Rotisseurs Wine Club of Newport Beach` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | La Chaine des Rotisseurs Wine Club of Newport Beach | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | La Chaine des Rotisseurs Wine Club of Ne | 82.60% | 0.783 | 0.925 | 0.00 | High word-for-word overlap. |
| 3 | Newport Beach Wine Festival | 52.80% | 0.390 | 0.852 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Southern Trace Wine Club | 49.10% | 0.345 | 0.832 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Cleveland Wine Club | 42.40% | 0.246 | 0.838 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Diversity Wine Club | 41.80% | 0.246 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Queen Wine Club | 41.70% | 0.246 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | The Virginia Wine Club | 41.00% | 0.246 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Groupe Wine Club | 40.70% | 0.246 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Wine La | 39.90% | 0.217 | 0.823 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Wine Club | 35.30% | 0.100 | 0.945 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Vintage West Wine Marketing | 33.90% | 0.133 | 0.819 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Classic Wine of California | 33.20% | 0.135 | 0.793 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Wine Connoisseur Magazine | 33.20% | 0.131 | 0.800 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Bowler Wine Merchants | 32.70% | 0.120 | 0.808 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: La Chaine des Rotisseurs Wine Club of Newport Beach**
> None

**Rank #2: La Chaine des Rotisseurs Wine Club of Ne**
> None

**Rank #3: Newport Beach Wine Festival**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: La Chaine des Rotisseurs Wine Club of Newport Beach (100.00%)   │
│  Match #2: La Chaine des Rotisseurs Wine Club of Ne (82.60%)               │
│                                                                            │
│  Score Difference: 17.40%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 82. Query: `Sumner & Ryan, LLC`

✅ **Exact Match Found:** `Sumner & Ryan, LLC` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Sumner & Ryan, LLC | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Miller Ryan LLC | 77.20% | 0.744 | 0.837 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Ryan Moving LLC | 77.10% | 0.744 | 0.835 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Ryan Companies | 76.40% | 0.744 | 0.811 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Sumner 360 | 75.10% | 0.744 | 0.768 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Ryan Sellers | 75.00% | 0.744 | 0.764 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Ryan Specialty, LLC | 74.90% | 0.744 | 0.760 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Ryan Law Firm | 74.80% | 0.744 | 0.760 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Ryan Ryan Law Firm | 74.80% | 0.744 | 0.757 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Ryan McCall | 74.50% | 0.744 | 0.747 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Ryan Racing LLC | 74.20% | 0.744 | 0.737 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Ryan Contracting | 73.90% | 0.744 | 0.729 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Ryan Read, LLC | 73.50% | 0.744 | 0.716 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Ryan Enterprise | 73.50% | 0.744 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Sumner Baseball | 73.50% | 0.744 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Sumner & Ryan, LLC**
> None

**Rank #2: Miller Ryan LLC**
> None

**Rank #3: Ryan Moving LLC**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Sumner & Ryan, LLC (100.00%)                                    │
│  Match #2: Miller Ryan LLC (77.20%)                                        │
│                                                                            │
│  Score Difference: 22.80%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 83. Query: `Tilt Creative & Production`

✅ **Exact Match Found:** `Tilt Creative & Production` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Tilt Creative & Production | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Tilt Creative + Production | 95.10% | 0.960 | 0.929 | 0.00 | High word-for-word overlap. |
| 3 | Creative Production Design | 79.80% | 0.810 | 0.772 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Creative Production Incentives | 77.10% | 0.810 | 0.681 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Tilt Production | 74.50% | 0.668 | 0.925 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Full Tilt Marketing | 57.50% | 0.528 | 0.684 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | ON TILT ENTERTAINMENT | 57.40% | 0.528 | 0.682 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Creative Talent Endeavors | 56.60% | 0.528 | 0.654 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Tilt Promotions | 53.00% | 0.435 | 0.750 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Creative Productions | 52.00% | 0.443 | 0.700 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Brightly Creative | 52.00% | 0.435 | 0.716 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Creative Direction | 51.90% | 0.435 | 0.714 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Creative Creations | 51.70% | 0.435 | 0.707 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Creative Fusion | 51.60% | 0.435 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Creative Works | 51.40% | 0.435 | 0.696 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Tilt Creative & Production**
> None

**Rank #2: Tilt Creative + Production**
> None

**Rank #3: Creative Production Design**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Tilt Creative & Production (100.00%)                            │
│  Match #2: Tilt Creative + Production (95.10%)                             │
│                                                                            │
│  Score Difference: 4.90%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 84. Query: `Cerberus Capital`

✅ **Exact Match Found:** `Cerberus Capital` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Cerberus Capital | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Cerberus Capital Management | 90.20% | 0.900 | 0.907 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 3 | Cerberus Capital Management L | 88.00% | 0.900 | 0.833 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 4 | Cerberus Capital Management LP | 87.40% | 0.900 | 0.813 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 5 | *Cerberus Capital | 82.50% | 0.787 | 0.912 | 0.00 | Substring match (target contains query text). |
| 6 | Cerberus Capitol Management | 75.30% | 0.744 | 0.773 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Cerberus Global Investments | 74.70% | 0.744 | 0.756 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Cerberus Law | 74.50% | 0.744 | 0.748 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | CEC Capital | 74.30% | 0.744 | 0.742 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Siemens Cerberus | 73.00% | 0.744 | 0.699 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | CERBERUS (CHRYSLER FINANCIAL) | 72.80% | 0.744 | 0.692 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Cerberus Sentinel | 72.20% | 0.744 | 0.670 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Merus Capital | 71.30% | 0.744 | 0.642 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | CEC Capital Group | 70.80% | 0.744 | 0.624 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Siemens Cerberus Division | 70.70% | 0.744 | 0.620 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Cerberus Capital**
> None

**Rank #2: Cerberus Capital Management**
> None

**Rank #3: Cerberus Capital Management L**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Cerberus Capital (100.00%)                                      │
│  Match #2: Cerberus Capital Management (90.20%)                            │
│                                                                            │
│  Score Difference: 9.80%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 85. Query: `Institute of Health Technology Transformation`

✅ **Exact Match Found:** `Institute of Health Technology Transformation` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Institute of Health Technology Transformation | 100.00% | 1.000 | 0.984 | 0.00 | Perfect character-for-character match. |
| 2 | Institute for Health Technology Transformation | 95.80% | 0.955 | 0.967 | 0.00 | High word-for-word overlap. |
| 3 | INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION | 93.00% | 0.955 | 0.850 | 0.00 | High word-for-word overlap. |
| 4 | IHTT | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'IHTT'. |
| 5 | Health Science Technology Education | 77.30% | 0.744 | 0.842 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Achieve Health Care Technology | 75.70% | 0.744 | 0.789 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Technology Health Experience | 72.80% | 0.651 | 0.908 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Unified Health Technology | 72.00% | 0.651 | 0.883 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 9 | Health Technology Association | 71.60% | 0.651 | 0.868 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | HEALTH TECHNOLOGY ASSESSMENT | 71.20% | 0.651 | 0.855 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Health Technology Center | 71.10% | 0.651 | 0.852 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Health Technology Exchange | 70.90% | 0.651 | 0.844 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Health Information Technology | 70.70% | 0.651 | 0.839 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Future of Health Technology | 70.50% | 0.651 | 0.833 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Health Technology | 68.60% | 0.551 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |

### Match Narratives

**Rank #1: Institute of Health Technology Transformation**
> None

**Rank #2: Institute for Health Technology Transformation**
> None

**Rank #3: INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Institute of Health Technology Transformation (100.00%)         │
│  Match #2: Institute for Health Technology Transformation (95.80%)         │
│                                                                            │
│  Score Difference: 4.20%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 86. Query: `The Jones Assembly`

✅ **Exact Match Found:** `The Jones Assembly` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | The Jones Assembly | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | JA | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'JA'. |
| 3 | The Jones Assembly Presents | 87.80% | 0.900 | 0.825 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 4 | General Assembly | 74.40% | 0.744 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | 1st Assembly | 74.30% | 0.744 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Bill Jones | 74.10% | 0.744 | 0.736 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Jones Power | 74.00% | 0.744 | 0.732 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | First Assembly | 73.80% | 0.744 | 0.726 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Jones Institute | 73.60% | 0.744 | 0.719 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Jones Ag | 73.60% | 0.744 | 0.718 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Jones Companies | 73.40% | 0.744 | 0.712 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Assembly Required | 73.30% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | ACTS Assembly | 73.20% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | JONES & JONES PRODUCTION LTD | 73.20% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Morgan Jones | 73.20% | 0.744 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: The Jones Assembly**
> None

**Rank #2: JA**
> None

**Rank #3: The Jones Assembly Presents**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: The Jones Assembly (100.00%)                                    │
│  Match #2: JA (90.00%)                                                     │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 87. Query: `American Black Film Insitutute`

✅ **Exact Match Found:** `American Black Film Insitutute` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | American Black Film Insitutute | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | American Black Film Festival | 81.60% | 0.844 | 0.751 | 0.00 | High word-for-word overlap. |
| 3 | American Black Film Festival Ventures | 80.20% | 0.844 | 0.704 | 0.00 | High word-for-word overlap. |
| 4 | Black Women Film Preservation | 74.00% | 0.744 | 0.733 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | HOLLYWOOD BLACK FILM FESTIVAL | 72.60% | 0.744 | 0.684 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | International Black Film Festival | 72.30% | 0.744 | 0.676 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | National Black Film Festival | 72.10% | 0.744 | 0.667 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Independent Black Film Festival | 71.50% | 0.744 | 0.648 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | CANADIAN BLACK FILM FESTIVAL | 71.10% | 0.744 | 0.636 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | BLACK FILM SPACE | 69.80% | 0.651 | 0.807 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Black Film Initiative | 69.30% | 0.651 | 0.791 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | American Film Works | 66.70% | 0.651 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | American Independent Film | 66.10% | 0.651 | 0.685 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | American Film Convention | 64.80% | 0.651 | 0.640 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | American Film Market | 64.60% | 0.651 | 0.636 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: American Black Film Insitutute**
> None

**Rank #2: American Black Film Festival**
> None

**Rank #3: American Black Film Festival Ventures**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: American Black Film Insitutute (100.00%)                        │
│  Match #2: American Black Film Festival (81.60%)                           │
│                                                                            │
│  Score Difference: 18.40%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 88. Query: `Berk Tek`

✅ **Exact Match Found:** `Berk Tek` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Berk Tek | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Berk-Tek | 100.00% | 0.900 | 0.879 | 0.00 | High word-for-word overlap. |
| 3 | Berk Tek / Leviton | 97.00% | 0.900 | 0.659 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 4 | BT | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'BT'. |
| 5 | Berk Tck | 89.60% | 0.744 | 0.750 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Berk Technologies | 88.30% | 0.744 | 0.707 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Paul Berk Travel | 84.70% | 0.744 | 0.639 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | TEK Source | 73.20% | 0.744 | 0.706 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | World Tek | 72.30% | 0.744 | 0.674 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Tek Travel | 71.80% | 0.744 | 0.659 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | New Tek | 71.50% | 0.744 | 0.646 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Tek Systems | 71.20% | 0.744 | 0.637 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Berk Communications | 71.10% | 0.744 | 0.636 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Tek Interests | 71.10% | 0.744 | 0.635 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Tek Productions | 71.00% | 0.744 | 0.631 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Berk Tek**
> None

**Rank #2: Berk-Tek**
> None

**Rank #3: Berk Tek / Leviton**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Berk Tek (100.00%)                                              │
│  Match #2: Berk-Tek (100.00%)                                              │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 89. Query: `Northbridge Travel`

✅ **Exact Match Found:** `Northbridge Travel` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Northbridge Travel | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | NT | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'NT'. |
| 3 | Northbridge Communities | 77.20% | 0.744 | 0.839 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Bridge Travel | 77.20% | 0.787 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | WestBridge Travel | 77.10% | 0.744 | 0.836 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Northbridge Environmental | 76.80% | 0.744 | 0.826 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Skybridge Travel | 76.10% | 0.744 | 0.801 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Seabridge Travel | 75.80% | 0.744 | 0.792 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Northbridge Insurance | 75.50% | 0.744 | 0.780 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | KingsBridge Travel | 75.30% | 0.744 | 0.775 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Northbridge Church | 75.10% | 0.744 | 0.768 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Northbridge Trading | 74.90% | 0.744 | 0.761 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Northbridge Financial | 74.80% | 0.744 | 0.758 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Travel Bridge | 74.40% | 0.744 | 0.743 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | River North Travel | 74.10% | 0.744 | 0.734 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Northbridge Travel**
> None

**Rank #2: NT**
> None

**Rank #3: Northbridge Communities**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Northbridge Travel (100.00%)                                    │
│  Match #2: NT (85.50%)                                                     │
│                                                                            │
│  Score Difference: 14.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 90. Query: `Kohler 2024`

✅ **Exact Match Found:** `Kohler 2024` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Kohler 2024 | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | K2 | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'K2'. |
| 3 | Destination Kohler | 81.40% | 0.850 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Kohler Distributing | 81.40% | 0.850 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Kohler Fixtures | 80.40% | 0.850 | 0.697 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Kohler Generators | 80.10% | 0.850 | 0.688 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Kohler Recreation | 80.10% | 0.850 | 0.688 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Destination Kohler Retail | 80.00% | 0.850 | 0.684 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Kohler Lighting | 80.00% | 0.850 | 0.684 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Kohler Germany | 79.90% | 0.850 | 0.680 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Kohler Wholesale | 79.80% | 0.850 | 0.678 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Kohler Energy | 79.80% | 0.850 | 0.676 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Kohler Expos | 79.80% | 0.850 | 0.676 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Kohler Landscape | 79.70% | 0.850 | 0.673 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Kohler Services | 79.60% | 0.850 | 0.669 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Kohler 2024**
> None

**Rank #2: K2**
> None

**Rank #3: Destination Kohler**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Kohler 2024 (100.00%)                                           │
│  Match #2: K2 (90.00%)                                                     │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 91. Query: `Louisiana State University Swim`

✅ **Exact Match Found:** `Louisiana State University Swim` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Louisiana State University Swim | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Louisiana State Univ Swim | 88.40% | 0.900 | 0.846 | 0.00 | High word-for-word overlap. |
| 3 | LSUS | 85.50% | 1.000 | 1.000 | 0.70 | Matched based on generated acronym 'LSUS'. |
| 4 | LOUISIANA STATE UNIVERSITY ATHLETICS | 80.20% | 0.825 | 0.747 | 0.00 | High word-for-word overlap. |
| 5 | Louisiana State University USA | 80.10% | 0.825 | 0.746 | 0.00 | High word-for-word overlap. |
| 6 | Louisiana State University Health | 79.80% | 0.825 | 0.736 | 0.00 | High word-for-word overlap. |
| 7 | Louisiana State University Football | 79.30% | 0.825 | 0.719 | 0.00 | High word-for-word overlap. |
| 8 | Louisiana State University Trips | 79.10% | 0.825 | 0.711 | 0.00 | High word-for-word overlap. |
| 9 | Louisiana State University Education Alumni | 78.70% | 0.825 | 0.699 | 0.00 | High word-for-word overlap. |
| 10 | Louisiana State University System | 78.70% | 0.825 | 0.698 | 0.00 | High word-for-word overlap. |
| 11 | Southeastern Louisiana State University | 78.10% | 0.825 | 0.679 | 0.00 | High word-for-word overlap. |
| 12 | Louisiana State University Press | 78.00% | 0.825 | 0.675 | 0.00 | High word-for-word overlap. |
| 13 | Florida State Swim Team | 76.00% | 0.779 | 0.717 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Florida State Swim Meet | 75.10% | 0.779 | 0.687 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Louisiana State University | 74.60% | 0.722 | 0.804 | 0.00 | High word-for-word overlap. |

### Match Narratives

**Rank #1: Louisiana State University Swim**
> None

**Rank #2: Louisiana State Univ Swim**
> None

**Rank #3: LSUS**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Louisiana State University Swim (100.00%)                       │
│  Match #2: Louisiana State Univ Swim (88.40%)                              │
│                                                                            │
│  Score Difference: 11.60%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 92. Query: `X DO NOT USE - FRANCIS PARKER SCHOOL`

✅ **Exact Match Found:** `X DO NOT USE - FRANCIS PARKER SCHOOL` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | X DO NOT USE - FRANCIS PARKER SCHOOL | 100.00% | 1.000 | 0.999 | 0.00 | Perfect character-for-character match. |
| 2 | Francis Parker School | 65.20% | 0.503 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 3 | Francis Parker School English Department | 58.60% | 0.489 | 0.811 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Francis Parker School of San Diego | 57.90% | 0.489 | 0.790 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Francis Parker High School | 57.40% | 0.430 | 0.910 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 6 | Francis Parker School Prom | 55.90% | 0.430 | 0.861 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Francis Parker | 55.00% | 0.437 | 0.813 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Francis W. Parker School, Chicago | 54.70% | 0.489 | 0.680 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Francis W. Parker School | 54.60% | 0.430 | 0.815 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | St. Francis Xavier Secondary School | 49.70% | 0.421 | 0.675 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Parker High School Class of 1984 | 49.20% | 0.421 | 0.658 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | St Francis Xavier School | 47.40% | 0.370 | 0.716 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Terry Parker High School | 46.50% | 0.370 | 0.685 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | St Francis High School | 46.30% | 0.370 | 0.680 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Robert C. Parker School | 46.30% | 0.370 | 0.678 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: X DO NOT USE - FRANCIS PARKER SCHOOL**
> None

**Rank #2: Francis Parker School**
> None

**Rank #3: Francis Parker School English Department**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: X DO NOT USE - FRANCIS PARKER SCHOOL (100.00%)                  │
│  Match #2: Francis Parker School (65.20%)                                  │
│                                                                            │
│  Score Difference: 34.80%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 93. Query: `Mitsubishi Motor Sales of America, Incorporated`

✅ **Exact Match Found:** `Mitsubishi Motor Sales of America, Incorporated` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Mitsubishi Motor Sales of America, Incorporated | 100.00% | 1.000 | 0.868 | 0.00 | Perfect character-for-character match. |
| 2 | Mitsubishi Motor Sales of America | 97.70% | 1.000 | 0.925 | 0.00 | High word-for-word overlap. |
| 3 | Mitsubishi Electronic Sales America | 85.40% | 0.844 | 0.879 | 0.00 | High word-for-word overlap. |
| 4 | Mitsubishi Motors Sales of America | 85.40% | 0.844 | 0.879 | 0.00 | High word-for-word overlap. |
| 5 | Mitsubishi Motor Sales of Caribbean | 83.70% | 0.844 | 0.820 | 0.00 | High word-for-word overlap. |
| 6 | Mitsubishi Electric Sales of America | 83.50% | 0.844 | 0.815 | 0.00 | High word-for-word overlap. |
| 7 | Mitsubishi Motor Sales Of Amer | 83.50% | 0.844 | 0.814 | 0.00 | High word-for-word overlap. |
| 8 | Mitsubishi Motor Sales of Canada, Incorporated | 83.50% | 0.844 | 0.814 | 0.00 | High word-for-word overlap. |
| 9 | Mitsubishi Motors Sales Caribbean | 77.90% | 0.744 | 0.861 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | Mitsubishi Electric Sales Canada | 77.40% | 0.744 | 0.844 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Mitsubishi Motors N America Inc | 77.00% | 0.744 | 0.832 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Mitsubishi Electric Automotive America | 76.50% | 0.744 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Mitsubishi Sales Caribbean | 71.20% | 0.651 | 0.854 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 14 | MITSUBISHI ELECTRIC AMERICA | 70.00% | 0.651 | 0.813 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | MITSUBISHI MOTOR | 68.60% | 0.551 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |

### Match Narratives

**Rank #1: Mitsubishi Motor Sales of America, Incorporated**
> None

**Rank #2: Mitsubishi Motor Sales of America**
> None

**Rank #3: Mitsubishi Electronic Sales America**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Mitsubishi Motor Sales of America, Incorporated (100.00%)       │
│  Match #2: Mitsubishi Motor Sales of America (97.70%)                      │
│                                                                            │
│  Score Difference: 2.30%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 94. Query: `Energy Distribution Partners Holdings'`

✅ **Exact Match Found:** `Energy Distribution Partners Holdings'` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Energy Distribution Partners Holdings' | 100.00% | 1.000 | 0.951 | 0.00 | Perfect character-for-character match. |
| 2 | Energy Distribution Partners | 80.50% | 0.722 | 1.000 | 0.00 | High word-for-word overlap. |
| 3 | Energy Distribution Partners Holdings L.P. | 80.30% | 0.825 | 0.752 | 0.00 | High word-for-word overlap. |
| 4 | Energy Products Distribution | 71.60% | 0.682 | 0.795 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Distribution Energy Financial Group | 71.50% | 0.682 | 0.793 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Energy Distributors Partners | 69.40% | 0.620 | 0.869 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 7 | Energy Distribution Holdings | 69.10% | 0.577 | 0.956 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 8 | Energy Power Partners | 68.80% | 0.620 | 0.848 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Total Energy Partners | 68.80% | 0.620 | 0.846 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Resource Energy Partners | 68.20% | 0.620 | 0.829 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Energy Impact Partners | 67.80% | 0.620 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Energy Trust Partners | 67.70% | 0.620 | 0.811 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Energy Solution Partners | 67.70% | 0.620 | 0.810 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Global Energy Partners | 67.30% | 0.620 | 0.797 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Direct Energy Partners | 67.30% | 0.620 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Energy Distribution Partners Holdings'**
> None

**Rank #2: Energy Distribution Partners**
> None

**Rank #3: Energy Distribution Partners Holdings L.P.**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Energy Distribution Partners Holdings' (100.00%)                │
│  Match #2: Energy Distribution Partners (80.50%)                           │
│                                                                            │
│  Score Difference: 19.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 95. Query: `ThinkAdvisor`

✅ **Exact Match Found:** `ThinkAdvisor` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | ThinkAdvisor | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Planadvisor | 46.50% | 0.313 | 0.819 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | TRIPADVISOR | 45.70% | 0.352 | 0.701 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | NeXtAdvisors | 45.30% | 0.313 | 0.780 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Invisors | 43.30% | 0.332 | 0.671 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | TripAdvisor LLC | 43.10% | 0.352 | 0.613 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Proadvisors | 42.40% | 0.286 | 0.745 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | ChannelAdvisor | 42.00% | 0.312 | 0.673 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | ChannelAdvisor Corporation | 42.00% | 0.312 | 0.672 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Homeadvisor | 41.30% | 0.313 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | PlanAdviser | 41.30% | 0.274 | 0.736 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | CHANNELADVISORS | 41.20% | 0.312 | 0.647 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Envisor | 41.00% | 0.284 | 0.704 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | NeXtAdvisors LLC | 40.90% | 0.313 | 0.632 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | ScoutAdvisor Corporation | 39.30% | 0.300 | 0.609 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: ThinkAdvisor**
> None

**Rank #2: Planadvisor**
> None

**Rank #3: TRIPADVISOR**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: ThinkAdvisor (100.00%)                                          │
│  Match #2: Planadvisor (46.50%)                                            │
│                                                                            │
│  Score Difference: 53.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 96. Query: `Jump on it Outreach`

✅ **Exact Match Found:** `Jump on it Outreach` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Jump on it Outreach | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Air Force Outreach Program | 60.70% | 0.525 | 0.798 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Above N Beyond Outreach | 60.70% | 0.525 | 0.798 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Outreach | 59.80% | 0.433 | 0.983 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | OutReach Program Development | 56.70% | 0.459 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Community Outreach Program | 56.20% | 0.459 | 0.801 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Friendly Outreach Mission | 55.60% | 0.459 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | New Creation Outreach | 55.40% | 0.459 | 0.775 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Jump | 55.30% | 0.433 | 0.833 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Outreach Youth Training | 55.20% | 0.459 | 0.769 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Community Outreach Center | 55.00% | 0.459 | 0.761 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | General Human Outreach | 55.00% | 0.459 | 0.760 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | Human Outreach Project | 54.90% | 0.459 | 0.759 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Breakthrough Outreach Center | 54.90% | 0.459 | 0.759 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Community Outreach Workshop | 54.80% | 0.459 | 0.754 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Jump on it Outreach**
> None

**Rank #2: Air Force Outreach Program**
> None

**Rank #3: Above N Beyond Outreach**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Jump on it Outreach (100.00%)                                   │
│  Match #2: Air Force Outreach Program (60.70%)                             │
│                                                                            │
│  Score Difference: 39.30%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 97. Query: `The Association of Ringside Consultants (ARC)`

✅ **Exact Match Found:** `The Association of Ringside Consultants (ARC)` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | The Association of Ringside Consultants (ARC) | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Association of Ringside Physicians | 69.00% | 0.651 | 0.783 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Ringside | 56.40% | 0.433 | 0.871 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | Arc-Consultants | 54.60% | 0.367 | 0.963 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 5 | Ringside, Incorporated | 54.20% | 0.433 | 0.796 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | ARC CONSULTANTS,  INC. | 50.50% | 0.367 | 0.827 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Arcadis Consultants | 50.20% | 0.367 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Circle Consultants | 50.10% | 0.367 | 0.811 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Diamond Consultants | 49.70% | 0.367 | 0.798 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | Ringside Talent | 49.20% | 0.367 | 0.782 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | ARCOM Association | 45.70% | 0.315 | 0.789 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Arc Bridge Consultant Inc | 41.20% | 0.240 | 0.811 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | 5 Rings Consulting Inc. | 38.60% | 0.211 | 0.795 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | ARC Consulting | 38.60% | 0.139 | 0.962 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 15 | Arc Consultancy LTD | 37.40% | 0.161 | 0.870 | 0.00 | Matched via strong semantic/conceptual similarity. |

### Match Narratives

**Rank #1: The Association of Ringside Consultants (ARC)**
> None

**Rank #2: Association of Ringside Physicians**
> None

**Rank #3: Ringside**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: The Association of Ringside Consultants (ARC) (100.00%)         │
│  Match #2: Association of Ringside Physicians (69.00%)                     │
│                                                                            │
│  Score Difference: 31.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 98. Query: `SFA HASA`

✅ **Exact Match Found:** `SFA HASA` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | SFA HASA | 100.00% | 1.000 | 0.995 | 0.00 | Perfect character-for-character match. |
| 2 | SH | 90.00% | 1.000 | 1.000 | 1.00 | Matched based on generated acronym 'SH'. |
| 3 | SFA Companies | 77.60% | 0.744 | 0.851 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 4 | SFA Leads | 77.30% | 0.744 | 0.841 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Sfa Charter | 76.40% | 0.744 | 0.810 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | SFA Partners | 76.30% | 0.744 | 0.810 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | SFA System Account | 75.80% | 0.744 | 0.790 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | SFA Opportunity | 75.20% | 0.744 | 0.770 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | SFA Design | 74.90% | 0.744 | 0.760 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | SFA Designs | 74.20% | 0.744 | 0.738 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | SFA Training | 74.00% | 0.744 | 0.731 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Stalla SFA | 73.90% | 0.744 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | SFA Saniflo | 73.80% | 0.744 | 0.726 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | SFA | 68.60% | 0.551 | 1.000 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 15 | SFA Inc | 64.60% | 0.551 | 0.866 | 0.00 | Matched via strong semantic/conceptual similarity. |

### Match Narratives

**Rank #1: SFA HASA**
> None

**Rank #2: SH**
> None

**Rank #3: SFA Companies**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: SFA HASA (100.00%)                                              │
│  Match #2: SH (90.00%)                                                     │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 99. Query: `Grupo Duracell Ene 2025`

✅ **Exact Match Found:** `Grupo Duracell Ene 2025` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Grupo Duracell Ene 2025 | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Grupo GT5 Brasil | 50.60% | 0.438 | 0.665 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 3 | Duracell Research Development | 50.20% | 0.438 | 0.651 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 4 | Grupo Brasil DPE | 50.10% | 0.438 | 0.648 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Duracell Brasil | 49.40% | 0.350 | 0.830 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | Grupo CLC 2024 | 49.20% | 0.438 | 0.621 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Grupo GT 5 | 49.20% | 0.438 | 0.621 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | 2025 Ag Event | 47.60% | 0.394 | 0.669 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | 2025 LEMA Affiliates | 47.30% | 0.394 | 0.659 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 10 | 2025 OLL Trip | 46.40% | 0.394 | 0.628 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 11 | Duracell Chile | 46.30% | 0.350 | 0.727 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 12 | Convene Columbus 2025 | 46.20% | 0.394 | 0.623 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 13 | DURACELL USA | 46.20% | 0.350 | 0.722 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | Duracell Services | 45.90% | 0.350 | 0.715 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Professional Duracell | 45.60% | 0.350 | 0.703 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: Grupo Duracell Ene 2025**
> None

**Rank #2: Grupo GT5 Brasil**
> None

**Rank #3: Duracell Research Development**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Grupo Duracell Ene 2025 (100.00%)                               │
│  Match #2: Grupo GT5 Brasil (50.60%)                                       │
│                                                                            │
│  Score Difference: 49.40%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 100. Query: `World Association of Medical Law`

✅ **Exact Match Found:** `World Association of Medical Law` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | World Association of Medical Law | 100.00% | 1.000 | 0.998 | 0.00 | Perfect character-for-character match. |
| 2 | World Association for Medical Law | 96.50% | 0.950 | 1.000 | 0.00 | High word-for-word overlap. |
| 3 | World Medical Association | 78.10% | 0.689 | 0.995 | 0.00 | High word-for-word overlap. |
| 4 | International Law Association | 70.10% | 0.651 | 0.817 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 5 | Pacific Medical Law | 70.00% | 0.651 | 0.814 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 6 | California Medical Legal Association | 69.80% | 0.637 | 0.840 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 7 | Professional Medical Education Association | 69.60% | 0.637 | 0.833 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 8 | Association Hospital Medical Education | 69.30% | 0.637 | 0.824 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 9 | Mass Medical Association | 66.20% | 0.558 | 0.905 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 10 | MEDICAL DOCTORS ASSOCIATION | 66.20% | 0.558 | 0.905 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 11 | Physicians Medical Association | 65.70% | 0.558 | 0.888 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 12 | Medical Research Association | 65.70% | 0.558 | 0.888 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 13 | National Medical Association | 65.40% | 0.558 | 0.879 | 0.00 | Matched via strong semantic/conceptual similarity. |
| 14 | American Medical Association | 64.50% | 0.558 | 0.849 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | Medical Association Management | 64.50% | 0.558 | 0.848 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: World Association of Medical Law**
> None

**Rank #2: World Association for Medical Law**
> None

**Rank #3: World Medical Association**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: World Association of Medical Law (100.00%)                      │
│  Match #2: World Association for Medical Law (96.50%)                      │
│                                                                            │
│  Score Difference: 3.50%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 101. Query: `ABA`

✅ **Exact Match Found:** `ABA` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | ABA | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | AMER BRIDGE ASSN | 97.80% | 0.500 | 0.277 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 3 | Am Bridge Assn | 97.80% | 0.500 | 0.273 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 4 | Arrowood Business Association | 97.70% | 0.500 | 0.238 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 5 | AZ Business Assn | 97.70% | 0.500 | 0.235 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 6 | ACL Business Assurance | 97.60% | 0.500 | 0.214 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 7 | AG Bell Association | 97.60% | 0.500 | 0.209 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 8 | A Brodi Abroad | 97.60% | 0.500 | 0.205 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 9 | Aprende Business Academy | 97.60% | 0.500 | 0.192 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 10 | Amcol Bio Ag | 97.60% | 0.500 | 0.191 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 11 | Andreini Benefit Advantage | 97.60% | 0.500 | 0.184 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 12 | Ancienne Belgique - AB | 97.60% | 0.500 | 0.185 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 13 | Am Business Advantage | 97.50% | 0.500 | 0.179 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 14 | Anderson Brule Architects | 97.50% | 0.500 | 0.180 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |
| 15 | Arthur Bell and Associates | 97.50% | 0.500 | 0.172 | 1.00 | Detected as a literal expansion of acronym 'ABA'. |

### Match Narratives

**Rank #1: ABA**
> None

**Rank #2: AMER BRIDGE ASSN**
> None

**Rank #3: Am Bridge Assn**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: ABA (100.00%)                                                   │
│  Match #2: AMER BRIDGE ASSN (97.80%)                                       │
│                                                                            │
│  Score Difference: 2.20%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 102. Query: `PDMA`

✅ **Exact Match Found:** `PDMA` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | PDMA | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | Prescription Drug Marketing Act | 94.00% | 0.500 | 0.187 | 0.70 | Detected as a literal expansion of acronym 'PDMA'. |
| 3 | Public Debt Management Agency | 93.90% | 0.500 | 0.174 | 0.70 | Detected as a literal expansion of acronym 'PDMA'. |
| 4 | Product Development Management Association | 93.80% | 0.500 | 0.124 | 0.70 | Detected as a literal expansion of acronym 'PDMA'. |
| 5 | PRA Destination Management Atlanta | 93.60% | 0.500 | 0.066 | 0.70 | Detected as a literal expansion of acronym 'PDMA'. |
| 6 | Penn Dental Medicine Alumni | 93.50% | 0.500 | 0.035 | 0.70 | Detected as a literal expansion of acronym 'PDMA'. |
| 7 | PRODUCT DEVEL MGMT ASSN | 93.40% | 0.500 | -0.040 | 0.70 | Detected as a literal expansion of acronym 'PDMA'. |
| 8 | Producttank Des Moines Ames | 93.40% | 0.500 | -0.084 | 0.70 | Detected as a literal expansion of acronym 'PDMA'. |
| 9 | PDMA inc | 89.00% | 0.900 | 0.867 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 10 | PDMA Association | 88.90% | 0.900 | 0.862 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 11 | PDMA Corporation | 88.50% | 0.900 | 0.849 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 12 | PDMA Alliance | 84.10% | 0.900 | 0.704 | 0.00 | Direct prefix match (target contains extra trailing words). |
| 13 | PDA | 51.50% | 0.375 | 0.841 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 14 | PDM | 50.10% | 0.375 | 0.794 | 0.00 | Hybrid match based on combined lexical and semantic features. |
| 15 | PDA CORP | 48.30% | 0.375 | 0.736 | 0.00 | Hybrid match based on combined lexical and semantic features. |

### Match Narratives

**Rank #1: PDMA**
> None

**Rank #2: Prescription Drug Marketing Act**
> None

**Rank #3: Public Debt Management Agency**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: PDMA (100.00%)                                                  │
│  Match #2: Prescription Drug Marketing Act (94.00%)                        │
│                                                                            │
│  Score Difference: 6.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 103. Query: `IBM`

✅ **Exact Match Found:** `IBM` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | IBM | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | IBM | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 3 | International Business Machines | 98.30% | 0.500 | 0.440 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 4 | International Business Machine | 98.30% | 0.500 | 0.420 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 5 | Intel Board Meeting | 98.10% | 0.500 | 0.365 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 6 | intechRx Business Meeting | 97.90% | 0.500 | 0.298 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 7 | International Boiler Makers | 97.80% | 0.500 | 0.275 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 8 | Innovation Business Media | 97.80% | 0.500 | 0.268 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 9 | International Business Management | 97.70% | 0.500 | 0.249 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 10 | INTERNATIONAL BUSINESS MACH | 97.70% | 0.500 | 0.250 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 11 | Italian Business Mission | 97.70% | 0.500 | 0.246 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 12 | Internet Business Mastery | 97.70% | 0.500 | 0.248 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 13 | ISO BRAND MARKETING | 97.70% | 0.500 | 0.240 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 14 | Inspired Business Media | 97.70% | 0.500 | 0.237 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |
| 15 | Integral Blue Meeting | 97.70% | 0.500 | 0.224 | 1.00 | Detected as a literal expansion of acronym 'IBM'. |

### Match Narratives

**Rank #1: IBM**
> None

**Rank #2: IBM**
> None

**Rank #3: International Business Machines**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: IBM (100.00%)                                                   │
│  Match #2: IBM (100.00%)                                                   │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 104. Query: `GE`

✅ **Exact Match Found:** `GE` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | GE | 100.00% | 1.000 | 1.000 | 0.00 | Perfect character-for-character match. |
| 2 | GMG Education | 98.00% | 0.500 | 0.331 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 3 | Gaia Experience | 97.90% | 0.500 | 0.316 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 4 | GAD e.G. | 97.90% | 0.500 | 0.312 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 5 | Gould Evans | 97.90% | 0.500 | 0.312 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 6 | GU Energy | 97.90% | 0.500 | 0.307 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 7 | G+G Enterprises | 97.90% | 0.500 | 0.302 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 8 | G&G Enterprises | 97.90% | 0.500 | 0.297 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 9 | Gud Energy | 97.90% | 0.500 | 0.296 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 10 | Guardian Education | 97.90% | 0.500 | 0.293 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 11 | G E | 97.90% | 0.500 | 0.294 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 12 | Grupo EP&A | 97.90% | 0.500 | 0.290 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 13 | Girard Elementary | 97.90% | 0.500 | 0.289 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 14 | G.U.M.B.O Enterprises | 97.90% | 0.500 | 0.286 | 1.00 | Detected as a literal expansion of acronym 'GE'. |
| 15 | Govbr Educacional | 97.90% | 0.500 | 0.287 | 1.00 | Detected as a literal expansion of acronym 'GE'. |

### Match Narratives

**Rank #1: GE**
> None

**Rank #2: GMG Education**
> None

**Rank #3: Gaia Experience**
> None

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: GE (100.00%)                                                    │
│  Match #2: GMG Education (98.00%)                                          │
│                                                                            │
│  Score Difference: 2.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

