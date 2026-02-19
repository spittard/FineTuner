# Company Matching Control Set - Ultra Detailed Report

**Generated:** 2026-01-02 11:26:52

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

## 1. Query: `NIH`

✅ **Exact Match Found:** `NIH` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | NIH | 100.40% | 1.000 | 0.403 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | NIH | 100.00% | 1.000 | 1.000 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 3 | NIH | 99.40% | 1.000 | 0.475 | 0.00 | Excellent Match (99%) — Based on exact analysis. |
| 4 | NIH | 98.20% | 1.000 | 0.420 | 0.00 | Excellent Match (98%) — Based on exact analysis. |
| 5 | NIH - NIDDK | 97.50% | 0.818 | 0.444 | 0.00 | Excellent Match (98%) — Based on hybrid analysis. |
| 6 | NIH - NICHD | 97.20% | 0.818 | 0.405 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 7 | NIH - NIAID | 96.80% | 0.818 | 0.445 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 8 | NIH - Ninds | 96.70% | 0.818 | 0.489 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 9 | NIH | 96.60% | 1.000 | 0.500 | 0.00 | Excellent Match (97%) — Based on exact analysis. |
| 10 | NIH - National Institute on Aging (Nih-N | 96.30% | 0.562 | 0.389 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 11 | NIH - NIAMS | 96.20% | 0.818 | 0.460 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 12 | NIH - NIBIB | 96.20% | 0.818 | 0.409 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 13 | NIH - NINDS | 96.20% | 0.818 | 0.447 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 14 | NIH - NEI | 96.20% | 0.818 | 0.424 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 15 | NIH | 96.20% | 1.000 | 0.438 | 0.00 | Excellent Match (96%) — Based on exact analysis. |

### Match Narratives

**Rank #1: NIH**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟠 FAIR (40%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (81%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Wilmington, NC but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+0.4% Boost</b> — Higher confidence due to 2 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Chicago 30.8%, ✅ Pennsylvania 29.6%<br>• Insight: The model detects a strong 'Chicago' influence in the company's semantic vector.<br>

**Rank #2: NIH**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (100%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>• Industry: ✅ Food 28.8%<br>

**Rank #3: NIH**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (99%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟠 FAIR (48%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (85%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Bethesda, MD but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+4.4% Boost</b> — Higher confidence due to 197 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Industry: ✅ Medical 26.6%<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: NIH (100.40%)                                                   │
│  Match #2: NIH (100.00%)                                                   │
│                                                                            │
│  Score Difference: 0.40%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Semantic Preference: #1 has stronger conceptual link.                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 2. Query: `Ohio University`

✅ **Exact Match Found:** `Ohio University` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Ohio University | 101.00% | 1.000 | 1.000 | 0.00 | Excellent Match (101%) — Based on exact analysis. |
| 2 | Ohio University | 101.00% | 1.000 | 0.783 | 0.00 | Excellent Match (101%) — Based on exact analysis. |
| 3 | Ohio University | 99.00% | 1.000 | 0.633 | 0.00 | Excellent Match (99%) — Based on exact analysis. |
| 4 | Ohio University | 97.70% | 1.000 | 0.916 | 0.00 | Excellent Match (98%) — Based on exact analysis. |
| 5 | Ohio University Athletics | 97.00% | 0.818 | 0.625 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 6 | Ohio University Alumni Association | 96.90% | 0.750 | 0.615 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 7 | University of Ohio | 96.80% | 1.000 | 0.871 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 8 | Miami of Ohio University | 96.70% | 0.818 | 0.657 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 9 | Ohio University Foundation | 96.70% | 0.826 | 0.582 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 10 | Ohio State University Arena | 96.60% | 0.792 | 0.692 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 11 | Ohio University Athletics Department | 96.60% | 0.750 | 0.578 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 12 | Ohio State University 4-H Foundation | 96.50% | 0.679 | 0.557 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 13 | UNIVERSITY SYSTEM OF OHIO | 96.20% | 0.864 | 0.669 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 14 | Ohio State University College of Optometry | 96.20% | 0.731 | 0.545 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 15 | The Ohio State University, School of Communication | 95.90% | 0.731 | 0.552 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: Ohio University**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (101%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (100%) — Reflects highly aligned industries.<br>• <b>Entity Popularity:</b> 🟢 <b>+1.0% Boost</b> — Higher confidence due to 2 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 81.8%, ✅ Illinois 39.4%, ✅ Chicago 33.9%, ✅ California 31.9%, ✅ Miami 27.6%<br>• Industry: ✅ Education 41.9%<br>• Nature: ✅ Professional 27.6%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>

**Rank #2: Ohio University**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (101%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 GOOD (78%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (99%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Columbus, OH but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+1.0% Boost</b> — Higher confidence due to 14 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 72.1%, ✅ Illinois 32.4%, ✅ Miami 30.8%, ✅ Chicago 30.5%, ✅ New York 28.0%, ✅ California 27.0%, ✅ Pennsylvania 25.6%<br>• Industry: ✅ Education 33.0%<br>• Nature: ✅ Professional 26.9%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>

**Rank #3: Ohio University**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (99%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟡 MODERATE (63%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (94%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Athens, OH but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+4.0% Boost</b> — Higher confidence due to 119 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 53.7%, ✅ Illinois 29.8%<br>• Industry: ✅ Education 34.2%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Ohio University (101.00%)                                       │
│  Match #2: Ohio University (101.00%)                                       │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 3. Query: `Western University`

✅ **Exact Match Found:** `Western University` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Western University | 100.00% | 1.000 | 1.000 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Western University | 100.00% | 1.000 | 0.600 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 3 | Western University | 99.70% | 1.000 | 0.706 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 4 | Western University | 96.70% | 1.000 | 0.630 | 0.00 | Excellent Match (97%) — Based on exact analysis. |
| 5 | Western University | 96.20% | 1.000 | 0.814 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 6 | Western University | 95.90% | 1.000 | 0.598 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 7 | Ivey Business School Western University | 95.90% | 0.692 | 0.552 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 8 | Western University | 95.90% | 1.000 | 0.612 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 9 | Western University | 95.90% | 1.000 | 0.836 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 10 | Western Washington University | 95.00% | 0.864 | 0.821 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 11 | Western University | 95.00% | 1.000 | 0.565 | 0.00 | Excellent Match (95%) — Based on exact analysis. |
| 12 | University Western Australia | 95.00% | 0.864 | 0.786 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 13 | Western New England University | 95.00% | 0.792 | 0.735 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 14 | University of Western Australia | 95.00% | 0.864 | 0.688 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 15 | UWO Western University | 95.00% | 0.818 | 0.796 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: Western University**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (100%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ London 32.9%, ✅ Ohio 31.9%, ✅ California 27.5%<br>• Industry: ✅ Education 46.3%<br>• Nature: ✅ Professional 32.0%, ✅ Local 27.3%<br>• Insight: The model detects a strong 'Education' influence in the company's semantic vector.<br>

**Rank #2: Western University**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟡 MODERATE (60%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (84%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Springfield, MA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Illinois 35.8%, ✅ Ohio 33.0%, ✅ Chicago 29.9%<br>• Industry: ✅ Education 27.4%<br>• Insight: The model detects a strong 'Illinois' influence in the company's semantic vector.<br>

**Rank #3: Western University**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 GOOD (71%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (91%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found London, ON but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+4.7% Boost</b> — Higher confidence due to 284 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ London 69.8%, ✅ New York 27.8%<br>• Industry: ✅ Education 35.0%<br>• Nature: ✅ Local 28.0%, ✅ Professional 27.8%, ✅ Global 25.3%<br>• Insight: The vector is strongly pulled toward geographic anchors, resolving potential ambiguity.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Western University (100.00%)                                    │
│  Match #2: Western University (100.00%)                                    │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 4. Query: `Kruger Products`

✅ **Exact Match Found:** `Kruger Products` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Kruger Products | 100.00% | 1.000 | 1.000 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Kruger Products | 100.00% | 1.000 | 0.461 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 3 | Kruger Products | 99.00% | 1.000 | 0.586 | 0.00 | Excellent Match (99%) — Based on exact analysis. |
| 4 | Kruger Products | 96.50% | 1.000 | 0.629 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 5 | Kruger Products USA Inc | 96.20% | 0.818 | 0.461 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 6 | Kruger Products | 95.90% | 1.000 | 0.725 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 7 | Kruger Products USA, Inc. | 95.90% | 0.818 | 0.408 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 8 | Kruger Products, Ltd. | 95.00% | 1.000 | 0.702 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 9 | Kruger Products L. P. | 95.00% | 0.750 | 0.520 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 10 | Kruger Paper Products | 95.00% | 0.864 | 0.774 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 11 | Kruger Products L.P. / Produits Kruger s.e.c. | 95.00% | 0.643 | 0.450 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 12 | Kruger Consumer Products Canada, Incorporated | 95.00% | 0.792 | 0.467 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 13 | Kruger Products | 95.00% | 1.000 | 0.581 | 0.00 | Excellent Match (95%) — Based on exact analysis. |
| 14 | Kruger Products LP | 95.00% | 0.826 | 0.658 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 15 | Kruger Products | 95.00% | 1.000 | 0.638 | 0.00 | Excellent Match (95%) — Based on exact analysis. |

### Match Narratives

**Rank #1: Kruger Products**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (100%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>• Industry: ✅ Food 27.4%<br>

**Rank #2: Kruger Products**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟠 FAIR (46%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (79%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Minnetonka, MN but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #3: Kruger Products**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (99%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟡 MODERATE (59%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (80%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Bentonville, AR but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+4.0% Boost</b> — Higher confidence due to 127 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Kruger Products (100.00%)                                       │
│  Match #2: Kruger Products (100.00%)                                       │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 5. Query: `Vision America`

✅ **Exact Match Found:** `Vision America` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Vision America | 100.00% | 1.000 | 1.000 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Vision America | 100.00% | 1.000 | 0.695 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 3 | Vision America | 99.50% | 1.000 | 0.525 | 0.00 | Excellent Match (99%) — Based on exact analysis. |
| 4 | Vision Council of America | 96.60% | 0.864 | 0.497 | 0.00 | Excellent Match (97%) — Based on hybrid analysis. |
| 5 | Vision America | 95.90% | 1.000 | 0.562 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 6 | Vision action america | 95.90% | 0.864 | 0.539 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 7 | Vision America | 95.90% | 1.000 | 0.580 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 8 | Vision America Action | 95.00% | 0.818 | 0.869 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 9 | Vision America | 95.00% | 1.000 | 0.763 | 0.00 | Excellent Match (95%) — Based on exact analysis. |
| 10 | Vision Council of America | 95.00% | 0.864 | 0.673 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 11 | The Vision Council of America | 95.00% | 0.864 | 0.607 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 12 | Vision Council Of America | 95.00% | 0.864 | 0.570 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 13 | Laser Vision Institute of America | 95.00% | 0.792 | 0.581 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 14 | Hanwha Vision America | 95.00% | 0.818 | 0.635 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 15 | VISION AMERICA | 95.00% | 1.000 | 0.599 | 0.00 | Excellent Match (95%) — Based on exact analysis. |

### Match Narratives

**Rank #1: Vision America**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (100%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ New York 34.7%, ✅ Canada 28.8%, ✅ Miami 28.8%, ✅ California 28.2%, ✅ Paris 26.3%<br>• Structure: ✅ Government 26.1%<br>• Nature: ✅ Global 42.9%, ✅ Consumer 32.2%<br>• Insight: The model detects a strong 'Global' influence in the company's semantic vector.<br>

**Rank #2: Vision America**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟡 MODERATE (69%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (94%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Birmingham, AL but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ California 28.5%, ✅ New York 28.5%<br>• Nature: ✅ Global 27.9%<br>

**Rank #3: Vision America**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (99%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟡 MODERATE (52%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (89%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Nacogdoches, TX but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+4.5% Boost</b> — Higher confidence due to 216 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Texas 31.7%<br>• Insight: The model detects a strong 'Texas' influence in the company's semantic vector.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Vision America (100.00%)                                        │
│  Match #2: Vision America (100.00%)                                        │
│                                                                            │
│  Score Difference: 0.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Virtual Tie: Negligible difference in score components.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 6. Query: `PDMA Association`

✅ **Exact Match Found:** `PDMA Association` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | PDMA Association | 100.00% | 1.000 | 0.586 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Association Headquarters-PDMA | 95.00% | 0.864 | 0.449 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 3 | PDMA Alliance | 90.90% | 0.850 | 0.544 | 0.00 | Excellent Match (91%) — Based on hybrid analysis. |
| 4 | PDMA ALLIANCE | 90.00% | 0.850 | 0.810 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 5 | PDMA ALLIANCE | 90.00% | 0.850 | 0.690 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 6 | PDMA ALLIANCE | 90.00% | 0.850 | 0.556 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 7 | PDMA Alliance Inc. | 90.00% | 0.850 | 0.483 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 8 | PDMA Alliance | 90.00% | 0.850 | 0.458 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 9 | PDMA Alliance | 90.00% | 0.850 | 0.475 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 10 | PDMA | 84.60% | 0.720 | 1.000 | 0.00 | Strong Match (85%) — Based on hybrid analysis. |
| 11 | PDMA inc | 82.70% | 0.720 | 0.913 | 0.00 | Strong Match (83%) — Based on hybrid analysis. |
| 12 | PDMA | 77.00% | 0.720 | 0.810 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |
| 13 | PDS User Group Association | 70.00% | 0.580 | 0.765 | 0.00 | Moderate Match (70%) — Based on hybrid analysis. |
| 14 | ASSOCIATION ACCOUNTS | 67.80% | 0.637 | 0.595 | 0.00 | Moderate Match (68%) — Based on hybrid analysis. |
| 15 | Joint Association | 67.70% | 0.637 | 0.630 | 0.00 | Moderate Match (68%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: PDMA Association**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟡 MODERATE (59%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (55%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Mount Laurel, NJ but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #2: Association Headquarters-PDMA**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (95%)</span><br><span style='color:#eee; font-size:1.1em;'>High Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('association') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (86%) — Based on strong character overlap.<br>• <b>Semantic Link:</b> 🟠 FAIR (45%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (61%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Mount Laurel, NJ but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #3: PDMA Alliance**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (91%)</span><br><span style='color:#eee; font-size:1.1em;'>High Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('pdma') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (85%) — Based on strong character overlap.<br>• <b>Semantic Link:</b> 🟡 MODERATE (54%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (57%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found York, SC but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+0.9% Boost</b> — Higher confidence due to 2 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ New York 39.3%<br>• Insight: The model detects a strong 'New York' influence in the company's semantic vector.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: PDMA Association (100.00%)                                      │
│  Match #2: Association Headquarters-PDMA (95.00%)                          │
│                                                                            │
│  Score Difference: 5.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 7. Query: `Nicolas/Sanchez Wedding`

✅ **Exact Match Found:** `Nicolas/Sanchez Wedding` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Nicolas/Sanchez Wedding | 100.00% | 1.000 | 0.723 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Sanchez/Justin Wedding | 81.20% | 0.744 | 0.916 | 0.00 | Strong Match (81%) — Based on hybrid analysis. |
| 3 | Sanchez Wedding | 81.10% | 0.787 | 0.788 | 0.00 | Strong Match (81%) — Based on hybrid analysis. |
| 4 | Garcia Sanchez Wedding | 79.30% | 0.676 | 1.000 | 0.00 | Strong Match (79%) — Based on hybrid analysis. |
| 5 | Sanchez/Cohen Wedding | 79.10% | 0.744 | 0.788 | 0.00 | Strong Match (79%) — Based on hybrid analysis. |
| 6 | Gibson Sanchez Wedding | 78.90% | 0.676 | 0.975 | 0.00 | Strong Match (79%) — Based on hybrid analysis. |
| 7 | Sanchez/Ramirez Wedding | 78.70% | 0.744 | 0.875 | 0.00 | Strong Match (79%) — Based on hybrid analysis. |
| 8 | Alequin Sanchez Wedding | 77.90% | 0.676 | 0.917 | 0.00 | Strong Match (78%) — Based on hybrid analysis. |
| 9 | Sanchez/Puerto Wedding | 77.60% | 0.744 | 0.811 | 0.00 | Strong Match (78%) — Based on hybrid analysis. |
| 10 | Sanchez/Naranjo Wedding | 77.40% | 0.744 | 0.779 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |
| 11 | Sanchez and Alfonso Wedding | 77.20% | 0.676 | 0.888 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |
| 12 | Rodriguez/ Sanchez Wedding | 76.60% | 0.676 | 0.907 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |
| 13 | Carlos Wedding | 76.30% | 0.637 | 0.940 | 0.00 | Strong Match (76%) — Based on hybrid analysis. |
| 14 | Sanchez/Fuentes Wedding | 76.20% | 0.744 | 0.778 | 0.00 | Strong Match (76%) — Based on hybrid analysis. |
| 15 | Sanchez/Ferree Wedding | 75.50% | 0.744 | 0.687 | 0.00 | Strong Match (76%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: Nicolas/Sanchez Wedding**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 GOOD (72%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (61%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Port Chester, NY but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #2: Sanchez/Justin Wedding**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (81%)</span><br><span style='color:#eee; font-size:1.1em;'>High Semantic Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('wedding') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (74%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (92%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 GOOD (85%) — Reflects related business categories.<br><br><br><b>Concept Analysis:</b><br>

**Rank #3: Sanchez Wedding**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (81%)</span><br><span style='color:#eee; font-size:1.1em;'>Moderate Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('wedding') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (79%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟢 GOOD (79%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟢 GOOD (88%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Marietta, GA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Nicolas/Sanchez Wedding (100.00%)                               │
│  Match #2: Sanchez/Justin Wedding (81.20%)                                 │
│                                                                            │
│  Score Difference: 18.80%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 8. Query: `Kehilat Ariel Synagogue`

✅ **Exact Match Found:** `Kehilat Ariel Synagogue` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Kehilat Ariel Synagogue | 100.00% | 1.000 | 0.935 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Kehilat Ariel Messianic Synagogue | 95.00% | 0.864 | 0.823 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 3 | Kehilat Ariel Passover | 90.00% | 0.883 | 0.556 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 4 | Kehilat Ariel | 90.00% | 0.818 | 0.589 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 5 | Jewish Synagogue | 69.10% | 0.432 | 1.000 | 0.00 | Moderate Match (69%) — Based on hybrid analysis. |
| 6 | KAS | 69.00% | 1.000 | 1.000 | 0.70 | Moderate Match (69%) — Based on acronym_reverse analysis. |
| 7 | Central Synagogue | 67.60% | 0.432 | 0.947 | 0.00 | Moderate Match (68%) — Based on hybrid analysis. |
| 8 | Beth Sholom Synagogue | 67.00% | 0.475 | 0.839 | 0.00 | Moderate Match (67%) — Based on hybrid analysis. |
| 9 | Synagogue 3000 Organization | 66.90% | 0.475 | 0.909 | 0.00 | Moderate Match (67%) — Based on hybrid analysis. |
| 10 | ICC Synagogue | 66.30% | 0.432 | 0.899 | 0.00 | Moderate Match (66%) — Based on hybrid analysis. |
| 11 | Park Synagogue | 65.10% | 0.432 | 0.921 | 0.00 | Moderate Match (65%) — Based on hybrid analysis. |
| 12 | Community Synagogue | 65.00% | 0.432 | 0.909 | 0.00 | Moderate Match (65%) — Based on hybrid analysis. |
| 13 | Temple Sinai Synagogue | 63.50% | 0.475 | 0.867 | 0.00 | Moderate Match (63%) — Based on hybrid analysis. |
| 14 | National Synagogue Youth | 62.90% | 0.475 | 0.800 | 0.00 | Moderate Match (63%) — Based on hybrid analysis. |
| 15 | Synagogue 2000 | 62.80% | 0.432 | 0.824 | 0.00 | Moderate Match (63%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: Kehilat Ariel Synagogue**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (93%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 GOOD (70%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found San Diego, CA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ California 29.7%<br>

**Rank #2: Kehilat Ariel Messianic Synagogue**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (95%)</span><br><span style='color:#eee; font-size:1.1em;'>High Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('ariel, kehilat, synagogue') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (86%) — Based on strong character overlap.<br>• <b>Semantic Link:</b> 🟢 GOOD (82%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟢 GOOD (79%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found San Diego, CA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #3: Kehilat Ariel Passover**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (90%)</span><br><span style='color:#eee; font-size:1.1em;'>High Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('ariel, kehilat') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (88%) — Based on strong character overlap.<br>• <b>Semantic Link:</b> 🟡 MODERATE (56%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟠 FAIR (47%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found San Diego, CA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ California 30.9%<br>• Insight: The model detects a strong 'California' influence in the company's semantic vector.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Kehilat Ariel Synagogue (100.00%)                               │
│  Match #2: Kehilat Ariel Messianic Synagogue (95.00%)                      │
│                                                                            │
│  Score Difference: 5.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 9. Query: `Next Level Events`

✅ **Exact Match Found:** `Next Level Events` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Next Level Events | 100.00% | 1.000 | 1.000 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Next Level Site Selection & Events | 96.30% | 0.792 | 0.583 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 3 | Next Level Events | 96.20% | 1.000 | 0.596 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 4 | Next Level Plus Events | 95.90% | 0.864 | 0.542 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 5 | Next Level Events | 95.90% | 1.000 | 0.592 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 6 | Next Level Events | 95.90% | 1.000 | 0.661 | 0.00 | Excellent Match (96%) — Based on exact analysis. |
| 7 | Next Level Meetings & Events (NLME) | 95.00% | 0.792 | 0.577 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 8 | Next Level Site Selection & Events | 95.00% | 0.792 | 0.664 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 9 | Next Level Events & Marketing | 95.00% | 0.818 | 0.657 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 10 | Next Level Events | 95.00% | 1.000 | 0.609 | 0.00 | Excellent Match (95%) — Based on exact analysis. |
| 11 | Next Level Events Inc | 95.00% | 1.000 | 0.534 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 12 | Next Level Events Inc | 95.00% | 1.000 | 0.617 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 13 | Next level Site Selection & Events | 95.00% | 0.792 | 0.540 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 14 | Next Level Cycling Events | 95.00% | 0.864 | 0.619 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 15 | Next Level Events | 95.00% | 1.000 | 0.629 | 0.00 | Excellent Match (95%) — Based on exact analysis. |

### Match Narratives

**Rank #1: Next Level Events**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (100%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>• Nature: ✅ Global 30.7%<br>• Insight: The model detects a strong 'Global' influence in the company's semantic vector.<br>

**Rank #2: Next Level Site Selection & Events**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (96%)</span><br><span style='color:#eee; font-size:1.1em;'>Moderate Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('events, level, next') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (79%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟡 MODERATE (58%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (85%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Washington, DC but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+1.3% Boost</b> — Higher confidence due to 4 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Chicago 27.5%, ✅ Pennsylvania 27.4%, ✅ New York 26.4%, ✅ London 25.5%<br>

**Rank #3: Next Level Events**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (96%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟡 MODERATE (60%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (67%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Austin, TX but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+1.2% Boost</b> — Higher confidence due to 3 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Texas 49.9%<br>• Nature: ✅ Local 26.6%<br>• Insight: The model detects a strong 'Texas' influence in the company's semantic vector.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Next Level Events (100.00%)                                     │
│  Match #2: Next Level Site Selection & Events (96.30%)                     │
│                                                                            │
│  Score Difference: 3.70%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 10. Query: `Site Foundation Golf Tournament`

✅ **Exact Match Found:** `Site Foundation Golf Tournament` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Site Foundation Golf Tournament | 100.00% | 1.000 | 0.955 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Golf Tournament | 80.50% | 0.656 | 1.000 | 0.00 | Strong Match (81%) — Based on hybrid analysis. |
| 3 | Fore County Golf Tournament | 79.80% | 0.744 | 0.819 | 0.00 | Strong Match (80%) — Based on hybrid analysis. |
| 4 | National Youth Golf Foundation | 79.70% | 0.744 | 0.782 | 0.00 | Strong Match (80%) — Based on hybrid analysis. |
| 5 | National Golf Foundation | 79.60% | 0.676 | 0.910 | 0.00 | Strong Match (80%) — Based on hybrid analysis. |
| 6 | House Victory Golf Tournament | 79.10% | 0.744 | 0.804 | 0.00 | Strong Match (79%) — Based on hybrid analysis. |
| 7 | World Golf Foundation | 78.90% | 0.676 | 0.933 | 0.00 | Strong Match (79%) — Based on hybrid analysis. |
| 8 | David Maus Foundation Golf Tournament | 78.70% | 0.716 | 0.793 | 0.00 | Strong Match (79%) — Based on hybrid analysis. |
| 9 | Women In Golf Foundation | 78.40% | 0.744 | 0.798 | 0.00 | Strong Match (78%) — Based on hybrid analysis. |
| 10 | Golf Coast Junior Golf Foundation, Incorporated | 78.10% | 0.744 | 0.728 | 0.00 | Strong Match (78%) — Based on hybrid analysis. |
| 11 | PT Golf Tournament | 77.50% | 0.676 | 0.845 | 0.00 | Strong Match (78%) — Based on hybrid analysis. |
| 12 | Golf Tournament Association of America | 77.20% | 0.744 | 0.740 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |
| 13 | IMG Junior Golf Tournament | 77.10% | 0.744 | 0.750 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |
| 14 | Hearth Foundation Golf Tournament | 77.10% | 0.787 | 0.629 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |
| 15 | Golf Tournament-Central Florida | 77.10% | 0.744 | 0.750 | 0.00 | Strong Match (77%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: Site Foundation Golf Tournament**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (95%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 GOOD (76%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Miami, FL but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Miami 42.9%<br>• Insight: The model detects a strong 'Miami' influence in the company's semantic vector.<br>

**Rank #2: Golf Tournament**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (81%)</span><br><span style='color:#eee; font-size:1.1em;'>High Semantic Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('golf, tournament') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟡 MODERATE (66%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (91%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>

**Rank #3: Fore County Golf Tournament**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (80%)</span><br><span style='color:#eee; font-size:1.1em;'>Moderate Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('golf, tournament') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (74%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟢 GOOD (82%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟢 GOOD (88%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Site Foundation Golf Tournament (100.00%)                       │
│  Match #2: Golf Tournament (80.50%)                                        │
│                                                                            │
│  Score Difference: 19.50%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 11. Query: `Interim WG Meeting - BIER`

✅ **Exact Match Found:** `Interim WG Meeting - BIER` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Interim WG Meeting - BIER | 100.00% | 1.000 | 0.811 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | AACP 2012 Interim Meeting | 71.10% | 0.708 | 0.637 | 0.00 | Moderate Match (71%) — Based on hybrid analysis. |
| 3 | Interim Healthcare TEAMM Meeting | 70.60% | 0.708 | 0.642 | 0.00 | Moderate Match (71%) — Based on hybrid analysis. |
| 4 | Legislative Interim Meeting | 68.70% | 0.644 | 0.681 | 0.00 | Moderate Match (69%) — Based on hybrid analysis. |
| 5 | BI Meeting | 66.20% | 0.375 | 1.000 | 0.00 | Moderate Match (66%) — Based on hybrid analysis. |
| 6 | Bi Annual Meeting | 64.70% | 0.409 | 0.900 | 0.00 | Moderate Match (65%) — Based on hybrid analysis. |
| 7 | ETW Meeting Management | 62.80% | 0.409 | 0.766 | 0.00 | Moderate Match (63%) — Based on hybrid analysis. |
| 8 | HRW Meeting Services | 62.50% | 0.409 | 0.751 | 0.00 | Moderate Match (62%) — Based on hybrid analysis. |
| 9 | Bim Object Meeting | 62.40% | 0.409 | 0.793 | 0.00 | Moderate Match (62%) — Based on hybrid analysis. |
| 10 | HRC Advisory Board meeting | 62.20% | 0.450 | 0.734 | 0.00 | Moderate Match (62%) — Based on hybrid analysis. |
| 11 | WG Consulting | 61.70% | 0.417 | 0.739 | 0.00 | Moderate Match (62%) — Based on hybrid analysis. |
| 12 | Hrg Event & Meeting Management | 61.30% | 0.450 | 0.661 | 0.00 | Moderate Match (61%) — Based on hybrid analysis. |
| 13 | Biz Library January Meeting | 61.30% | 0.450 | 0.713 | 0.00 | Moderate Match (61%) — Based on hybrid analysis. |
| 14 | Executive Advisory Board Meeting | 61.30% | 0.450 | 0.671 | 0.00 | Moderate Match (61%) — Based on hybrid analysis. |
| 15 | American Biz Meeting | 61.00% | 0.409 | 0.784 | 0.00 | Moderate Match (61%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: Interim WG Meeting - BIER**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 GOOD (81%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (91%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Salem, OR but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #2: AACP 2012 Interim Meeting**
> <div style='border: 1px solid #ffaa00; border-left: 10px solid #ffaa00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#ffaa00;'>⚠️ MODERATE MATCH (71%)</span><br><span style='color:#eee; font-size:1.1em;'>Moderate Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('interim, meeting') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (71%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟡 MODERATE (64%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (79%) — Reflects related business categories.<br><br><br><b>Concept Analysis:</b><br>• Structure: ✅ Corporate 25.7%<br>

**Rank #3: Interim Healthcare TEAMM Meeting**
> <div style='border: 1px solid #ffaa00; border-left: 10px solid #ffaa00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#ffaa00;'>⚠️ MODERATE MATCH (71%)</span><br><span style='color:#eee; font-size:1.1em;'>Moderate Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('interim, meeting') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (71%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟡 MODERATE (64%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (77%) — Reflects related business categories.<br><br><br><b>Concept Analysis:</b><br>• Industry: ✅ Medical 39.9%<br>• Nature: ✅ Professional 28.9%<br>• Insight: The model detects a strong 'Medical' influence in the company's semantic vector.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Interim WG Meeting - BIER (100.00%)                             │
│  Match #2: AACP 2012 Interim Meeting (71.10%)                              │
│                                                                            │
│  Score Difference: 28.90%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 12. Query: `DermaQuest Inc`

✅ **Exact Match Found:** `DermaQuest Inc` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | DermaQuest Inc | 100.00% | 1.000 | 0.881 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Dermaquest Skin Care | 95.00% | 0.750 | 1.000 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 3 | Dermaquest, Incorporated | 95.00% | 1.000 | 0.724 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 4 | Dermaquest Skin Therapy | 95.00% | 0.750 | 0.638 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 5 | Dermapen | 57.00% | 0.300 | 0.947 | 0.00 | Moderate Match (57%) — Based on hybrid analysis. |
| 6 | DERMA E | 56.90% | 0.289 | 0.971 | 0.00 | Moderate Match (57%) — Based on hybrid analysis. |
| 7 | ENTREQUEST | 56.80% | 0.315 | 0.767 | 0.00 | Moderate Match (57%) — Based on hybrid analysis. |
| 8 | Interquest | 56.60% | 0.315 | 0.831 | 0.00 | Moderate Match (57%) — Based on hybrid analysis. |
| 9 | ProQuest | 55.90% | 0.300 | 0.798 | 0.00 | Moderate Match (56%) — Based on hybrid analysis. |
| 10 | medQuest | 55.70% | 0.300 | 0.783 | 0.00 | Moderate Match (56%) — Based on hybrid analysis. |
| 11 | Perquest | 55.30% | 0.350 | 0.775 | 0.00 | Moderate Match (55%) — Based on hybrid analysis. |
| 12 | SPECIALQUEST | 55.20% | 0.286 | 0.828 | 0.00 | Moderate Match (55%) — Based on hybrid analysis. |
| 13 | Carequest | 54.60% | 0.284 | 0.783 | 0.00 | Moderate Match (55%) — Based on hybrid analysis. |
| 14 | Crownquest | 54.20% | 0.270 | 0.854 | 0.00 | Moderate Match (54%) — Based on hybrid analysis. |
| 15 | Spaquest | 53.90% | 0.300 | 0.757 | 0.00 | Moderate Match (54%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: DermaQuest Inc**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 GOOD (88%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (54%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Hayward, CA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #2: Dermaquest Skin Care**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (95%)</span><br><span style='color:#eee; font-size:1.1em;'>High Semantic Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('dermaquest') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (75%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (63%) — Reflects related business categories.<br><br><br><b>Concept Analysis:</b><br>• Industry: ✅ Medical 26.7%<br>

**Rank #3: Dermaquest, Incorporated**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (95%)</span><br><span style='color:#eee; font-size:1.1em;'>High Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Pure Semantic Match</b>. There is no direct text overlap; the connection is based entirely on the underlying business context and meaning.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 GOOD (72%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (57%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Hayward, CA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: DermaQuest Inc (100.00%)                                        │
│  Match #2: Dermaquest Skin Care (95.00%)                                   │
│                                                                            │
│  Score Difference: 5.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 13. Query: `Ellwood Group Inc`

✅ **Exact Match Found:** `Ellwood Group Inc` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | Ellwood Group Inc | 100.00% | 1.000 | 1.000 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Ellwood Associates | 95.90% | 0.900 | 0.724 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 3 | Ellwood TX Forge Houston | 95.90% | 0.692 | 0.650 | 0.00 | Excellent Match (96%) — Based on hybrid analysis. |
| 4 | Ellwood Community Church | 95.00% | 0.750 | 0.826 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 5 | Ellwood TX Forge | 95.00% | 0.750 | 0.696 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 6 | Ellwood Rose Machine | 95.00% | 0.750 | 0.703 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 7 | Ellwood Specialty Steel | 95.00% | 0.750 | 0.682 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 8 | Ellwood City Area School District (inc) | 95.00% | 0.600 | 0.666 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 9 | Ellwood TX Forge Houston | 95.00% | 0.692 | 0.692 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 10 | Ellwood TX Forge | 95.00% | 0.750 | 0.656 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 11 | Ellwood Closed Die Group | 95.00% | 0.750 | 0.688 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 12 | EGI | 75.00% | 1.000 | 1.000 | 1.00 | Strong Match (75%) — Based on acronym_reverse analysis. |
| 13 | Delwood | 60.30% | 0.375 | 0.852 | 0.00 | Moderate Match (60%) — Based on hybrid analysis. |
| 14 | Kenwood Group | 59.00% | 0.321 | 0.861 | 0.00 | Moderate Match (59%) — Based on hybrid analysis. |
| 15 | Reignwood Group | 57.50% | 0.281 | 0.958 | 0.00 | Moderate Match (57%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: Ellwood Group Inc**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 GOOD (84%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Ellwood City, PA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #2: Ellwood Associates**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (96%)</span><br><span style='color:#eee; font-size:1.1em;'>High Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('ellwood') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (90%) — Based on strong character overlap.<br>• <b>Semantic Link:</b> 🟢 GOOD (72%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (67%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Chicago, IL but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+0.9% Boost</b> — Higher confidence due to 2 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Chicago 42.3%, ✅ Illinois 32.5%<br>• Insight: The model detects a strong 'Chicago' influence in the company's semantic vector.<br>

**Rank #3: Ellwood TX Forge Houston**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (96%)</span><br><span style='color:#eee; font-size:1.1em;'>Partial Composite Match</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('ellwood') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟡 MODERATE (69%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟡 MODERATE (65%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (65%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Houston, TX but no boost was warranted.<br>• <b>Entity Popularity:</b> 🟢 <b>+0.9% Boost</b> — Higher confidence due to 2 occurrences in master set.<br><br><br><b>Concept Analysis:</b><br>• Geography: ✅ Texas 38.2%<br>• Insight: The model detects a strong 'Texas' influence in the company's semantic vector.<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: Ellwood Group Inc (100.00%)                                     │
│  Match #2: Ellwood Associates (95.90%)                                     │
│                                                                            │
│  Score Difference: 4.10%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 14. Query: `American Miniature Horse Registry`

✅ **Exact Match Found:** `American Miniature Horse Registry` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | American Miniature Horse Registry | 100.00% | 1.000 | 1.000 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | American Miniature Horse Association | 90.00% | 0.844 | 0.592 | 0.00 | Excellent Match (90%) — Based on hybrid analysis. |
| 3 | American Miniature Horse Association Headquarters | 79.50% | 0.767 | 0.761 | 0.00 | Strong Match (80%) — Based on hybrid analysis. |
| 4 | American Saddle Horse Association | 72.30% | 0.744 | 0.572 | 0.00 | Moderate Match (72%) — Based on hybrid analysis. |
| 5 | American Youth & Horse Council | 72.00% | 0.744 | 0.538 | 0.00 | Moderate Match (72%) — Based on hybrid analysis. |
| 6 | Miniature Horse & Pony Show | 71.90% | 0.744 | 0.543 | 0.00 | Moderate Match (72%) — Based on hybrid analysis. |
| 7 | American Miniature Hores Association | 71.30% | 0.744 | 0.524 | 0.00 | Moderate Match (71%) — Based on hybrid analysis. |
| 8 | American Youth Horse Council | 70.90% | 0.744 | 0.526 | 0.00 | Moderate Match (71%) — Based on hybrid analysis. |
| 9 | Arabian Horse Registry of America, Inc. | 70.90% | 0.744 | 0.494 | 0.00 | Moderate Match (71%) — Based on hybrid analysis. |
| 10 | American Horse Publication | 70.60% | 0.676 | 0.610 | 0.00 | Moderate Match (71%) — Based on hybrid analysis. |
| 11 | American Horse Council | 70.00% | 0.676 | 0.632 | 0.00 | Moderate Match (70%) — Based on hybrid analysis. |
| 12 | American Hackney Horse Society | 69.60% | 0.744 | 0.482 | 0.00 | Moderate Match (70%) — Based on hybrid analysis. |
| 13 | American Shire Horse Association | 69.60% | 0.744 | 0.433 | 0.00 | Moderate Match (70%) — Based on hybrid analysis. |
| 14 | Norwegian Fjord Horse Registry | 69.40% | 0.744 | 0.480 | 0.00 | Moderate Match (69%) — Based on hybrid analysis. |
| 15 | American Youth Horse Council | 69.30% | 0.744 | 0.492 | 0.00 | Moderate Match (69%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: American Miniature Horse Registry**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (100%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (100%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>

**Rank #2: American Miniature Horse Association**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (90%)</span><br><span style='color:#eee; font-size:1.1em;'>Moderate Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('american, horse, miniature') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (84%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟡 MODERATE (59%) — Detected via moderate meaning-based connection.<br>• <b>Concept Alignment:</b> 🟢 GOOD (81%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Alvarado, TX but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #3: American Miniature Horse Association Headquarters**
> <div style='border: 1px solid #00cc00; border-left: 10px solid #00cc00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00cc00;'>✅ STRONG MATCH (80%)</span><br><span style='color:#eee; font-size:1.1em;'>Moderate Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('american, horse, miniature') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 GOOD (77%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟢 GOOD (76%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟢 GOOD (89%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>• Structure: ✅ Small Business 26.2%<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: American Miniature Horse Registry (100.00%)                     │
│  Match #2: American Miniature Horse Association (90.00%)                   │
│                                                                            │
│  Score Difference: 10.00%                                                  │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## 15. Query: `YADA ENTERPRISES, INC`

✅ **Exact Match Found:** `YADA ENTERPRISES, INC` (100.0%)

### Top 15 Search Results

| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |
|------|--------------|-------|--------|----------|---------|---------------|
| 1 | YADA ENTERPRISES, INC | 100.00% | 1.000 | 0.883 | 0.00 | Excellent Match (100%) — Based on exact analysis. |
| 2 | Yada Yada | 95.00% | 0.900 | 0.786 | 0.00 | Excellent Match (95%) — Based on hybrid analysis. |
| 3 | Yasuda Corporation Limited | 63.50% | 0.360 | 0.929 | 0.00 | Moderate Match (63%) — Based on hybrid analysis. |
| 4 | Yama Enterprises | 62.10% | 0.337 | 0.880 | 0.00 | Moderate Match (62%) — Based on hybrid analysis. |
| 5 | Ya | 59.70% | 0.300 | 1.000 | 0.00 | Moderate Match (60%) — Based on hybrid analysis. |
| 6 | Yamato Corporation | 59.00% | 0.270 | 0.877 | 0.00 | Moderate Match (59%) — Based on hybrid analysis. |
| 7 | Yama | 58.10% | 0.337 | 0.954 | 0.00 | Moderate Match (58%) — Based on hybrid analysis. |
| 8 | Yamas | 57.70% | 0.337 | 0.874 | 0.00 | Moderate Match (58%) — Based on hybrid analysis. |
| 9 | Adani Enterprises | 56.60% | 0.300 | 0.746 | 0.00 | Moderate Match (57%) — Based on hybrid analysis. |
| 10 | Yara | 56.20% | 0.337 | 0.901 | 0.00 | Moderate Match (56%) — Based on hybrid analysis. |
| 11 | YATA | 55.90% | 0.337 | 0.883 | 0.00 | Moderate Match (56%) — Based on hybrid analysis. |
| 12 | Yapu Limited Corporation | 55.60% | 0.225 | 0.854 | 0.00 | Moderate Match (56%) — Based on hybrid analysis. |
| 13 | Yau Yiu Company | 54.80% | 0.149 | 0.934 | 0.00 | Moderate Match (55%) — Based on hybrid analysis. |
| 14 | ALD Enterprises | 54.70% | 0.257 | 0.743 | 0.00 | Moderate Match (55%) — Based on hybrid analysis. |
| 15 | Yap, Inc. | 54.40% | 0.257 | 0.754 | 0.00 | Moderate Match (54%) — Based on hybrid analysis. |

### Match Narratives

**Rank #1: YADA ENTERPRISES, INC**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (100%)</span><br><span style='color:#eee; font-size:1.1em;'>Exact Name Match</span></div><br><br><b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (100%) — Based on identical strings.<br>• <b>Semantic Link:</b> 🟢 GOOD (88%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 EXCELLENT (93%) — Reflects highly aligned industries.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Redfield, SD but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #2: Yada Yada**
> <div style='border: 1px solid #00ff00; border-left: 10px solid #00ff00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#00ff00;'>✅ EXCELLENT MATCH (95%)</span><br><span style='color:#eee; font-size:1.1em;'>High Lexical Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('yada') despite differences in overall string structure.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟢 EXCELLENT (90%) — Based on strong character overlap.<br>• <b>Semantic Link:</b> 🟢 GOOD (79%) — Detected via strong contextual link.<br>• <b>Concept Alignment:</b> 🟡 MODERATE (57%) — Reflects related business categories.<br>• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found Kirkland, WA but no boost was warranted.<br><br><br><b>Concept Analysis:</b><br>

**Rank #3: Yasuda Corporation Limited**
> <div style='border: 1px solid #ffaa00; border-left: 10px solid #ffaa00; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'><span style='font-size:1.4em; font-weight:bold; color:#ffaa00;'>⚠️ MODERATE MATCH (63%)</span><br><span style='color:#eee; font-size:1.1em;'>High Semantic Similarity</span></div><br><br><b>Relationship:</b> This is a <b>Pure Semantic Match</b>. There is no direct text overlap; the connection is based entirely on the underlying business context and meaning.<br><br><b>Evidence Analysis:</b><br>• <b>Name Similarity:</b> 🟠 FAIR (36%) — Based on partial character alignment.<br>• <b>Semantic Link:</b> 🟢 EXCELLENT (93%) — Detected via synonymous concepts.<br>• <b>Concept Alignment:</b> 🟢 GOOD (89%) — Reflects highly aligned industries.<br><br><br><b>Concept Analysis:</b><br>• Structure: ✅ Small Business 25.5%<br>

### Ranking Rationale
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANKING: MATCH #1 VS MATCH #2                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Match #1: YADA ENTERPRISES, INC (100.00%)                                 │
│  Match #2: Yada Yada (95.00%)                                              │
│                                                                            │
│  Score Difference: 5.00%                                                   │
│                                                                            │
│  RANKING RATIONALE:                                                        │
│  • Text Identity: #1 is a perfect character match.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

