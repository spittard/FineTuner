# Company Matching Control Set Report

**Generated:** 2025-12-28 12:18:02

---

## Matching Scenarios Handled

This system is designed to handle the following real-world company matching challenges:

### 1. **Exact Matches**
Perfect text matching when query exactly equals company name.
- Example: `"IBM"` → `"IBM"` (100% match)

### 2. **Acronym Expansions**
Matching acronyms to their full company names.
- Example: `"IBM"` → `"International Business Machines"` (98%+ match)
- Example: `"ABA"` → `"American Bar Association"` (98%+ match)

### 3. **Typo Tolerance**
Fuzzy matching handles common spelling errors.
- Example: `"Microsft"` → `"Microsoft"` (94%+ match)
- Example: `"Gogle"` → `"Google"` (92%+ match)

### 4. **Abbreviation Variations**
Recognizes common business abbreviations.
- Example: `"Corp"` ↔ `"Corporation"`
- Example: `"Inc"` ↔ `"Incorporated"`
- Example: `"Intl"` ↔ `"International"`

### 5. **Plural/Singular Variations**
Handles grammatical number differences.
- Example: `"International Business Machine"` vs `"International Business Machines"`

### 6. **Word Order Variations**
Matches despite different word arrangements.
- Example: `"Bank First National"` → `"First National Bank"`

### 7. **Partial Name Matches**
Finds matches when only part of the company name is provided.
- Example: `"Acme"` → `"Acme Corporation"`

### 8. **Legal Entity Suffix Variations**
Handles different legal entity designations.
- Example: `"Acme LLC"` vs `"Acme Inc"` vs `"Acme Corporation"`

---

## Scoring Formula

```
Base Score = (String Similarity × 70%) + (Semantic Similarity × 30%)
Final Score = Base Score + Acronym Fidelity Boost (up to +15%)
```

## Acronym Fidelity Score Explained

The **Acronym Fidelity Score** measures how well a company name expands an acronym using **pure algorithmic pattern matching**. It analyzes whether the first letters of significant words match the acronym letters in order.

### How It Works

The algorithm extracts the first letter of each significant word (excluding common words like 'the', 'of', 'and') and compares them to the acronym:

**Example 1: Perfect Match (Fidelity = 1.00)**
```
Query: "IBM"
Match: "International Business Machines"

Step 1: Extract first letters
  Words: [International, Business, Machines]
  First letters: [I, B, M]
  Joined: "IBM"

Step 2: Compare to query
  Query: "IBM"
  Word starts: "IBM"
  Match: EXACT ✓

Step 3: Check for overlaps
  Does any word contain multiple acronym letters? NO ✓

Result: Fidelity = 1.00 (Perfect Expansion)
```

**Example 2: Prefix Match (Fidelity = 0.95)**
```
Query: "IBM"
Match: "International Business Machines Corporation"

First letters: [I, B, M, C] → "IBMC"
Query: "IBM"
Pattern: "IBM" is a prefix of "IBMC" ✓

Result: Fidelity = 0.95 (Acronym matches start, extra words after)
```

**Example 3: Subsequence Match (Fidelity = 0.90)**
```
Query: "IBM"
Match: "International Bureau of Management"

First letters: [I, B, M] → "IBM" (skipping 'of')
Pattern: "IBM" appears as subsequence in word starts ✓

Result: Fidelity = 0.90 (Subsequence match)
```

### Fidelity Score Reference

| Score | Pattern | Example |
|-------|---------|----------|
| 1.00 | Perfect: Each letter = first letter of distinct word, no overlaps | IBM → International Business Machines |
| 0.95 | Prefix: Acronym matches start, extra words after | IBM → International Business Machines Corp |
| 0.90 | Subsequence: Acronym appears in word-starts | IBM → International Bureau of Management |
| 0.70 | Collision: Words overlap with acronym letters | (penalized) |
| 0.65 | Word prefix: First word starts with acronym | IBMA → IBM... |
| 0.40 | Partial: Some letters match out of order | (scaled down) |

---

## Control Set Results

### 1. PDMA Association

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `PDMA Alliance` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 0.7332 | 30% | 0.2200 |
| **Base Score** | **0.8150** | - | **81.50%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. PDMA Alliance (90.0%)
3. Association Headquarters-PDMA (90.0%)
4. PDMA (77.6%)
5. PDMA inc (75.2%)
6. PDMA Corporation (74.8%)

---

### 2. Nicolas/Sanchez Wedding

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Sanchez Wedding` (85.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7875 | 70% | 0.5512 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8512** | - | **85.12%** |
| **Final Score** | **0.8510** | - | **85.10%** |

**Top 5 Non-Self Matches:**

2. Sanchez Wedding (85.1%)
3. Sanchez/Flores Wedding (76.3%)
4. Sanchez/Ramirez Wedding (76.0%)
5. Garcia Sanchez Wedding (76.0%)
6. Sanchez/Puerto Wedding (75.8%)

---

### 3. Kehilat Ariel Synagogue

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Kehilat Ariel Messianic Synagogue` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 0.8174 | 30% | 0.2452 |
| **Base Score** | **0.8498** | - | **84.98%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Kehilat Ariel Messianic Synagogue (90.0%)
3. Kehilat Ariel (90.0%)
4. KAS (69.0%)
5. Kehilath Israel Synagogue (59.6%)
6. Beth Israel Synagogue (55.5%)

---

### 4. Next Level Events

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Next Level Events Inc` (95.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.8369 | 30% | 0.2511 |
| **Base Score** | **0.9511** | - | **95.11%** |
| **Final Score** | **0.9510** | - | **95.10%** |

**Top 5 Non-Self Matches:**

2. Next Level Events Inc (95.1%)
3. Next Level Plus Events (90.0%)
4. Next Level (90.0%)
5. Next Level Now (90.0%)
6. Next Level Games (90.0%)

---

### 5. Site Foundation Golf Tournament

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Hearth Foundation Golf Tournament` (80.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7875 | 70% | 0.5512 |
| Semantic Similarity | 0.8309 | 30% | 0.2493 |
| **Base Score** | **0.8005** | - | **80.05%** |
| **Final Score** | **0.8010** | - | **80.10%** |

**Top 5 Non-Self Matches:**

2. Hearth Foundation Golf Tournament (80.1%)
3. Tournament Golf Foundation Incorporated (76.5%)
4. Golf League Amateur Golf Tournament (74.2%)
5. Midwest Classic Golf Tournament (74.2%)
6. World Amateur Golf Tournament (74.2%)

---

### 6. Interim WG Meeting - BIER

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `BI Meeting` (53.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.3750 | 70% | 0.2625 |
| Semantic Similarity | 0.9159 | 30% | 0.2748 |
| **Base Score** | **0.5373** | - | **53.73%** |
| **Final Score** | **0.5370** | - | **53.70%** |

**Top 5 Non-Self Matches:**

2. BI Meeting (53.7%)
3. Bi Annual Meeting (53.4%)
4. Biz meeting (50.7%)
5. WMS Meeting (50.5%)
6. Bim Object Meeting (50.4%)

---

### 7. DermaQuest Inc

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Dermaquest, Incorporated` (94.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.8076 | 30% | 0.2423 |
| **Base Score** | **0.9423** | - | **94.23%** |
| **Final Score** | **0.9420** | - | **94.20%** |

**Top 5 Non-Self Matches:**

2. Dermaquest, Incorporated (94.2%)
3. Dermaquest Skin Care (74.5%)
4. Dermaquest Skin Therapy (73.1%)
5. Perquest Inc (42.9%)
6. EQuest (42.8%)

---

### 8. Ellwood Group Inc

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Ellwood Associates` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.7737 | 30% | 0.2321 |
| **Base Score** | **0.8621** | - | **86.21%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Ellwood Associates (90.0%)
3. EGI (75.0%)
4. Ellwood Community Church (74.1%)
5. Delwood (48.5%)
6. Sellwood (47.4%)

---

### 9. American Miniature Horse Registry

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `American Miniature Horse Association` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 0.8153 | 30% | 0.2446 |
| **Base Score** | **0.8352** | - | **83.52%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. American Miniature Horse Association (90.0%)
3. American Miniature Horse Association Headquarters (76.5%)
4. Miniature Horse & Pony Show (73.1%)
5. American Horse Show Association (70.7%)
6. American Youth Horse Council (70.1%)

---

### 10. YADA ENTERPRISES, INC

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Yada Yada` (90.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.9163 | 30% | 0.2749 |
| **Base Score** | **0.9049** | - | **90.49%** |
| **Final Score** | **0.9050** | - | **90.50%** |

**Top 5 Non-Self Matches:**

2. Yada Yada (90.5%)
3. Yama Enterprises (52.9%)
4. Yacada (47.6%)
5. Yadea Group (46.8%)
6. Yasuda Corporation Limited (46.6%)

---

### 11. Seafood Nutrition Partnership

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Sustainable Seafood Partnership` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.8210 | 30% | 0.2463 |
| **Base Score** | **0.8131** | - | **81.31%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Sustainable Seafood Partnership (90.0%)
3. SEAFOOD NUTRITION (81.4%)
4. Seafood Products Association (59.5%)
5. Seafood Choices Alliance (59.5%)
6. Seafood Choices Alliances (58.8%)

---

### 12. AVIAKOMPANIYA SIBIR, PAO

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `ASP` (69.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **1.0000** | - | **100.00%** |
| Acronym Fidelity Boost | 0.7000 | 15% max | +0.1050 |
| **Final Score** | **0.6900** | - | **69.00%** |

**Top 5 Non-Self Matches:**

2. ASP (69.0%)
3. AVIAKOMPANIYA MIZHNARODNI AVIA (59.0%)
4. Faizan Kabir (35.1%)
5. Shibir  Desai (34.5%)
6. Kabira (34.0%)

---

### 13. Hartford Hospital School of Nursing

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Hartford Hospital Offices USA` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 0.7980 | 30% | 0.2394 |
| **Base Score** | **0.8344** | - | **83.44%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Hartford Hospital Offices USA (90.0%)
3. Hartford Hospital (82.5%)
4. Hartford School District (77.0%)
5. New Hartford School (76.3%)
6. Hartford Elementary School (75.9%)

---

### 14. Internal J&J Meeting and Breakfast

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Breakfast Meeting` (65.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6500** | - | **65.00%** |
| **Final Score** | **0.6500** | - | **65.00%** |

**Top 5 Non-Self Matches:**

2. Breakfast Meeting (65.0%)
3. AEP Breakfast Meeting (64.4%)
4. Breakfast Meeting NYC (64.2%)
5. Bisnow Breakfast Meeting (63.3%)
6. GMCVB Breakfast & Meeting (62.7%)

---

### 15. Spina Bifida Coalition of Cincinnati

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Spina Bifida Association of Cincinnati, Inc.` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 0.7544 | 30% | 0.2263 |
| **Base Score** | **0.8169** | - | **81.69%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Spina Bifida Association of Cincinnati, Inc. (90.0%)
3. SBCC (75.0%)
4. Spina Bifida Association of Michigan (75.0%)
5. Illinois Spina Bifida Association (74.7%)
6. Spina Bifida Association of Kentucky (74.3%)

---

### 16. THE SOCA GROUP ORGANIZATION

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Team SOCA` (78.9%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8942 | 30% | 0.2682 |
| **Base Score** | **0.7889** | - | **78.89%** |
| **Final Score** | **0.7890** | - | **78.90%** |

**Top 5 Non-Self Matches:**

2. Team SOCA (78.9%)
3. Soca Society (78.0%)
4. System Organization Group (75.9%)
5. SOCA Convention (75.8%)
6. Soca Takeover (75.4%)

---

### 17. Shiroyama Junior High School

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Junior High School 45` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 0.7318 | 30% | 0.2195 |
| **Base Score** | **0.7970** | - | **79.70%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Junior High School 45 (90.0%)
3. Haga Junior High School (90.0%)
4. Jubail Junior High School (90.0%)
5. CARROLL JUNIOR HIGH SCHOOL (90.0%)
6. Junior High School 22 (90.0%)

---

### 18. National Home Health

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `National Home Healthcare` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.9004 | 30% | 0.2701 |
| **Base Score** | **0.9001** | - | **90.01%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. National Home Healthcare (90.0%)
3. Home Health (90.0%)
4. National Home Health Care (90.0%)
5. Community Home Health (90.0%)
6. RESIDENTIAL HOME HEALTH (90.0%)

---

### 19. American News Women's Club

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `American Women's Club` (79.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7670 | 70% | 0.5369 |
| Semantic Similarity | 0.8713 | 30% | 0.2614 |
| **Base Score** | **0.7983** | - | **79.83%** |
| **Final Score** | **0.7980** | - | **79.80%** |

**Top 5 Non-Self Matches:**

2. American Women's Club (79.8%)
3. American Women Club (77.3%)
4. DC Democratic Women's Club (73.4%)
5. UM Women's Club (70.4%)
6. Women's International Club (69.9%)

---

### 20. Denise Roberge

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Tamara Denise` (73.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.7021 | 30% | 0.2106 |
| **Base Score** | **0.7313** | - | **73.13%** |
| **Final Score** | **0.7310** | - | **73.10%** |

**Top 5 Non-Self Matches:**

2. Tamara Denise (73.1%)
3. Denise Long (72.6%)
4. Tasha Denise (72.4%)
5. Sandra Denise (72.4%)
6. Denise O (71.5%)

---

### 21. Synergy Soccer Club

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Synergy Football Club` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.8363 | 30% | 0.2509 |
| **Base Score** | **0.8177** | - | **81.77%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Synergy Football Club (90.0%)
3. Synergy Volleyball Club (90.0%)
4. Alliance Soccer Club (90.0%)
5. Advantage Soccer Club (90.0%)
6. Kitsap Soccer Club (90.0%)

---

### 22. NFC Forum

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `NFC Forum         .` (96.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.8866 | 30% | 0.2660 |
| **Base Score** | **0.9660** | - | **96.60%** |
| **Final Score** | **0.9660** | - | **96.60%** |

**Top 5 Non-Self Matches:**

2. NFC Forum         . (96.6%)
3. NFC Forum Members (90.0%)
4. NFC Fighting (73.8%)
5. NFC Orientation (73.4%)
6. NFC Consulting (73.2%)

---

### 23. A Better Choice Limousine & Concierge

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `First Choice Limousine Services` (76.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8223 | 30% | 0.2467 |
| **Base Score** | **0.7673** | - | **76.73%** |
| **Final Score** | **0.7670** | - | **76.70%** |

**Top 5 Non-Self Matches:**

2. First Choice Limousine Services (76.7%)
3. Prestige Limousine (57.7%)
4. Capital Travel Limousine (56.6%)
5. Executive Limousine (56.3%)
6. Limousine Livery (56.0%)

---

### 24. Danish Sisterhood of America

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `The Danish Sisterhood of America` (97.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.9134 | 30% | 0.2740 |
| **Base Score** | **0.9740** | - | **97.40%** |
| **Final Score** | **0.9740** | - | **97.40%** |

**Top 5 Non-Self Matches:**

2. The Danish Sisterhood of America (97.4%)
3. Danish Sisterhood of the Americas (90.0%)
4. Danish Sisterhood and Brotherhood of America (90.0%)
5. Danish Brotherhood & Danish Sisterhood of America (90.0%)
6. DANISH BROTHERHOOD AND DANISH SISTERHOOD OF AMERICA (90.0%)

---

### 25. Brooklyn Comics Club

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Brooklyn Baseball Club` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.7758 | 30% | 0.2327 |
| **Base Score** | **0.7995** | - | **79.95%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Brooklyn Baseball Club (90.0%)
3. Brooklyn Ski Club (90.0%)
4. Brooklyn Conversation Club (90.0%)
5. Brooklyn Football Club (90.0%)
6. Brooklyn Barbell Club (90.0%)

---

### 26. Global Interagency Security Forum

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Cyber Security Collaboration Forum` (76.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 0.7402 | 30% | 0.2221 |
| **Base Score** | **0.7675** | - | **76.75%** |
| **Final Score** | **0.7670** | - | **76.70%** |

**Top 5 Non-Self Matches:**

2. Cyber Security Collaboration Forum (76.7%)
3. Security Network Forum (76.5%)
4. Security Forum (76.5%)
5. Infrastructure Security and Resilience Forum (76.4%)
6. The Cyber Security Forum Initiative (76.3%)

---

### 27. Lancet Software

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Lancet Technology` (76.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8254 | 30% | 0.2476 |
| **Base Score** | **0.7683** | - | **76.83%** |
| **Final Score** | **0.7680** | - | **76.80%** |

**Top 5 Non-Self Matches:**

2. Lancet Technology (76.8%)
3. Lancet Technology, Incorporated (72.4%)
4. JAT Software (71.4%)
5. CAS Software (71.3%)
6. Agile Software (70.3%)

---

### 28. Our Lady of the Lakes Catholic Church and School

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `OUR LADY OF THE LAKES CATHOLIC CHURCH` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8498 | 70% | 0.5949 |
| Semantic Similarity | 0.9616 | 30% | 0.2885 |
| **Base Score** | **0.8833** | - | **88.33%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. OUR LADY OF THE LAKES CATHOLIC CHURCH (90.0%)
3. Our Lady of the Lakes Catholic School (90.0%)
4. Our Lady of the Lake Roman Catholic Church (90.0%)
5. Our Lady of the Lake Catholic Church (79.9%)
6. Our Lady Guadalupe Catholic Church (77.1%)

---

### 29. Broadway Bound International

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Broadway Bound` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.9657 | 30% | 0.2897 |
| **Base Score** | **0.8624** | - | **86.24%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Broadway Bound (90.0%)
3. Broadway Bound West (90.0%)
4. Broadway Bound Kids (90.0%)
5. Bound Four Broadway (90.0%)
6. Broadway Bound Studio (90.0%)

---

### 30. E. H. Wachs

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `EHW` (75.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **1.0000** | - | **100.00%** |
| Acronym Fidelity Boost | 1.0000 | 15% max | +0.1500 |
| **Final Score** | **0.7500** | - | **75.00%** |

**Top 5 Non-Self Matches:**

2. EHW (75.0%)
3. E.H. Wachs (64.5%)
4. Wachs Services (60.3%)
5. Elen Wachs (60.1%)
6. Wachs Water Services (58.2%)

---

### 31. Marine Corps Fox 2/5

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `United State Marine Corps` (77.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 0.7498 | 30% | 0.2249 |
| **Base Score** | **0.7704** | - | **77.04%** |
| **Final Score** | **0.7700** | - | **77.00%** |

**Top 5 Non-Self Matches:**

2. United State Marine Corps (77.0%)
3. Marine Corps Personnel Support (76.9%)
4. American US Marine Corps (76.9%)
5. MARINE CORPS BASE CAMP (76.9%)
6. Marine Corps Community Service (76.8%)

---

### 32. Fantasia Turistica

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Fantasia Travels` (77.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8496 | 30% | 0.2549 |
| **Base Score** | **0.7755** | - | **77.55%** |
| **Final Score** | **0.7750** | - | **77.50%** |

**Top 5 Non-Self Matches:**

2. Fantasia Travels (77.5%)
3. Fantasia Travel (76.7%)
4. Fantasia Accessry (75.4%)
5. Ferrari Fantasia (75.1%)
6. Fantasia (74.1%)

---

### 33. Esoterix

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Esoterix Headquarters` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.7630 | 30% | 0.2289 |
| **Base Score** | **0.8016** | - | **80.16%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Esoterix Headquarters (90.0%)
3. Esoterix Integrated Genetics (73.2%)
4. CENTRIX (42.6%)
5. Metrix (42.5%)
6. Verix (42.0%)

---

### 34. Coker Group

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `The Coker Group` (90.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.9026 | 30% | 0.2708 |
| **Base Score** | **0.9008** | - | **90.08%** |
| **Final Score** | **0.9010** | - | **90.10%** |

**Top 5 Non-Self Matches:**

2. The Coker Group (90.1%)
3. Coker College (90.0%)
4. Coker Consultants (90.0%)
5. Coker University (90.0%)
6. Coker Law (90.0%)

---

### 35. GILEAD IT

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Gilead Productions` (77.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8405 | 30% | 0.2522 |
| **Base Score** | **0.7728** | - | **77.28%** |
| **Final Score** | **0.7730** | - | **77.30%** |

**Top 5 Non-Self Matches:**

2. Gilead Productions (77.3%)
3. Gilead Science (77.3%)
4. Gilead Sciences (77.0%)
5. Gilead 1N (76.8%)
6. Gilead Services (76.2%)

---

### 36. 4143 Affiliate INDA 2016

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `1528 Affiliate INDA 2016` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.7036 | 30% | 0.2111 |
| **Base Score** | **0.8411** | - | **84.11%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. 1528 Affiliate INDA 2016 (90.0%)
3. 4143 Affiliate Aan 2017 (75.6%)
4. 430 Affiliate ALA 2016 (72.2%)
5. 1035 Affiliate CASE 2016 (71.3%)
6. 1035 AFSA Affiliate 2016 (70.3%)

---

### 37. Pipe and Plant Solutions

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Pipe & Plant` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8264 | 70% | 0.5785 |
| Semantic Similarity | 0.8998 | 30% | 0.2699 |
| **Base Score** | **0.8485** | - | **84.85%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Pipe & Plant (90.0%)
3. Advanced Pipe Solutions (78.9%)
4. TV Pipe Solutions (73.8%)
5. Infra Pipe Solutions (73.7%)
6. Plant Solutions Limited (72.7%)

---

### 38. Stephen Rourke

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Rourke` (73.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6300 | 70% | 0.4410 |
| Semantic Similarity | 0.9754 | 30% | 0.2926 |
| **Base Score** | **0.7336** | - | **73.36%** |
| **Final Score** | **0.7340** | - | **73.40%** |

**Top 5 Non-Self Matches:**

2. Rourke (73.4%)
3. Rourke Publishing (73.4%)
4. Damon Rourke (72.7%)
5. Rourke Manufacturing (71.5%)
6. Rourke Rooms (68.9%)

---

### 39. MIT Initiative on the Digital Economy

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `MIT Energy Initiative` (58.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4813 | 70% | 0.3369 |
| Semantic Similarity | 0.8125 | 30% | 0.2438 |
| **Base Score** | **0.5806** | - | **58.06%** |
| **Final Score** | **0.5810** | - | **58.10%** |

**Top 5 Non-Self Matches:**

2. MIT Energy Initiative (58.1%)
3. MIT Information Services and Technology (44.4%)
4. MIT INFORMATION SERVICES (43.2%)
5. MIT Information Systems (41.8%)
6. MIT Global Initiatives (41.6%)

---

### 40. Urx Community USA

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `USA Community Service Commission` (67.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7027 | 70% | 0.4919 |
| Semantic Similarity | 0.5990 | 30% | 0.1797 |
| **Base Score** | **0.6716** | - | **67.16%** |
| **Final Score** | **0.6720** | - | **67.20%** |

**Top 5 Non-Self Matches:**

2. USA Community Service Commission (67.2%)
3. Florida Urological Society USA (56.0%)
4. URX Conference (55.2%)
5. Sites USA (54.6%)
6. URENCO USA (54.3%)

---

### 41. Spredfast Engage

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Spredfast Events` (77.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8534 | 30% | 0.2560 |
| **Base Score** | **0.7766** | - | **77.66%** |
| **Final Score** | **0.7770** | - | **77.70%** |

**Top 5 Non-Self Matches:**

2. Spredfast Events (77.7%)
3. Spredfast Product (73.9%)
4. Spredfast (73.0%)
5. Shredfast (41.6%)
6. Redfast (40.5%)

---

### 42. City of Dallas-Parks & Recreation

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `City of Dallas Park & Recreation` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 0.9326 | 30% | 0.2798 |
| **Base Score** | **0.8704** | - | **87.04%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. City of Dallas Park & Recreation (90.0%)
3. Dallas Parks and Recreation Department (90.0%)
4. Dallas Parks and Recreation Dept (90.0%)
5. Baltimore City Recreation and Parks (90.0%)
6. City of Miami Parks & Recreation (90.0%)

---

### 43. Kai Pono Builders, Inc.

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Kai Pono Builders` (95.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9375 | 70% | 0.6562 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.9563** | - | **95.62%** |
| **Final Score** | **0.9560** | - | **95.60%** |

**Top 5 Non-Self Matches:**

2. Kai Pono Builders (95.6%)
3. Pono Kai Resort (73.3%)
4. Pono Kai (72.2%)
5. Kai Partners (56.1%)
6. S Kai (55.3%)

---

### 44. MUSICFIRST COALITION

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `music FIRST Coalition` (71.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.7975 | 30% | 0.2393 |
| **Base Score** | **0.7126** | - | **71.26%** |
| **Final Score** | **0.7130** | - | **71.30%** |

**Top 5 Non-Self Matches:**

2. music FIRST Coalition (71.3%)
3. CDFI Coalition (70.4%)
4. Musicfirst (69.6%)
5. Future of Music Coalition (69.5%)
6. Coalition of Music Stores (65.9%)

---

### 45. Frontier Power Products

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Frontier Business Products` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.7442 | 30% | 0.2232 |
| **Base Score** | **0.7901** | - | **79.01%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Frontier Business Products (90.0%)
3. Frontier Natural Products (90.0%)
4. Advanced Power Products (90.0%)
5. Worldwide Power Products (90.0%)
6. Power Service Products (90.0%)

---

### 46. 1960

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `District 1960` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.7453 | 30% | 0.2236 |
| **Base Score** | **0.7963** | - | **79.63%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. District 1960 (90.0%)
3. 60 (90.0%)
4. Playhouse 1960 (90.0%)
5. 1960 Family Practice (69.7%)
6. PAGE CLASS OF 1960 (68.9%)

---

### 47. Pacific Northwest Diabetes Research Inst

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Pacific Northwest Diabetes Research` (84.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7773 | 70% | 0.5441 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8441** | - | **84.41%** |
| **Final Score** | **0.8440** | - | **84.40%** |

**Top 5 Non-Self Matches:**

2. Pacific Northwest Diabetes Research (84.4%)
3. Diabetes Research (65.9%)
4. Diabetes Research Wellness Foundation (57.7%)
5. Junior Diabetes Research Foundation (57.4%)
6. Diabetes Research Institute Foundation (57.2%)

---

### 48. Mentors & Mentees

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `RE Mentors` (75.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.7865 | 30% | 0.2359 |
| **Base Score** | **0.7566** | - | **75.66%** |
| **Final Score** | **0.7570** | - | **75.70%** |

**Top 5 Non-Self Matches:**

2. RE Mentors (75.7%)
3. TRUE Mentors (75.6%)
4. Master Mentors (75.5%)
5. 3 Mentors (75.2%)
6. Mentors Inc. (66.7%)

---

### 49. NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06` (69.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5931 | 70% | 0.4151 |
| Semantic Similarity | 0.9302 | 30% | 0.2791 |
| **Base Score** | **0.6942** | - | **69.42%** |
| **Final Score** | **0.6940** | - | **69.40%** |

**Top 5 Non-Self Matches:**

2. NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06 (69.4%)
3. NaLA 2023 fall conference M01674661235470 01-25-23 10:40:54 (67.2%)
4. NALSC 2024 Annual Conference M01680804350265 04-06-23 14:05:56 (44.7%)
5. HBMA 2024 Fall Conference (40.6%)
6. Ascend Annual Conference 2023 M01689096187969 07-11-23 13:23:11 (39.3%)

---

### 50. Donnelley Work Session

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `SMDS Work Session` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.5326 | 30% | 0.1598 |
| **Base Score** | **0.7266** | - | **72.66%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. SMDS Work Session (90.0%)
3. DWS (75.0%)
4. Donnelley Financial Services (56.5%)
5. Experience Session 5 (55.6%)
6. Donnelley Financial Solutions (54.6%)

---

### 51. North Shore Senior Center

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Northshore Senior Center` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8905 | 70% | 0.6234 |
| Semantic Similarity | 0.8152 | 30% | 0.2446 |
| **Base Score** | **0.8679** | - | **86.79%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Northshore Senior Center (90.0%)
3. Coastal North Town Center (90.0%)
4. North Shore Elder Services (90.0%)
5. North Shore Cancer Center (90.0%)
6. North Shore Community College (90.0%)

---

### 52. Singles Who Like Food & Fun

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Fun Asian Singles` (57.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4675 | 70% | 0.3273 |
| Semantic Similarity | 0.8264 | 30% | 0.2479 |
| **Base Score** | **0.5752** | - | **57.52%** |
| **Final Score** | **0.5750** | - | **57.50%** |

**Top 5 Non-Self Matches:**

2. Fun Asian Singles (57.5%)
3. Christian Singles Fun Events (55.5%)
4. Fun Social Singles 35+ (55.3%)
5. Food Fun & Fellowship (53.0%)
6. Singles Who Dance (52.9%)

---

### 53. Zen Meetings & Events

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Zen Events México` (77.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 0.7814 | 30% | 0.2344 |
| **Base Score** | **0.7755** | - | **77.55%** |
| **Final Score** | **0.7750** | - | **77.50%** |

**Top 5 Non-Self Matches:**

2. Zen Events México (77.5%)
3. Zen Events Group (77.1%)
4. Zen Events, LLC (71.4%)
5. EVENT ZEN (64.6%)
6. Zen (63.4%)

---

### 54. Chicago South Swim Club

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `South Carolina Swim Club` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 0.8227 | 30% | 0.2468 |
| **Base Score** | **0.8243** | - | **82.43%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. South Carolina Swim Club (90.0%)
3. Baltimore City Swim Club (78.0%)
4. Detroit Recreation Swim Club (77.3%)
5. Ohio State Swim Club (77.0%)
6. Southern Kentucky Swim Club (76.1%)

---

### 55. Edna, Dabra@SAP.IO

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Edna Owusu` (75.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.7871 | 30% | 0.2361 |
| **Base Score** | **0.7567** | - | **75.67%** |
| **Final Score** | **0.7570** | - | **75.70%** |

**Top 5 Non-Self Matches:**

2. Edna Owusu (75.7%)
3. Edna Rose (75.0%)
4. Edna Travel (74.5%)
5. Edna ISD (74.1%)
6. City of Edna (73.9%)

---

### 56. Boys and Girls Club of Dawson Community Centre

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `GIRLS CLUB` (60.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5464 | 70% | 0.3825 |
| Semantic Similarity | 0.7287 | 30% | 0.2186 |
| **Base Score** | **0.6011** | - | **60.11%** |
| **Final Score** | **0.6010** | - | **60.10%** |

**Top 5 Non-Self Matches:**

2. GIRLS CLUB (60.1%)
3. Dawson Community College (53.7%)
4. DAWSON COMMUNITY BLUES (53.1%)
5. GIRLS Bridge Club (50.7%)
6. Dawson County Schools (38.8%)

---

### 57. Beissbarth

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Beissbarth GmbH` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.8375 | 30% | 0.2513 |
| **Base Score** | **0.8240** | - | **82.40%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Beissbarth GmbH (90.0%)
3. Breitbart (42.4%)
4. Bitbar (42.3%)
5. Brietbart (40.3%)
6. Ziebart (39.5%)

---

### 58. US Night Vision

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `PM Night Vision` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.8272 | 30% | 0.2482 |
| **Base Score** | **0.8150** | - | **81.50%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. PM Night Vision (90.0%)
3. US Vision Care (90.0%)
4. Night Vision Entertainment (90.0%)
5. Night Vision Manufacturers (90.0%)
6. WORLD VISION US (90.0%)

---

### 59. Amedysis, Incorporated

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Amedysis, Inc.` (95.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.8615 | 30% | 0.2584 |
| **Base Score** | **0.9584** | - | **95.84%** |
| **Final Score** | **0.9580** | - | **95.80%** |

**Top 5 Non-Self Matches:**

2. Amedysis, Inc. (95.8%)
3. Amedysis Home Health (79.0%)
4. Medysis (55.2%)
5. Avysis (46.0%)
6. Lysis (44.5%)

---

### 60. Taiyo Air Service Co.,Ltd

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Taiyo Air Services Co.` (80.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7244 | 70% | 0.5071 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8071** | - | **80.71%** |
| **Final Score** | **0.8070** | - | **80.70%** |

**Top 5 Non-Self Matches:**

2. Taiyo Air Services Co. (80.7%)
3. Fuyo Air Service Co. Ltd. (75.6%)
4. CITS Taikoo Air Service Ltd (72.8%)
5. TASC (69.0%)
6. Air Service (68.2%)

---

### 61. National Conference on Race & Ethnicity in American Higher E

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8421 | 70% | 0.5895 |
| Semantic Similarity | 0.9178 | 30% | 0.2753 |
| **Base Score** | **0.8648** | - | **86.48%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER (90.0%)
3. National Conference On Race & Ethnicity In America Higher Ed (90.0%)
4. National Conference on Race & Ethnicity in AM Higher Education (90.0%)
5. NCORE NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER EDUCATION (90.0%)
6. National Conference on Race & Ethnicity in Am. Higher Educ (90.0%)

---

### 62. Reminger Law Firm

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Reminger & Reminger Law Firm` (97.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.9011 | 30% | 0.2703 |
| **Base Score** | **0.9703** | - | **97.03%** |
| **Final Score** | **0.9700** | - | **97.00%** |

**Top 5 Non-Self Matches:**

2. Reminger & Reminger Law Firm (97.0%)
3. Withers Law Firm (90.0%)
4. S Law Firm (90.0%)
5. Speer Law Firm (90.0%)
6. Didier Law Firm (90.0%)

---

### 63. SEMMOA BOD

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `SEMMOA AACM` (74.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.7306 | 30% | 0.2192 |
| **Base Score** | **0.7398** | - | **73.98%** |
| **Final Score** | **0.7400** | - | **74.00%** |

**Top 5 Non-Self Matches:**

2. SEMMOA AACM (74.0%)
3. Bod Pro (73.5%)
4. SEMMOA Coop (73.2%)
5. CSA BOD (72.2%)
6. Semmoa (71.3%)

---

### 64. Telefonica Global Solutions

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Telefonica Multinational Solutions` (90.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 0.9424 | 30% | 0.2827 |
| **Base Score** | **0.9011** | - | **90.11%** |
| **Final Score** | **0.9010** | - | **90.10%** |

**Top 5 Non-Self Matches:**

2. Telefonica Multinational Solutions (90.1%)
3. Telefonica Global Solutions USA Inc. (90.0%)
4. TGS (75.0%)
5. TELEFONICA INTERNATIONAL USA (71.7%)
6. Telefonica (71.0%)

---

### 65. Travel Leaders - Dube Travel

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Dube Travel Leaders` (100.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **1.0000** | - | **100.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Non-Self Matches:**

2. Dube Travel Leaders (100.0%)
3. Travel Leaders Go (90.0%)
4. Travel Leaders Travel Quest (90.0%)
5. TRAVEL LEADERS INTERNATIONAL (90.0%)
6. Travel Leaders Travel Agency (90.0%)

---

### 66. Hi- Tours

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Hi Tours` (93.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.9300** | - | **93.00%** |
| **Final Score** | **0.9300** | - | **93.00%** |

**Top 5 Non-Self Matches:**

2. Hi Tours (93.0%)
3. Hi-Tours (90.0%)
4. Hi Life Tours (90.0%)
5. HI LITE TOURS (90.0%)
6. Hi Tour (81.7%)

---

### 67. Volkswagen Group China

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Volkswagen China` (99.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.9681 | 30% | 0.2904 |
| **Base Score** | **0.9904** | - | **99.04%** |
| **Final Score** | **0.9900** | - | **99.00%** |

**Top 5 Non-Self Matches:**

2. Volkswagen China (99.0%)
3. VOLKSWAGEN CHINA INVESTMENT COMPANY LTD (90.0%)
4. Volkswagen Group Japan (76.6%)
5. Volkswagen Group Australia (75.7%)
6. VOLKSWAGEN KOREA (75.2%)

---

### 68. Sun Tx

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Standard Sun` (73.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.7216 | 30% | 0.2165 |
| **Base Score** | **0.7371** | - | **73.71%** |
| **Final Score** | **0.7370** | - | **73.70%** |

**Top 5 Non-Self Matches:**

2. Standard Sun (73.7%)
3. Sun City (73.5%)
4. Sun Am (73.4%)
5. Sun Outdoors (73.4%)
6. sun coast (73.3%)

---

### 69. Southern Vermont Deerfield Valley Chamber of commerce

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Deerfield Beach Chamber of Commerce` (67.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6198 | 70% | 0.4339 |
| Semantic Similarity | 0.7934 | 30% | 0.2380 |
| **Base Score** | **0.6719** | - | **67.19%** |
| **Final Score** | **0.6720** | - | **67.20%** |

**Top 5 Non-Self Matches:**

2. Deerfield Beach Chamber of Commerce (67.2%)
3. Chamber of Commerce Mid-Ohio Valley (65.4%)
4. Deerfield Chamber of Commerce (64.3%)
5. Vermont Chamber of Commerce (62.8%)
6. Northwest Valley Chamber of Commerce (62.7%)

---

### 70. DGR Ministries

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `DG Ministries` (76.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8084 | 30% | 0.2425 |
| **Base Score** | **0.7631** | - | **76.31%** |
| **Final Score** | **0.7630** | - | **76.30%** |

**Top 5 Non-Self Matches:**

2. DG Ministries (76.3%)
3. Power Ministries (74.5%)
4. Impact Ministries (73.5%)
5. Progressive Ministries (73.2%)
6. Empowered Ministries (73.1%)

---

### 71. Impacto 6

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Impacto 52` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 0.8093 | 30% | 0.2428 |
| **Base Score** | **0.8378** | - | **83.78%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Impacto 52 (90.0%)
3. Impacto Strategies (90.0%)
4. Impacto Vital (90.0%)
5. Triple Impacto (90.0%)
6. Impacto Tactico (90.0%)

---

### 72. Neos Therapeutics, Inc.

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Neos Therapeutics` (93.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 0.9836 | 30% | 0.2951 |
| **Base Score** | **0.9368** | - | **93.68%** |
| **Final Score** | **0.9370** | - | **93.70%** |

**Top 5 Non-Self Matches:**

2. Neos Therapeutics (93.7%)
3. Neos Therapeutics LP (79.9%)
4. Neogene Therapeutics, Inc. (76.9%)
5. Neogene Therapeutics (74.0%)
6. Neos Partners (73.1%)

---

### 73. International Tax Institute

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `INTERNATIONAL TAX INSTITUTE INC` (97.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.9239 | 30% | 0.2772 |
| **Base Score** | **0.9772** | - | **97.72%** |
| **Final Score** | **0.9770** | - | **97.70%** |

**Top 5 Non-Self Matches:**

2. INTERNATIONAL TAX INSTITUTE INC (97.7%)
3. National Tax Institute (90.4%)
4. International Property Tax Institute (90.0%)
5. Professional Tax Institute (90.0%)
6. Tax Research Institute (90.0%)

---

### 74. Mitsubishi M501G

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `NEC Mitsubishi` (78.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8674 | 30% | 0.2602 |
| **Base Score** | **0.7809** | - | **78.09%** |
| **Final Score** | **0.7810** | - | **78.10%** |

**Top 5 Non-Self Matches:**

2. NEC Mitsubishi (78.1%)
3. Mitsubishi Power (77.9%)
4. Mitsubishi Motors (77.9%)
5. MITSUBISHI MOTOR (77.8%)
6. Mitsubishi Electronics (77.7%)

---

### 75. Huskies Sports

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Huskies Basketball` (77.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8346 | 30% | 0.2504 |
| **Base Score** | **0.7710** | - | **77.10%** |
| **Final Score** | **0.7710** | - | **77.10%** |

**Top 5 Non-Self Matches:**

2. Huskies Basketball (77.1%)
3. Miami Huskies (75.9%)
4. Mass Huskies (75.6%)
5. NH Huskies (73.7%)
6. Howard Huskies (72.7%)

---

### 76. Acacia Pharma Group Inc.

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `ACACIA PHARMA, Inc.` (94.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.8045 | 30% | 0.2413 |
| **Base Score** | **0.9413** | - | **94.13%** |
| **Final Score** | **0.9410** | - | **94.10%** |

**Top 5 Non-Self Matches:**

2. ACACIA PHARMA, Inc. (94.1%)
3. Acacia Pharma (92.2%)
4. Acacia Pharma Ltd (91.7%)
5. Acacia Research Group (77.4%)
6. Acacia Network Inc. (77.2%)

---

### 77. Acumatica Summit 2017 Z7NWPDKS625

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Acumatica Asia` (54.9%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4167 | 70% | 0.2917 |
| Semantic Similarity | 0.8586 | 30% | 0.2576 |
| **Base Score** | **0.5492** | - | **54.92%** |
| **Final Score** | **0.5490** | - | **54.90%** |

**Top 5 Non-Self Matches:**

2. Acumatica Asia (54.9%)
3. Acumatica User Group Southeast (54.0%)
4. Contact Center Compliance Summit (51.7%)
5. ACUMATICA PRESIDENTS CLUB (50.8%)
6. Acumatica - Sales Office (50.7%)

---

### 78. Linklaters CIS

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `CIS Partners` (76.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8052 | 30% | 0.2415 |
| **Base Score** | **0.7622** | - | **76.22%** |
| **Final Score** | **0.7620** | - | **76.20%** |

**Top 5 Non-Self Matches:**

2. CIS Partners (76.2%)
3. Cis GmbH (75.8%)
4. One Cis (75.7%)
5. CIS Method (75.6%)
6. Cis 22 (75.2%)

---

### 79. Christian Girls Family Ministry

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Christian Girls Family Ministry Training` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.8461 | 30% | 0.2538 |
| **Base Score** | **0.8266** | - | **82.66%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Christian Girls Family Ministry Training (90.0%)
3. Christian Family Fellowship Church (74.5%)
4. FRIENDSHIP CHRISTIAN CHURCH MINISTRY (74.4%)
5. Community Outreach Christian Ministry (73.8%)
6. Christian Growth Family Ministries (73.3%)

---

### 80. Alosa Foundation

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Formosa Foundation` (70.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6375 | 70% | 0.4462 |
| Semantic Similarity | 0.8631 | 30% | 0.2589 |
| **Base Score** | **0.7052** | - | **70.52%** |
| **Final Score** | **0.7050** | - | **70.50%** |

**Top 5 Non-Self Matches:**

2. Formosa Foundation (70.5%)
3. CL Foundation (66.4%)
4. CAP Foundation (66.1%)
5. FIRST FOUNDATION (66.0%)
6. UH FOUNDATION (65.8%)

---

### 81. La Chaine des Rotisseurs Wine Club of Newport Beach

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `La Chaine des Rotisseurs Wine Club of Ne` (80.9%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7597 | 70% | 0.5318 |
| Semantic Similarity | 0.9248 | 30% | 0.2775 |
| **Base Score** | **0.8093** | - | **80.93%** |
| **Final Score** | **0.8090** | - | **80.90%** |

**Top 5 Non-Self Matches:**

2. La Chaine des Rotisseurs Wine Club of Ne (80.9%)
3. Newport Beach Wine Festival (53.4%)
4. Southern Trace Wine Club (49.6%)
5. Cleveland Wine Club (43.5%)
6. Diversity Wine Club (42.9%)

---

### 82. Sumner & Ryan, LLC

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Miller Ryan LLC` (77.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8366 | 30% | 0.2510 |
| **Base Score** | **0.7716** | - | **77.16%** |
| **Final Score** | **0.7720** | - | **77.20%** |

**Top 5 Non-Self Matches:**

2. Miller Ryan LLC (77.2%)
3. Ryan Moving LLC (77.1%)
4. Ryan Companies (76.4%)
5. Sumner 360 (75.1%)
6. Ryan Sellers (75.0%)

---

### 83. Tilt Creative & Production

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Tilt Creative + Production` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8727 | 70% | 0.6109 |
| Semantic Similarity | 0.9294 | 30% | 0.2788 |
| **Base Score** | **0.8897** | - | **88.97%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Tilt Creative + Production (90.0%)
3. Creative Production Design (90.0%)
4. Creative Production Incentives (90.0%)
5. Tilt Production (79.3%)
6. Full Tilt Marketing (57.5%)

---

### 84. Cerberus Capital

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Cerberus Capital Management` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.9071 | 30% | 0.2721 |
| **Base Score** | **0.8449** | - | **84.49%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Cerberus Capital Management (90.0%)
3. *Cerberus Capital (82.5%)
4. Cerberus Capital Management L (77.5%)
5. Cerberus Capital Management LP (76.9%)
6. Cerberus Law (74.5%)

---

### 85. Institute of Health Technology Transformation

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Institute for Health Technology Transformation` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8678 | 70% | 0.6074 |
| Semantic Similarity | 0.9668 | 30% | 0.2900 |
| **Base Score** | **0.8975** | - | **89.75%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Institute for Health Technology Transformation (90.0%)
3. INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION (90.0%)
4. Health Science Technology Education (77.3%)
5. Health Technology (75.9%)
6. Achieve Health Care Technology (75.7%)

---

### 86. The Jones Assembly

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `The Jones Assembly Presents` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.8252 | 30% | 0.2475 |
| **Base Score** | **0.8203** | - | **82.03%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. The Jones Assembly Presents (90.0%)
3. General Assembly (74.4%)
4. 1st Assembly (74.3%)
5. Bill Jones (74.1%)
6. Jones Power (74.0%)

---

### 87. American Black Film Insitutute

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `American Black Film Festival` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 0.7505 | 30% | 0.2252 |
| **Base Score** | **0.8158** | - | **81.58%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. American Black Film Festival (90.0%)
3. American Black Film Festival Ventures (74.8%)
4. Black Women Film Preservation (74.0%)
5. HOLLYWOOD BLACK FILM FESTIVAL (72.6%)
6. International Black Film Festival (72.3%)

---

### 88. Berk Tek

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Berk-Tek` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.8789 | 30% | 0.2637 |
| **Base Score** | **0.8937** | - | **89.37%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Berk-Tek (90.0%)
3. Berk Tck (74.6%)
4. Berk Technologies (73.3%)
5. TEK Source (73.2%)
6. Berk Tek / Leviton (72.3%)

---

### 89. Northbridge Travel

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Northbridge Communities` (77.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8388 | 30% | 0.2517 |
| **Base Score** | **0.7723** | - | **77.23%** |
| **Final Score** | **0.7720** | - | **77.20%** |

**Top 5 Non-Self Matches:**

2. Northbridge Communities (77.2%)
3. Bridge Travel (77.2%)
4. WestBridge Travel (77.1%)
5. Northbridge Environmental (76.8%)
6. Skybridge Travel (76.1%)

---

### 90. Kohler 2024

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Destination Kohler` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 0.7313 | 30% | 0.2194 |
| **Base Score** | **0.8144** | - | **81.44%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Destination Kohler (90.0%)
3. Kohler Distributing (90.0%)
4. Kohler Fixtures (90.0%)
5. Kohler Generators (90.0%)
6. Kohler Recreation (90.0%)

---

### 91. Louisiana State University Swim

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Louisiana State Univ Swim` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.8455 | 30% | 0.2537 |
| **Base Score** | **0.8837** | - | **88.37%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Louisiana State Univ Swim (90.0%)
3. LOUISIANA STATE UNIVERSITY ATHLETICS (90.0%)
4. Louisiana State University USA (90.0%)
5. Louisiana State University Health (90.0%)
6. Louisiana State University Football (90.0%)

---

### 92. X DO NOT USE - FRANCIS PARKER SCHOOL

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Francis Parker School` (68.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5464 | 70% | 0.3825 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6825** | - | **68.25%** |
| **Final Score** | **0.6820** | - | **68.20%** |

**Top 5 Non-Self Matches:**

2. Francis Parker School (68.2%)
3. Francis Parker (60.1%)
4. Francis Parker High School (58.0%)
5. Francis Parker School English Department (57.6%)
6. Francis Parker School of San Diego (57.0%)

---

### 93. Mitsubishi Motor Sales of America, Incorporated

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Mitsubishi Motor Sales of America` (97.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 0.9248 | 30% | 0.2774 |
| **Base Score** | **0.9774** | - | **97.74%** |
| **Final Score** | **0.9770** | - | **97.70%** |

**Top 5 Non-Self Matches:**

2. Mitsubishi Motor Sales of America (97.7%)
3. Mitsubishi Electronic Sales America (90.0%)
4. Mitsubishi Motors Sales of America (90.0%)
5. Mitsubishi Motor Sales of Caribbean (90.0%)
6. Mitsubishi Electric Sales of America (90.0%)

---

### 94. Energy Distribution Partners Holdings'

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Energy Distribution Partners Holdings L.P.` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 0.7515 | 30% | 0.2255 |
| **Base Score** | **0.8030** | - | **80.30%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Energy Distribution Partners Holdings L.P. (90.0%)
3. Energy Distribution Partners (82.5%)
4. Energy Distribution Holdings (76.8%)
5. Energy Products Distribution (73.4%)
6. Distribution Energy Financial Group (73.4%)

---

### 95. ThinkAdvisor

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Planadvisor` (46.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.3130 | 70% | 0.2191 |
| Semantic Similarity | 0.8190 | 30% | 0.2457 |
| **Base Score** | **0.4648** | - | **46.48%** |
| **Final Score** | **0.4650** | - | **46.50%** |

**Top 5 Non-Self Matches:**

2. Planadvisor (46.5%)
3. TRIPADVISOR (45.7%)
4. NeXtAdvisors (45.3%)
5. Invisors (43.3%)
6. TripAdvisor LLC (43.1%)

---

### 96. Jump on it Outreach

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Outreach` (61.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4500 | 70% | 0.3150 |
| Semantic Similarity | 0.9830 | 30% | 0.2949 |
| **Base Score** | **0.6099** | - | **60.99%** |
| **Final Score** | **0.6100** | - | **61.00%** |

**Top 5 Non-Self Matches:**

2. Outreach (61.0%)
3. Air Force Outreach Program (60.7%)
4. Above N Beyond Outreach (60.7%)
5. OutReach Program Development (57.9%)
6. Community Outreach Program (57.4%)

---

### 97. The Association of Ringside Consultants (ARC)

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Association of Ringside Physicians` (70.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.7827 | 30% | 0.2348 |
| **Base Score** | **0.7081** | - | **70.81%** |
| **Final Score** | **0.7080** | - | **70.80%** |

**Top 5 Non-Self Matches:**

2. Association of Ringside Physicians (70.8%)
3. Arc-Consultants (59.5%)
4. Ringside (57.6%)
5. ARC CONSULTANTS,  INC. (55.4%)
6. Ringside, Incorporated (55.4%)

---

### 98. SFA HASA

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `SFA Companies` (77.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8513 | 30% | 0.2554 |
| **Base Score** | **0.7760** | - | **77.60%** |
| **Final Score** | **0.7760** | - | **77.60%** |

**Top 5 Non-Self Matches:**

2. SFA Companies (77.6%)
3. SFA Leads (77.3%)
4. Sfa Charter (76.4%)
5. SFA Partners (76.3%)
6. SFA Opportunity (75.2%)

---

### 99. Grupo Duracell Ene 2025

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Duracell Brasil` (54.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4167 | 70% | 0.2917 |
| Semantic Similarity | 0.8303 | 30% | 0.2491 |
| **Base Score** | **0.5408** | - | **54.08%** |
| **Final Score** | **0.5410** | - | **54.10%** |

**Top 5 Non-Self Matches:**

2. Duracell Brasil (54.1%)
3. Grupo GT5 Brasil (51.8%)
4. Duracell Research Development (51.3%)
5. Grupo Brasil DPE (51.3%)
6. Duracell Chile (51.0%)

---

### 100. World Association of Medical Law

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `World Association for Medical Law` (90.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.9045** | - | **90.45%** |
| **Final Score** | **0.9050** | - | **90.50%** |

**Top 5 Non-Self Matches:**

2. World Association for Medical Law (90.5%)
3. World Medical Association (80.0%)
4. International Law Association (71.8%)
5. Pacific Medical Law (71.7%)
6. California Medical Legal Association (69.8%)

---

### 101. ABA

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `AMER BRIDGE ASSN` (92.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 0.2767 | 30% | 0.0830 |
| **Base Score** | **0.4330** | - | **43.30%** |
| Acronym Fidelity Boost | 1.0000 | 15% max | +0.1500 |
| **Final Score** | **0.9280** | - | **92.80%** |

**Top 5 Non-Self Matches:**

2. AMER BRIDGE ASSN (92.8%)
3. Am Bridge Assn (92.7%)
4. Arrowood Business Association (92.4%)
5. AZ Business Assn (92.3%)
6. ACL Business Assurance (92.1%)

---

### 102. PDMA

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `PDMA inc` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 0.8667 | 30% | 0.2600 |
| **Base Score** | **0.8900** | - | **89.00%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. PDMA inc (90.0%)
3. PDMA Association (90.0%)
4. PDMA Corporation (90.0%)
5. PDMA Alliance (90.0%)
6. Prescription Drug Marketing Act (85.9%)

---

### 103. IBM

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `International Business Machines` (94.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 0.4402 | 30% | 0.1321 |
| **Base Score** | **0.4821** | - | **48.21%** |
| Acronym Fidelity Boost | 1.0000 | 15% max | +0.1500 |
| **Final Score** | **0.9440** | - | **94.40%** |

**Top 5 Non-Self Matches:**

2. International Business Machines (94.4%)
3. International Business Machine (94.2%)
4. Intel Board Meeting (93.7%)
5. intechRx Business Meeting (93.0%)
6. International Boiler Makers (92.7%)

---

### 104. GE

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `GMG Education` (93.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 0.3305 | 30% | 0.0992 |
| **Base Score** | **0.4492** | - | **44.92%** |
| Acronym Fidelity Boost | 1.0000 | 15% max | +0.1500 |
| **Final Score** | **0.9330** | - | **93.30%** |

**Top 5 Non-Self Matches:**

2. GMG Education (93.3%)
3. Gaia Experience (93.2%)
4. GAD e.G. (93.1%)
5. Gould Evans (93.1%)
6. GU Energy (93.1%)

---

