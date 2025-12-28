# Company Matching Control Set Report

**Generated:** 2025-12-28 05:46:51

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

✅ **Exact Match:** `PDMA Association` (100.0%)

---

### 2. Nicolas/Sanchez Wedding

✅ **Exact Match:** `Nicolas/Sanchez Wedding` (100.0%)

---

### 3. Kehilat Ariel Synagogue

✅ **Exact Match:** `Kehilat Ariel Synagogue` (100.0%)

---

### 4. Next Level Events

✅ **Exact Match:** `Next Level Events` (100.0%)

---

### 5. Site Foundation Golf Tournament

✅ **Exact Match:** `Site Foundation Golf Tournament` (100.0%)

---

### 6. Interim WG Meeting - BIER

✅ **Exact Match:** `Interim WG Meeting - BIER` (100.0%)

---

### 7. DermaQuest Inc

**Top Match:** `DI` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. DI (100.0%)
2. DermaQuest Inc (100.0%)
3. Dermaquest, Incorporated (94.2%)
4. Dermaquest Skin Care (85.0%)
5. Dermaquest Skin Therapy (83.6%)

---

### 8. Ellwood Group Inc

**Top Match:** `EGI` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. EGI (100.0%)
2. Ellwood Group Inc (100.0%)
3. Ellwood Associates (86.2%)
4. Ellwood Community Church (84.6%)
5. Delwood (48.5%)

---

### 9. American Miniature Horse Registry

✅ **Exact Match:** `American Miniature Horse Registry` (100.0%)

---

### 10. YADA ENTERPRISES, INC

✅ **Exact Match:** `YADA ENTERPRISES, INC` (100.0%)

---

### 11. Seafood Nutrition Partnership

✅ **Exact Match:** `Seafood Nutrition Partnership` (100.0%)

---

### 12. AVIAKOMPANIYA SIBIR, PAO

✅ **Exact Match:** `AVIAKOMPANIYA SIBIR, PAO` (100.0%)

---

### 13. Hartford Hospital School of Nursing

✅ **Exact Match:** `Hartford Hospital School of Nursing` (100.0%)

---

### 14. Internal J&J Meeting and Breakfast

✅ **Exact Match:** `Internal J&J Meeting and Breakfast` (100.0%)

---

### 15. Spina Bifida Coalition of Cincinnati

**Top Match:** `SBCC` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. SBCC (100.0%)
2. Spina Bifida Coalition of Cincinnati (100.0%)
3. Spina Bifida Association of Cincinnati, Inc. (81.7%)
4. Spina Bifida Association of Michigan (75.0%)
5. Illinois Spina Bifida Association (74.7%)

---

### 16. THE SOCA GROUP ORGANIZATION

✅ **Exact Match:** `THE SOCA GROUP ORGANIZATION` (100.0%)

---

### 17. Shiroyama Junior High School

✅ **Exact Match:** `Shiroyama Junior High School` (100.0%)

---

### 18. National Home Health

✅ **Exact Match:** `National Home Health` (100.0%)

---

### 19. American News Women's Club

✅ **Exact Match:** `American News Women's Club` (100.0%)

---

### 20. Denise Roberge

✅ **Exact Match:** `Denise Roberge` (100.0%)

---

### 21. Synergy Soccer Club

✅ **Exact Match:** `Synergy Soccer Club` (100.0%)

---

### 22. NFC Forum

✅ **Exact Match:** `NFC Forum` (100.0%)

---

### 23. A Better Choice Limousine & Concierge

✅ **Exact Match:** `A Better Choice Limousine & Concierge` (100.0%)

---

### 24. Danish Sisterhood of America

✅ **Exact Match:** `Danish Sisterhood of America` (100.0%)

---

### 25. Brooklyn Comics Club

✅ **Exact Match:** `Brooklyn Comics Club` (100.0%)

---

### 26. Global Interagency Security Forum

**Top Match:** `GISF` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. GISF (100.0%)
2. Global Interagency Security Forum (100.0%)
3. Cyber Security Collaboration Forum (76.7%)
4. Infrastructure Security and Resilience Forum (76.4%)
5. The Cyber Security Forum Initiative (76.3%)

---

### 27. Lancet Software

✅ **Exact Match:** `Lancet Software` (100.0%)

---

### 28. Our Lady of the Lakes Catholic Church and School

✅ **Exact Match:** `Our Lady of the Lakes Catholic Church and School` (100.0%)

---

### 29. Broadway Bound International

**Top Match:** `BBI` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. BBI (100.0%)
2. Broadway Bound International (100.0%)
3. Broadway Bound West (87.2%)
4. Broadway Bound Kids (85.8%)
5. Bound Four Broadway (85.7%)

---

### 30. E. H. Wachs

**Top Match:** `EHW` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. EHW (100.0%)
2. E. H. Wachs (100.0%)
3. E.H. Wachs (61.3%)
4. Wachs Water Services (58.2%)
5. Wachs / Russell Wedding (57.4%)

---

### 31. Marine Corps Fox 2/5

✅ **Exact Match:** `Marine Corps Fox 2/5` (100.0%)

---

### 32. Fantasia Turistica

✅ **Exact Match:** `Fantasia Turistica` (100.0%)

---

### 33. Esoterix

✅ **Exact Match:** `Esoterix` (100.0%)

---

### 34. Coker Group

**Top Match:** `CG` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. CG (100.0%)
2. Coker Group (100.0%)
3. The Coker Group (100.0%)
4. Coker Grp (98.5%)
5. Coker Cheerleading Group (97.5%)

---

### 35. GILEAD IT

✅ **Exact Match:** `GILEAD IT` (100.0%)

---

### 36. 4143 Affiliate INDA 2016

✅ **Exact Match:** `4143 Affiliate INDA 2016` (100.0%)

---

### 37. Pipe and Plant Solutions

✅ **Exact Match:** `Pipe and Plant Solutions` (100.0%)

---

### 38. Stephen Rourke

✅ **Exact Match:** `Stephen Rourke` (100.0%)

---

### 39. MIT Initiative on the Digital Economy

✅ **Exact Match:** `MIT Initiative on the Digital Economy` (100.0%)

---

### 40. Urx Community USA

✅ **Exact Match:** `Urx Community USA` (100.0%)

---

### 41. Spredfast Engage

✅ **Exact Match:** `Spredfast Engage` (100.0%)

---

### 42. City of Dallas-Parks & Recreation

✅ **Exact Match:** `City of Dallas-Parks & Recreation` (100.0%)

---

### 43. Kai Pono Builders, Inc.

✅ **Exact Match:** `Kai Pono Builders, Inc.` (100.0%)

---

### 44. MUSICFIRST COALITION

✅ **Exact Match:** `MUSICFIRST COALITION` (100.0%)

---

### 45. Frontier Power Products

**Top Match:** `FPP` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. FPP (100.0%)
2. Frontier Power Products (100.0%)
3. Frontier Business Products (79.0%)
4. Frontier Natural Products (76.7%)
5. Advanced Power Products (76.4%)

---

### 46. 1960

✅ **Exact Match:** `1960` (100.0%)

---

### 47. Pacific Northwest Diabetes Research Inst

✅ **Exact Match:** `Pacific Northwest Diabetes Research Inst` (100.0%)

---

### 48. Mentors & Mentees

✅ **Exact Match:** `Mentors & Mentees` (100.0%)

---

### 49. NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46

✅ **Exact Match:** `NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46` (100.0%)

---

### 50. Donnelley Work Session

**Top Match:** `DWS` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. DWS (100.0%)
2. Donnelley Work Session (100.0%)
3. SMDS Work Session (72.7%)
4. Donnelley Financial Services (56.5%)
5. Experience Session 5 (55.6%)

---

### 51. North Shore Senior Center

**Top Match:** `NSSC` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. NSSC (100.0%)
2. North Shore Senior Center (100.0%)
3. North Shore Cancer Center (84.8%)
4. Northshore Senior Center (84.5%)
5. Coastal North Town Center (82.3%)

---

### 52. Singles Who Like Food & Fun

✅ **Exact Match:** `Singles Who Like Food & Fun` (100.0%)

---

### 53. Zen Meetings & Events

✅ **Exact Match:** `Zen Meetings & Events` (100.0%)

---

### 54. Chicago South Swim Club

✅ **Exact Match:** `Chicago South Swim Club` (100.0%)

---

### 55. Edna, Dabra@SAP.IO

✅ **Exact Match:** `Edna, Dabra@SAP.IO` (100.0%)

---

### 56. Boys and Girls Club of Dawson Community Centre

✅ **Exact Match:** `Boys and Girls Club of Dawson Community Centre` (100.0%)

---

### 57. Beissbarth

✅ **Exact Match:** `Beissbarth` (100.0%)

---

### 58. US Night Vision

✅ **Exact Match:** `US Night Vision` (100.0%)

---

### 59. Amedysis, Incorporated

✅ **Exact Match:** `Amedysis, Incorporated` (100.0%)

---

### 60. Taiyo Air Service Co.,Ltd

✅ **Exact Match:** `Taiyo Air Service Co.,Ltd` (100.0%)

---

### 61. National Conference on Race & Ethnicity in American Higher E

✅ **Exact Match:** `National Conference on Race & Ethnicity in American Higher E` (100.0%)

---

### 62. Reminger Law Firm

**Top Match:** `RLF` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. RLF (100.0%)
2. Reminger Law Firm (100.0%)
3. Reminger & Reminger Law Firm (97.0%)
4. Withers Law Firm (82.3%)
5. S Law Firm (81.9%)

---

### 63. SEMMOA BOD

✅ **Exact Match:** `SEMMOA BOD` (100.0%)

---

### 64. Telefonica Global Solutions

**Top Match:** `TGS` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. TGS (100.0%)
2. Telefonica Global Solutions (100.0%)
3. Telefonica Multinational Solutions (90.1%)
4. Telefonica Global Solutions USA Inc. (90.1%)
5. TELEFONICA INTERNATIONAL USA (71.7%)

---

### 65. Travel Leaders - Dube Travel

**Top Match:** `Dube Travel Leaders` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. Dube Travel Leaders (100.0%)
2. Travel Leaders - Dube Travel (100.0%)
3. Dube Travel / Travel Leaders (93.4%)
4. Dube / Travel Leaders (93.0%)
5. Dube Travel/Travel Leaders (88.7%)

---

### 66. Hi- Tours

**Top Match:** `HT` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. HT (100.0%)
2. Hi Tours (100.0%)
3. Hi-Tours (100.0%)
4. Hi- Tours (100.0%)
5. Hi Tour (96.7%)

---

### 67. Volkswagen Group China

✅ **Exact Match:** `Volkswagen Group China` (100.0%)

---

### 68. Sun Tx

✅ **Exact Match:** `Sun Tx` (100.0%)

---

### 69. Southern Vermont Deerfield Valley Chamber of commerce

✅ **Exact Match:** `Southern Vermont Deerfield Valley Chamber of commerce` (100.0%)

---

### 70. DGR Ministries

**Top Match:** `DM` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. DM (100.0%)
2. DGR Ministries (100.0%)
3. DG Ministries (76.3%)
4. Power Ministries (74.5%)
5. Impact Ministries (73.5%)

---

### 71. Impacto 6

✅ **Exact Match:** `Impacto 6` (100.0%)

---

### 72. Neos Therapeutics, Inc.

✅ **Exact Match:** `Neos Therapeutics, Inc.` (100.0%)

---

### 73. International Tax Institute

✅ **Exact Match:** `International Tax Institute` (100.0%)

---

### 74. Mitsubishi M501G

✅ **Exact Match:** `Mitsubishi M501G` (100.0%)

---

### 75. Huskies Sports

✅ **Exact Match:** `Huskies Sports` (100.0%)

---

### 76. Acacia Pharma Group Inc.

✅ **Exact Match:** `Acacia Pharma Group Inc.` (100.0%)

---

### 77. Acumatica Summit 2017 Z7NWPDKS625

✅ **Exact Match:** `Acumatica Summit 2017 Z7NWPDKS625` (100.0%)

---

### 78. Linklaters CIS

**Top Match:** `LC` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. LC (100.0%)
2. Linklaters CIS (100.0%)
3. CIS Partners (76.2%)
4. Cis GmbH (75.8%)
5. One Cis (75.7%)

---

### 79. Christian Girls Family Ministry

✅ **Exact Match:** `Christian Girls Family Ministry` (100.0%)

---

### 80. Alosa Foundation

**Top Match:** `AF` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. AF (100.0%)
2. Alosa Foundation (100.0%)
3. Formosa Foundation (70.5%)
4. CL Foundation (66.4%)
5. CAP Foundation (66.1%)

---

### 81. La Chaine des Rotisseurs Wine Club of Newport Beach

✅ **Exact Match:** `La Chaine des Rotisseurs Wine Club of Newport Beach` (100.0%)

---

### 82. Sumner & Ryan, LLC

✅ **Exact Match:** `Sumner & Ryan, LLC` (100.0%)

---

### 83. Tilt Creative & Production

✅ **Exact Match:** `Tilt Creative & Production` (100.0%)

---

### 84. Cerberus Capital

✅ **Exact Match:** `Cerberus Capital` (100.0%)

---

### 85. Institute of Health Technology Transformation

✅ **Exact Match:** `Institute of Health Technology Transformation` (100.0%)

---

### 86. The Jones Assembly

**Top Match:** `JA` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. JA (100.0%)
2. The Jones Assembly (100.0%)
3. The Jones Assembly Presents (87.8%)
4. General Assembly (74.4%)
5. 1st Assembly (74.3%)

---

### 87. American Black Film Insitutute

✅ **Exact Match:** `American Black Film Insitutute` (100.0%)

---

### 88. Berk Tek

**Top Match:** `BT` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. BT (100.0%)
2. Berk Tek (100.0%)
3. Berk-Tek (100.0%)
4. Berk Tek / Leviton (97.0%)
5. Berk Tck (89.6%)

---

### 89. Northbridge Travel

✅ **Exact Match:** `Northbridge Travel` (100.0%)

---

### 90. Kohler 2024

**Top Match:** `K2` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. K2 (100.0%)
2. Kohler 2024 (100.0%)
3. Destination Kohler (81.4%)
4. Kohler Distributing (81.4%)
5. Kohler Fixtures (80.4%)

---

### 91. Louisiana State University Swim

✅ **Exact Match:** `Louisiana State University Swim` (100.0%)

---

### 92. X DO NOT USE - FRANCIS PARKER SCHOOL

✅ **Exact Match:** `X DO NOT USE - FRANCIS PARKER SCHOOL` (100.0%)

---

### 93. Mitsubishi Motor Sales of America, Incorporated

✅ **Exact Match:** `Mitsubishi Motor Sales of America, Incorporated` (100.0%)

---

### 94. Energy Distribution Partners Holdings'

✅ **Exact Match:** `Energy Distribution Partners Holdings'` (100.0%)

---

### 95. ThinkAdvisor

✅ **Exact Match:** `ThinkAdvisor` (100.0%)

---

### 96. Jump on it Outreach

✅ **Exact Match:** `Jump on it Outreach` (100.0%)

---

### 97. The Association of Ringside Consultants (ARC)

✅ **Exact Match:** `The Association of Ringside Consultants (ARC)` (100.0%)

---

### 98. SFA HASA

**Top Match:** `SH` (100.0%)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.0000 | 70% | 0.0000 |
| Semantic Similarity | 0.0000 | 30% | 0.0000 |
| **Base Score** | **0.0000** | - | **0.00%** |
| **Final Score** | **1.0000** | - | **100.00%** |

**Top 5 Matches:**

1. SH (100.0%)
2. SFA HASA (100.0%)
3. SFA Companies (77.6%)
4. SFA Leads (77.3%)
5. Sfa Charter (76.4%)

---

### 99. Grupo Duracell Ene 2025

✅ **Exact Match:** `Grupo Duracell Ene 2025` (100.0%)

---

### 100. World Association of Medical Law

✅ **Exact Match:** `World Association of Medical Law` (100.0%)

---

### 101. ABA

✅ **Exact Match:** `ABA` (100.0%)

---

### 102. PDMA

✅ **Exact Match:** `PDMA` (100.0%)

---

### 103. IBM

✅ **Exact Match:** `IBM` (100.0%)

---

### 104. GE

✅ **Exact Match:** `GE` (100.0%)

---

