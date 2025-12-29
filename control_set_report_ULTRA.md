# Company Matching Control Set Report (Location-Aware)

**Generated:** 2025-12-29 17:20:38

---

## Matching Scenarios Handled

This system is designed to handle the following real-world company matching challenges:

### 1. **Exact Matches**
Perfect character-for-character matching.
- `"IBM"` → `"IBM"` (100%)
- `"Microsoft"` → `"Microsoft"` (100%)
- `"Apple Inc."` → `"Apple Inc."` (100%)
- `"Google"` → `"Google"` (100%)
- `"Amazon.com"` → `"Amazon.com"` (100%)

### 2. **Acronym Expansions**
Matching acronyms to their full company names or vice-versa.
- `"IBM"` → `"International Business Machines"` (Strong expansion)
- `"AWS"` → `"Amazon Web Services"` (Strong expansion)
- `"GE"` → `"General Electric"` (Strong expansion)
- `"AT&T"` → `"American Telephone and Telegraph"` (Strong expansion)
- `"FedEx"` → `"Federal Express"` (Strong expansion)

### 3. **Location-Aware Matching (NEW)**
Using city/state context to resolve ambiguity between identical or similar names.
- `"Acme"` (Chicago) → `"Acme Corp"` (Chicago, IL) vs (Miami, FL)
- `"Northwestern"` (Evanston) → `"Northwestern University"` (Evanston, IL) vs `"Northwestern Mutual"` (Milwaukee, WI)
- `"Pizza Hut"` (London, KY) → `"Pizza Hut"` (London, KY) vs `"Pizza Hut"` (London, UK)
- `"Springfield Power"` (Springfield, IL) → Resolved to Illinois entity over Massachusetts
- `"Regency Hotel"` (Paris, TX) → Resolved to Texas entity over France or Nevada

### 4. **Popularity/Frequency Bias (NEW)**
Using occurrence counts to break ties, prioritizing major entities over obscure ones.
- `"McDonalds"` → Global chain (5,000+ records) vs `"McDonalds Hardware"` (1 record)
- `"Starbucks"` → National brand vs `"Starbucks Coffee Roasters"` (local shop)
- `"Walmart"` → Major retailer vs `"Walmarts Antiques"` (single entry)
- `"Chase"` → `"JP Morgan Chase"` (Bank) vs `"Chase & Sons Trucking"` 
- `"Ford"` → `"Ford Motor Company"` vs `"Ford's Diner"`

---

## Advanced Scoring Formula

```
Base Score = (String Similarity × 70%) + (Semantic Similarity × 30%)
Fidelity Boost = Acronym Fidelity × 15%
Location Boost = Location Score × 5% (Post-Inference)
Final Score = Base Score + Fidelity Boost + Location Boost + Popularity Boost
```

---

## Control Set Results

## 1. PDMA Association

**Query:** `PDMA Association` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Association Headquarters-PDMA (Mount Laurel, NJ) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity (Normalized) | 0.4491 | 30% | 0.1347 |
| Semantic Similarity (Raw) | 2.8192 | - | - |
| **Base Score** | **0.7393** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (FAIR):** Some meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• association

**Your Search Also Includes:**
• pdma

**Company Name Also Includes:**
• headquarters-pdma

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.8636 (Weight: 70%)
• Semantic Similarity: 0.4491 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Association Headquarters-PDMA (Mount Laurel, NJ) - 95.6% • *High word-for-word overlap.*
2. PDMA Alliance (York, SC) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Association Headquarters-PDMA"

**Score Difference:** 0.0471 (4.71 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.4491 vs 0.5445 (Δ -0.0954)


</details>
3. PDMA ALLIANCE - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "PDMA Alliance"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5445 vs 0.8098 (Δ -0.2654)


</details>
4. PDMA ALLIANCE (, FL) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "PDMA ALLIANCE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. PDMA ALLIANCE (CHARLOTTE, NC) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "PDMA ALLIANCE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. PDMA Alliance Inc. (Charlotte, NC) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "PDMA ALLIANCE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8098 vs 0.4832 (Δ +0.3267)


</details>
7. PDMA Alliance (Valhalla, NY) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "PDMA Alliance Inc."

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.4832 vs 0.5445 (Δ -0.0613)


</details>
8. PDMA Alliance (Valballa, NY) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "PDMA Alliance"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. PDMA - 80.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "PDMA Alliance"

**Score Difference:** 0.0966 (9.66 percentage points)

**Key Differentiators:**
- String Similarity: 0.8500 vs 0.7200 (Δ +0.1300)
- Semantic Similarity: 0.5445 vs 1.0000 (Δ -0.4555)


</details>

---

## 2. Nicolas/Sanchez Wedding

**Query:** `Nicolas/Sanchez Wedding` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Sanchez/Justin Wedding • **Score:** 80.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9159 | 30% | 0.2748 |
| Semantic Similarity (Raw) | 4.6285 | - | - |
| **Base Score** | **0.7954** | - | - |
| **FINAL SCORE** | **0.8002** | - | **80.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8002
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• wedding

**Your Search Also Includes:**
• nicolas/sanchez

**Company Name Also Includes:**
• sanchez/justin

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.9159 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Sanchez/Justin Wedding - 80.0% • *Matched via strong semantic/conceptual similarity.*
2. Sanchez Wedding (Marietta, GA) - 79.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Sanchez/Justin Wedding"

**Score Difference:** 0.0077 (0.77 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9159 vs 0.7885 (Δ +0.1274)


</details>
3. Sanchez/Ramirez Wedding (Miami, FL) - 78.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Sanchez Wedding"

**Score Difference:** 0.0046 (0.46 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7885 vs 0.8753 (Δ -0.0868)


</details>
4. Garcia Sanchez Wedding (Miami, FL) - 77.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Sanchez/Ramirez Wedding"

**Score Difference:** 0.0100 (1.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.8753 vs 1.0000 (Δ -0.1247)


</details>
5. Gibson Sanchez Wedding - 77.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Garcia Sanchez Wedding"

**Score Difference:** 0.0076 (0.76 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Sanchez/Puerto Wedding (Miami Beach, Fl) - 76.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Gibson Sanchez Wedding"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.9749 vs 0.8113 (Δ +0.1637)


</details>
7. Sanchez/Cohen Wedding (Hollywood, FL) - 76.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Sanchez/Puerto Wedding"

**Score Difference:** 0.0069 (0.69 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Sanchez/Naranjo Wedding (Miami, FL) - 75.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Sanchez/Cohen Wedding"

**Score Difference:** 0.0029 (0.29 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Sanchez/Fuentes Wedding (Dallas, TX) - 75.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Sanchez/Naranjo Wedding"

**Score Difference:** 0.0002 (0.02 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 3. Kehilat Ariel Synagogue (Los Angeles, CA)

**Query:** `Kehilat Ariel Synagogue` • **Location:** Los Angeles, CA • **Self-Match:** ✅ Found & Filtered

**Top Match:** Kehilat Ariel Synagogue (San Diego, CA) • **Score:** 102.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.9348 | 30% | 0.2805 |
| Semantic Similarity (Raw) | 4.4039 | - | - |
| **Base Score** | **0.9805** | - | - |
| Location Match Boost | 40.0000 | 5% max | +2.0000 |
| **FINAL SCORE** | **1.0224** | - | **102.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0224
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Kehilat Ariel Synagogue (San Diego, CA) - 102.2% • *Perfect character-for-character match.*
2. Kehilat Ariel Messianic Synagogue (San Diego, CA) - 84.6% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Kehilat Ariel Synagogue"

**Score Difference:** 0.1767 (17.67 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8636 (Δ +0.1364)
- Semantic Similarity: 0.9348 vs 0.8225 (Δ +0.1123)


</details>
3. Kehilat Ariel (San Diego, CA) - 80.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Kehilat Ariel Messianic Synagogue"

**Score Difference:** 0.0403 (4.03 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8225 vs 0.5886 (Δ +0.2339)


</details>
4. Kehilat Ariel Passover (San Diego, CA) - 80.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Kehilat Ariel"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.8833 (Δ -0.0652)


</details>
5. KAS - 69.0% • *Matched based on generated acronym 'KAS'.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Kehilat Ariel Passover"

**Score Difference:** 0.1155 (11.55 percentage points)

**Key Differentiators:**
- String Similarity: 0.8833 vs 1.0000 (Δ -0.1167)
- Semantic Similarity: 0.5558 vs 1.0000 (Δ -0.4442)
- Acronym Fidelity: 0.0000 vs 0.7000 (Δ -0.7000)
- Location Score: 40.0000 vs 0.0000 (Δ +40.0000)


</details>
6. Ohel Moshe Synagogue (Los Angeles, CA) - 62.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "KAS"

**Score Difference:** 0.0613 (6.13 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.4750 (Δ +0.5250)
- Semantic Similarity: 1.0000 vs 0.6645 (Δ +0.3355)
- Acronym Fidelity: 0.7000 vs 0.0000 (Δ +0.7000)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
7. Sephardic Temple-Synagogue (Los Angeles, CA) - 62.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Ohel Moshe Synagogue"

**Score Difference:** 0.0091 (0.91 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Synagogue 3000 (Los Angeles, CA) - 61.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Sephardic Temple-Synagogue"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6269 vs 0.7146 (Δ -0.0877)


</details>
9. Temple Sinai Synagogue (Oakland, CA) - 52.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Synagogue 3000"

**Score Difference:** 0.0931 (9.31 percentage points)

**Key Differentiators:**
- Location Score: 100.0000 vs 40.0000 (Δ +60.0000)


</details>
10. Synagogue Temple Aliyah (Woodland Hills, CA) - 50.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "Temple Sinai Synagogue"

**Score Difference:** 0.0177 (1.77 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7069 vs 0.6519 (Δ +0.0551)


</details>

---

## 4. Next Level Events

**Query:** `Next Level Events` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Next Level Events (New York, NY) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.5961 | 30% | 0.1788 |
| Semantic Similarity (Raw) | 3.5914 | - | - |
| **Base Score** | **0.8788** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Next Level Events (New York, NY) - 100.4% • *Perfect character-for-character match.*
2. Next Level Events (Lehi, UT) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Next Level Events"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Next Level Events - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Next Level Events"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Next Level Events (Atlanta, GA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Next Level Events"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. NEXT LEVEL EVENTS (Dallas, TX) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Next Level Events"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Next Level Events (Elizabeth, NJ) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "NEXT LEVEL EVENTS"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. NEXT LEVEL EVENTS (LOS ANGELES, CA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Next Level Events"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Next Level Events (Woodbridge, VA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "NEXT LEVEL EVENTS"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Next Level Events (Salt Lake City, UT) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Next Level Events"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 5. Site Foundation Golf Tournament

**Query:** `Site Foundation Golf Tournament` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Fore County Golf Tournament • **Score:** 77.1%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8191 | 30% | 0.2457 |
| Semantic Similarity (Raw) | 4.6740 | - | - |
| **Base Score** | **0.7664** | - | - |
| **FINAL SCORE** | **0.7710** | - | **77.1%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7710
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• golf, tournament

**Your Search Also Includes:**
• foundation, site

**Company Name Also Includes:**
• county, fore

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.8191 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Fore County Golf Tournament - 77.1% • *Hybrid match based on combined lexical and semantic features.*
2. House Victory Golf Tournament - 76.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Fore County Golf Tournament"

**Score Difference:** 0.0047 (0.47 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Women In Golf Foundation - 76.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "House Victory Golf Tournament"

**Score Difference:** 0.0016 (0.16 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. ANNIKA Foundation - Golf Tournament - 76.4% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Women In Golf Foundation"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7983 vs 0.6945 (Δ +0.1038)


</details>
5. Golf Tournament - 76.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "ANNIKA Foundation - Golf Tournament"

**Score Difference:** 0.0002 (0.02 percentage points)

**Key Differentiators:**
- String Similarity: 0.7875 vs 0.6562 (Δ +0.1312)
- Semantic Similarity: 0.6945 vs 1.0000 (Δ -0.3055)


</details>
6. Bunker To Bunker Golf Tournament - 76.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Golf Tournament"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:**
- String Similarity: 0.6562 vs 0.7438 (Δ -0.0875)
- Semantic Similarity: 1.0000 vs 0.7885 (Δ +0.2115)


</details>
7. National Youth Golf Foundation - 76.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Bunker To Bunker Golf Tournament"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. World Golf Foundation - 75.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "National Youth Golf Foundation"

**Score Difference:** 0.0020 (0.20 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.7825 vs 0.9335 (Δ -0.1510)


</details>
9. National Golf Foundation - 75.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "World Golf Foundation"

**Score Difference:** 0.0070 (0.70 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 6. Interim WG Meeting - BIER

**Query:** `Interim WG Meeting - BIER` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Interim Healthcare TEAMM Meeting • **Score:** 69.3%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity (Normalized) | 0.6420 | 30% | 0.1926 |
| Semantic Similarity (Raw) | 2.9424 | - | - |
| **Base Score** | **0.6884** | - | - |
| **FINAL SCORE** | **0.6926** | - | **69.3%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.6926
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• interim, meeting

**Your Search Also Includes:**
• -, bier, wg

**Company Name Also Includes:**
• healthcare, teamm

**Match Strength:**
• 40% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7083 (Weight: 70%)
• Semantic Similarity: 0.6420 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Interim Healthcare TEAMM Meeting - 69.3% • *Hybrid match based on combined lexical and semantic features.*
2. AACP 2012 Interim Meeting - 69.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Interim Healthcare TEAMM Meeting"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Legislative Interim Meeting (Weston, WV) - 65.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "AACP 2012 Interim Meeting"

**Score Difference:** 0.0322 (3.22 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.6439 (Δ +0.0644)


</details>
4. BI Meeting - 56.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Legislative Interim Meeting"

**Score Difference:** 0.0931 (9.31 percentage points)

**Key Differentiators:**
- String Similarity: 0.6439 vs 0.3750 (Δ +0.2689)
- Semantic Similarity: 0.6808 vs 1.0000 (Δ -0.3192)


</details>
5. Bi Annual Meeting - 56.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "BI Meeting"

**Score Difference:** 0.0060 (0.60 percentage points)

**Key Differentiators:**
- Semantic Similarity: 1.0000 vs 0.9004 (Δ +0.0996)


</details>
6. HRC Advisory Board meeting - 53.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Bi Annual Meeting"

**Score Difference:** 0.0213 (2.13 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9004 vs 0.7343 (Δ +0.1661)


</details>
7. Biz Library January Meeting - 53.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "HRC Advisory Board meeting"

**Score Difference:** 0.0066 (0.66 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Bim Object Meeting - 52.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Biz Library January Meeting"

**Score Difference:** 0.0045 (0.45 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7125 vs 0.7930 (Δ -0.0805)


</details>
9. American Biz Meeting - 52.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Bim Object Meeting"

**Score Difference:** 0.0027 (0.27 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 7. DermaQuest Inc

**Query:** `DermaQuest Inc` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Dermaquest Skin Care • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 4.3077 | - | - |
| **Base Score** | **0.8250** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• dermaquest

**Your Search Also Includes:**
• inc

**Company Name Also Includes:**
• care, skin

**Match Strength:**
• 33% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7500 (Weight: 70%)
• Semantic Similarity: 1.0000 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Dermaquest Skin Care - 95.6% • *High word-for-word overlap.*
2. Dermaquest, Incorporated (Hayward, CA) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Dermaquest Skin Care"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 1.0000 (Δ -0.2500)
- Semantic Similarity: 1.0000 vs 0.7240 (Δ +0.2760)


</details>
3. Dermaquest Skin Therapy (Hayward, CA) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Dermaquest, Incorporated"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7500 (Δ +0.2500)
- Semantic Similarity: 0.7240 vs 0.6384 (Δ +0.0856)


</details>
4. Dermapen - 49.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Dermaquest Skin Therapy"

**Score Difference:** 0.4569 (45.69 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.3000 (Δ +0.4500)
- Semantic Similarity: 0.6384 vs 0.9470 (Δ -0.3087)


</details>
5. DERMA E - 49.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Dermapen"

**Score Difference:** 0.0025 (0.25 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Mapquest - 48.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "DERMA E"

**Score Difference:** 0.0102 (1.02 percentage points)

**Key Differentiators:**
- String Similarity: 0.2888 vs 0.3500 (Δ -0.0612)
- Semantic Similarity: 0.9707 vs 0.7941 (Δ +0.1766)


</details>
7. Perquest - 48.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Mapquest"

**Score Difference:** 0.0058 (0.58 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Interquest - 47.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Perquest"

**Score Difference:** 0.0077 (0.77 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7750 vs 0.8312 (Δ -0.0562)


</details>
9. RamQuest - 46.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Interquest"

**Score Difference:** 0.0120 (1.20 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8312 vs 0.7097 (Δ +0.1215)


</details>

---

## 8. Ellwood Group Inc (Chicago, IL)

**Query:** `Ellwood Group Inc` • **Location:** Chicago, IL • **Self-Match:** ✅ Found & Filtered

**Top Match:** Ellwood Group Inc (Ellwood City, PA) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.2426 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Ellwood Group Inc (Ellwood City, PA) - 100.2% • *Perfect character-for-character match.*
2. Ellwood Associates (Chicago, IL) - 96.9% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Ellwood Group Inc"

**Score Difference:** 0.0333 (3.33 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.9000 (Δ +0.1000)
- Semantic Similarity: 1.0000 vs 0.7235 (Δ +0.2765)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
3. Ellwood TX Forge Houston (Houston, TX) - 76.9% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Ellwood Associates"

**Score Difference:** 0.2000 (20.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.9000 vs 0.6923 (Δ +0.2077)
- Semantic Similarity: 0.7235 vs 0.6505 (Δ +0.0730)
- Location Score: 100.0000 vs 0.0000 (Δ +100.0000)


</details>
4. Ellwood Community Church - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Ellwood TX Forge Houston"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- String Similarity: 0.6923 vs 0.7500 (Δ -0.0577)
- Semantic Similarity: 0.6505 vs 0.8259 (Δ -0.1754)


</details>
5. Ellwood Rose Machine (Houston, TX) - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Ellwood Community Church"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8259 vs 0.7028 (Δ +0.1231)


</details>
6. Ellwood TX Forge - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Ellwood Rose Machine"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Ellwood TX Forge Houston - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Ellwood TX Forge"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.6923 (Δ +0.0577)


</details>
8. Ellwood Closed Die Group (Houston, TX) - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Ellwood TX Forge Houston"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.6923 vs 0.7500 (Δ -0.0577)


</details>
9. Ellwood Specialty Steel (Ellwood City, PA) - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Ellwood Closed Die Group"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
10. Ellwood City Area School District (inc) (Ellwood City, PA) - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "Ellwood Specialty Steel"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.6000 (Δ +0.1500)


</details>

---

## 9. American Miniature Horse Registry

**Query:** `American Miniature Horse Registry` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** American Miniature Horse Association (Alvarado, TX) • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 0.5919 | 30% | 0.1776 |
| Semantic Similarity (Raw) | 3.7158 | - | - |
| **Base Score** | **0.7682** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• american, horse, miniature

**Your Search Also Includes:**
• registry

**Company Name Also Includes:**
• association

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8438 (Weight: 70%)
• Semantic Similarity: 0.5919 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. American Miniature Horse Association (Alvarado, TX) - 90.5% • *High word-for-word overlap.*
2. American Miniature Horse Association Headquarters - 77.0% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "American Miniature Horse Association"

**Score Difference:** 0.1356 (13.56 percentage points)

**Key Differentiators:**
- String Similarity: 0.8438 vs 0.7670 (Δ +0.0767)
- Semantic Similarity: 0.5919 vs 0.7608 (Δ -0.1689)


</details>
3. American Saddle Horse Association - 69.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "American Miniature Horse Association Headquarters"

**Score Difference:** 0.0735 (7.35 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7608 vs 0.5717 (Δ +0.1892)


</details>
4. American Horse Defense Fund - 69.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "American Saddle Horse Association"

**Score Difference:** 0.0049 (0.49 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Miniature Horse & Pony Show (Farr West, UT) - 68.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "American Horse Defense Fund"

**Score Difference:** 0.0037 (0.37 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. American Youth & Horse Council - 68.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Miniature Horse & Pony Show"

**Score Difference:** 0.0016 (0.16 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. American Youth Horse Council (Storrs, CT) - 68.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "American Youth & Horse Council"

**Score Difference:** 0.0035 (0.35 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. American Miniature Hores Association - 68.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "American Youth Horse Council"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. American Youth Horse Council (Lexington, KY) - 67.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "American Miniature Hores Association"

**Score Difference:** 0.0030 (0.30 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 10. YADA ENTERPRISES, INC

**Query:** `YADA ENTERPRISES, INC` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Yada Yada (Kirkland, WA) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity (Normalized) | 0.7863 | 30% | 0.2359 |
| Semantic Similarity (Raw) | 3.3808 | - | - |
| **Base Score** | **0.8659** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• yada

**Your Search Also Includes:**
• enterprises,, inc

**Match Strength:**
• 33% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.9000 (Weight: 70%)
• Semantic Similarity: 0.7863 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Yada Yada (Kirkland, WA) - 95.6% • *High word-for-word overlap.*
2. Yasuda Corporation Limited - 53.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Yada Yada"

**Score Difference:** 0.4217 (42.17 percentage points)

**Key Differentiators:**
- String Similarity: 0.9000 vs 0.3600 (Δ +0.5400)
- Semantic Similarity: 0.7863 vs 0.9295 (Δ -0.1432)


</details>
3. Yama - 52.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Yasuda Corporation Limited"

**Score Difference:** 0.0083 (0.83 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Yama Group - 51.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Yama"

**Score Difference:** 0.0096 (0.96 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Ya - 51.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Yama Group"

**Score Difference:** 0.0030 (0.30 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9225 vs 1.0000 (Δ -0.0775)


</details>
6. Yara - 51.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Ya"

**Score Difference:** 0.0035 (0.35 percentage points)

**Key Differentiators:**
- Semantic Similarity: 1.0000 vs 0.9008 (Δ +0.0992)


</details>
7. YATA - 50.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Yara"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Yama Enterprises (Smyrna, GA) - 50.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "YATA"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Yamas - 50.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Yama Enterprises"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 11. Seafood Nutrition Partnership

**Query:** `Seafood Nutrition Partnership` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Seafood Nutrition Partnership • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6985 | 30% | 0.2095 |
| Semantic Similarity (Raw) | 4.7158 | - | - |
| **Base Score** | **0.9095** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Seafood Nutrition Partnership - 100.2% • *Perfect character-for-character match.*
2. Seafood Nutrition Partnership (Durham, CT) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Seafood Nutrition Partnership"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Seafood Nutrition Partnership (Bellevue, WA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Seafood Nutrition Partnership"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Sustainable Seafood Partnership (Bellingham, WA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Seafood Nutrition Partnership"

**Score Difference:** 0.0970 (9.70 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8097 (Δ +0.1903)
- Semantic Similarity: 0.6985 vs 0.5755 (Δ +0.1230)


</details>
5. SEAFOOD NUTRITION (ARLINGTON, VA) - 72.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Sustainable Seafood Partnership"

**Score Difference:** 0.1853 (18.53 percentage points)

**Key Differentiators:**
- String Similarity: 0.8097 vs 0.7500 (Δ +0.0597)
- Semantic Similarity: 0.5755 vs 0.6360 (Δ -0.0606)


</details>
6. Seafood Choices Alliance - 59.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "SEAFOOD NUTRITION"

**Score Difference:** 0.1217 (12.17 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.5278 (Δ +0.2222)
- Semantic Similarity: 0.6360 vs 0.7513 (Δ -0.1153)


</details>
7. Pet Nutrition Alliance - 56.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Seafood Choices Alliance"

**Score Difference:** 0.0353 (3.53 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7513 vs 0.6344 (Δ +0.1169)


</details>
8. East Coast Seafood - 55.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Pet Nutrition Alliance"

**Score Difference:** 0.0049 (0.49 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. International Boston Seafood - 55.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "East Coast Seafood"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 12. AVIAKOMPANIYA SIBIR, PAO

**Query:** `AVIAKOMPANIYA SIBIR, PAO` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** AVIAKOMPANIYA MIZHNARODNI AVIA • **Score:** 59.3%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity (Normalized) | 0.7340 | 30% | 0.2202 |
| Semantic Similarity (Raw) | 3.2724 | - | - |
| **Base Score** | **0.5897** | - | - |
| **FINAL SCORE** | **0.5932** | - | **59.3%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.5932
```

### Component Analysis

- **String Similarity (FAIR):** Some lexical similarity - partial word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• aviakompaniya

**Your Search Also Includes:**
• pao, sibir,

**Company Name Also Includes:**
• avia, mizhnarodni

**Match Strength:**
• 33% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.5278 (Weight: 70%)
• Semantic Similarity: 0.7340 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. AVIAKOMPANIYA MIZHNARODNI AVIA - 59.3% • *Hybrid match based on combined lexical and semantic features.*
2. AVIAKOMPANIYA MIZHNARODNI AVIA (KYIV, ) - 58.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "AVIAKOMPANIYA MIZHNARODNI AVIA"

**Score Difference:** 0.0109 (1.09 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. AVIAKOMPANIYA AEROSVIT, PRYVAT (SELO GORA, ) - 54.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "AVIAKOMPANIYA MIZHNARODNI AVIA"

**Score Difference:** 0.0409 (4.09 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7340 vs 0.5625 (Δ +0.1715)


</details>
4. AVIAKOMPANIYA AEROSVIT, PRYVATNE AT (SELO GORA, ) - 49.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "AVIAKOMPANIYA AEROSVIT, PRYVAT"

**Score Difference:** 0.0498 (4.98 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5625 vs 0.5095 (Δ +0.0530)


</details>
5. Shibir  Desai - 34.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "AVIAKOMPANIYA AEROSVIT, PRYVATNE AT"

**Score Difference:** 0.1445 (14.45 percentage points)

**Key Differentiators:**
- String Similarity: 0.4798 vs 0.1636 (Δ +0.3162)
- Semantic Similarity: 0.5095 vs 0.7685 (Δ -0.2590)


</details>
6. AVIPAM Sao Paulo - 33.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Shibir  Desai"

**Score Difference:** 0.0094 (0.94 percentage points)

**Key Differentiators:**
- String Similarity: 0.1636 vs 0.2538 (Δ -0.0902)
- Semantic Similarity: 0.7685 vs 0.5269 (Δ +0.2416)


</details>
7. Avyaya Integrated - 33.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "AVIPAM Sao Paulo"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:**
- String Similarity: 0.2538 vs 0.1841 (Δ +0.0698)
- Semantic Similarity: 0.5269 vs 0.6823 (Δ -0.1554)


</details>
8. Salaha Kabir - 33.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Avyaya Integrated"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. AMANDA MAHABIR - 33.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Salaha Kabir"

**Score Difference:** 0.0017 (0.17 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7287 vs 0.6406 (Δ +0.0881)


</details>

---

## 13. Hartford Hospital School of Nursing (Hartford, CT)

**Query:** `Hartford Hospital School of Nursing` • **Location:** Hartford, CT • **Self-Match:** ✅ Found & Filtered

**Top Match:** Hartford Hospital School of Nursing (Wethersfield, CT) • **Score:** 102.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.7613 | 30% | 0.2284 |
| Semantic Similarity (Raw) | 4.4633 | - | - |
| **Base Score** | **0.9284** | - | - |
| Location Match Boost | 40.0000 | 5% max | +2.0000 |
| **FINAL SCORE** | **1.0224** | - | **102.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0224
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Hartford Hospital School of Nursing (Wethersfield, CT) - 102.2% • *Perfect character-for-character match.*
2. Hartford Public High School (Hartford, Ct) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Hartford Hospital School of Nursing"

**Score Difference:** 0.0970 (9.70 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8500 (Δ +0.1500)
- Location Score: 40.0000 vs 100.0000 (Δ -60.0000)


</details>
3. Hartford Magnet Middle School (Hartford, CT) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Hartford Public High School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7692 vs 0.6747 (Δ +0.0944)


</details>
4. East Hartford Middle School (East Hartford, CT) - 91.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Hartford Magnet Middle School"

**Score Difference:** 0.0106 (1.06 percentage points)

**Key Differentiators:**
- Location Score: 100.0000 vs 93.1000 (Δ +6.9000)


</details>
5. Hartford Public High School (West Hartford, CT) - 91.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "East Hartford Middle School"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6898 vs 0.7692 (Δ -0.0793)


</details>
6. West Hartford Public School (West Hartford, CT) - 91.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Hartford Public High School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Hartford Hospital (Hartford, CT) - 86.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "West Hartford Public School"

**Score Difference:** 0.0466 (4.66 percentage points)

**Key Differentiators:**
- String Similarity: 0.8500 vs 0.7500 (Δ +0.1000)
- Semantic Similarity: 0.7268 vs 1.0000 (Δ -0.2732)
- Location Score: 93.1000 vs 100.0000 (Δ -6.9000)


</details>
8. Hartford Union High School (Hartford, WI) - 84.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Hartford Hospital"

**Score Difference:** 0.0195 (1.95 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.8500 (Δ -0.1000)
- Semantic Similarity: 1.0000 vs 0.6911 (Δ +0.3089)
- Location Score: 100.0000 vs 60.0000 (Δ +40.0000)


</details>
9. Hartford School District (Hartford, CT) - 84.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Hartford Union High School"

**Score Difference:** 0.0038 (0.38 percentage points)

**Key Differentiators:**
- String Similarity: 0.8500 vs 0.7727 (Δ +0.0773)
- Semantic Similarity: 0.6911 vs 0.8504 (Δ -0.1593)
- Location Score: 60.0000 vs 100.0000 (Δ -40.0000)


</details>
10. Hartford Elementary School (Hartford, CT) - 83.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "Hartford School District"

**Score Difference:** 0.0059 (0.59 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 14. Internal J&J Meeting and Breakfast

**Query:** `Internal J&J Meeting and Breakfast` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Internal J&J Meeting and Breakfast (Somerville, NJ) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8794 | 30% | 0.2638 |
| Semantic Similarity (Raw) | 3.9747 | - | - |
| **Base Score** | **0.9638** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Internal J&J Meeting and Breakfast (Somerville, NJ) - 100.2% • *Perfect character-for-character match.*
2. ASCO Internal Pre Meeting - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Internal J&J Meeting and Breakfast"

**Score Difference:** 0.0970 (9.70 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8500 (Δ +0.1500)
- Semantic Similarity: 0.8794 vs 0.5930 (Δ +0.2865)


</details>
3. IT Management Internal Meeting - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "ASCO Internal Pre Meeting"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Atea Internal Meeting - 74.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "IT Management Internal Meeting"

**Score Difference:** 0.1588 (15.88 percentage points)

**Key Differentiators:**
- String Similarity: 0.8500 vs 0.7727 (Δ +0.0773)
- Semantic Similarity: 0.5585 vs 0.6709 (Δ -0.1124)


</details>
5. Nov. Internal Meeting - 71.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Atea Internal Meeting"

**Score Difference:** 0.0367 (3.67 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6709 vs 0.5493 (Δ +0.1216)


</details>
6. Internal Meeting - 68.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Nov. Internal Meeting"

**Score Difference:** 0.0237 (2.37 percentage points)

**Key Differentiators:**
- String Similarity: 0.7727 vs 0.6667 (Δ +0.1061)
- Semantic Similarity: 0.5493 vs 0.7183 (Δ -0.1691)


</details>
7. Greg Tolliver Breakfast Meeting - 65.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Internal Meeting"

**Score Difference:** 0.0327 (3.27 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. GMCVB Breakfast & Meeting - 65.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Greg Tolliver Breakfast Meeting"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:**
- String Similarity: 0.6375 vs 0.5795 (Δ +0.0580)
- Semantic Similarity: 0.6781 vs 0.8116 (Δ -0.1334)


</details>
9. DXC Technology Breakfast Meeting - 65.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "GMCVB Breakfast & Meeting"

**Score Difference:** 0.0026 (0.26 percentage points)

**Key Differentiators:**
- String Similarity: 0.5795 vs 0.6375 (Δ -0.0580)
- Semantic Similarity: 0.8116 vs 0.6678 (Δ +0.1438)


</details>

---

## 15. Spina Bifida Coalition of Cincinnati (Cincinnati, OH)

**Query:** `Spina Bifida Coalition of Cincinnati` • **Location:** Cincinnati, OH • **Self-Match:** ✅ Found & Filtered

**Top Match:** Spina Bifida Association of Cincinnati, Inc. (Cincinnati, OH) • **Score:** 92.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 0.7990 | 30% | 0.2397 |
| Semantic Similarity (Raw) | 4.0340 | - | - |
| **Base Score** | **0.8303** | - | - |
| Location Match Boost | 100.0000 | 5% max | +5.0000 |
| **FINAL SCORE** | **0.9255** | - | **92.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 100.0000 × 0.05 = 5.0000

Final Score = Base Score + Location Boost = 0.9255
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 100.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• bifida, of, spina

**Your Search Also Includes:**
• cincinnati, coalition

**Company Name Also Includes:**
• association, cincinnati,, inc.

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.8438 (Weight: 70%)
• Semantic Similarity: 0.7990 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Spina Bifida Association of Cincinnati, Inc. (Cincinnati, OH) - 92.5% • *High word-for-word overlap.*
2. SBCC - 75.0% • *Matched based on generated acronym 'SBCC'.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Spina Bifida Association of Cincinnati, Inc."

**Score Difference:** 0.1755 (17.55 percentage points)

**Key Differentiators:**
- String Similarity: 0.8438 vs 1.0000 (Δ -0.1562)
- Semantic Similarity: 0.7990 vs 1.0000 (Δ -0.2010)
- Acronym Fidelity: 0.0000 vs 1.0000 (Δ -1.0000)
- Location Score: 100.0000 vs 0.0000 (Δ +100.0000)


</details>
3. Greater Cincinnati Good Food Coalition (Cincinnati, OH) - 72.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "SBCC"

**Score Difference:** 0.0228 (2.28 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.6761 (Δ +0.3239)
- Semantic Similarity: 1.0000 vs 0.6024 (Δ +0.3976)
- Acronym Fidelity: 1.0000 vs 0.0000 (Δ +1.0000)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
4. Alliance Cincinnati (Cincinnati, OH) - 63.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Greater Cincinnati Good Food Coalition"

**Score Difference:** 0.0929 (9.29 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.4062 (Δ +0.2699)
- Semantic Similarity: 0.6024 vs 0.8481 (Δ -0.2457)


</details>
5. Urban Appalachian Community Coalition (Cincinnati, OH) - 63.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Alliance Cincinnati"

**Score Difference:** 0.0043 (0.43 percentage points)

**Key Differentiators:**
- String Similarity: 0.4062 vs 0.4875 (Δ -0.0812)
- Semantic Similarity: 0.8481 vs 0.6406 (Δ +0.2075)


</details>
6. Health Alliance of Greater Cincinnati (Cincinnati, OH) - 62.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Urban Appalachian Community Coalition"

**Score Difference:** 0.0045 (0.45 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. AAA Allied Group Cincinnati (Cincinnati, OH) - 62.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Health Alliance of Greater Cincinnati"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6220 vs 0.7226 (Δ -0.1007)


</details>
8. Cincinnati Reds Baseball Club (Cincinnati, OH) - 62.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "AAA Allied Group Cincinnati"

**Score Difference:** 0.0002 (0.02 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7226 vs 0.6185 (Δ +0.1042)


</details>
9. Cincinnati North IMA Chapter (Cincinnati, OH) - 62.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Cincinnati Reds Baseball Club"

**Score Difference:** 0.0031 (0.31 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 16. THE SOCA GROUP ORGANIZATION

**Query:** `THE SOCA GROUP ORGANIZATION` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Team SOCA • **Score:** 79.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8942 | 30% | 0.2682 |
| Semantic Similarity (Raw) | 5.3999 | - | - |
| **Base Score** | **0.7889** | - | - |
| **FINAL SCORE** | **0.7937** | - | **79.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7937
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• soca

**Your Search Also Includes:**
• group, organization, the

**Company Name Also Includes:**
• team

**Match Strength:**
• 25% word overlap
• This is a WEAK match - may be coincidental
• Action: Verify carefully before using

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.8942 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Team SOCA - 79.4% • *Matched via strong semantic/conceptual similarity.*
2. Soca Society - 78.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Team SOCA"

**Score Difference:** 0.0091 (0.91 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Organization Management Group - 74.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Soca Society"

**Score Difference:** 0.0396 (3.96 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8639 vs 0.7326 (Δ +0.1313)


</details>
4. Four Organization - 73.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Organization Management Group"

**Score Difference:** 0.0081 (0.81 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. System Organization - 73.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Four Organization"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Organization Meeting - 73.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "System Organization"

**Score Difference:** 0.0048 (0.48 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. International organization - 72.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Organization Meeting"

**Score Difference:** 0.0031 (0.31 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. social organization - 72.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "International organization"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Organization Management - 72.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "social organization"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 17. Shiroyama Junior High School

**Query:** `Shiroyama Junior High School` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Brooks Junior High School • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity (Normalized) | 0.8103 | 30% | 0.2431 |
| Semantic Similarity (Raw) | 3.7100 | - | - |
| **Base Score** | **0.8206** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• high, junior, school

**Your Search Also Includes:**
• shiroyama

**Company Name Also Includes:**
• brooks

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8250 (Weight: 70%)
• Semantic Similarity: 0.8103 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Brooks Junior High School - 90.5% • *High word-for-word overlap.*
2. Kenmore Junior High School - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Brooks Junior High School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Sargent Junior High School - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Kenmore Junior High School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Greenspun Junior High School - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Sargent Junior High School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Junior High School #275 - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Greenspun Junior High School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Nimitz Junior High School - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Junior High School #275"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. CARROLL JUNIOR HIGH SCHOOL (Southlake, TX) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Nimitz Junior High School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Junior High School 45 (New York, NY) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "CARROLL JUNIOR HIGH SCHOOL"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Hardin Junior High School (Hardin, TX) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Junior High School 45"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 18. National Home Health (Washington, DC)

**Query:** `National Home Health` • **Location:** Washington, DC • **Self-Match:** ✅ Found & Filtered

**Top Match:** National Home Health (Herndon, VA) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.5904 | 30% | 0.1771 |
| Semantic Similarity (Raw) | 3.7280 | - | - |
| **Base Score** | **0.8771** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. National Home Health (Herndon, VA) - 100.4% • *Perfect character-for-character match.*
2. National Home Health - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "National Home Health"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. National Home Health Care (Washington, DC) - 96.9% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "National Home Health"

**Score Difference:** 0.0333 (3.33 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8225 (Δ +0.1775)
- Semantic Similarity: 0.5904 vs 0.7439 (Δ -0.1535)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
4. Legacy Home Health Care (Washington, DC) - 92.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "National Home Health Care"

**Score Difference:** 0.0405 (4.05 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7439 vs 0.6663 (Δ +0.0775)


</details>
5. National Association of Home Care (Washington, DC) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Legacy Home Health Care"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6663 vs 0.5686 (Δ +0.0977)


</details>
6. Human Touch Home Health (Washington, DC) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "National Association of Home Care"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Community Home Health (Arlington, VA) - 81.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Human Touch Home Health"

**Score Difference:** 0.1116 (11.16 percentage points)

**Key Differentiators:**
- String Similarity: 0.8030 vs 0.8833 (Δ -0.0803)
- Semantic Similarity: 0.5533 vs 0.6849 (Δ -0.1316)
- Location Score: 100.0000 vs 44.2000 (Δ +55.8000)


</details>
8. National Home Health Care - 76.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Community Home Health"

**Score Difference:** 0.0481 (4.81 percentage points)

**Key Differentiators:**
- String Similarity: 0.8833 vs 0.8225 (Δ +0.0608)
- Semantic Similarity: 0.6849 vs 0.7439 (Δ -0.0589)
- Location Score: 44.2000 vs 0.0000 (Δ +44.2000)


</details>
9. National Home Health Holiday Party - 76.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "National Home Health Care"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8225 vs 0.7500 (Δ +0.0725)


</details>
10. National Association of Home Health Care Providers - 76.6% • *High word-for-word overlap.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "National Home Health Holiday Party"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7499 vs 0.6530 (Δ +0.0969)


</details>

---

## 19. American News Women's Club

**Query:** `American News Women's Club` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Danish-American Women's Club (San Jose, CA) • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 0.5520 | 30% | 0.1656 |
| Semantic Similarity (Raw) | 2.9119 | - | - |
| **Base Score** | **0.7562** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• club, women's

**Your Search Also Includes:**
• american, news

**Company Name Also Includes:**
• danish-american

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.8438 (Weight: 70%)
• Semantic Similarity: 0.5520 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Danish-American Women's Club (San Jose, CA) - 90.5% • *High word-for-word overlap.*
2. American Slavic Women's Club (Maple Valley, WA) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Danish-American Women's Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. American Women's Club - 80.3% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "American Slavic Women's Club"

**Score Difference:** 0.1023 (10.23 percentage points)

**Key Differentiators:**
- String Similarity: 0.8438 vs 0.7670 (Δ +0.0767)
- Semantic Similarity: 0.5476 vs 0.8713 (Δ -0.3237)


</details>
4. American Women Club - 77.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "American Women's Club"

**Score Difference:** 0.0252 (2.52 percentage points)

**Key Differentiators:**
- String Similarity: 0.7670 vs 0.6761 (Δ +0.0909)
- Semantic Similarity: 0.8713 vs 1.0000 (Δ -0.1287)


</details>
5. Indo American Press Club - 72.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "American Women Club"

**Score Difference:** 0.0526 (5.26 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 1.0000 vs 0.6681 (Δ +0.3319)


</details>
6. Thousand Oaks Women's Club - 72.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Indo American Press Club"

**Score Difference:** 0.0029 (0.29 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Women's Club Board Meeting - 72.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Thousand Oaks Women's Club"

**Score Difference:** 0.0026 (0.26 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. San Jose Women's Club - 71.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Women's Club Board Meeting"

**Score Difference:** 0.0015 (0.15 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Los Prados Women's Club - 71.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "San Jose Women's Club"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 20. Denise Roberge

**Query:** `Denise Roberge` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Denise Abril • **Score:** 81.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9680 | 30% | 0.2904 |
| Semantic Similarity (Raw) | 4.0800 | - | - |
| **Base Score** | **0.8110** | - | - |
| **FINAL SCORE** | **0.8160** | - | **81.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8160
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• denise

**Your Search Also Includes:**
• roberge

**Company Name Also Includes:**
• abril

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.9680 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Denise Abril - 81.6% • *Matched via strong semantic/conceptual similarity.*
2. Denise Beard - 81.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Denise Abril"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Denise White - 81.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Denise Beard"

**Score Difference:** 0.0011 (0.11 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Charmaine Denise - 81.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Denise White"

**Score Difference:** 0.0040 (0.40 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Denise Wallack - 80.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Charmaine Denise"

**Score Difference:** 0.0028 (0.28 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. CeCi Denise - 80.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Denise Wallack"

**Score Difference:** 0.0008 (0.08 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Denise Ivy - 80.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "CeCi Denise"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Denise Martin - 79.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Denise Ivy"

**Score Difference:** 0.0118 (1.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. sylvia denise - 79.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Denise Martin"

**Score Difference:** 0.0027 (0.27 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 21. Synergy Soccer Club

**Query:** `Synergy Soccer Club` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Nordic Soccer Club (Colchester, VT) • **Score:** 90.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity (Normalized) | 0.6126 | 30% | 0.1838 |
| Semantic Similarity (Raw) | 3.0237 | - | - |
| **Base Score** | **0.7506** | - | - |
| **FINAL SCORE** | **0.9087** | - | **90.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9087
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• club, soccer

**Your Search Also Includes:**
• synergy

**Company Name Also Includes:**
• nordic

**Match Strength:**
• 67% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8097 (Weight: 70%)
• Semantic Similarity: 0.6126 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Nordic Soccer Club (Colchester, VT) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
2. Alliance Soccer Club (Reynoldsburg, OH) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Nordic Soccer Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Club Ohio Soccer (Dublin, OH) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Alliance Soccer Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Synergy Volleyball Club (Toronto, ON) - 90.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Club Ohio Soccer"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5791 vs 0.8526 (Δ -0.2734)


</details>
5. Club Soccer Event - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Synergy Volleyball Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Sting Soccer Club - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Club Soccer Event"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Magic Soccer Club - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Sting Soccer Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. International Soccer Club - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Magic Soccer Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Classic Soccer Club - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "International Soccer Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 22. NFC Forum

**Query:** `NFC Forum` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** NFC Forum (Wakefield, MA) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6788 | 30% | 0.2036 |
| Semantic Similarity (Raw) | 4.8315 | - | - |
| **Base Score** | **0.9036** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. NFC Forum (Wakefield, MA) - 100.4% • *Perfect character-for-character match.*
2. NFC Forum - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "NFC Forum"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. NFC Forum (Minneapolis, MN) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "NFC Forum"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. NFC Forum (Woodville, WI) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "NFC Forum"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. NFC Forum (Escondido, CA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "NFC Forum"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. NFC Forum         . (Wakfield, MA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "NFC Forum"

**Score Difference:** 0.0467 (4.67 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. NFC Forum Members (Wakefield, MA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "NFC Forum         ."

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8182 (Δ +0.1818)


</details>
8. NFC Forum         . (Wakefield, MA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "NFC Forum Members"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 1.0000 (Δ -0.1818)


</details>
9. NFC Consulting - 73.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "NFC Forum         ."

**Score Difference:** 0.2190 (21.90 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7438 (Δ +0.2562)
- Semantic Similarity: 0.6543 vs 0.7055 (Δ -0.0512)


</details>

---

## 23. A Better Choice Limousine & Concierge

**Query:** `A Better Choice Limousine & Concierge` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** First Choice Limousine Services (Dorchester, MA) • **Score:** 72.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.6505 | 30% | 0.1952 |
| Semantic Similarity (Raw) | 3.7201 | - | - |
| **Base Score** | **0.7158** | - | - |
| **FINAL SCORE** | **0.7201** | - | **72.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7201
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• choice, limousine

**Your Search Also Includes:**
• &, a, better, concierge

**Company Name Also Includes:**
• first, services

**Match Strength:**
• 33% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.6505 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. First Choice Limousine Services (Dorchester, MA) - 72.0% • *Hybrid match based on combined lexical and semantic features.*
2. Better Choice Travel - 65.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "First Choice Limousine Services"

**Score Difference:** 0.0673 (6.73 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6505 vs 0.5852 (Δ +0.0653)


</details>
3. Better Choice Travel (Cleveland, OH) - 63.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Better Choice Travel"

**Score Difference:** 0.0210 (2.10 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Executive Limousine - 56.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Better Choice Travel"

**Score Difference:** 0.0650 (6.50 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.4062 (Δ +0.2699)
- Semantic Similarity: 0.5852 vs 0.9300 (Δ -0.3448)


</details>
5. Chicago Limousine Transportation - 55.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Executive Limousine"

**Score Difference:** 0.0091 (0.91 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9300 vs 0.8138 (Δ +0.1162)


</details>
6. Journey Limousine - 55.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Chicago Limousine Transportation"

**Score Difference:** 0.0017 (0.17 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8138 vs 0.8943 (Δ -0.0805)


</details>
7. Metropolitan Limousine - 55.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Journey Limousine"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Greater Atlanta Limousine - 55.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Metropolitan Limousine"

**Score Difference:** 0.0002 (0.02 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8879 vs 0.8011 (Δ +0.0868)


</details>
9. Alliance Limousine - 55.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Greater Atlanta Limousine"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8011 vs 0.8871 (Δ -0.0860)


</details>

---

## 24. Danish Sisterhood of America

**Query:** `Danish Sisterhood of America` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Danish Sisterhood and Brotherhood of America (Burbank, Ca) • **Score:** 95.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity (Normalized) | 0.6466 | 30% | 0.1940 |
| Semantic Similarity (Raw) | 3.6582 | - | - |
| **Base Score** | **0.7985** | - | - |
| **FINAL SCORE** | **0.9592** | - | **95.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9592
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
ALL WORDS MATCHED

**What This Means:**
Every word in your search 'Danish Sisterhood of America' was found in this company name.

**Matching Words:**
• america, danish, of, sisterhood

**Company Name Also Includes:**
• and, brotherhood

**Match Strength:**
• 67% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8636 (Weight: 70%)
• Semantic Similarity: 0.6466 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Danish Sisterhood and Brotherhood of America (Burbank, Ca) - 95.9% • *High word-for-word overlap.*
2. The Danish Sisterhood of America (Arvada, CO) - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Danish Sisterhood and Brotherhood of America"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- String Similarity: 0.8636 vs 1.0000 (Δ -0.1364)


</details>
3. DANISH BROTHERHOOD AND DANISH SISTERHOOD OF AMERICA (Palatine, IL) - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "The Danish Sisterhood of America"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8182 (Δ +0.1818)


</details>
4. Danish Brotherhood & Danish Sisterhood of America (Hamden, CT) - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "DANISH BROTHERHOOD AND DANISH SISTERHOOD OF AMERICA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6455 vs 0.5719 (Δ +0.0736)


</details>
5. Danish Sisterhood of America National Board Mtg (Los Angeles, CA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Danish Brotherhood & Danish Sisterhood of America"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.6923 (Δ +0.1259)


</details>
6. Danish Sisterhood of the Americas - 90.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Danish Sisterhood of America National Board Mtg"

**Score Difference:** 0.0503 (5.03 percentage points)

**Key Differentiators:**
- String Similarity: 0.6923 vs 0.8250 (Δ -0.1327)
- Semantic Similarity: 0.5320 vs 0.8532 (Δ -0.3212)


</details>
7. The Dansih Sisterhood of America - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Danish Sisterhood of the Americas"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8532 vs 0.7079 (Δ +0.1453)


</details>
8. Danish Sisterhood of Amercia (Mundelain, IL) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "The Dansih Sisterhood of America"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7079 vs 0.5316 (Δ +0.1763)


</details>
9. The Dansih Sisterhood of America (Arvada, CO) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Danish Sisterhood of Amercia"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5316 vs 0.7079 (Δ -0.1763)


</details>

---

## 25. Brooklyn Comics Club (Brooklyn, NY)

**Query:** `Brooklyn Comics Club` • **Location:** Brooklyn, NY • **Self-Match:** ✅ Found & Filtered

**Top Match:** Brooklyn Comics Club • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.6312 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Brooklyn Comics Club - 100.2% • *Perfect character-for-character match.*
2. Cathedral Club of Brooklyn (Brooklyn, NY) - 93.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Brooklyn Comics Club"

**Score Difference:** 0.0683 (6.83 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8097 (Δ +0.1903)
- Semantic Similarity: 1.0000 vs 0.6648 (Δ +0.3352)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
3. Brooklyn Barbell Club (Brooklyn, NY) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Cathedral Club of Brooklyn"

**Score Difference:** 0.0087 (0.87 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6648 vs 0.7223 (Δ -0.0575)


</details>
4. Brooklyn Wallyball Club (Brooklyn, NY) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Brooklyn Barbell Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7223 vs 0.6463 (Δ +0.0761)


</details>
5. Rotary Club of Brooklyn (Brooklyn, NY) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Brooklyn Wallyball Club"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Brooklyn Book Club Meetup (Brooklyn, NY) - 78.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Rotary Club of Brooklyn"

**Score Difference:** 0.1376 (13.76 percentage points)

**Key Differentiators:**
- String Similarity: 0.8097 vs 0.7361 (Δ +0.0736)
- Semantic Similarity: 0.6225 vs 0.7134 (Δ -0.0909)


</details>
7. Brooklyn College Diversity Club (Brooklyn, NY) - 78.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Brooklyn Book Club Meetup"

**Score Difference:** 0.0021 (0.21 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. South Brooklyn Running Club (Brooklyn, NY) - 78.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Brooklyn College Diversity Club"

**Score Difference:** 0.0044 (0.44 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Brooklyn Bridge Rotary Club (Brooklyn, NY) - 77.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "South Brooklyn Running Club"

**Score Difference:** 0.0058 (0.58 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
10. North Brooklyn Comic Book Club (Brooklyn, NY) - 77.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "Brooklyn Bridge Rotary Club"

**Score Difference:** 0.0049 (0.49 percentage points)

**Key Differentiators:**
- String Similarity: 0.7361 vs 0.6748 (Δ +0.0613)
- Semantic Similarity: 0.6627 vs 0.7855 (Δ -0.1228)


</details>

---

## 26. Global Interagency Security Forum

**Query:** `Global Interagency Security Forum` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Infrastructure Security and Resilience Forum • **Score:** 81.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity (Normalized) | 0.8919 | 30% | 0.2676 |
| Semantic Similarity (Raw) | 4.0410 | - | - |
| **Base Score** | **0.8130** | - | - |
| **FINAL SCORE** | **0.8179** | - | **81.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8179
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• forum, security

**Your Search Also Includes:**
• global, interagency

**Company Name Also Includes:**
• and, infrastructure, resilience

**Match Strength:**
• 40% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7792 (Weight: 70%)
• Semantic Similarity: 0.8919 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Infrastructure Security and Resilience Forum - 81.8% • *Matched via strong semantic/conceptual similarity.*
2. NY Cyber Security Forum - 81.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Infrastructure Security and Resilience Forum"

**Score Difference:** 0.0065 (0.65 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. BITS Security Forum - 80.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "NY Cyber Security Forum"

**Score Difference:** 0.0108 (1.08 percentage points)

**Key Differentiators:**
- String Similarity: 0.7792 vs 0.7083 (Δ +0.0708)
- Semantic Similarity: 0.8704 vs 1.0000 (Δ -0.1296)


</details>
4. Privacy + Security Forum - 79.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "BITS Security Forum"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.7792 (Δ -0.0708)
- Semantic Similarity: 1.0000 vs 0.8303 (Δ +0.1697)


</details>
5. Halifax International Security Forum - 79.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Privacy + Security Forum"

**Score Difference:** 0.0040 (0.40 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Information Security Leadership Forum - 79.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Halifax International Security Forum"

**Score Difference:** 0.0047 (0.47 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Information Security Forum - 79.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Information Security Leadership Forum"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:**
- String Similarity: 0.7792 vs 0.7083 (Δ +0.0708)
- Semantic Similarity: 0.7921 vs 0.9554 (Δ -0.1633)


</details>
8. The Cyber Security Forum Initiative (Baltimore, MD) - 78.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Information Security Forum"

**Score Difference:** 0.0082 (0.82 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.7792 (Δ -0.0708)
- Semantic Similarity: 0.9554 vs 0.7722 (Δ +0.1833)


</details>
9. The Learning Forum Security Council - 78.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "The Cyber Security Forum Initiative"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 27. Lancet Software

**Query:** `Lancet Software` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Lancet Technology (Boston, MA) • **Score:** 77.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8438 | 30% | 0.2531 |
| Semantic Similarity (Raw) | 3.7700 | - | - |
| **Base Score** | **0.7738** | - | - |
| **FINAL SCORE** | **0.7785** | - | **77.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7785
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• lancet

**Your Search Also Includes:**
• software

**Company Name Also Includes:**
• technology

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.8438 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Lancet Technology (Boston, MA) - 77.8% • *Hybrid match based on combined lexical and semantic features.*
2. Quest Software - 76.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Lancet Technology"

**Score Difference:** 0.0106 (1.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Tech Software - 76.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Quest Software"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. FRS Software - 76.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Tech Software"

**Score Difference:** 0.0024 (0.24 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. ET Software - 76.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "FRS Software"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7985 vs 0.6945 (Δ +0.1039)


</details>
6. Software Professionals - 76.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "ET Software"

**Score Difference:** 0.0029 (0.29 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6945 vs 0.7870 (Δ -0.0925)


</details>
7. Jaguar Software (Sullivan, ) - 75.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Software Professionals"

**Score Difference:** 0.0027 (0.27 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. CDT Software - 75.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Jaguar Software"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Riptide Software - 75.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "CDT Software"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 28. Our Lady of the Lakes Catholic Church and School (Miami Lakes, FL)

**Query:** `Our Lady of the Lakes Catholic Church and School` • **Location:** Miami Lakes, FL • **Self-Match:** ✅ Found & Filtered

**Top Match:** Our Lady of the Lakes Catholic Church (Miami Lakes, FL) • **Score:** 92.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8498 | 70% | 0.5949 |
| Semantic Similarity (Normalized) | 0.8340 | 30% | 0.2502 |
| Semantic Similarity (Raw) | 4.4298 | - | - |
| **Base Score** | **0.8451** | - | - |
| Location Match Boost | 100.0000 | 5% max | +5.0000 |
| **FINAL SCORE** | **0.9287** | - | **92.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 100.0000 × 0.05 = 5.0000

Final Score = Base Score + Location Boost = 0.9287
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 100.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
7 word(s) match exactly between your search and this company.

**Matching Words:**
• catholic, church, lady, lakes, of, our, the

**Your Search Also Includes:**
• and, school

**Match Strength:**
• 78% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8498 (Weight: 70%)
• Semantic Similarity: 0.8340 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Our Lady of the Lakes Catholic Church (Miami Lakes, FL) - 92.9% • *High word-for-word overlap.*
2. Our Lady of the Lakes Catholic School (Miami Lakes, FL) - 92.9% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Our Lady of the Lakes Catholic Church"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8340 vs 0.7771 (Δ +0.0569)


</details>
3. Our Lady of The Lakes Catholic Church (Miami, FL) - 90.6% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Our Lady of the Lakes Catholic School"

**Score Difference:** 0.0228 (2.28 percentage points)

**Key Differentiators:**
- Location Score: 100.0000 vs 90.2000 (Δ +9.8000)


</details>
4. Our Lady of the Lakes Catholic School (Miami, FL) - 90.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Our Lady of The Lakes Catholic Church"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Our Lady of the Lakes Catholic School Miami (Miami, FL) - 90.6% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Our Lady of the Lakes Catholic School"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8498 vs 0.9167 (Δ -0.0669)
- Semantic Similarity: 0.7771 vs 0.6554 (Δ +0.1218)


</details>
6. Our Lady of Lourdes Catholic Church & School (Miami, FL) - 90.6% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Our Lady of the Lakes Catholic School Miami"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.9167 vs 0.8594 (Δ +0.0573)


</details>
7. Our Lady of the Lakes Catholic Church (Miramar, FL) - 80.5% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Our Lady of Lourdes Catholic Church & School"

**Score Difference:** 0.1004 (10.04 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6539 vs 0.8340 (Δ -0.1801)
- Location Score: 90.2000 vs 40.0000 (Δ +50.2000)


</details>
8. Our Lady of the Lakes Catholic School (Hialeah, FL) - 80.5% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Our Lady of the Lakes Catholic Church"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8340 vs 0.7771 (Δ +0.0569)


</details>
9. Our Lady of The Lakes Church (Miami Lakes, FL) - 78.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Our Lady of the Lakes Catholic School"

**Score Difference:** 0.0258 (2.58 percentage points)

**Key Differentiators:**
- String Similarity: 0.8498 vs 0.6901 (Δ +0.1597)
- Location Score: 40.0000 vs 100.0000 (Δ -60.0000)


</details>

---

## 29. Broadway Bound International

**Query:** `Broadway Bound International` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Broadway Bound International • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6929 | 30% | 0.2079 |
| Semantic Similarity (Raw) | 4.1892 | - | - |
| **Base Score** | **0.9079** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Broadway Bound International - 100.2% • *Perfect character-for-character match.*
2. Broadway Bound Kids - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Broadway Bound International"

**Score Difference:** 0.0970 (9.70 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8833 (Δ +0.1167)
- Semantic Similarity: 0.6929 vs 0.7976 (Δ -0.1046)


</details>
3. Broadway Bound Kidz - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Broadway Bound Kids"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7976 vs 0.6944 (Δ +0.1032)


</details>
4. Broadway Bound (New Orleans, LA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Broadway Bound Kidz"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8833 vs 0.8182 (Δ +0.0652)
- Semantic Similarity: 0.6944 vs 0.6412 (Δ +0.0532)


</details>
5. Broadway Bound (Merrimack, NH) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Broadway Bound"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Broadway Bound West (Seattle, WA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Broadway Bound"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.8833 (Δ -0.0652)


</details>
7. Broadway Bound Childrens Theatre (Seattle, WA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Broadway Bound West"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8833 vs 0.8030 (Δ +0.0803)


</details>
8. Broadway Bound Dance (Media, PA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Broadway Bound Childrens Theatre"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8030 vs 0.8833 (Δ -0.0803)


</details>
9. BROADWAY BOUND DANCE CENTRE (New Albany, OH) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Broadway Bound Dance"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8833 vs 0.8030 (Δ +0.0803)


</details>

---

## 30. E. H. Wachs

**Query:** `E. H. Wachs` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** E. H. Wachs (Chicago, IL) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8582 | 30% | 0.2575 |
| Semantic Similarity (Raw) | 4.3398 | - | - |
| **Base Score** | **0.9575** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. E. H. Wachs (Chicago, IL) - 100.2% • *Perfect character-for-character match.*
2. EHW - 75.0% • *Matched based on generated acronym 'EHW'.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "E. H. Wachs"

**Score Difference:** 0.2524 (25.24 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8582 vs 1.0000 (Δ -0.1418)
- Acronym Fidelity: 0.0000 vs 1.0000 (Δ -1.0000)


</details>
3. E H Smith - 66.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "EHW"

**Score Difference:** 0.0853 (8.53 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.6625 (Δ +0.3375)
- Semantic Similarity: 1.0000 vs 0.6564 (Δ +0.3436)
- Acronym Fidelity: 1.0000 vs 0.0000 (Δ +1.0000)


</details>
4. Wachs Services - 62.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "E H Smith"

**Score Difference:** 0.0412 (4.12 percentage points)

**Key Differentiators:**
- String Similarity: 0.6625 vs 0.4798 (Δ +0.1827)
- Semantic Similarity: 0.6564 vs 0.9463 (Δ -0.2899)


</details>
5. Elen Wachs - 62.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Wachs Services"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. H-E Parts - 61.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Elen Wachs"

**Score Difference:** 0.0063 (0.63 percentage points)

**Key Differentiators:**
- String Similarity: 0.4798 vs 0.6625 (Δ -0.1827)
- Semantic Similarity: 0.9424 vs 0.4952 (Δ +0.4472)


</details>
7. E.H. Wachs (Lincolnshire, IL) - 60.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "H-E Parts"

**Score Difference:** 0.0102 (1.02 percentage points)

**Key Differentiators:**
- String Similarity: 0.6625 vs 0.4924 (Δ +0.1701)
- Semantic Similarity: 0.4952 vs 0.8582 (Δ -0.3630)


</details>
8. Wachs Water Services (Buffalo Grove, IL) - 53.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "E.H. Wachs"

**Score Difference:** 0.0719 (7.19 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8582 vs 0.5373 (Δ +0.3209)


</details>
9. Wachs Wedding Room Block - 53.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Wachs Water Services"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5373 vs 0.6420 (Δ -0.1047)


</details>

---

## 31. Marine Corps Fox 2/5

**Query:** `Marine Corps Fox 2/5` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** MARINE CORPS BASE CAMP • **Score:** 80.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity (Normalized) | 0.8602 | 30% | 0.2581 |
| Semantic Similarity (Raw) | 4.4003 | - | - |
| **Base Score** | **0.8035** | - | - |
| **FINAL SCORE** | **0.8084** | - | **80.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8084
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• corps, marine

**Your Search Also Includes:**
• 2/5, fox

**Company Name Also Includes:**
• base, camp

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7792 (Weight: 70%)
• Semantic Similarity: 0.8602 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. MARINE CORPS BASE CAMP - 80.8% • *Matched via strong semantic/conceptual similarity.*
2. MARINE CORPS COMMUNITY SERVICE - 80.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "MARINE CORPS BASE CAMP"

**Score Difference:** 0.0011 (0.11 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Navy Marine Corps Ball - 80.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "MARINE CORPS COMMUNITY SERVICE"

**Score Difference:** 0.0043 (0.43 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. MARINE CORPS LOGISTICS BASE - 80.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Navy Marine Corps Ball"

**Score Difference:** 0.0027 (0.27 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. 25th US Marine Corps - 80.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "MARINE CORPS LOGISTICS BASE"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. MARINE CORPS MILITARY REUNION - 79.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "25th US Marine Corps"

**Score Difference:** 0.0062 (0.62 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Marine Corps Air Transport - 79.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "MARINE CORPS MILITARY REUNION"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. MARINE CORPS AIR GROUND - 79.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Marine Corps Air Transport"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. US Marine Corps Training - 79.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "MARINE CORPS AIR GROUND"

**Score Difference:** 0.0023 (0.23 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 32. Fantasia Turistica

**Query:** `Fantasia Turistica` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Fantasia • **Score:** 74.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.6300 | 70% | 0.4410 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.6617 | - | - |
| **Base Score** | **0.7410** | - | - |
| **FINAL SCORE** | **0.7455** | - | **74.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7455
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• fantasia

**Your Search Also Includes:**
• turistica

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.6300 (Weight: 70%)
• Semantic Similarity: 1.0000 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Fantasia - 74.6% • *Matched via strong semantic/conceptual similarity.*
2. Fantasia Travels (London, ) - 72.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Fantasia"

**Score Difference:** 0.0170 (1.70 percentage points)

**Key Differentiators:**
- String Similarity: 0.6300 vs 0.7438 (Δ -0.1138)
- Semantic Similarity: 1.0000 vs 0.6782 (Δ +0.3218)


</details>
3. Noreen Fantasia - 72.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Fantasia Travels"

**Score Difference:** 0.0044 (0.44 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Fantasia Travel (Bensalem, PA) - 71.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Noreen Fantasia"

**Score Difference:** 0.0104 (1.04 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Fantasia Accessry (New York, NY) - 71.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Fantasia Travel"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Ferrari Fantasia (Naples, FL) - 70.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Fantasia Accessry"

**Score Difference:** 0.0050 (0.50 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Fantasia Travels (Richmond, VA) - 70.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Ferrari Fantasia"

**Score Difference:** 0.0011 (0.11 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6068 vs 0.6782 (Δ -0.0714)


</details>
8. Fantasia Veneziana (New York, NY) - 69.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Fantasia Travels"

**Score Difference:** 0.0077 (0.77 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6782 vs 0.5779 (Δ +0.1003)


</details>
9. Operadora Turistica - 68.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Fantasia Veneziana"

**Score Difference:** 0.0102 (1.02 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 33. Esoterix

**Query:** `Esoterix` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Esoterix Headquarters • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity (Normalized) | 0.7630 | 30% | 0.2289 |
| Semantic Similarity (Raw) | 4.7814 | - | - |
| **Base Score** | **0.8016** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PREFIX MATCH

**What This Means:**
This company name starts with 'Esoterix' and has additional information added.

**Action Required:**
• This is likely the same company with extra details
• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')
• If yes, use this match

**Why This Happens:**
• Someone entered just the core company name
• Your system has the full legal name
• Common in business databases where legal names include extra terms

**Top 10 Matches:**
1. Esoterix Headquarters - 95.6% • *Direct prefix match (target contains extra trailing words).*
2. Esoterix Integrated Genetics (Hull, MA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Esoterix Headquarters"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.7500 (Δ +0.0682)
- Semantic Similarity: 0.7630 vs 0.5352 (Δ +0.2278)


</details>
3. Esoterix Integrated Genetics (Westborough, MA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Esoterix Integrated Genetics"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Esoterix Genetic Laboratories, LLC (Westborough, MA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Esoterix Integrated Genetics"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5352 vs 0.3914 (Δ +0.1438)


</details>
5. Esoterix Clinical Trials Services (Research Triangle Park, NC) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Esoterix Genetic Laboratories, LLC"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.6923 (Δ +0.0577)


</details>
6. ESOTERIX GENETC LABORATORIES, LLC (Westborough, MA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Esoterix Clinical Trials Services"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.6923 vs 0.7500 (Δ -0.0577)


</details>
7. Centrix - 42.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "ESOTERIX GENETC LABORATORIES, LLC"

**Score Difference:** 0.5274 (52.74 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.3000 (Δ +0.4500)
- Semantic Similarity: 0.3632 vs 0.7193 (Δ -0.3561)


</details>
8. Verix - 42.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Centrix"

**Score Difference:** 0.0062 (0.62 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Netrix - 42.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Verix"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7525 vs 0.6470 (Δ +0.1055)


</details>

---

## 34. Coker Group

**Query:** `Coker Group` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Coker Group • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6928 | 30% | 0.2078 |
| Semantic Similarity (Raw) | 4.4074 | - | - |
| **Base Score** | **0.9078** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Coker Group - 100.2% • *Perfect character-for-character match.*
2. Coker Group (North Canton, OH) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Coker Group"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Coker Group (Arlington, VA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Coker Group"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Coker Group (Smyrna, GA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Coker Group"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Coker Group (Gainesville, GA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Coker Group"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Coker Capital (Charlotte, NC) - 96.2% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Coker Group"

**Score Difference:** 0.0409 (4.09 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8182 (Δ +0.1818)
- Semantic Similarity: 0.6928 vs 0.5601 (Δ +0.1327)


</details>
7. Jackson & Coker (Alpharetta, GA) - 96.2% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Coker Capital"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5601 vs 0.4554 (Δ +0.1047)


</details>
8. Coker Capital - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Jackson & Coker"

**Score Difference:** 0.0024 (0.24 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.4554 vs 0.5601 (Δ -0.1047)


</details>
9. Kristen Coker (San Diego, CA) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Coker Capital"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 35. GILEAD IT

**Query:** `GILEAD IT` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Gilead Productions • **Score:** 77.7%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8405 | 30% | 0.2522 |
| Semantic Similarity (Raw) | 5.7787 | - | - |
| **Base Score** | **0.7728** | - | - |
| **FINAL SCORE** | **0.7775** | - | **77.7%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7775
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• gilead

**Your Search Also Includes:**
• it

**Company Name Also Includes:**
• productions

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.8405 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Gilead Productions - 77.7% • *Hybrid match based on combined lexical and semantic features.*
2. Gilead Sciences - 77.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Gilead Productions"

**Score Difference:** 0.0029 (0.29 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Gilead 1N - 77.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Gilead Sciences"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Gilead Services - 76.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Gilead 1N"

**Score Difference:** 0.0059 (0.59 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. GILEAD MEDICAL - 76.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Gilead Services"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Gilead Media - 75.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "GILEAD MEDICAL"

**Score Difference:** 0.0078 (0.78 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Gilead Finance - 75.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Gilead Media"

**Score Difference:** 0.0021 (0.21 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Baltimore Gilead - 75.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Gilead Finance"

**Score Difference:** 0.0052 (0.52 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Gilead - 75.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Baltimore Gilead"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6300 (Δ +0.1138)
- Semantic Similarity: 0.7499 vs 1.0000 (Δ -0.2501)


</details>

---

## 36. 4143 Affiliate INDA 2016

**Query:** `4143 Affiliate INDA 2016` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** 1528 Affiliate INDA 2016 (Houston, TX) • **Score:** 90.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity (Normalized) | 0.8384 | 30% | 0.2515 |
| Semantic Similarity (Raw) | 4.0353 | - | - |
| **Base Score** | **0.8815** | - | - |
| **FINAL SCORE** | **0.9087** | - | **90.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9087
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• 2016, affiliate, inda

**Your Search Also Includes:**
• 4143

**Company Name Also Includes:**
• 1528

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.9000 (Weight: 70%)
• Semantic Similarity: 0.8384 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. 1528 Affiliate INDA 2016 (Houston, TX) - 90.9% • *High word-for-word overlap.*
2. 1528 Affiliate INDA 2016 (Chesterfield, VA) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "1528 Affiliate INDA 2016"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. 1528 Affiliate INDA 2016 (Osaka, Osaka) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "1528 Affiliate INDA 2016"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. 1528 Affiliate AFA 2016 (Dallas, TX) - 73.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "1528 Affiliate INDA 2016"

**Score Difference:** 0.1683 (16.83 percentage points)

**Key Differentiators:**
- String Similarity: 0.9000 vs 0.7438 (Δ +0.1562)
- Semantic Similarity: 0.8384 vs 0.7069 (Δ +0.1315)


</details>
5. 2016 Google Affiliate - 73.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "1528 Affiliate AFA 2016"

**Score Difference:** 0.0039 (0.39 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.7069 vs 0.8517 (Δ -0.1448)


</details>
6. 1035 Affiliate CASE 2016 (Washington, DC) - 73.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "2016 Google Affiliate"

**Score Difference:** 0.0023 (0.23 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.8517 vs 0.6865 (Δ +0.1653)


</details>
7. 1035 JPMorgan Affiliate 2016 - 73.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "1035 Affiliate CASE 2016"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. AACR AFFILIATE 2016 - 72.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "1035 JPMorgan Affiliate 2016"

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6821 vs 0.8369 (Δ -0.1548)


</details>
9. 1035 Affiliate CASE 2016 (Boston, MA) - 72.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "AACR AFFILIATE 2016"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.8369 vs 0.6865 (Δ +0.1504)


</details>

---

## 37. Pipe and Plant Solutions

**Query:** `Pipe and Plant Solutions` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Pipe & Plant • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8264 | 70% | 0.5785 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.3401 | - | - |
| **Base Score** | **0.8785** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• pipe, plant

**Your Search Also Includes:**
• and, solutions

**Company Name Also Includes:**
• &

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.8264 (Weight: 70%)
• Semantic Similarity: 1.0000 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Pipe & Plant - 90.5% • *Matched via strong semantic/conceptual similarity.*
2. Advanced Pipe Solutions (Findlay, OH) - 75.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Pipe & Plant"

**Score Difference:** 0.1521 (15.21 percentage points)

**Key Differentiators:**
- String Similarity: 0.8264 vs 0.7729 (Δ +0.0535)
- Semantic Similarity: 1.0000 vs 0.6927 (Δ +0.3073)


</details>
3. Infra Pipe Solutions (Mississauga, ON) - 72.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Advanced Pipe Solutions"

**Score Difference:** 0.0324 (3.24 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6927 vs 0.5854 (Δ +0.1073)


</details>
4. Plant Solutions Limited (Debe, ) - 70.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Infra Pipe Solutions"

**Score Difference:** 0.0191 (1.91 percentage points)

**Key Differentiators:**
- String Similarity: 0.7729 vs 0.7159 (Δ +0.0570)
- Semantic Similarity: 0.5854 vs 0.6550 (Δ -0.0696)


</details>
5. PPS - 69.0% • *Matched based on generated acronym 'PPS'.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Plant Solutions Limited"

**Score Difference:** 0.0119 (1.19 percentage points)

**Key Differentiators:**
- String Similarity: 0.7159 vs 1.0000 (Δ -0.2841)
- Semantic Similarity: 0.6550 vs 1.0000 (Δ -0.3450)
- Acronym Fidelity: 0.0000 vs 0.7000 (Δ -0.7000)


</details>
6. TV Pipe Solutions (Boise, ID) - 68.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "PPS"

**Score Difference:** 0.0026 (0.26 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7729 (Δ +0.2271)
- Semantic Similarity: 1.0000 vs 0.4660 (Δ +0.5340)
- Acronym Fidelity: 0.7000 vs 0.0000 (Δ +0.7000)


</details>
7. Plant Operations - 60.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "TV Pipe Solutions"

**Score Difference:** 0.0802 (8.02 percentage points)

**Key Differentiators:**
- String Similarity: 0.7729 vs 0.5038 (Δ +0.2691)
- Semantic Similarity: 0.4660 vs 0.8363 (Δ -0.3703)


</details>
8. Cal Pipe Industries - 60.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Plant Operations"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:**
- String Similarity: 0.5038 vs 0.5542 (Δ -0.0504)
- Semantic Similarity: 0.8363 vs 0.7149 (Δ +0.1214)


</details>
9. Independent Concrete Pipe - 60.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Cal Pipe Industries"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 38. Stephen Rourke

**Query:** `Stephen Rourke` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Rourke Publishing (Vero Beach, FL) • **Score:** 75.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.7715 | 30% | 0.2314 |
| Semantic Similarity (Raw) | 3.6038 | - | - |
| **Base Score** | **0.7521** | - | - |
| **FINAL SCORE** | **0.7593** | - | **75.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7593
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• rourke

**Your Search Also Includes:**
• stephen

**Company Name Also Includes:**
• publishing

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.7715 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Rourke Publishing (Vero Beach, FL) - 75.9% • *Hybrid match based on combined lexical and semantic features.*
2. Stephen Scott - 75.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Rourke Publishing"

**Score Difference:** 0.0080 (0.80 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Stephen Oh - 73.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Stephen Scott"

**Score Difference:** 0.0131 (1.31 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Stephen Lacy - 73.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Stephen Oh"

**Score Difference:** 0.0033 (0.33 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Stephen Michael - 72.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Stephen Lacy"

**Score Difference:** 0.0089 (0.89 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Stephen Madden - 71.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Stephen Michael"

**Score Difference:** 0.0092 (0.92 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Stephen McConnell - 71.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Stephen Madden"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Eliza Stephen - 71.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Stephen McConnell"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Stephen Kent - 71.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Eliza Stephen"

**Score Difference:** 0.0033 (0.33 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 39. MIT Initiative on the Digital Economy (Cambridge, MA)

**Query:** `MIT Initiative on the Digital Economy` • **Location:** Cambridge, MA • **Self-Match:** ✅ Found & Filtered

**Top Match:** MIT Initiative on the Digital Economy • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.2696 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. MIT Initiative on the Digital Economy - 100.2% • *Perfect character-for-character match.*
2. MIT Office of Digital Learning (Cambridge, MA) - 62.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "MIT Initiative on the Digital Economy"

**Score Difference:** 0.3775 (37.75 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.5250 (Δ +0.4750)
- Semantic Similarity: 1.0000 vs 0.5322 (Δ +0.4678)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
3. MIT Office of Digital Learning (formerly OEIT) (Cambridge, MA) - 61.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "MIT Office of Digital Learning"

**Score Difference:** 0.0072 (0.72 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. MIT Energy Initiative (Cambridge, MA) - 61.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "MIT Office of Digital Learning (formerly OEIT)"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5023 vs 0.5823 (Δ -0.0801)


</details>
5. MIT Lean Advancement Initiative (Cambridge, MA) - 61.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "MIT Energy Initiative"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5823 vs 0.4926 (Δ +0.0897)


</details>
6. MIT Information Services and Technology (Cambridge, MA) - 50.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "MIT Lean Advancement Initiative"

**Score Difference:** 0.1069 (10.69 percentage points)

**Key Differentiators:**
- String Similarity: 0.5250 vs 0.2936 (Δ +0.2314)
- Semantic Similarity: 0.4926 vs 0.5903 (Δ -0.0977)


</details>
7. MIT Office of Digital Learning (Concord, MA) - 50.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "MIT Information Services and Technology"

**Score Difference:** 0.0035 (0.35 percentage points)

**Key Differentiators:**
- String Similarity: 0.2936 vs 0.5250 (Δ -0.2314)
- Semantic Similarity: 0.5903 vs 0.5322 (Δ +0.0581)
- Location Score: 100.0000 vs 40.0000 (Δ +60.0000)


</details>
8. MIT Information Systems (Cambridge, MA) - 48.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "MIT Office of Digital Learning"

**Score Difference:** 0.0166 (1.66 percentage points)

**Key Differentiators:**
- String Similarity: 0.5250 vs 0.2558 (Δ +0.2692)
- Semantic Similarity: 0.5322 vs 0.5952 (Δ -0.0631)
- Location Score: 40.0000 vs 100.0000 (Δ -60.0000)


</details>
9. MIT Education Studies Program (Cambridge, MA) - 47.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "MIT Information Systems"

**Score Difference:** 0.0091 (0.91 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
10. MIT Office of Digital Learning - 47.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "MIT Education Studies Program"

**Score Difference:** 0.0037 (0.37 percentage points)

**Key Differentiators:**
- String Similarity: 0.2554 vs 0.5250 (Δ -0.2696)
- Location Score: 100.0000 vs 0.0000 (Δ +100.0000)


</details>

---

## 40. Urx Community USA

**Query:** `Urx Community USA` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Community Alliance USA • **Score:** 71.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity (Normalized) | 0.5628 | 30% | 0.1688 |
| Semantic Similarity (Raw) | 3.4157 | - | - |
| **Base Score** | **0.7099** | - | - |
| **FINAL SCORE** | **0.7142** | - | **71.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7142
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• community, usa

**Your Search Also Includes:**
• urx

**Company Name Also Includes:**
• alliance

**Match Strength:**
• 67% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.7729 (Weight: 70%)
• Semantic Similarity: 0.5628 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Community Alliance USA - 71.4% • *Hybrid match based on combined lexical and semantic features.*
2. Community Brands USA - 71.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Community Alliance USA"

**Score Difference:** 0.0003 (0.03 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. UiPath Community USA - 71.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Community Brands USA"

**Score Difference:** 0.0041 (0.41 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Community Events LLC USA - 69.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "UiPath Community USA"

**Score Difference:** 0.0138 (1.38 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Community Bridges Inc USA - 69.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Community Events LLC USA"

**Score Difference:** 0.0020 (0.20 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Umoja Community USA - 69.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Community Bridges Inc USA"

**Score Difference:** 0.0025 (0.25 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Community Development Society USA - 67.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Umoja Community USA"

**Score Difference:** 0.0184 (1.84 percentage points)

**Key Differentiators:**
- String Similarity: 0.7729 vs 0.7027 (Δ +0.0703)
- Semantic Similarity: 0.4879 vs 0.5908 (Δ -0.1029)


</details>
8. Community Information Exchange USA - 67.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Community Development Society USA"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Oregon Community Trees USA - 65.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Community Information Exchange USA"

**Score Difference:** 0.0204 (2.04 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5844 vs 0.5167 (Δ +0.0677)


</details>

---

## 41. Spredfast Engage

**Query:** `Spredfast Engage` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Spredfast Product • **Score:** 79.7%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9037 | 30% | 0.2711 |
| Semantic Similarity (Raw) | 4.1403 | - | - |
| **Base Score** | **0.7917** | - | - |
| **FINAL SCORE** | **0.7965** | - | **79.7%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7965
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• spredfast

**Your Search Also Includes:**
• engage

**Company Name Also Includes:**
• product

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.9037 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Spredfast Product - 79.7% • *Matched via strong semantic/conceptual similarity.*
2. Spredfast Events (New York, NY) - 72.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Spredfast Product"

**Score Difference:** 0.0672 (6.72 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9037 vs 0.6810 (Δ +0.2226)


</details>
3. Spredfast Events (Austin, TX) - 72.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Spredfast Events"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. 8 Engage - 69.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Spredfast Events"

**Score Difference:** 0.0302 (3.02 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6810 vs 0.5786 (Δ +0.1024)


</details>
5. Engage Point - 68.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "8 Engage"

**Score Difference:** 0.0091 (0.91 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. engage fi - 67.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Engage Point"

**Score Difference:** 0.0100 (1.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. CU Engage - 67.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "engage fi"

**Score Difference:** 0.0089 (0.89 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Engage R+D - 66.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "CU Engage"

**Score Difference:** 0.0078 (0.78 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Engage 360 - 66.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Engage R+D"

**Score Difference:** 0.0002 (0.02 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 42. City of Dallas-Parks & Recreation (Dallas, TX)

**Query:** `City of Dallas-Parks & Recreation` • **Location:** Dallas, TX • **Self-Match:** ✅ Found & Filtered

**Top Match:** Dallas Parks and Recreation Dept (Dallas, TX) • **Score:** 92.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.2681 | - | - |
| **Base Score** | **0.8906** | - | - |
| Location Match Boost | 100.0000 | 5% max | +5.0000 |
| **FINAL SCORE** | **0.9255** | - | **92.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 100.0000 × 0.05 = 5.0000

Final Score = Base Score + Location Boost = 0.9255
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection
- **Location Match (EXCELLENT):** 100.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• recreation

**Your Search Also Includes:**
• &, city, dallas-parks, of

**Company Name Also Includes:**
• and, dallas, dept, parks

**Match Strength:**
• 20% word overlap
• This is a WEAK match - may be coincidental
• Action: Verify carefully before using

**Score Breakdown:**
• Lexical Similarity: 0.8438 (Weight: 70%)
• Semantic Similarity: 1.0000 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Dallas Parks and Recreation Dept (Dallas, TX) - 92.5% • *High word-for-word overlap.*
2. Dallas Parks and Recreation Department (Dallas, TX) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Dallas Parks and Recreation Dept"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. City of Dallas Park & Recreation (Dallas, TX) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Dallas Parks and Recreation Department"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. PARKS FOR DOWNTOWN DALLAS (Dallas, TX) - 85.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "City of Dallas Park & Recreation"

**Score Difference:** 0.0697 (6.97 percentage points)

**Key Differentiators:**
- String Similarity: 0.8438 vs 0.7438 (Δ +0.1000)


</details>
5. DALLAS PARK & RECREATION DEPARTMENT (Dallas, TX) - 84.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "PARKS FOR DOWNTOWN DALLAS"

**Score Difference:** 0.0079 (0.79 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. City Career Fair Dallas (Dallas, TX) - 81.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "DALLAS PARK & RECREATION DEPARTMENT"

**Score Difference:** 0.0326 (3.26 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9435 vs 0.8086 (Δ +0.1350)


</details>
7. Dallas Parks Foundation (Dallas, TX) - 79.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "City Career Fair Dallas"

**Score Difference:** 0.0212 (2.12 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.8086 vs 0.8785 (Δ -0.0700)


</details>
8. CITY OF DALLAS POLICE DEPARTMENT (Dallas, TX) - 78.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Dallas Parks Foundation"

**Score Difference:** 0.0057 (0.57 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.8785 vs 0.6971 (Δ +0.1814)


</details>
9. City Year Dallas (Dallas, TX) - 78.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "CITY OF DALLAS POLICE DEPARTMENT"

**Score Difference:** 0.0081 (0.81 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6971 vs 0.8108 (Δ -0.1137)


</details>

---

## 43. Kai Pono Builders, Inc. (Honolulu, HI)

**Query:** `Kai Pono Builders, Inc.` • **Location:** Honolulu, HI • **Self-Match:** ✅ Found & Filtered

**Top Match:** Kai Pono Builders, Inc. (Kamuela, HI) • **Score:** 102.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8623 | 30% | 0.2587 |
| Semantic Similarity (Raw) | 4.0814 | - | - |
| **Base Score** | **0.9587** | - | - |
| Location Match Boost | 40.0000 | 5% max | +2.0000 |
| **FINAL SCORE** | **1.0224** | - | **102.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0224
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Kai Pono Builders, Inc. (Kamuela, HI) - 102.2% • *Perfect character-for-character match.*
2. Kai Pono Builders (Honolulu, HI) - 97.1% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Kai Pono Builders, Inc."

**Score Difference:** 0.0516 (5.16 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.9375 (Δ +0.0625)
- Semantic Similarity: 0.8623 vs 1.0000 (Δ -0.1377)
- Location Score: 40.0000 vs 100.0000 (Δ -60.0000)


</details>
3. Pono Kai Resort (Kapaau, HI) - 67.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Kai Pono Builders"

**Score Difference:** 0.2921 (29.21 percentage points)

**Key Differentiators:**
- String Similarity: 0.9375 vs 0.7729 (Δ +0.1646)
- Semantic Similarity: 1.0000 vs 0.6725 (Δ +0.3275)
- Location Score: 100.0000 vs 40.0000 (Δ +60.0000)


</details>
4. Pono Kai (Kapaa, HI) - 65.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Pono Kai Resort"

**Score Difference:** 0.0221 (2.21 percentage points)

**Key Differentiators:**
- String Similarity: 0.7729 vs 0.7027 (Δ +0.0703)
- Semantic Similarity: 0.6725 vs 0.7448 (Δ -0.0724)


</details>
5. NA Kama Kai (Honolulu, HI) - 64.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Pono Kai"

**Score Difference:** 0.0096 (0.96 percentage points)

**Key Differentiators:**
- String Similarity: 0.7027 vs 0.5146 (Δ +0.1881)
- Semantic Similarity: 0.7448 vs 0.6396 (Δ +0.1052)
- Location Score: 40.0000 vs 100.0000 (Δ -60.0000)


</details>
6. Kai Hawaii Company (Honolulu, HI) - 63.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "NA Kama Kai"

**Score Difference:** 0.0124 (1.24 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6396 vs 0.7056 (Δ -0.0660)


</details>
7. ASSOCIATED BUILDERS, INC. (Honolulu, HI) - 63.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Kai Hawaii Company"

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7056 vs 0.6180 (Δ +0.0876)


</details>
8. KAI Hawaii (Honolulu, HI) - 62.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "ASSOCIATED BUILDERS, INC."

**Score Difference:** 0.0137 (1.37 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Associated Builders & Contractors Hawaii (Honolulu, HI) - 61.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "KAI Hawaii"

**Score Difference:** 0.0066 (0.66 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
10. Jac Builders, Incorporated (Honolulu, HI) - 61.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "Associated Builders & Contractors Hawaii"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 44. MUSICFIRST COALITION

**Query:** `MUSICFIRST COALITION` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Wind Coalition • **Score:** 73.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.6832 | 30% | 0.2050 |
| Semantic Similarity (Raw) | 3.5596 | - | - |
| **Base Score** | **0.7256** | - | - |
| **FINAL SCORE** | **0.7300** | - | **73.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7300
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• coalition

**Your Search Also Includes:**
• musicfirst

**Company Name Also Includes:**
• wind

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.6832 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Wind Coalition - 73.0% • *Hybrid match based on combined lexical and semantic features.*
2. Unconventional Coalition - 72.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Wind Coalition"

**Score Difference:** 0.0035 (0.35 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Human Coalition - 71.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Unconventional Coalition"

**Score Difference:** 0.0123 (1.23 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Founders Coalition - 71.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Human Coalition"

**Score Difference:** 0.0043 (0.43 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Tech Coalition - 70.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Founders Coalition"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Union Coalition - 70.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Tech Coalition"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Data Coalition - 70.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Union Coalition"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. District New Music Coalition - 69.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Data Coalition"

**Score Difference:** 0.0025 (0.25 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6198 (Δ +0.1240)
- Semantic Similarity: 0.5898 vs 0.8706 (Δ -0.2808)


</details>
9. Coalition International - 69.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "District New Music Coalition"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:**
- String Similarity: 0.6198 vs 0.7438 (Δ -0.1240)
- Semantic Similarity: 0.8706 vs 0.5799 (Δ +0.2907)


</details>

---

## 45. Frontier Power Products

**Query:** `Frontier Power Products` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Worldwide Power Products • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity (Normalized) | 0.8671 | 30% | 0.2601 |
| Semantic Similarity (Raw) | 4.4807 | - | - |
| **Base Score** | **0.8269** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• power, products

**Your Search Also Includes:**
• frontier

**Company Name Also Includes:**
• worldwide

**Match Strength:**
• 67% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8097 (Weight: 70%)
• Semantic Similarity: 0.8671 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Worldwide Power Products - 90.5% • *Matched via strong semantic/conceptual similarity.*
2. Power Management Products - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Worldwide Power Products"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8671 vs 0.7555 (Δ +0.1116)


</details>
3. Western Power Products Inc - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Power Management Products"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Zenith Power Products LLC (Bristol, VA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Western Power Products Inc"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7516 vs 0.7000 (Δ +0.0516)


</details>
5. Frontier Business Products (Denver, CO) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Zenith Power Products LLC"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Advanced Power Products (Orlando, FL) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Frontier Business Products"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Residential and Power Products - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Advanced Power Products"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Pacific Power Products (Tacoma, WA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Residential and Power Products"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6599 vs 0.6049 (Δ +0.0550)


</details>
9. Frontier Natural Products (Norway, IA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Pacific Power Products"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 46. 1960

**Query:** `1960` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** 1960 (Tenafly, NJ) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.9894 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. 1960 (Tenafly, NJ) - 100.2% • *Perfect character-for-character match.*
2. 1960 (Milwaukee, WI) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "1960"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. 1960 (Hayward, CA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "1960"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. 1960 Family Practice (Houston, TX) - 96.2% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "1960"

**Score Difference:** 0.0409 (4.09 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7500 (Δ +0.2500)
- Semantic Similarity: 1.0000 vs 0.4402 (Δ +0.5598)


</details>
5. District 1960 (Houston, TX) - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "1960 Family Practice"

**Score Difference:** 0.0058 (0.58 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.8182 (Δ -0.0682)


</details>
6. PAGE CLASS OF 1960 (WASHINGTON, DC) - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "District 1960"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.7500 (Δ +0.0682)


</details>
7. Playhouse 1960 (Houston, TX) - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "PAGE CLASS OF 1960"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.8182 (Δ -0.0682)


</details>
8. Miami Beach Class 1960 - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Playhouse 1960"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.6923 (Δ +0.1259)


</details>
9. 1960 Hope Center (Houston, TX) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Miami Beach Class 1960"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.6923 vs 0.7500 (Δ -0.0577)


</details>

---

## 47. Pacific Northwest Diabetes Research Inst (Seattle, WA)

**Query:** `Pacific Northwest Diabetes Research Inst` • **Location:** Seattle, WA • **Self-Match:** ✅ Found & Filtered

**Top Match:** Pacific Northwest Diabetes Research (Seattle, WA) • **Score:** 88.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7773 | 70% | 0.5441 |
| Semantic Similarity (Normalized) | 0.9997 | 30% | 0.2999 |
| Semantic Similarity (Raw) | 4.9265 | - | - |
| **Base Score** | **0.8440** | - | - |
| Location Match Boost | 100.0000 | 5% max | +5.0000 |
| **FINAL SCORE** | **0.8803** | - | **88.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 100.0000 × 0.05 = 5.0000

Final Score = Base Score + Location Boost = 0.8803
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection
- **Location Match (EXCELLENT):** 100.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
4 word(s) match exactly between your search and this company.

**Matching Words:**
• diabetes, northwest, pacific, research

**Your Search Also Includes:**
• inst

**Match Strength:**
• 80% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.7773 (Weight: 70%)
• Semantic Similarity: 0.9997 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Pacific Northwest Diabetes Research (Seattle, WA) - 88.0% • *High word-for-word overlap.*
2. Pacific Northwest Research Institute (Seattle, WA) - 74.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Pacific Northwest Diabetes Research"

**Score Difference:** 0.1399 (13.99 percentage points)

**Key Differentiators:**
- String Similarity: 0.7773 vs 0.7118 (Δ +0.0655)
- Semantic Similarity: 0.9997 vs 0.5740 (Δ +0.4257)


</details>
3. Pacific Northwest Gastroenterology Society (Seattle, WA) - 61.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Pacific Northwest Research Institute"

**Score Difference:** 0.1284 (12.84 percentage points)

**Key Differentiators:**
- String Similarity: 0.7118 vs 0.5100 (Δ +0.2018)
- Semantic Similarity: 0.5740 vs 0.5142 (Δ +0.0599)


</details>
4. Diabetes Research - 57.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Pacific Northwest Gastroenterology Society"

**Score Difference:** 0.0382 (3.82 percentage points)

**Key Differentiators:**
- String Similarity: 0.5100 vs 0.5885 (Δ -0.0785)
- Semantic Similarity: 0.5142 vs 1.0000 (Δ -0.4858)
- Location Score: 100.0000 vs 0.0000 (Δ +100.0000)


</details>
5. Pacific Northwest Research Foundation - 56.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Diabetes Research"

**Score Difference:** 0.0071 (0.71 percentage points)

**Key Differentiators:**
- String Similarity: 0.5885 vs 0.7118 (Δ -0.1234)
- Semantic Similarity: 1.0000 vs 0.6829 (Δ +0.3171)


</details>
6. DIABETES RESEARCH INST FND (Hollywood, FL) - 55.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Pacific Northwest Research Foundation"

**Score Difference:** 0.0133 (1.33 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6829 vs 0.6281 (Δ +0.0548)


</details>
7. Pacific Northwest Research Institite - 54.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "DIABETES RESEARCH INST FND"

**Score Difference:** 0.0077 (0.77 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. GHNS62F3GFQ, Diabetes Pacific Northwest District Meeting - 54.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Pacific Northwest Research Institite"

**Score Difference:** 0.0056 (0.56 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Diabetes Research (Washington, DC) - 53.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "GHNS62F3GFQ, Diabetes Pacific Northwest District Meeting"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:**
- String Similarity: 0.7118 vs 0.5885 (Δ +0.1234)
- Semantic Similarity: 0.5730 vs 1.0000 (Δ -0.4270)


</details>

---

## 48. Mentors & Mentees

**Query:** `Mentors & Mentees` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** TRUE Mentors • **Score:** 76.1%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.7847 | 30% | 0.2354 |
| Semantic Similarity (Raw) | 4.6289 | - | - |
| **Base Score** | **0.7560** | - | - |
| **FINAL SCORE** | **0.7606** | - | **76.1%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7606
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• mentors

**Your Search Also Includes:**
• &, mentees

**Company Name Also Includes:**
• true

**Match Strength:**
• 33% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.7847 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. TRUE Mentors - 76.1% • *Hybrid match based on combined lexical and semantic features.*
2. 3 Mentors - 75.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "TRUE Mentors"

**Score Difference:** 0.0041 (0.41 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. SCORE Mentors - 74.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "3 Mentors"

**Score Difference:** 0.0144 (1.44 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Green Mentors - 73.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "SCORE Mentors"

**Score Difference:** 0.0061 (0.61 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Oregon Mentors - 73.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Green Mentors"

**Score Difference:** 0.0023 (0.23 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. 3-Mentors - 73.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Oregon Mentors"

**Score Difference:** 0.0025 (0.25 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. 3-Mentors, Inc. - 69.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "3-Mentors"

**Score Difference:** 0.0347 (3.47 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6870 vs 0.5721 (Δ +0.1150)


</details>
8. The Marketing Mentors - 69.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "3-Mentors, Inc."

**Score Difference:** 0.0020 (0.20 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. CDL Mentors - 69.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "The Marketing Mentors"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 49. NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46

**Query:** `NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06 • **Score:** 69.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.5931 | 70% | 0.4151 |
| Semantic Similarity (Normalized) | 0.9302 | 30% | 0.2791 |
| Semantic Similarity (Raw) | 4.3577 | - | - |
| **Base Score** | **0.6942** | - | - |
| **FINAL SCORE** | **0.6984** | - | **69.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.6984
```

### Component Analysis

- **String Similarity (FAIR):** Some lexical similarity - partial word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• conference, fall, nala

**Your Search Also Includes:**
• 02-29-24, 12:03:46, 2024, m01709226216947

**Company Name Also Includes:**
• 01-24-23, 09:04:06, 2023, m01674569043113

**Match Strength:**
• 43% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.5931 (Weight: 70%)
• Semantic Similarity: 0.9302 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06 - 69.8% • *Matched via strong semantic/conceptual similarity.*
2. NaLA 2023 fall conference M01674661235470 01-25-23 10:40:54 - 67.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06"

**Score Difference:** 0.0225 (2.25 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. 2024 CPF Conference - CRDF Global M01712246836200 04-04-24 12:07:23 - 52.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "NaLA 2023 fall conference M01674661235470 01-25-23 10:40:54"

**Score Difference:** 0.1537 (15.37 percentage points)

**Key Differentiators:**
- String Similarity: 0.5542 vs 0.4750 (Δ +0.0792)
- Semantic Similarity: 0.9464 vs 0.6218 (Δ +0.3246)


</details>
4. TCN Worldwide 2025 Fall Conference M01712007730091 04-01-24 17:42:23 - 51.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "2024 CPF Conference - CRDF Global M01712246836200 04-04-24 12:07:23"

**Score Difference:** 0.0037 (0.37 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6218 vs 0.7103 (Δ -0.0885)


</details>
5. AE Events Corporate Conference M01724936557111 08-29-24 09:02:41 - 51.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "TCN Worldwide 2025 Fall Conference M01712007730091 04-01-24 17:42:23"

**Score Difference:** 0.0088 (0.88 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7103 vs 0.5804 (Δ +0.1299)


</details>
6. General Contractors 2024 Conference M01712000754964 04-01-24 15:46:00 - 51.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "AE Events Corporate Conference M01724936557111 08-29-24 09:02:41"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Request for Availability of a Conference Room M01733155625999 12-02-24 11:07:12 - 50.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "General Contractors 2024 Conference M01712000754964 04-01-24 15:46:00"

**Score Difference:** 0.0080 (0.80 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5799 vs 0.6542 (Δ -0.0743)


</details>
8. Fall Exchange 2024 M01704922221245 01-10-24 16:30:36 - 49.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Request for Availability of a Conference Room M01733155625999 12-02-24 11:07:12"

**Score Difference:** 0.0056 (0.56 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Region II Conference M01706895307768 02-02-24 12:35:11 - 48.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Fall Exchange 2024 M01704922221245 01-10-24 16:30:36"

**Score Difference:** 0.0120 (1.20 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 50. Donnelley Work Session

**Query:** `Donnelley Work Session` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** SMDS Work Session • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity (Normalized) | 0.5326 | 30% | 0.1598 |
| Semantic Similarity (Raw) | 3.2162 | - | - |
| **Base Score** | **0.7266** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• session, work

**Your Search Also Includes:**
• donnelley

**Company Name Also Includes:**
• smds

**Match Strength:**
• 67% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8097 (Weight: 70%)
• Semantic Similarity: 0.5326 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. SMDS Work Session - 90.5% • *Hybrid match based on combined lexical and semantic features.*
2. DWS - 75.0% • *Matched based on generated acronym 'DWS'.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "SMDS Work Session"

**Score Difference:** 0.1555 (15.55 percentage points)

**Key Differentiators:**
- String Similarity: 0.8097 vs 1.0000 (Δ -0.1903)
- Semantic Similarity: 0.5326 vs 1.0000 (Δ -0.4674)
- Acronym Fidelity: 0.0000 vs 1.0000 (Δ -1.0000)


</details>
3. Donnelley Financial Solutions - 54.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "DWS"

**Score Difference:** 0.2009 (20.09 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.5278 (Δ +0.4722)
- Semantic Similarity: 1.0000 vs 0.5879 (Δ +0.4121)
- Acronym Fidelity: 1.0000 vs 0.0000 (Δ +1.0000)


</details>
4. Professional Development Session - 54.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Donnelley Financial Solutions"

**Score Difference:** 0.0070 (0.70 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. RH Donnelley Headquarters - 53.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Professional Development Session"

**Score Difference:** 0.0049 (0.49 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Session One - 53.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "RH Donnelley Headquarters"

**Score Difference:** 0.0015 (0.15 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5484 vs 0.6552 (Δ -0.1068)


</details>
7. Executive Breakfast Session - 53.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Session One"

**Score Difference:** 0.0042 (0.42 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6552 vs 0.5295 (Δ +0.1257)


</details>
8. Session M - 53.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Executive Breakfast Session"

**Score Difference:** 0.0008 (0.08 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5295 vs 0.6388 (Δ -0.1093)


</details>
9. Strategy Session - 52.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Session M"

**Score Difference:** 0.0026 (0.26 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 51. North Shore Senior Center

**Query:** `North Shore Senior Center` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** North Shore Senior Center • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6961 | 30% | 0.2088 |
| Semantic Similarity (Raw) | 4.2566 | - | - |
| **Base Score** | **0.9088** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. North Shore Senior Center - 100.2% • *Perfect character-for-character match.*
2. South Shore Cultural Center (Chicago, IL) - 91.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "North Shore Senior Center"

**Score Difference:** 0.0883 (8.83 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8500 (Δ +0.1500)
- Semantic Similarity: 0.6961 vs 0.5007 (Δ +0.1954)


</details>
3. North Shore Elder Services (Danvers, MA) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "South Shore Cultural Center"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5007 vs 0.5554 (Δ -0.0547)


</details>
4. National Institute Senior Center (Jamaica, NY) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "North Shore Elder Services"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. North Carolina Solar Center (Raleigh, NC) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "National Institute Senior Center"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Coastal North Town Center - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "North Carolina Solar Center"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.4863 vs 0.7606 (Δ -0.2743)


</details>
7. North SHore Community College - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Coastal North Town Center"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. North Shore Community Bank - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "North SHore Community College"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Glen Cove Senior Center - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "North Shore Community Bank"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 52. Singles Who Like Food & Fun

**Query:** `Singles Who Like Food & Fun` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Fun Asian Singles • **Score:** 61.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.4675 | 70% | 0.3273 |
| Semantic Similarity (Normalized) | 0.9302 | 30% | 0.2791 |
| Semantic Similarity (Raw) | 4.3279 | - | - |
| **Base Score** | **0.6063** | - | - |
| **FINAL SCORE** | **0.6100** | - | **61.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.6100
```

### Component Analysis

- **String Similarity (FAIR):** Some lexical similarity - partial word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• fun, singles

**Your Search Also Includes:**
• &, food, like, who

**Company Name Also Includes:**
• asian

**Match Strength:**
• 33% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.4675 (Weight: 70%)
• Semantic Similarity: 0.9302 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Fun Asian Singles - 61.0% • *Matched via strong semantic/conceptual similarity.*
2. American Singles Who Love Asian - 58.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Fun Asian Singles"

**Score Difference:** 0.0210 (2.10 percentage points)

**Key Differentiators:**
- String Similarity: 0.4675 vs 0.5610 (Δ -0.0935)
- Semantic Similarity: 0.9302 vs 0.6425 (Δ +0.2878)


</details>
3. Fun Social Singles 35+ - 58.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "American Singles Who Love Asian"

**Score Difference:** 0.0076 (0.76 percentage points)

**Key Differentiators:**
- String Similarity: 0.5610 vs 0.5100 (Δ +0.0510)
- Semantic Similarity: 0.6425 vs 0.7364 (Δ -0.0939)


</details>
4. LGBT Food N' Fun Social Group (Chicago, IL) - 57.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Fun Social Singles 35+"

**Score Difference:** 0.0046 (0.46 percentage points)

**Key Differentiators:**
- String Similarity: 0.5100 vs 0.5610 (Δ -0.0510)
- Semantic Similarity: 0.7364 vs 0.6022 (Δ +0.1343)


</details>
5. Fun and Awesome Adventures for Singles and Couples - 56.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "LGBT Food N' Fun Social Group"

**Score Difference:** 0.0141 (1.41 percentage points)

**Key Differentiators:**
- String Similarity: 0.5610 vs 0.5100 (Δ +0.0510)
- Semantic Similarity: 0.6022 vs 0.6744 (Δ -0.0722)


</details>
6. Lesbians who love literature & food (Portland, OR) - 56.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Fun and Awesome Adventures for Singles and Couples"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:**
- String Similarity: 0.5100 vs 0.5610 (Δ -0.0510)
- Semantic Similarity: 0.6744 vs 0.5494 (Δ +0.1249)


</details>
7. NYC SINGLES FUN EVENTS - 56.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Lesbians who love literature & food"

**Score Difference:** 0.0011 (0.11 percentage points)

**Key Differentiators:**
- String Similarity: 0.5610 vs 0.5100 (Δ +0.0510)
- Semantic Similarity: 0.5494 vs 0.6647 (Δ -0.1152)


</details>
8. Food Fun & Fellowship - 55.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "NYC SINGLES FUN EVENTS"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6647 vs 0.7625 (Δ -0.0978)


</details>
9. Singles Who Dance - 55.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Food Fun & Fellowship"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 53. Zen Meetings & Events

**Query:** `Zen Meetings & Events` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Zen Meetings & Events (Langley, SL3 6EZ) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8655 | 30% | 0.2596 |
| Semantic Similarity (Raw) | 5.2728 | - | - |
| **Base Score** | **0.9596** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Zen Meetings & Events (Langley, SL3 6EZ) - 100.4% • *Perfect character-for-character match.*
2. Zen Meetings & Events - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Zen Meetings & Events"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Zen Meetings & Events (Coulsdon, Surrey) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Zen Meetings & Events"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Zen Meetings & Events (Slough, Slough) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Zen Meetings & Events"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Zen Events México - 78.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Zen Meetings & Events"

**Score Difference:** 0.2223 (22.23 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7729 (Δ +0.2271)
- Semantic Similarity: 0.8655 vs 0.7814 (Δ +0.0841)


</details>
6. UHG Meetings & Events - 73.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Zen Events México"

**Score Difference:** 0.0426 (4.26 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7814 vs 0.6403 (Δ +0.1411)


</details>
7. Exclusive Meetings Events - 73.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "UHG Meetings & Events"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Elements Meetings Events - 73.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Exclusive Meetings Events"

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Strategic Meetings & Events - 73.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Elements Meetings Events"

**Score Difference:** 0.0021 (0.21 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 54. Chicago South Swim Club (Chicago, IL)

**Query:** `Chicago South Swim Club` • **Location:** Chicago, IL • **Self-Match:** ✅ Found & Filtered

**Top Match:** Maverick Swim Club (Chicago, IL) • **Score:** 79.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity (Normalized) | 0.8054 | 30% | 0.2416 |
| Semantic Similarity (Raw) | 4.5485 | - | - |
| **Base Score** | **0.7374** | - | - |
| Location Match Boost | 100.0000 | 5% max | +5.0000 |
| **FINAL SCORE** | **0.7944** | - | **79.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 100.0000 × 0.05 = 5.0000

Final Score = Base Score + Location Boost = 0.7944
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 100.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• club, swim

**Your Search Also Includes:**
• chicago, south

**Company Name Also Includes:**
• maverick

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7083 (Weight: 70%)
• Semantic Similarity: 0.8054 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Maverick Swim Club (Chicago, IL) - 79.4% • *Hybrid match based on combined lexical and semantic features.*
2. MYST Swim Club (Chicago, IL) - 78.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Maverick Swim Club"

**Score Difference:** 0.0086 (0.86 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Chicago Wolfpack Aquatic Club (Chicago, IL) - 78.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "MYST Swim Club"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.7792 (Δ -0.0708)
- Semantic Similarity: 0.7699 vs 0.5937 (Δ +0.1762)


</details>
4. Chicago City Soccer Club (Chicago, IL) - 78.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Chicago Wolfpack Aquatic Club"

**Score Difference:** 0.0021 (0.21 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Uptown Chicago Tennis Club (Chicago, IL) - 78.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Chicago City Soccer Club"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Chicago Bears Football Club (Chicago, IL) - 78.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Uptown Chicago Tennis Club"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Chicago Yacht Club (Chicago, IL) - 78.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Chicago Bears Football Club"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- String Similarity: 0.7792 vs 0.7083 (Δ +0.0708)
- Semantic Similarity: 0.5930 vs 0.7066 (Δ -0.1136)


</details>
8. Chicago Adventure Travel Club (Chicago, IL) - 77.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Chicago Yacht Club"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.7792 (Δ -0.0708)
- Semantic Similarity: 0.7066 vs 0.5779 (Δ +0.1287)


</details>
9. Dance Lovers Club of Chicago (Chicago, IL) - 77.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Chicago Adventure Travel Club"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 55. Edna, Dabra@SAP.IO

**Query:** `Edna, Dabra@SAP.IO` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Edna Rose • **Score:** 79.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8955 | 30% | 0.2687 |
| Semantic Similarity (Raw) | 3.8702 | - | - |
| **Base Score** | **0.7893** | - | - |
| **FINAL SCORE** | **0.7941** | - | **79.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7941
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
LINGUISTIC MATCH

**What This Means:**
The names look different but are linguistically related.

**Key Relationships Found:**
• 'edna,' ↔ 'edna' (abbreviation/expansion)

**Details:**
• 'edna' is abbreviation of 'edna,'

**Real-World Scenario:**
• Your system has the short form 'edna' but someone wrote 'edna,'

**Action Required:**
• Verify if this variation makes sense
• Likely the same company

**Top 10 Matches:**
1. Edna Rose - 79.4% • *Matched via strong semantic/conceptual similarity.*
2. Edna ISD - 78.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Edna Rose"

**Score Difference:** 0.0112 (1.12 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Edna ISD (Edna, TX) - 77.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Edna ISD"

**Score Difference:** 0.0091 (0.91 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. EDNA LUMBER COMPANY - 76.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Edna ISD"

**Score Difference:** 0.0092 (0.92 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8585 vs 0.7980 (Δ +0.0605)


</details>
5. City of Edna (Edna, TX) - 76.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "EDNA LUMBER COMPANY"

**Score Difference:** 0.0028 (0.28 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Edna Owusu (Hawthorne, NJ) - 74.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "City of Edna"

**Score Difference:** 0.0135 (1.35 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Edna Sawyer - 72.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Edna Owusu"

**Score Difference:** 0.0227 (2.27 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7440 vs 0.6686 (Δ +0.0754)


</details>
8. Edna 80th Celebration - 68.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Edna Sawyer"

**Score Difference:** 0.0392 (3.92 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)


</details>
9. Viajes Edna S.A. - 67.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Edna 80th Celebration"

**Score Difference:** 0.0144 (1.44 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 56. Boys and Girls Club of Dawson Community Centre

**Query:** `Boys and Girls Club of Dawson Community Centre` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Boys and Girls Club Services of Greater Victoria • **Score:** 72.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7650 | 70% | 0.5355 |
| Semantic Similarity (Normalized) | 0.6298 | 30% | 0.1890 |
| Semantic Similarity (Raw) | 2.9247 | - | - |
| **Base Score** | **0.7245** | - | - |
| **FINAL SCORE** | **0.7289** | - | **72.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7289
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
5 word(s) match exactly between your search and this company.

**Matching Words:**
• and, boys, club, girls, of

**Your Search Also Includes:**
• centre, community, dawson

**Company Name Also Includes:**
• greater, services, victoria

**Match Strength:**
• 62% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.7650 (Weight: 70%)
• Semantic Similarity: 0.6298 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Boys and Girls Club Services of Greater Victoria - 72.9% • *Hybrid match based on combined lexical and semantic features.*
2. Boys & Girls Club Metro Phoenix Area - 72.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Boys and Girls Club Services of Greater Victoria"

**Score Difference:** 0.0030 (0.30 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Boys & Girls Club in Orange County (Fullerton, CA) - 71.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Boys & Girls Club Metro Phoenix Area"

**Score Difference:** 0.0114 (1.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Boys & Girls Club of Hilton Head Island - 71.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Boys & Girls Club in Orange County"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Boys & Girls Club of Greater High Point - 70.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Boys & Girls Club of Hilton Head Island"

**Score Difference:** 0.0075 (0.75 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Pacific Youth Foundation Boys & Girls Club (Santa Monica, CA) - 70.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Boys & Girls Club of Greater High Point"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. North Omaha Boys & Girls Club - 69.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Pacific Youth Foundation Boys & Girls Club"

**Score Difference:** 0.0131 (1.31 percentage points)

**Key Differentiators:**
- String Similarity: 0.7650 vs 0.6955 (Δ +0.0695)
- Semantic Similarity: 0.5500 vs 0.6688 (Δ -0.1188)


</details>
8. Boys and Girls Club of Green Bay (Green Bay, WI) - 69.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "North Omaha Boys & Girls Club"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Boys & Girls Club of Washington DC - 68.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Boys and Girls Club of Green Bay"

**Score Difference:** 0.0027 (0.27 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 57. Beissbarth

**Query:** `Beissbarth` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Beissbarth GmbH • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 4.6947 | - | - |
| **Base Score** | **0.8727** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PREFIX MATCH

**What This Means:**
This company name starts with 'Beissbarth' and has additional information added.

**Action Required:**
• This is likely the same company with extra details
• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')
• If yes, use this match

**Why This Happens:**
• Someone entered just the core company name
• Your system has the full legal name
• Common in business databases where legal names include extra terms

**Top 10 Matches:**
1. Beissbarth GmbH - 95.6% • *Direct prefix match (target contains extra trailing words).*
2. Bitbar - 47.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Beissbarth GmbH"

**Score Difference:** 0.4855 (48.55 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.2812 (Δ +0.5369)
- Semantic Similarity: 1.0000 vs 0.9018 (Δ +0.0982)


</details>
3. Beiss Barth - 44.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Bitbar"

**Score Difference:** 0.0299 (2.99 percentage points)

**Key Differentiators:**
- String Similarity: 0.2812 vs 0.3409 (Δ -0.0597)
- Semantic Similarity: 0.9018 vs 0.6635 (Δ +0.2382)


</details>
4. Ziebart - 43.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Beiss Barth"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:**
- String Similarity: 0.3409 vs 0.2647 (Δ +0.0762)
- Semantic Similarity: 0.6635 vs 0.8349 (Δ -0.1714)


</details>
5. ISOBAR - 41.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Ziebart"

**Score Difference:** 0.0194 (1.94 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8349 vs 0.7322 (Δ +0.1027)


</details>
6. SideBar - 40.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "ISOBAR"

**Score Difference:** 0.0112 (1.12 percentage points)

**Key Differentiators:**
- String Similarity: 0.2812 vs 0.2118 (Δ +0.0695)
- Semantic Similarity: 0.7322 vs 0.8573 (Δ -0.1251)


</details>
7. MakerBar - 40.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "SideBar"

**Score Difference:** 0.0024 (0.24 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. backbar - 40.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "MakerBar"

**Score Difference:** 0.0037 (0.37 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Agbar - 39.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "backbar"

**Score Difference:** 0.0035 (0.35 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8371 vs 0.8998 (Δ -0.0627)


</details>

---

## 58. US Night Vision

**Query:** `US Night Vision` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Association of U.S. Night Vision Manufacturers (Roanoke, VA) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity (Normalized) | 0.5572 | 30% | 0.1672 |
| Semantic Similarity (Raw) | 3.0268 | - | - |
| **Base Score** | **0.6922** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• night, vision

**Your Search Also Includes:**
• us

**Company Name Also Includes:**
• association, manufacturers, of, u.s.

**Match Strength:**
• 33% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7500 (Weight: 70%)
• Semantic Similarity: 0.5572 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Association of U.S. Night Vision Manufacturers (Roanoke, VA) - 95.6% • *High word-for-word overlap.*
2. WORLD VISION US (New York, NY) - 91.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Association of U.S. Night Vision Manufacturers"

**Score Difference:** 0.0416 (4.16 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.8097 (Δ -0.0597)
- Semantic Similarity: 0.5572 vs 0.6549 (Δ -0.0977)


</details>
3. Night Vision Entertainment (Hollywood, CA) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "WORLD VISION US"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6549 vs 0.7310 (Δ -0.0761)


</details>
4. WORLD VISION US (Washington, DC) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Night Vision Entertainment"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7310 vs 0.6549 (Δ +0.0761)


</details>
5. World Vision US (Federal Way, WA) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "WORLD VISION US"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Night Vision Entertainment - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "World Vision US"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. WORLD VISION US - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Night Vision Entertainment"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7310 vs 0.6549 (Δ +0.0761)


</details>
8. ITT Night Vision - 90.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "WORLD VISION US"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6549 vs 0.8552 (Δ -0.2003)


</details>
9. Night Vision Systems, LLC - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "ITT Night Vision"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8552 vs 0.7531 (Δ +0.1022)


</details>

---

## 59. Amedysis, Incorporated

**Query:** `Amedysis, Incorporated` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Amedysis, Inc. • **Score:** 100.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 4.7399 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0061** | - | **100.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0061
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• amedysis,

**Your Search Also Includes:**
• incorporated

**Company Name Also Includes:**
• inc.

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 1.0000 (Weight: 70%)
• Semantic Similarity: 1.0000 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Amedysis, Inc. - 100.6% • *High word-for-word overlap.*
2. Amedysis, Incorporated (Anchorage, AK) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Amedysis, Inc."

**Score Difference:** 0.0037 (0.37 percentage points)

**Key Differentiators:**
- Semantic Similarity: 1.0000 vs 0.8923 (Δ +0.1077)


</details>
3. Amedysis Home Health (Severna Park, MD) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Amedysis, Incorporated"

**Score Difference:** 0.0467 (4.67 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7500 (Δ +0.2500)
- Semantic Similarity: 0.8923 vs 0.7338 (Δ +0.1584)


</details>
4. Dialysis Corporation - 47.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Amedysis Home Health"

**Score Difference:** 0.4839 (48.39 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.2812 (Δ +0.4688)
- Semantic Similarity: 0.7338 vs 0.9072 (Δ -0.1733)


</details>
5. Personalysis Corporation - 44.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Dialysis Corporation"

**Score Difference:** 0.0228 (2.28 percentage points)

**Key Differentiators:**
- String Similarity: 0.2812 vs 0.2250 (Δ +0.0563)
- Semantic Similarity: 0.9072 vs 0.9628 (Δ -0.0557)


</details>
6. Medysis (Montreal, QC) - 44.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Personalysis Corporation"

**Score Difference:** 0.0066 (0.66 percentage points)

**Key Differentiators:**
- String Similarity: 0.2250 vs 0.3750 (Δ -0.1500)
- Semantic Similarity: 0.9628 vs 0.5909 (Δ +0.3719)


</details>
7. Cardialysis (Rotterdam, ) - 42.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Medysis"

**Score Difference:** 0.0181 (1.81 percentage points)

**Key Differentiators:**
- String Similarity: 0.3750 vs 0.2842 (Δ +0.0908)
- Semantic Similarity: 0.5909 vs 0.7427 (Δ -0.1518)


</details>
8. Avysis (Austin, TX) - 42.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Cardialysis"

**Score Difference:** 0.0030 (0.30 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7427 vs 0.6409 (Δ +0.1018)


</details>
9. Dialysis Centers Incorporated - 40.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Avysis"

**Score Difference:** 0.0141 (1.41 percentage points)

**Key Differentiators:**
- String Similarity: 0.3214 vs 0.1705 (Δ +0.1510)
- Semantic Similarity: 0.6409 vs 0.9515 (Δ -0.3105)


</details>

---

## 60. Taiyo Air Service Co.,Ltd

**Query:** `Taiyo Air Service Co.,Ltd` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Taiyo Air Services Co. • **Score:** 81.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7244 | 70% | 0.5071 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.2796 | - | - |
| **Base Score** | **0.8071** | - | - |
| **FINAL SCORE** | **0.8120** | - | **81.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8120
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• air, taiyo

**Your Search Also Includes:**
• co.,ltd, service

**Company Name Also Includes:**
• co., services

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7244 (Weight: 70%)
• Semantic Similarity: 1.0000 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Taiyo Air Services Co. - 81.2% • *Matched via strong semantic/conceptual similarity.*
2. Fuyo Air Service Co. Ltd. (Tokyo, ) - 74.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Taiyo Air Services Co."

**Score Difference:** 0.0676 (6.76 percentage points)

**Key Differentiators:**
- Semantic Similarity: 1.0000 vs 0.7760 (Δ +0.2240)


</details>
3. CITS Taikoo Air Service Ltd - 73.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Fuyo Air Service Co. Ltd."

**Score Difference:** 0.0123 (1.23 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7760 vs 0.6903 (Δ +0.0857)


</details>
4. Tec Air Service Co. Ltd. (tokyo) (Tokyo, ) - 73.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "CITS Taikoo Air Service Ltd"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.7969 (Δ -0.0531)
- Semantic Similarity: 0.6903 vs 0.5660 (Δ +0.1243)


</details>
5. GUANGZHOU GZL AIR SERVICE CO., LTD. - 72.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Tec Air Service Co. Ltd. (tokyo)"

**Score Difference:** 0.0114 (1.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. WORLD-AIR SEA SERVICE CO., LTD. (Tokyo, ) - 71.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "GUANGZHOU GZL AIR SERVICE CO., LTD."

**Score Difference:** 0.0023 (0.23 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Union Air Service Co. Ltd. (Japan) (Tokyo, ) - 71.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "WORLD-AIR SEA SERVICE CO., LTD."

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Yumen Air & Sea Service Co., Ltd. - 71.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Union Air Service Co. Ltd. (Japan)"

**Score Difference:** 0.0044 (0.44 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Ryowa Diamond Air Service Co., Ltd. (Shinjuku, Tokyo) - 71.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Yumen Air & Sea Service Co., Ltd."

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 61. National Conference on Race & Ethnicity in American Higher E (Norman, OK)

**Query:** `National Conference on Race & Ethnicity in American Higher E` • **Location:** Norman, OK • **Self-Match:** ✅ Found & Filtered

**Top Match:** National Conference On Race & Ethnicity In America Higher Ed (Norman, OK) • **Score:** 92.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8490 | 70% | 0.5943 |
| Semantic Similarity (Normalized) | 0.7814 | 30% | 0.2344 |
| Semantic Similarity (Raw) | 3.7917 | - | - |
| **Base Score** | **0.8287** | - | - |
| Location Match Boost | 100.0000 | 5% max | +5.0000 |
| **FINAL SCORE** | **0.9287** | - | **92.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 100.0000 × 0.05 = 5.0000

Final Score = Base Score + Location Boost = 0.9287
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 100.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
8 word(s) match exactly between your search and this company.

**Matching Words:**
• &, conference, ethnicity, higher, in, national, on, race

**Your Search Also Includes:**
• american, e

**Company Name Also Includes:**
• america, ed

**Match Strength:**
• 80% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8490 (Weight: 70%)
• Semantic Similarity: 0.7814 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. National Conference On Race & Ethnicity In America Higher Ed (Norman, OK) - 92.9% • *High word-for-word overlap.*
2. National Conference on Race & Ethnicity in Am. Higher Educ (Norman, OK) - 92.9% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "National Conference On Race & Ethnicity In America Higher Ed"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7814 vs 0.7089 (Δ +0.0725)


</details>
3. NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER (Norman, OK) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "National Conference on Race & Ethnicity in Am. Higher Educ"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7089 vs 0.7802 (Δ -0.0713)


</details>
4. National Conference on Race & Ethnicity in AM Higher Education (Norman, OK) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7802 vs 0.7102 (Δ +0.0700)


</details>
5. NATIONAL CONF. ON RACE & ETHNICITY IN AMERICAN HIGHER EDUC. (Norman, OK) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "National Conference on Race & Ethnicity in AM Higher Education"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8490 vs 0.9056 (Δ -0.0566)
- Semantic Similarity: 0.7102 vs 0.6182 (Δ +0.0920)


</details>
6. NATL CONF ON RACE & ETHNICITY IN AMERICAN HIGHER EDUCATION (Norman, OK) - 92.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "NATIONAL CONF. ON RACE & ETHNICITY IN AMERICAN HIGHER EDUC."

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6182 vs 0.5645 (Δ +0.0537)


</details>
7. National Conference on Race & Ethnicity in Higher Education (Norman, OK) - 80.9% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "NATL CONF ON RACE & ETHNICITY IN AMERICAN HIGHER EDUCATION"

**Score Difference:** 0.1161 (11.61 percentage points)

**Key Differentiators:**
- String Similarity: 0.8833 vs 0.7718 (Δ +0.1116)
- Semantic Similarity: 0.5645 vs 0.7190 (Δ -0.1545)


</details>
8. National Conference on Race & Ethnicity in America in Higher Education (NCORE) (Norman, OK) - 79.5% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "National Conference on Race & Ethnicity in Higher Education"

**Score Difference:** 0.0146 (1.46 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7190 vs 0.6476 (Δ +0.0713)


</details>
9. NCORE Nat. Conf. for Race & Ethnicity in American Higher Ed (Norman, OK) - 77.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "National Conference on Race & Ethnicity in America in Higher Education (NCORE)"

**Score Difference:** 0.0179 (1.79 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6476 vs 0.5482 (Δ +0.0994)


</details>

---

## 62. Reminger Law Firm

**Query:** `Reminger Law Firm` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Reminger & Reminger Law Firm (Sandusky, OH) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.7389 | 30% | 0.2217 |
| Semantic Similarity (Raw) | 3.8080 | - | - |
| **Base Score** | **0.9217** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
SUBSTRING MATCH

**What This Means:**
This company name contains 'Reminger Law Firm' somewhere within it.

**Action Required:**
• This is likely the same company
• Check if the surrounding words make sense
• If yes, use this match

**Why This Happens:**
• Someone entered a partial company name
• Your system has the complete name
• Common when people remember only part of a company name

**Top 10 Matches:**
1. Reminger & Reminger Law Firm (Sandusky, OH) - 95.6% • *Substring match (target contains query text).*
2. Lanier Law Firm (Los Angeles, CA) - 91.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Reminger & Reminger Law Firm"

**Score Difference:** 0.0431 (4.31 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8097 (Δ +0.1903)
- Semantic Similarity: 0.7389 vs 0.6021 (Δ +0.1367)


</details>
3. McNair Law Firm (Columbia, SC) - 91.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Lanier Law Firm"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Levin Law Firm (Pensacola, FL) - 91.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "McNair Law Firm"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Hood Law Firm (Charleston, SC) - 91.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Levin Law Firm"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5576 vs 0.7253 (Δ -0.1677)


</details>
6. Lanier Law Firm (Houston, TX) - 91.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Hood Law Firm"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7253 vs 0.6021 (Δ +0.1232)


</details>
7. Cordell Law Firm (Saint Louis, MO) - 91.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Lanier Law Firm"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Ackerman Law Firm (Miami, FL) - 91.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Cordell Law Firm"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Lauro Law Firm (Tampa, FL) - 91.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Ackerman Law Firm"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 63. SEMMOA BOD

**Query:** `SEMMOA BOD` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** World Bod • **Score:** 75.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.7698 | 30% | 0.2309 |
| Semantic Similarity (Raw) | 3.3018 | - | - |
| **Base Score** | **0.7516** | - | - |
| **FINAL SCORE** | **0.7561** | - | **75.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7561
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• bod

**Your Search Also Includes:**
• semmoa

**Company Name Also Includes:**
• world

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.7698 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. World Bod - 75.6% • *Hybrid match based on combined lexical and semantic features.*
2. BOD HD - 75.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "World Bod"

**Score Difference:** 0.0047 (0.47 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. BOD - 74.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "BOD HD"

**Score Difference:** 0.0059 (0.59 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6300 (Δ +0.1138)
- Semantic Similarity: 0.7541 vs 1.0000 (Δ -0.2459)


</details>
4. SEMMOA AACM (Detroit, MI) - 73.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "BOD"

**Score Difference:** 0.0092 (0.92 percentage points)

**Key Differentiators:**
- String Similarity: 0.6300 vs 0.7438 (Δ -0.1138)
- Semantic Similarity: 1.0000 vs 0.7043 (Δ +0.2957)


</details>
5. Regions BOD - 73.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "SEMMOA AACM"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. BOD Consulting - 72.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Regions BOD"

**Score Difference:** 0.0086 (0.86 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. BOD Meeding - 72.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "BOD Consulting"

**Score Difference:** 0.0027 (0.27 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. CSG BOD - 71.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "BOD Meeding"

**Score Difference:** 0.0065 (0.65 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. BOD Meeting - 71.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "CSG BOD"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 64. Telefonica Global Solutions

**Query:** `Telefonica Global Solutions` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Telefonica Global Solutions (Miami, FL) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.4480 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Telefonica Global Solutions (Miami, FL) - 100.2% • *Perfect character-for-character match.*
2. Telefonica Global Solutions USA Inc. (Miami, FL) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Telefonica Global Solutions"

**Score Difference:** 0.0467 (4.67 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8464 (Δ +0.1536)
- Semantic Similarity: 1.0000 vs 0.6381 (Δ +0.3619)


</details>
3. Telefonica Multinational Solutions - 90.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Telefonica Global Solutions USA Inc."

**Score Difference:** 0.0492 (4.92 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6381 vs 0.9424 (Δ -0.3043)


</details>
4. Telefonica Multinational Solutions (New York, NY) - 90.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Telefonica Multinational Solutions"

**Score Difference:** 0.0011 (0.11 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. TGS - 75.0% • *Matched based on generated acronym 'TGS'.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Telefonica Multinational Solutions"

**Score Difference:** 0.1555 (15.55 percentage points)

**Key Differentiators:**
- String Similarity: 0.8833 vs 1.0000 (Δ -0.1167)
- Semantic Similarity: 0.9424 vs 1.0000 (Δ -0.0576)
- Acronym Fidelity: 0.0000 vs 1.0000 (Δ -1.0000)


</details>
6. 02 Telefonica - 67.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "TGS"

**Score Difference:** 0.0789 (7.89 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.5758 (Δ +0.4242)
- Semantic Similarity: 1.0000 vs 0.8801 (Δ +0.1199)
- Acronym Fidelity: 1.0000 vs 0.0000 (Δ +1.0000)


</details>
7. Telefonica Del Peru - 66.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "02 Telefonica"

**Score Difference:** 0.0076 (0.76 percentage points)

**Key Differentiators:**
- String Similarity: 0.5758 vs 0.6333 (Δ -0.0576)
- Semantic Similarity: 0.8801 vs 0.7206 (Δ +0.1595)


</details>
8. Telefonica España - 66.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Telefonica Del Peru"

**Score Difference:** 0.0040 (0.40 percentage points)

**Key Differentiators:**
- String Similarity: 0.6333 vs 0.5758 (Δ +0.0576)
- Semantic Similarity: 0.7206 vs 0.8416 (Δ -0.1210)


</details>
9. TELEFONICA INTERNATIONAL USA (New York, NY) - 65.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Telefonica España"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:**
- String Similarity: 0.5758 vs 0.6333 (Δ -0.0576)
- Semantic Similarity: 0.8416 vs 0.7059 (Δ +0.1358)


</details>

---

## 65. Travel Leaders - Dube Travel

**Query:** `Travel Leaders - Dube Travel` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Dube Travel Leaders (Charlotte, NC) • **Score:** 95.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.7996 | 30% | 0.2399 |
| Semantic Similarity (Raw) | 4.5068 | - | - |
| **Base Score** | **0.9399** | - | - |
| **FINAL SCORE** | **0.9592** | - | **95.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9592
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• dube, leaders, travel

**Your Search Also Includes:**
• -

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 1.0000 (Weight: 70%)
• Semantic Similarity: 0.7996 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Dube Travel Leaders (Charlotte, NC) - 95.9% • *High word-for-word overlap.*
2. Dube Travel / Travel Leaders (Hallowell, ME) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Dube Travel Leaders"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8636 (Δ +0.1364)
- Semantic Similarity: 0.7996 vs 0.6794 (Δ +0.1202)


</details>
3. Dube Travel Leaders (Chicago, IL) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Dube Travel / Travel Leaders"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- String Similarity: 0.8636 vs 1.0000 (Δ -0.1364)
- Semantic Similarity: 0.6794 vs 0.7996 (Δ -0.1202)


</details>
4. dube Travel leaders (Topsham, ME) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Dube Travel Leaders"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Dube / Travel Leaders (Western Springs, IL) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "dube Travel leaders"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8636 (Δ +0.1364)
- Semantic Similarity: 0.7798 vs 0.6630 (Δ +0.1168)


</details>
6. Travel Leaders 365 - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Dube / Travel Leaders"

**Score Difference:** 0.0471 (4.71 percentage points)

**Key Differentiators:**
- String Similarity: 0.8636 vs 0.8097 (Δ +0.0539)
- Semantic Similarity: 0.6630 vs 0.7709 (Δ -0.1079)


</details>
7. Travel Leaders UK (London, ) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Travel Leaders 365"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7709 vs 0.7188 (Δ +0.0522)


</details>
8. Travel Leaders Network (PLYMOUTH, ) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Travel Leaders UK"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Advent Travel Leaders (Minneapolis, MN) - 90.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Travel Leaders Network"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7167 vs 0.6663 (Δ +0.0504)


</details>

---

## 66. Hi- Tours

**Query:** `Hi- Tours` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Hi tours • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.5987 | - | - |
| **Base Score** | **0.9300** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• tours

**Your Search Also Includes:**
• hi-

**Company Name Also Includes:**
• hi

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.9000 (Weight: 70%)
• Semantic Similarity: 1.0000 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Hi tours - 95.6% • *High word-for-word overlap.*
2. Hi-Tours - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Hi tours"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 1.0000 vs 0.8816 (Δ +0.1184)


</details>
3. Hi Life Tours - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Hi-Tours"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. R&C HI Tours - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Hi Life Tours"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8371 vs 0.6686 (Δ +0.1685)


</details>
5. Hi Tour - 82.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "R&C HI Tours"

**Score Difference:** 0.1342 (13.42 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6686 vs 0.8845 (Δ -0.2159)


</details>
6. Nice Tours - 77.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Hi Tour"

**Score Difference:** 0.0498 (4.98 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8845 vs 0.8215 (Δ +0.0630)


</details>
7. Sky Tours - 77.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Nice Tours"

**Score Difference:** 0.0017 (0.17 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Destination Tours - 76.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Sky Tours"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Fun Tours - 76.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Destination Tours"

**Score Difference:** 0.0044 (0.44 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 67. Volkswagen Group China

**Query:** `Volkswagen Group China` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** VOLKSWAGEN GROUP CHINA • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.6367 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. VOLKSWAGEN GROUP CHINA - 100.2% • *Perfect character-for-character match.*
2. Volkswagen China - 100.0% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "VOLKSWAGEN GROUP CHINA"

**Score Difference:** 0.0025 (0.25 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. FAW Volkswagen Automotive Co., Ltd. South China Branch (guangzhou, guangdong) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Volkswagen China"

**Score Difference:** 0.0408 (4.08 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.6786 (Δ +0.3214)
- Semantic Similarity: 0.9681 vs 0.5057 (Δ +0.4624)


</details>
4. VOLKSWAGEN CHINA INVESTMENT COMPANY LTD - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "FAW Volkswagen Automotive Co., Ltd. South China Branch"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- String Similarity: 0.6786 vs 0.8182 (Δ -0.1396)
- Semantic Similarity: 0.5057 vs 0.7162 (Δ -0.2106)


</details>
5. Volkswagen China Investment Company.Ltd (beijing, beijing) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "VOLKSWAGEN CHINA INVESTMENT COMPANY LTD"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.7500 (Δ +0.0682)
- Semantic Similarity: 0.7162 vs 0.6239 (Δ +0.0924)


</details>
6. FAW Volkswagen Automotive Co., Ltd. South China Branch - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Volkswagen China Investment Company.Ltd"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.6786 (Δ +0.0714)
- Semantic Similarity: 0.6239 vs 0.5057 (Δ +0.1182)


</details>
7. FAW Volkswagen Automotive Co., Ltd. South China Branch (Foshan, Guangdong) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "FAW Volkswagen Automotive Co., Ltd. South China Branch"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Volkswagen Group Japan - 77.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "FAW Volkswagen Automotive Co., Ltd. South China Branch"

**Score Difference:** 0.1855 (18.55 percentage points)

**Key Differentiators:**
- String Similarity: 0.6786 vs 0.7438 (Δ -0.0652)
- Semantic Similarity: 0.5057 vs 0.8165 (Δ -0.3109)


</details>
9. Volkswagen Group Australia - 76.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Volkswagen Group Japan"

**Score Difference:** 0.0086 (0.86 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 68. Sun Tx

**Query:** `Sun Tx` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Sun Coast (Houston, TX) • **Score:** 80.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9070 | 30% | 0.2721 |
| Semantic Similarity (Raw) | 5.1374 | - | - |
| **Base Score** | **0.7927** | - | - |
| **FINAL SCORE** | **0.8004** | - | **80.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8004
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• sun

**Your Search Also Includes:**
• tx

**Company Name Also Includes:**
• coast

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.9070 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Sun Coast (Houston, TX) - 80.0% • *Matched via strong semantic/conceptual similarity.*
2. SUN TRAVEL (Dallas, TX) - 79.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Sun Coast"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Mod Sun (Dallas, TX) - 78.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "SUN TRAVEL"

**Score Difference:** 0.0105 (1.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Sun Travel (Beaumont, TX) - 77.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Mod Sun"

**Score Difference:** 0.0105 (1.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Sun City - 77.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Sun Travel"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. sun coast - 77.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Sun City"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Rising Sun - 77.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "sun coast"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Sun City Texas (Sun City, TX) - 77.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Rising Sun"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.8237 vs 0.9773 (Δ -0.1536)


</details>
9. Sun Resorts (Dallas, TX) - 76.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Sun City Texas"

**Score Difference:** 0.0049 (0.49 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.9773 vs 0.8033 (Δ +0.1740)


</details>

---

## 69. Southern Vermont Deerfield Valley Chamber of commerce (Deerfield, VT)

**Query:** `Southern Vermont Deerfield Valley Chamber of commerce` • **Location:** Deerfield, VT • **Self-Match:** ✅ Found & Filtered

**Top Match:** Southern Vermont Deerfield Valley Chamber of commerce • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.6302 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Southern Vermont Deerfield Valley Chamber of commerce - 100.2% • *Perfect character-for-character match.*
2. Deerfield Bannockburn Riverwoods Chamber of Commerce (Deerfield, IL) - 65.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Southern Vermont Deerfield Valley Chamber of commerce"

**Score Difference:** 0.3464 (34.64 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.6761 (Δ +0.3239)
- Semantic Similarity: 1.0000 vs 0.6389 (Δ +0.3611)
- Location Score: 0.0000 vs 60.0000 (Δ -60.0000)


</details>
3. DEERFIELD BEACH CHAMBER OF COMMERCE (Deerfield Beach, FL) - 62.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Deerfield Bannockburn Riverwoods Chamber of Commerce"

**Score Difference:** 0.0333 (3.33 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.6198 (Δ +0.0563)
- Location Score: 60.0000 vs 52.8000 (Δ +7.2000)


</details>
4. Deerfield Chamber of Commerce (Deerfield, IL) - 61.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "DEERFIELD BEACH CHAMBER OF COMMERCE"

**Score Difference:** 0.0050 (0.50 percentage points)

**Key Differentiators:**
- String Similarity: 0.6198 vs 0.5385 (Δ +0.0813)
- Semantic Similarity: 0.6761 vs 0.8018 (Δ -0.1257)
- Location Score: 52.8000 vs 60.0000 (Δ -7.2000)


</details>
5. Regional Black Chamber of Commerce Southern California - 55.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Deerfield Chamber of Commerce"

**Score Difference:** 0.0587 (5.87 percentage points)

**Key Differentiators:**
- String Similarity: 0.5385 vs 0.7438 (Δ -0.2053)
- Semantic Similarity: 0.8018 vs 0.5762 (Δ +0.2256)
- Location Score: 60.0000 vs 0.0000 (Δ +60.0000)


</details>
6. Deerfield Beach Chamber of Commerce - 54.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Regional Black Chamber of Commerce Southern California"

**Score Difference:** 0.0174 (1.74 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6198 (Δ +0.1240)
- Semantic Similarity: 0.5762 vs 0.7934 (Δ -0.2172)


</details>
7. America-Israel Chamber of Commerce Chicago (Deerfield, IL) - 53.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Deerfield Beach Chamber of Commerce"

**Score Difference:** 0.0045 (0.45 percentage points)

**Key Differentiators:**
- String Similarity: 0.6198 vs 0.4678 (Δ +0.1520)
- Semantic Similarity: 0.7934 vs 0.6332 (Δ +0.1602)
- Location Score: 0.0000 vs 60.0000 (Δ -60.0000)


</details>
8. Vermont Chamber of Commerce (Montpelier, VT) - 53.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "America-Israel Chamber of Commerce Chicago"

**Score Difference:** 0.0008 (0.08 percentage points)

**Key Differentiators:**
- String Similarity: 0.4678 vs 0.5385 (Δ -0.0707)
- Location Score: 60.0000 vs 40.0000 (Δ +20.0000)


</details>
9. West Valley Warner Center Chamber of Commerce - 53.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Vermont Chamber of Commerce"

**Score Difference:** 0.0029 (0.29 percentage points)

**Key Differentiators:**
- String Similarity: 0.5385 vs 0.6906 (Δ -0.1522)
- Location Score: 40.0000 vs 0.0000 (Δ +40.0000)


</details>
10. Mill Valley Chamber of Commerce & Visitor Center - 52.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "West Valley Warner Center Chamber of Commerce"

**Score Difference:** 0.0061 (0.61 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 70. DGR Ministries

**Query:** `DGR Ministries` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Power Ministries • **Score:** 81.1%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9525 | 30% | 0.2858 |
| Semantic Similarity (Raw) | 4.9856 | - | - |
| **Base Score** | **0.8064** | - | - |
| **FINAL SCORE** | **0.8113** | - | **81.1%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8113
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• ministries

**Your Search Also Includes:**
• dgr

**Company Name Also Includes:**
• power

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.9525 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Power Ministries - 81.1% • *Matched via strong semantic/conceptual similarity.*
2. DG Ministries (, TX) - 80.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Power Ministries"

**Score Difference:** 0.0105 (1.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Empowered Ministries - 79.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "DG Ministries"

**Score Difference:** 0.0080 (0.80 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Legacy Ministries - 78.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Empowered Ministries"

**Score Difference:** 0.0040 (0.40 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Sure ministries - 78.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Legacy Ministries"

**Score Difference:** 0.0050 (0.50 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. CV Ministries - 78.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Sure ministries"

**Score Difference:** 0.0008 (0.08 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. SB Ministries - 78.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "CV Ministries"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Special Ministries - 78.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "SB Ministries"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. AG Ministries - 78.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Special Ministries"

**Score Difference:** 0.0011 (0.11 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 71. Impacto 6

**Query:** `Impacto 6` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Impacto Vital (San Juan, PR) • **Score:** 91.1%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity (Normalized) | 0.5219 | 30% | 0.1566 |
| Semantic Similarity (Raw) | 2.5618 | - | - |
| **Base Score** | **0.7516** | - | - |
| **FINAL SCORE** | **0.9110** | - | **91.1%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9110
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• impacto

**Your Search Also Includes:**
• 6

**Company Name Also Includes:**
• vital

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.8500 (Weight: 70%)
• Semantic Similarity: 0.5219 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Impacto Vital (San Juan, PR) - 91.1% • *Hybrid match based on combined lexical and semantic features.*
2. Impacto 52 - 90.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Impacto Vital"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5219 vs 1.0000 (Δ -0.4781)


</details>
3. Impacto EDL - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Impacto 52"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 1.0000 vs 0.8473 (Δ +0.1527)


</details>
4. Impacto Tactico (Lima, ) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Impacto EDL"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Impacto YOUTH - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Impacto Tactico"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8088 vs 0.7420 (Δ +0.0669)


</details>
6. Impacto Ejecutivo - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Impacto YOUTH"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7420 vs 0.6811 (Δ +0.0609)


</details>
7. Kiin Impacto - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Impacto Ejecutivo"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Impacto Strategies (Indianapolis, IN) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Kiin Impacto"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. IMPACTO Youth (Manassas, VA) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Impacto Strategies"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6386 vs 0.4712 (Δ +0.1674)


</details>

---

## 72. Neos Therapeutics, Inc.

**Query:** `Neos Therapeutics, Inc.` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Neos Therapeutics, Inc. (Grand Prairie, TX) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6741 | 30% | 0.2022 |
| Semantic Similarity (Raw) | 3.8159 | - | - |
| **Base Score** | **0.9022** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Neos Therapeutics, Inc. (Grand Prairie, TX) - 100.2% • *Perfect character-for-character match.*
2. Neos Therapeutics (Grand Prairie, TX) - 96.3% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Neos Therapeutics, Inc."

**Score Difference:** 0.0390 (3.90 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.9167 (Δ +0.0833)


</details>
3. Neos Therapeutics (Fairfax, VA) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Neos Therapeutics"

**Score Difference:** 0.0043 (0.43 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Neos Therapeutics (Blue Bell, PA) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Neos Therapeutics"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Neos Therapeutics (Trussville, AL) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Neos Therapeutics"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Neos Therapeutics - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Neos Therapeutics"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Neos Therapeutics (Canton, OH) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Neos Therapeutics"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Neos Therapeutics (Lincolnshire, IL) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Neos Therapeutics"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Neos Therapeutics LP (Grand Prairie, TX) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Neos Therapeutics"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.9167 vs 0.7658 (Δ +0.1509)


</details>

---

## 73. International Tax Institute

**Query:** `International Tax Institute` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** International Tax Institute (Lewiston, NY) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.7985 | 30% | 0.2396 |
| Semantic Similarity (Raw) | 4.1791 | - | - |
| **Base Score** | **0.9396** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. International Tax Institute (Lewiston, NY) - 100.2% • *Perfect character-for-character match.*
2. International Property Tax Institute (Toronto, ON) - 96.2% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "International Tax Institute"

**Score Difference:** 0.0409 (4.09 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8636 (Δ +0.1364)


</details>
3. International Property Tax Institute (Etobicoke, ON) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "International Property Tax Institute"

**Score Difference:** 0.0024 (0.24 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. INTERNATIONAL TAX INSTITUTE INC (New York, NY) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "International Property Tax Institute"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- String Similarity: 0.8636 vs 1.0000 (Δ -0.1364)


</details>
5. International Property Tax Institute (Bethesda, MD) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "INTERNATIONAL TAX INSTITUTE INC"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8636 (Δ +0.1364)


</details>
6. International Tax & Auditing Institute (Algonquin, IL) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "International Property Tax Institute"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7907 vs 0.6391 (Δ +0.1517)


</details>
7. Federal Tax Institute - 92.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "International Tax & Auditing Institute"

**Score Difference:** 0.0319 (3.19 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6391 vs 1.0000 (Δ -0.3609)


</details>
8. International Tax Form - 92.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Federal Tax Institute"

**Score Difference:** 0.0016 (0.16 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Sales Tax Institute (Chicago, IL) - 91.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "International Tax Form"

**Score Difference:** 0.0082 (0.82 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9948 vs 0.5532 (Δ +0.4416)


</details>

---

## 74. Mitsubishi M501G

**Query:** `Mitsubishi M501G` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Mitsubishi Power • **Score:** 78.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8774 | 30% | 0.2632 |
| Semantic Similarity (Raw) | 5.5224 | - | - |
| **Base Score** | **0.7838** | - | - |
| **FINAL SCORE** | **0.7886** | - | **78.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7886
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• mitsubishi

**Your Search Also Includes:**
• m501g

**Company Name Also Includes:**
• power

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.8774 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Mitsubishi Power - 78.9% • *Matched via strong semantic/conceptual similarity.*
2. Mitsubishi Motors - 78.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Mitsubishi Power"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Mitsubishi Securities - 77.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Mitsubishi Motors"

**Score Difference:** 0.0174 (1.74 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8769 vs 0.8193 (Δ +0.0576)


</details>
4. Mitsubishi Germany - 76.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Mitsubishi Securities"

**Score Difference:** 0.0025 (0.25 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Mitsubishi Digital - 76.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Mitsubishi Germany"

**Score Difference:** 0.0008 (0.08 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Mitsubishi Motor Company - 76.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Mitsubishi Digital"

**Score Difference:** 0.0062 (0.62 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Mitsubishi Paper - 76.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Mitsubishi Motor Company"

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Mitsubishi Meeting - 75.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Mitsubishi Paper"

**Score Difference:** 0.0026 (0.26 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Mitsubishi Estate - 75.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Mitsubishi Meeting"

**Score Difference:** 0.0020 (0.20 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 75. Huskies Sports

**Query:** `Huskies Sports` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Huskies Sports (Portland, OR) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.2045 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Huskies Sports (Portland, OR) - 100.4% • *Perfect character-for-character match.*
2. Huskies Basketball (Grand Rapids, MI) - 73.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Huskies Sports"

**Score Difference:** 0.2729 (27.29 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7438 (Δ +0.2562)
- Semantic Similarity: 1.0000 vs 0.6864 (Δ +0.3136)


</details>
3. Empire State Huskies - 72.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Huskies Basketball"

**Score Difference:** 0.0076 (0.76 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6864 vs 0.8189 (Δ -0.1325)


</details>
4. Miami Huskies (Florida City, FL) - 72.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Empire State Huskies"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.8189 vs 0.6578 (Δ +0.1611)


</details>
5. Wolverines Sports - 71.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Miami Huskies"

**Score Difference:** 0.0070 (0.70 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Mid Huron Huskies - 71.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Wolverines Sports"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6348 vs 0.7923 (Δ -0.1576)


</details>
7. NH Huskies (Auburn, NH) - 71.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Mid Huron Huskies"

**Score Difference:** 0.0043 (0.43 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.7923 vs 0.6204 (Δ +0.1719)


</details>
8. Sports Collegiate - 71.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "NH Huskies"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Cheer Sports - 70.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Sports Collegiate"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 76. Acacia Pharma Group Inc.

**Query:** `Acacia Pharma Group Inc.` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Acacia Pharma (Indianapolis, IN) • **Score:** 95.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity (Normalized) | 0.7326 | 30% | 0.2198 |
| Semantic Similarity (Raw) | 4.6050 | - | - |
| **Base Score** | **0.8615** | - | - |
| **FINAL SCORE** | **0.9592** | - | **95.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9592
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• acacia, pharma

**Your Search Also Includes:**
• group, inc.

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.9167 (Weight: 70%)
• Semantic Similarity: 0.7326 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Acacia Pharma (Indianapolis, IN) - 95.9% • *High word-for-word overlap.*
2. Acacia Pharma Ltd (Harston, ) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Acacia Pharma"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. ACACIA PHARMA, Inc. (Indianapolis, IN) - 95.9% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Acacia Pharma Ltd"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.9167 vs 1.0000 (Δ -0.0833)
- Semantic Similarity: 0.7047 vs 0.6484 (Δ +0.0563)


</details>
4. Acacia Pharma Ltd - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "ACACIA PHARMA, Inc."

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.9167 (Δ +0.0833)
- Semantic Similarity: 0.6484 vs 0.7047 (Δ -0.0563)


</details>
5. Acacia pharma (Cambridge, ) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Acacia Pharma Ltd"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7047 vs 0.8027 (Δ -0.0980)


</details>
6. Acacia Pharma (Franklin, TN) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Acacia pharma"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8027 vs 0.7326 (Δ +0.0701)


</details>
7. Acacia Pharma (Cambridge, Cambridgeshire) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Acacia Pharma"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Acacia Pharma (San Diego, CA) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Acacia Pharma"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Acacia Pharma (Carmel, IN) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Acacia Pharma"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 77. Acumatica Summit 2017 Z7NWPDKS625

**Query:** `Acumatica Summit 2017 Z7NWPDKS625` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** 2017 MISMO Fall Summit • **Score:** 65.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity (Normalized) | 0.5088 | 30% | 0.1526 |
| Semantic Similarity (Raw) | 2.2677 | - | - |
| **Base Score** | **0.6485** | - | - |
| **FINAL SCORE** | **0.6524** | - | **65.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.6524
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• 2017, summit

**Your Search Also Includes:**
• acumatica, z7nwpdks625

**Company Name Also Includes:**
• fall, mismo

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7083 (Weight: 70%)
• Semantic Similarity: 0.5088 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. 2017 MISMO Fall Summit - 65.2% • *Hybrid match based on combined lexical and semantic features.*
2. Acumatica User Group Southeast - 57.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "2017 MISMO Fall Summit"

**Score Difference:** 0.0810 (8.10 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.4545 (Δ +0.2538)
- Semantic Similarity: 0.5088 vs 0.8325 (Δ -0.3237)


</details>
3. Acumatica Asia (Singapore, ) - 54.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Acumatica User Group Southeast"

**Score Difference:** 0.0284 (2.84 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Contact Center Compliance Summit - 54.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Acumatica Asia"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:**
- String Similarity: 0.4167 vs 0.5000 (Δ -0.0833)
- Semantic Similarity: 0.8269 vs 0.6265 (Δ +0.2004)


</details>
5. 2024 Lung Summit KQN5K6PH2MW - 53.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Contact Center Compliance Summit"

**Score Difference:** 0.0052 (0.52 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. ACUMATICA PRESIDENTS CLUB - 53.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "2024 Lung Summit KQN5K6PH2MW"

**Score Difference:** 0.0015 (0.15 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6092 vs 0.7104 (Δ -0.1012)


</details>
7. Acumatica The Cloud ERP - 53.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "ACUMATICA PRESIDENTS CLUB"

**Score Difference:** 0.0041 (0.41 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Acumatica Asia (Seattle, WA) - 53.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Acumatica The Cloud ERP"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6969 vs 0.8269 (Δ -0.1300)


</details>
9. Healthcare IT Connect Summit - 52.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Acumatica Asia"

**Score Difference:** 0.0097 (0.97 percentage points)

**Key Differentiators:**
- String Similarity: 0.4167 vs 0.5000 (Δ -0.0833)
- Semantic Similarity: 0.8269 vs 0.5584 (Δ +0.2685)


</details>

---

## 78. Linklaters CIS

**Query:** `Linklaters CIS` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Cis GmbH • **Score:** 76.3%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.7915 | 30% | 0.2374 |
| Semantic Similarity (Raw) | 4.6453 | - | - |
| **Base Score** | **0.7581** | - | - |
| **FINAL SCORE** | **0.7627** | - | **76.3%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7627
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• cis

**Your Search Also Includes:**
• linklaters

**Company Name Also Includes:**
• gmbh

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.7915 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Cis GmbH - 76.3% • *Hybrid match based on combined lexical and semantic features.*
2. Cis 22 - 75.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Cis GmbH"

**Score Difference:** 0.0057 (0.57 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Cis Technologies - 74.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Cis 22"

**Score Difference:** 0.0117 (1.17 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. CIS Conference - 73.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Cis Technologies"

**Score Difference:** 0.0106 (1.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. STEELE CIS - 73.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "CIS Conference"

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. IEEE CIS - 71.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "STEELE CIS"

**Score Difference:** 0.0165 (1.65 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6959 vs 0.6411 (Δ +0.0548)


</details>
7. CIS ASIA - 71.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "IEEE CIS"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. CIS Travel - 71.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "CIS ASIA"

**Score Difference:** 0.0002 (0.02 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Deloitte CIS - 71.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "CIS Travel"

**Score Difference:** 0.0016 (0.16 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 79. Christian Girls Family Ministry

**Query:** `Christian Girls Family Ministry` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Christian Girls Family Ministry Training • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.2208 | - | - |
| **Base Score** | **0.8727** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PREFIX MATCH

**What This Means:**
This company name starts with 'Christian Girls Family Ministry' and has additional information added.

**Action Required:**
• This is likely the same company with extra details
• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')
• If yes, use this match

**Why This Happens:**
• Someone entered just the core company name
• Your system has the full legal name
• Common in business databases where legal names include extra terms

**Top 10 Matches:**
1. Christian Girls Family Ministry Training - 95.6% • *Direct prefix match (target contains extra trailing words).*
2. Christian Womens Ministry - 76.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Christian Girls Family Ministry Training"

**Score Difference:** 0.1882 (18.82 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.6761 (Δ +0.1420)


</details>
3. Kingdom Life Christian Ministry - 76.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Christian Womens Ministry"

**Score Difference:** 0.0027 (0.27 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.9655 vs 0.7987 (Δ +0.1668)


</details>
4. Christian Family Worship Center - 76.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Kingdom Life Christian Ministry"

**Score Difference:** 0.0021 (0.21 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Christian Marriage Ministry - 76.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Christian Family Worship Center"

**Score Difference:** 0.0023 (0.23 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.7919 vs 0.9420 (Δ -0.1501)


</details>
6. Christian Community Ministry - 75.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Christian Marriage Ministry"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. National Christian Ministry Association - 75.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Christian Community Ministry"

**Score Difference:** 0.0011 (0.11 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.9386 vs 0.7772 (Δ +0.1614)


</details>
8. NEW FAMILY CHRISTIAN CHURCH - 75.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "National Christian Ministry Association"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Christian Family Home Educators - 75.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "NEW FAMILY CHRISTIAN CHURCH"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 80. Alosa Foundation

**Query:** `Alosa Foundation` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Alosa Foundation • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.0562 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Alosa Foundation - 100.2% • *Perfect character-for-character match.*
2. Foundation Workshop - 66.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Alosa Foundation"

**Score Difference:** 0.3416 (34.16 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.6375 (Δ +0.3625)
- Semantic Similarity: 1.0000 vs 0.7020 (Δ +0.2980)


</details>
3. Alwan Foundation - 66.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Foundation Workshop"

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. NEST Foundation - 65.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Alwan Foundation"

**Score Difference:** 0.0021 (0.21 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Llosa's foundation - 65.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "NEST Foundation"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Foundation Building - 65.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Llosa's foundation"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. SEE Foundation - 65.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Foundation Building"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Ivy Foundation - 65.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "SEE Foundation"

**Score Difference:** 0.0023 (0.23 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Help Foundation - 65.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Ivy Foundation"

**Score Difference:** 0.0020 (0.20 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 81. La Chaine des Rotisseurs Wine Club of Newport Beach (Newport Beach, CA)

**Query:** `La Chaine des Rotisseurs Wine Club of Newport Beach` • **Location:** Newport Beach, CA • **Self-Match:** ✅ Found & Filtered

**Top Match:** La Chaine des Rotisseurs Wine Club of Newport Beach (Costa Mesa, CA) • **Score:** 102.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 4.5622 | - | - |
| **Base Score** | **1.0000** | - | - |
| Location Match Boost | 40.0000 | 5% max | +2.0000 |
| **FINAL SCORE** | **1.0224** | - | **102.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0224
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Same city and state

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. La Chaine des Rotisseurs Wine Club of Newport Beach (Costa Mesa, CA) - 102.2% • *Perfect character-for-character match.*
2. La Chaine des Rotisseurs Bailliage de Newport Beach (Los Angeles, CA) - 80.5% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "La Chaine des Rotisseurs Wine Club of Newport Beach"

**Score Difference:** 0.2170 (21.70 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8357 (Δ +0.1643)
- Semantic Similarity: 1.0000 vs 0.7702 (Δ +0.2298)


</details>
3. La Chaine des Rotisseurs Bailliage de Newport Beach (Costa Mesa, CA) - 80.5% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "La Chaine des Rotisseurs Bailliage de Newport Beach"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. La Chaine des Rotisseurs Wine Club of Ne (Costa Mesa, CA) - 73.1% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "La Chaine des Rotisseurs Bailliage de Newport Beach"

**Score Difference:** 0.0744 (7.44 percentage points)

**Key Differentiators:**
- String Similarity: 0.8357 vs 0.7597 (Δ +0.0760)
- Semantic Similarity: 0.7702 vs 0.9198 (Δ -0.1496)


</details>
5. Newport Beach Wine and Food Festival (Newport Beach, CA) - 64.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "La Chaine des Rotisseurs Wine Club of Ne"

**Score Difference:** 0.0904 (9.04 percentage points)

**Key Differentiators:**
- String Similarity: 0.7597 vs 0.4286 (Δ +0.3312)
- Semantic Similarity: 0.9198 vs 0.8225 (Δ +0.0973)
- Location Score: 40.0000 vs 100.0000 (Δ -60.0000)


</details>
6. Newport Beach Wine Festival (Newport Beach, CA) - 64.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Newport Beach Wine and Food Festival"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8225 vs 0.8934 (Δ -0.0708)


</details>
7. Newport Beach Food and Wine Festival (Newport Beach, CA) - 63.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Newport Beach Wine Festival"

**Score Difference:** 0.0047 (0.47 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8934 vs 0.8024 (Δ +0.0910)


</details>
8. Newport Beach Tennis Club (Newport Beach, CA) - 59.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Newport Beach Food and Wine Festival"

**Score Difference:** 0.0371 (3.71 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8024 vs 0.7203 (Δ +0.0820)


</details>
9. NEWPORT BEACH COUNTRY CLUB (Newport Beach, CA) - 59.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Newport Beach Tennis Club"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
10. La Chaine Des Rotisseurs Hillsborough Chapter (Los Angeles, CA) - 59.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "NEWPORT BEACH COUNTRY CLUB"

**Score Difference:** 0.0051 (0.51 percentage points)

**Key Differentiators:**
- String Similarity: 0.3980 vs 0.6071 (Δ -0.2092)
- Location Score: 100.0000 vs 40.0000 (Δ +60.0000)


</details>

---

## 82. Sumner & Ryan, LLC

**Query:** `Sumner & Ryan, LLC` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Ryan McCall • **Score:** 79.7%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9040 | 30% | 0.2712 |
| Semantic Similarity (Raw) | 3.8435 | - | - |
| **Base Score** | **0.7918** | - | - |
| **FINAL SCORE** | **0.7967** | - | **79.7%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7967
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
LINGUISTIC MATCH

**What This Means:**
The names look different but are linguistically related.

**Key Relationships Found:**
• 'ryan,' ↔ 'ryan' (abbreviation/expansion)

**Details:**
• 'ryan' is abbreviation of 'ryan,'

**Real-World Scenario:**
• Your system has the short form 'ryan' but someone wrote 'ryan,'

**Action Required:**
• Verify if this variation makes sense
• Likely the same company

**Top 10 Matches:**
1. Ryan McCall - 79.7% • *Matched via strong semantic/conceptual similarity.*
2. Sumner 360 (Washington, DC) - 79.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Ryan McCall"

**Score Difference:** 0.0048 (0.48 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Ryan Consulting - 78.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Sumner 360"

**Score Difference:** 0.0107 (1.07 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Ryan Mitchell Associates, LLC - 78.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Ryan Consulting"

**Score Difference:** 0.0005 (0.05 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Brianna Sumner - 77.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Ryan Mitchell Associates, LLC"

**Score Difference:** 0.0015 (0.15 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Sumner Baseball (, NC) - 77.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Brianna Sumner"

**Score Difference:** 0.0085 (0.85 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Ryan Henderson - 76.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Sumner Baseball"

**Score Difference:** 0.0035 (0.35 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Ryan Prospects - 76.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Ryan Henderson"

**Score Difference:** 0.0002 (0.02 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Ryan Staker - 76.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Ryan Prospects"

**Score Difference:** 0.0008 (0.08 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 83. Tilt Creative & Production

**Query:** `Tilt Creative & Production` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Tilt Creative + Production (Richmond, VA) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8727 | 70% | 0.6109 |
| Semantic Similarity (Normalized) | 0.8591 | 30% | 0.2577 |
| Semantic Similarity (Raw) | 3.7950 | - | - |
| **Base Score** | **0.8686** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• creative, production, tilt

**Your Search Also Includes:**
• &

**Company Name Also Includes:**
• +

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8727 (Weight: 70%)
• Semantic Similarity: 0.8591 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Tilt Creative + Production (Richmond, VA) - 95.6% • *High word-for-word overlap.*
2. Creative Production Incentives - 90.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Tilt Creative + Production"

**Score Difference:** 0.0503 (5.03 percentage points)

**Key Differentiators:**
- String Similarity: 0.8727 vs 0.8097 (Δ +0.0630)


</details>
3. Tilt Production (Richmond, VA) - 74.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Creative Production Incentives"

**Score Difference:** 0.1615 (16.15 percentage points)

**Key Differentiators:**
- String Similarity: 0.8097 vs 0.7361 (Δ +0.0736)
- Semantic Similarity: 0.8924 vs 0.7387 (Δ +0.1537)


</details>
4. Absolute Creative Design & Production (Vancouver, BC) - 71.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Tilt Production"

**Score Difference:** 0.0289 (2.89 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7387 vs 0.6517 (Δ +0.0869)


</details>
5. Bam Creative Production Pte Ltd (Singapore, ) - 71.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Absolute Creative Design & Production"

**Score Difference:** 0.0046 (0.46 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Shiloh Creative Production Studios Inc - 70.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Bam Creative Production Pte Ltd"

**Score Difference:** 0.0026 (0.26 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Upper Room Creative Production (Atlanta, GA) - 70.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Shiloh Creative Production Studios Inc"

**Score Difference:** 0.0001 (0.01 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Full Tilt Marketing - 64.2% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Upper Room Creative Production"

**Score Difference:** 0.0656 (6.56 percentage points)

**Key Differentiators:**
- String Similarity: 0.7361 vs 0.5278 (Δ +0.2083)
- Semantic Similarity: 0.6275 vs 0.8962 (Δ -0.2686)


</details>
9. CREATIVE THINKING PRODUCTIONS - 62.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Full Tilt Marketing"

**Score Difference:** 0.0170 (1.70 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8962 vs 0.8398 (Δ +0.0563)


</details>

---

## 84. Cerberus Capital

**Query:** `Cerberus Capital` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Cerberus Capital (New York, NY) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.8753 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Cerberus Capital (New York, NY) - 100.2% • *Perfect character-for-character match.*
2. YP-Cerberus Capital Mgmt (Glendale, CA) - 96.2% • *Substring match (target contains query text).*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Cerberus Capital"

**Score Difference:** 0.0409 (4.09 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7500 (Δ +0.2500)
- Semantic Similarity: 1.0000 vs 0.4639 (Δ +0.5361)


</details>
3. Cerberus Capital Management LP - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "YP-Cerberus Capital Mgmt"

**Score Difference:** 0.0058 (0.58 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.4639 vs 0.8131 (Δ -0.3492)


</details>
4. Cerberus Capital Management (New York, NY) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Cerberus Capital Management LP"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.8182 (Δ -0.0682)
- Semantic Similarity: 0.8131 vs 0.6828 (Δ +0.1304)


</details>
5. Cerberus Capital Management L (Herndon, VA) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Cerberus Capital Management"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.7500 (Δ +0.0682)


</details>
6. YP-Cerberus Capital Mgmt (Los Angeles, CA) - 95.6% • *Substring match (target contains query text).*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Cerberus Capital Management L"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6540 vs 0.4639 (Δ +0.1901)


</details>
7. CEC Capital - 74.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "YP-Cerberus Capital Mgmt"

**Score Difference:** 0.2080 (20.80 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.4639 vs 0.7422 (Δ -0.2783)


</details>
8. *Cerberus Capital (Minneapolis, MN) - 74.0% • *Substring match (target contains query text).*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "CEC Capital"

**Score Difference:** 0.0078 (0.78 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7422 vs 0.6142 (Δ +0.1281)


</details>
9. Cerberus - 71.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "*Cerberus Capital"

**Score Difference:** 0.0240 (2.40 percentage points)

**Key Differentiators:**
- String Similarity: 0.7875 vs 0.6300 (Δ +0.1575)
- Semantic Similarity: 0.6142 vs 0.9021 (Δ -0.2879)


</details>

---

## 85. Institute of Health Technology Transformation

**Query:** `Institute of Health Technology Transformation` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8678 | 70% | 0.6074 |
| Semantic Similarity (Normalized) | 0.9365 | 30% | 0.2810 |
| Semantic Similarity (Raw) | 4.7241 | - | - |
| **Base Score** | **0.8884** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
4 word(s) match exactly between your search and this company.

**Matching Words:**
• health, institute, technology, transformation

**Your Search Also Includes:**
• of

**Company Name Also Includes:**
• &, for

**Match Strength:**
• 67% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8678 (Weight: 70%)
• Semantic Similarity: 0.9365 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION - 95.6% • *High word-for-word overlap.*
2. Institute for Health Technology Transformation (Temecula, CA) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.9365 vs 0.7009 (Δ +0.2356)


</details>
3. Technology Health Experience - 77.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Institute for Health Technology Transformation"

**Score Difference:** 0.1778 (17.78 percentage points)

**Key Differentiators:**
- String Similarity: 0.8678 vs 0.6761 (Δ +0.1916)
- Semantic Similarity: 0.7009 vs 1.0000 (Δ -0.2991)


</details>
4. Health Technology Assessment International - 77.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Technology Health Experience"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 1.0000 vs 0.8289 (Δ +0.1711)


</details>
5. HEALTH TECHNOLOGY ASSESSMENT INT - 77.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Health Technology Assessment International"

**Score Difference:** 0.0070 (0.70 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Health Technology Association - 76.5% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "HEALTH TECHNOLOGY ASSESSMENT INT"

**Score Difference:** 0.0052 (0.52 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.8149 vs 0.9556 (Δ -0.1407)


</details>
7. Health Technology Assesment International - 76.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Health Technology Association"

**Score Difference:** 0.0024 (0.24 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.9556 vs 0.7899 (Δ +0.1657)


</details>
8. Health Technology Assessment - 76.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Health Technology Assesment International"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.7899 vs 0.9413 (Δ -0.1514)


</details>
9. Health Technology Exchange - 75.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Health Technology Assessment"

**Score Difference:** 0.0035 (0.35 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 86. The Jones Assembly

**Query:** `The Jones Assembly` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** The Jones Assembly Presents (Oklahoma City, OK) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity (Normalized) | 0.5728 | 30% | 0.1718 |
| Semantic Similarity (Raw) | 3.5947 | - | - |
| **Base Score** | **0.7446** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PREFIX MATCH

**What This Means:**
This company name starts with 'The Jones Assembly' and has additional information added.

**Action Required:**
• This is likely the same company with extra details
• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')
• If yes, use this match

**Why This Happens:**
• Someone entered just the core company name
• Your system has the full legal name
• Common in business databases where legal names include extra terms

**Top 10 Matches:**
1. The Jones Assembly Presents (Oklahoma City, OK) - 95.6% • *Direct prefix match (target contains extra trailing words).*
2. General Assembly - 74.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "The Jones Assembly Presents"

**Score Difference:** 0.2077 (20.77 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.7438 (Δ +0.0744)
- Semantic Similarity: 0.5728 vs 0.7430 (Δ -0.1702)


</details>
3. First Assembly - 74.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "General Assembly"

**Score Difference:** 0.0052 (0.52 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Jones Ag - 74.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "First Assembly"

**Score Difference:** 0.0022 (0.22 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Jones Companies - 73.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Jones Ag"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Jones Capital - 73.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Jones Companies"

**Score Difference:** 0.0044 (0.44 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. State Assembly - 73.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Jones Capital"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. MC Assembly - 73.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "State Assembly"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. John Jones - 72.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "MC Assembly"

**Score Difference:** 0.0039 (0.39 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 87. American Black Film Insitutute

**Query:** `American Black Film Insitutute` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** American Black Film Festival (New York, NY) • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 0.7347 | 30% | 0.2204 |
| Semantic Similarity (Raw) | 3.2801 | - | - |
| **Base Score** | **0.8110** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• american, black, film

**Your Search Also Includes:**
• insitutute

**Company Name Also Includes:**
• festival

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8438 (Weight: 70%)
• Semantic Similarity: 0.7347 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. American Black Film Festival (New York, NY) - 90.5% • *High word-for-word overlap.*
2. International Black Film Festival - 76.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "American Black Film Festival"

**Score Difference:** 0.1370 (13.70 percentage points)

**Key Differentiators:**
- String Similarity: 0.8438 vs 0.7438 (Δ +0.1000)
- Semantic Similarity: 0.7347 vs 0.8107 (Δ -0.0760)


</details>
3. Black Women Film Preservation (Atlanta, GA) - 76.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "International Black Film Festival"

**Score Difference:** 0.0045 (0.45 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Black Film Initiative - 76.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Black Women Film Preservation"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.7957 vs 0.9491 (Δ -0.1534)


</details>
5. American Black Film Festival Ventures (New York, NY) - 75.2% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Black Film Initiative"

**Score Difference:** 0.0109 (1.09 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7670 (Δ -0.0909)
- Semantic Similarity: 0.9491 vs 0.6922 (Δ +0.2569)


</details>
6. Black Star Film Festival - 74.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "American Black Film Festival Ventures"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6922 vs 0.7442 (Δ -0.0520)


</details>
7. American Film Works - 73.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Black Star Film Festival"

**Score Difference:** 0.0177 (1.77 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.7442 vs 0.8434 (Δ -0.0992)


</details>
8. Black Women Film Network (Atlanta, GA) - 72.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "American Film Works"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.8434 vs 0.6648 (Δ +0.1786)


</details>
9. American Black Film Festival Honors (New York, NY) - 72.8% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Black Women Film Network"

**Score Difference:** 0.0006 (0.06 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 88. Berk Tek

**Query:** `Berk Tek` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Berk Tek (Saint Louis, MO) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.7643 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Berk Tek (Saint Louis, MO) - 100.2% • *Perfect character-for-character match.*
2. Berk Tek (Chicago, IL) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Berk Tek"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Berk Tek (New Holland, PA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Berk Tek"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Berk-Tek (Lawrenceville, GA) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Berk Tek"

**Score Difference:** 0.0467 (4.67 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.9000 (Δ +0.1000)
- Semantic Similarity: 1.0000 vs 0.5303 (Δ +0.4697)


</details>
5. Berk Tek / Leviton (Chicago, IL) - 95.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Berk-Tek"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.9000 vs 0.7500 (Δ +0.1500)
- Semantic Similarity: 0.5303 vs 0.4762 (Δ +0.0542)


</details>
6. Berk-Tek,a Nexans Company (Denver, NC) - 90.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Berk Tek / Leviton"

**Score Difference:** 0.0503 (5.03 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.8182 (Δ -0.0682)
- Semantic Similarity: 0.4762 vs 0.4017 (Δ +0.0744)


</details>
7. Berk Tck - 75.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Berk-Tek,a Nexans Company"

**Score Difference:** 0.1553 (15.53 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.7438 (Δ +0.0744)
- Semantic Similarity: 0.4017 vs 0.7502 (Δ -0.3484)


</details>
8. Berk Technologies - 73.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Berk Tck"

**Score Difference:** 0.0129 (1.29 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. TEK Source - 73.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Berk Technologies"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 89. Northbridge Travel

**Query:** `Northbridge Travel` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Bridge Travel • **Score:** 78.7%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7875 | 70% | 0.5512 |
| Semantic Similarity (Normalized) | 0.7711 | 30% | 0.2313 |
| Semantic Similarity (Raw) | 5.0702 | - | - |
| **Base Score** | **0.7826** | - | - |
| **FINAL SCORE** | **0.7873** | - | **78.7%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7873
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• travel

**Your Search Also Includes:**
• northbridge

**Company Name Also Includes:**
• bridge

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7875 (Weight: 70%)
• Semantic Similarity: 0.7711 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Bridge Travel - 78.7% • *Hybrid match based on combined lexical and semantic features.*
2. Skybridge Travel - 77.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Bridge Travel"

**Score Difference:** 0.0096 (0.96 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7711 vs 0.8414 (Δ -0.0703)


</details>
3. Northbridge Insurance - 77.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Skybridge Travel"

**Score Difference:** 0.0066 (0.66 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. KingsBridge Travel - 76.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Northbridge Insurance"

**Score Difference:** 0.0018 (0.18 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Travel Bridge - 75.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "KingsBridge Travel"

**Score Difference:** 0.0100 (1.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Northbridge - 74.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Travel Bridge"

**Score Difference:** 0.0139 (1.39 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6300 (Δ +0.1138)
- Semantic Similarity: 0.7807 vs 1.0000 (Δ -0.2193)


</details>
7. Northbridge Financial Corp - 74.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Northbridge"

**Score Difference:** 0.0055 (0.55 percentage points)

**Key Differentiators:**
- String Similarity: 0.6300 vs 0.7438 (Δ -0.1138)
- Semantic Similarity: 1.0000 vs 0.7163 (Δ +0.2837)


</details>
8. Northbridge Insurance (Toronto, ON) - 73.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Northbridge Financial Corp"

**Score Difference:** 0.0024 (0.24 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7163 vs 0.8196 (Δ -0.1033)


</details>
9. Northbridge Environmental (Washington, DC) - 73.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Northbridge Insurance"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8196 vs 0.7071 (Δ +0.1125)


</details>

---

## 90. Kohler 2024 (Kohler, WI)

**Query:** `Kohler 2024` • **Location:** Kohler, WI • **Self-Match:** ✅ Found & Filtered

**Top Match:** Kohler 2024 (Franklin, TN) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8410 | 30% | 0.2523 |
| Semantic Similarity (Raw) | 3.8853 | - | - |
| **Base Score** | **0.9523** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Kohler 2024 (Franklin, TN) - 100.2% • *Perfect character-for-character match.*
2. Kohler Company Sales (Kohler, WI) - 94.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Kohler 2024"

**Score Difference:** 0.0616 (6.16 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8500 (Δ +0.1500)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
3. Kohler Company Marketing (Kohler, WI) - 93.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Kohler Company Sales"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Kohler Communications (Kohler, WI) - 93.7% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Kohler Company Marketing"

**Score Difference:** 0.0016 (0.16 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Kohler Engines (Kohler, WI) - 93.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Kohler Communications"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Kohler Company Finance (Kohler, WI) - 93.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Kohler Engines"

**Score Difference:** 0.0020 (0.20 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Kohler Schools (Kohler, WI) - 93.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Kohler Company Finance"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Kohler Energy (Kohler, WI) - 93.4% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Kohler Schools"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Kohler Company Accounting (Kohler, WI) - 93.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Kohler Energy"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
10. Kohler Fixtures (Kohler, WI) - 93.3% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "Kohler Company Accounting"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8366 vs 0.8967 (Δ -0.0601)


</details>

---

## 91. Louisiana State University Swim (Baton Rouge, LA)

**Query:** `Louisiana State University Swim` • **Location:** Baton Rouge, LA • **Self-Match:** ✅ Found & Filtered

**Top Match:** Louisiana State University Swim • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.4700 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. Louisiana State University Swim - 100.2% • *Perfect character-for-character match.*
2. Louisiana State University System (Baton Rouge, LA) - 94.2% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Louisiana State University Swim"

**Score Difference:** 0.0601 (6.01 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8250 (Δ +0.1750)
- Semantic Similarity: 1.0000 vs 0.5830 (Δ +0.4170)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
3. Louisiana State University Foundation (Baton Rouge, LA) - 92.9% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Louisiana State University System"

**Score Difference:** 0.0137 (1.37 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. Louisiana State University CCT (Baton Rouge, LA) - 92.9% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Louisiana State University Foundation"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Louisiana State Univ Swim (Baton Rouge, LA) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Louisiana State University CCT"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:**
- String Similarity: 0.8250 vs 0.9000 (Δ -0.0750)
- Semantic Similarity: 0.5351 vs 0.7069 (Δ -0.1718)


</details>
6. Louisiana State University Football (Baton Rouge, LA) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Louisiana State Univ Swim"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.9000 vs 0.8250 (Δ +0.0750)
- Semantic Similarity: 0.7069 vs 0.5986 (Δ +0.1083)


</details>
7. Louisiana State University Trips (Baton Rouge, LA) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Louisiana State University Football"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Louisiana State University Band (Baton Rouge, LA) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Louisiana State University Trips"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5878 vs 0.5141 (Δ +0.0737)


</details>
9. Louisiana State University System (New Orleans, LA) - 80.9% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Louisiana State University Band"

**Score Difference:** 0.1168 (11.68 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5141 vs 0.5830 (Δ -0.0689)
- Location Score: 100.0000 vs 40.0000 (Δ +60.0000)


</details>
10. Northwestern Louisiana State University (, LA) - 80.5% • *High word-for-word overlap.*
<details><summary><i>Why below #9?</i></summary>

## Relative Positioning Analysis (Rank #10)

### Why Ranked Below #9: "Louisiana State University System"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 92. X DO NOT USE - FRANCIS PARKER SCHOOL (San Diego, CA)

**Query:** `X DO NOT USE - FRANCIS PARKER SCHOOL` • **Location:** San Diego, CA • **Self-Match:** ✅ Found & Filtered

**Top Match:** X DO NOT USE - FRANCIS PARKER SCHOOL (Boston, MA) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8143 | 30% | 0.2443 |
| Semantic Similarity (Raw) | 3.9476 | - | - |
| **Base Score** | **0.9443** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. X DO NOT USE - FRANCIS PARKER SCHOOL (Boston, MA) - 100.4% • *Perfect character-for-character match.*
2. X DO NOT USE - FRANCIS PARKER SCHOOL - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "X DO NOT USE - FRANCIS PARKER SCHOOL"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. X DO NOT USE - FRANCIS PARKER SCHOOL (East Windsor, NJ) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "X DO NOT USE - FRANCIS PARKER SCHOOL"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. X DO NOT USE - FRANCIS PARKER SCHOOL (Brighton, MA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "X DO NOT USE - FRANCIS PARKER SCHOOL"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. X DO NOT USE - FRANCIS PARKER SCHOOL (Seattle, WA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "X DO NOT USE - FRANCIS PARKER SCHOOL"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. X DO NOT USE - FRANCIS PARKER SCHOOL (Swampscott, MA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "X DO NOT USE - FRANCIS PARKER SCHOOL"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. X DO NOT USE - CLAIREMONT HIGH SCHOOL (San Diego, CA) - 92.5% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "X DO NOT USE - FRANCIS PARKER SCHOOL"

**Score Difference:** 0.0770 (7.70 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8036 (Δ +0.1964)
- Semantic Similarity: 0.8143 vs 0.4520 (Δ +0.3623)
- Location Score: 0.0000 vs 100.0000 (Δ -100.0000)


</details>
8. Francis Parker School (San Diego, CA) - 68.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "X DO NOT USE - CLAIREMONT HIGH SCHOOL"

**Score Difference:** 0.2368 (23.68 percentage points)

**Key Differentiators:**
- String Similarity: 0.8036 vs 0.5464 (Δ +0.2571)
- Semantic Similarity: 0.4520 vs 0.7258 (Δ -0.2738)


</details>
9. Francis Parker School English Department (SAN DIEGO, CA) - 62.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Francis Parker School"

**Score Difference:** 0.0595 (5.95 percentage points)

**Key Differentiators:**
- String Similarity: 0.5464 vs 0.4756 (Δ +0.0708)
- Semantic Similarity: 0.7258 vs 0.6652 (Δ +0.0606)


</details>

---

## 93. Mitsubishi Motor Sales of America, Incorporated

**Query:** `Mitsubishi Motor Sales of America, Incorporated` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Mitsubishi Motor Sales of America (Cypress, CA) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8096 | 30% | 0.2429 |
| Semantic Similarity (Raw) | 4.9950 | - | - |
| **Base Score** | **0.9429** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
4 word(s) match exactly between your search and this company.

**Matching Words:**
• mitsubishi, motor, of, sales

**Your Search Also Includes:**
• america,, incorporated

**Company Name Also Includes:**
• america

**Match Strength:**
• 67% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 1.0000 (Weight: 70%)
• Semantic Similarity: 0.8096 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Mitsubishi Motor Sales of America (Cypress, CA) - 95.6% • *High word-for-word overlap.*
2. Mitsubishi Motor Sales of America, Inc. (Irvine, CA) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Mitsubishi Motor Sales of America"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8096 vs 0.6553 (Δ +0.1543)


</details>
3. Mitsubishi Electronic Sales America - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Mitsubishi Motor Sales of America, Inc."

**Score Difference:** 0.0503 (5.03 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.8438 (Δ +0.1562)
- Semantic Similarity: 0.6553 vs 0.8981 (Δ -0.2429)


</details>
4. Mitsubishi Motors Sales of America (Cypress, CA) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Mitsubishi Electronic Sales America"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8981 vs 0.7692 (Δ +0.1289)


</details>
5. Mitsubishi Electric Sales of America (Cypress, CA) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Mitsubishi Motors Sales of America"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Mitsubishi Motor Sales Of Amer (Cypress, CA) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Mitsubishi Electric Sales of America"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. Mitsubishi Motor Sales of Canada, Incorporated (Toronto, ON) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Mitsubishi Motor Sales Of Amer"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Mitsubishi Motor Sales of Caribbean (San Juan, PR) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Mitsubishi Motor Sales of Canada, Incorporated"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. MITSUBISHI MOTOR NORTH AMERICA, INC (Auburn Hills, MI) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Mitsubishi Motor Sales of Caribbean"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6807 vs 0.5829 (Δ +0.0978)


</details>

---

## 94. Energy Distribution Partners Holdings'

**Query:** `Energy Distribution Partners Holdings'` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Energy Distribution Partners (EDP) • **Score:** 90.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity (Normalized) | 0.7254 | 30% | 0.2176 |
| Semantic Similarity (Raw) | 4.1316 | - | - |
| **Base Score** | **0.7951** | - | - |
| **FINAL SCORE** | **0.9055** | - | **90.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9055
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• distribution, energy, partners

**Your Search Also Includes:**
• holdings'

**Company Name Also Includes:**
• (edp)

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8250 (Weight: 70%)
• Semantic Similarity: 0.7254 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Energy Distribution Partners (EDP) - 90.5% • *High word-for-word overlap.*
2. Energy Distribution Partners Holdings L.P. (Chicago, IL) - 90.5% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Energy Distribution Partners (EDP)"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.7254 vs 0.6671 (Δ +0.0583)


</details>
3. Energy Distribution Holdings - 78.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Energy Distribution Partners Holdings L.P."

**Score Difference:** 0.1195 (11.95 percentage points)

**Key Differentiators:**
- String Similarity: 0.8250 vs 0.6875 (Δ +0.1375)
- Semantic Similarity: 0.6671 vs 1.0000 (Δ -0.3329)


</details>
4. Energy Distribution Partners (Chicago, IL) - 75.6% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Energy Distribution Holdings"

**Score Difference:** 0.0295 (2.95 percentage points)

**Key Differentiators:**
- String Similarity: 0.6875 vs 0.7500 (Δ -0.0625)
- Semantic Similarity: 1.0000 vs 0.7364 (Δ +0.2636)


</details>
5. Energy Transfer Partners LP - 73.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Energy Distribution Partners"

**Score Difference:** 0.0251 (2.51 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Energy Power Partners - 72.1% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Energy Transfer Partners LP"

**Score Difference:** 0.0100 (1.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.6439 (Δ +0.0644)
- Semantic Similarity: 0.7703 vs 0.8874 (Δ -0.1171)


</details>
7. EIG Global Energy Partners - 72.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Energy Power Partners"

**Score Difference:** 0.0012 (0.12 percentage points)

**Key Differentiators:**
- String Similarity: 0.6439 vs 0.7083 (Δ -0.0644)
- Semantic Similarity: 0.8874 vs 0.7333 (Δ +0.1541)


</details>
8. Energy Impact Partners - 71.0% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "EIG Global Energy Partners"

**Score Difference:** 0.0097 (0.97 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.6439 (Δ +0.0644)
- Semantic Similarity: 0.7333 vs 0.8513 (Δ -0.1180)


</details>
9. Energy Transfer Partners LP (Houston, TX) - 70.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Energy Impact Partners"

**Score Difference:** 0.0030 (0.30 percentage points)

**Key Differentiators:**
- String Similarity: 0.6439 vs 0.7083 (Δ -0.0644)
- Semantic Similarity: 0.8513 vs 0.7703 (Δ +0.0810)


</details>

---

## 95. ThinkAdvisor

**Query:** `ThinkAdvisor` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Planadvisor • **Score:** 46.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.3130 | 70% | 0.2191 |
| Semantic Similarity (Normalized) | 0.8190 | 30% | 0.2457 |
| Semantic Similarity (Raw) | 4.6040 | - | - |
| **Base Score** | **0.4648** | - | - |
| **FINAL SCORE** | **0.4676** | - | **46.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.4676
```

### Component Analysis

- **String Similarity (WEAK):** Low lexical match - minimal word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
SEMANTIC MATCH (Score: 0.47)

**What This Means:**
The AI model found a meaning-based connection, but no direct word overlap.

**Score Breakdown:**
• Lexical Similarity: 0.3130 (Weight: 70%)
• Semantic Similarity: 0.8190 (Weight: 30%)

**Confidence Level:**
• Poor (40-49%) - Weak semantic relationship

**Action Required:**
• This is a LOWER confidence match
• CAREFULLY verify if these companies are actually related
• Check address and other details

**Top 10 Matches:**
1. Planadvisor - 46.8% • *Hybrid match based on combined lexical and semantic features.*
2. TripAdvisor - 46.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Planadvisor"

**Score Difference:** 0.0079 (0.79 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8190 vs 0.7015 (Δ +0.1175)


</details>
3. Invisors - 43.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "TripAdvisor"

**Score Difference:** 0.0238 (2.38 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. ChannelAdvisor - 42.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Invisors"

**Score Difference:** 0.0134 (1.34 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. HomeAdvisor - 41.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "ChannelAdvisor"

**Score Difference:** 0.0067 (0.67 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. Tripadvisor (Providence, RI) - 40.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "HomeAdvisor"

**Score Difference:** 0.0088 (0.88 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6470 vs 0.5265 (Δ +0.1205)


</details>
7. Tripadvisor (Ames, IA) - 40.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Tripadvisor"

**Score Difference:** 0.0032 (0.32 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. TRIPADVISOR (Mobile, AL) - 40.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Tripadvisor"

**Score Difference:** 0.0036 (0.36 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. ScoutAdvisor Corporation - 39.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "TRIPADVISOR"

**Score Difference:** 0.0051 (0.51 percentage points)

**Key Differentiators:**
- String Similarity: 0.3522 vs 0.3000 (Δ +0.0522)
- Semantic Similarity: 0.5041 vs 0.6088 (Δ -0.1047)


</details>

---

## 96. Jump on it Outreach

**Query:** `Jump on it Outreach` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Jump On It • **Score:** 69.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7159 | 70% | 0.5011 |
| Semantic Similarity (Normalized) | 0.6301 | 30% | 0.1890 |
| Semantic Similarity (Raw) | 3.6371 | - | - |
| **Base Score** | **0.6902** | - | - |
| **FINAL SCORE** | **0.6944** | - | **69.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.6944
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• it, jump, on

**Your Search Also Includes:**
• outreach

**Match Strength:**
• 75% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.7159 (Weight: 70%)
• Semantic Similarity: 0.6301 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Jump On It - 69.4% • *High word-for-word overlap.*
2. Evangelism on the Move Outreach Ministry - 62.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Jump On It"

**Score Difference:** 0.0742 (7.42 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6301 vs 0.4771 (Δ +0.1530)


</details>
3. Outreach - 61.9% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Evangelism on the Move Outreach Ministry"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.4500 (Δ +0.2261)
- Semantic Similarity: 0.4771 vs 1.0000 (Δ -0.5229)


</details>
4. Above N Beyond Outreach - 61.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Outreach"

**Score Difference:** 0.0040 (0.40 percentage points)

**Key Differentiators:**
- String Similarity: 0.4500 vs 0.5250 (Δ -0.0750)
- Semantic Similarity: 1.0000 vs 0.8118 (Δ +0.1882)


</details>
5. Community Connections Outreach Program - 59.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Above N Beyond Outreach"

**Score Difference:** 0.0232 (2.32 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.8118 vs 0.7348 (Δ +0.0770)


</details>
6. Outreach Strategies - 57.8% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Community Connections Outreach Program"

**Score Difference:** 0.0137 (1.37 percentage points)

**Key Differentiators:**
- String Similarity: 0.5250 vs 0.4375 (Δ +0.0875)
- Semantic Similarity: 0.7348 vs 0.8936 (Δ -0.1588)


</details>
7. Jump - 57.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Outreach Strategies"

**Score Difference:** 0.0051 (0.51 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Where are you? Outreach - 57.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "Jump"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:**
- String Similarity: 0.4500 vs 0.5250 (Δ -0.0750)
- Semantic Similarity: 0.8475 vs 0.6678 (Δ +0.1797)


</details>
9. Human Outreach Project - 56.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Where are you? Outreach"

**Score Difference:** 0.0021 (0.21 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6678 vs 0.7721 (Δ -0.1043)


</details>

---

## 97. The Association of Ringside Consultants (ARC)

**Query:** `The Association of Ringside Consultants (ARC)` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Virginia Association of Legal Consultants • **Score:** 71.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.6434 | 30% | 0.1930 |
| Semantic Similarity (Raw) | 3.0567 | - | - |
| **Base Score** | **0.7136** | - | - |
| **FINAL SCORE** | **0.7180** | - | **71.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7180
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
3 word(s) match exactly between your search and this company.

**Matching Words:**
• association, consultants, of

**Your Search Also Includes:**
• (arc), ringside, the

**Company Name Also Includes:**
• legal, virginia

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.6434 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. Virginia Association of Legal Consultants - 71.8% • *Hybrid match based on combined lexical and semantic features.*
2. Oklahoma Association of Personnel Consultants - 71.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "Virginia Association of Legal Consultants"

**Score Difference:** 0.0031 (0.31 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Association of Ringside Physicians - 71.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Oklahoma Association of Personnel Consultants"

**Score Difference:** 0.0024 (0.24 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6330 vs 0.7827 (Δ -0.1497)


</details>
4. Association of Charlotte Area Consultants - 70.8% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Association of Ringside Physicians"

**Score Difference:** 0.0045 (0.45 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.7827 vs 0.6016 (Δ +0.1811)


</details>
5. Professional Consultants Association - 69.7% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "Association of Charlotte Area Consultants"

**Score Difference:** 0.0109 (1.09 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6016 vs 0.7317 (Δ -0.1300)


</details>
6. Investment Management Consultants Association - 69.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Professional Consultants Association"

**Score Difference:** 0.0025 (0.25 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.7317 vs 0.5656 (Δ +0.1660)


</details>
7. American Association of Ringside Physicians (Darien, CT) - 69.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "Investment Management Consultants Association"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. Association of Professional Investment Consultants - 69.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "American Association of Ringside Physicians"

**Score Difference:** 0.0023 (0.23 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. Association of Ringside Physicians (ARP) (Madison, WI) - 69.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Association of Professional Investment Consultants"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 98. SFA HASA

**Query:** `SFA HASA` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** SFA Partners • **Score:** 76.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8096 | 30% | 0.2429 |
| Semantic Similarity (Raw) | 4.7743 | - | - |
| **Base Score** | **0.7635** | - | - |
| **FINAL SCORE** | **0.7681** | - | **76.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7681
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
1 word(s) match exactly between your search and this company.

**Matching Words:**
• sfa

**Your Search Also Includes:**
• hasa

**Company Name Also Includes:**
• partners

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7438 (Weight: 70%)
• Semantic Similarity: 0.8096 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. SFA Partners - 76.8% • *Hybrid match based on combined lexical and semantic features.*
2. SFA Opportunity - 75.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "SFA Partners"

**Score Difference:** 0.0119 (1.19 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. Sfa - 74.6% • *Matched via strong semantic/conceptual similarity.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "SFA Opportunity"

**Score Difference:** 0.0107 (1.07 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6300 (Δ +0.1138)
- Semantic Similarity: 0.7702 vs 1.0000 (Δ -0.2298)


</details>
4. SFA Training - 74.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Sfa"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:**
- String Similarity: 0.6300 vs 0.7438 (Δ -0.1138)
- Semantic Similarity: 1.0000 vs 0.7313 (Δ +0.2687)


</details>
5. Test Sfa - 73.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "SFA Training"

**Score Difference:** 0.0110 (1.10 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. SFA System Account - 71.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Test Sfa"

**Score Difference:** 0.0190 (1.90 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6948 vs 0.7897 (Δ -0.0948)


</details>
7. SFA Saniflo (Netherlands, ) - 71.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "SFA System Account"

**Score Difference:** 0.0003 (0.03 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.7897 vs 0.6308 (Δ +0.1589)


</details>
8. SFA Design (Santa Barbara, CA) - 70.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "SFA Saniflo"

**Score Difference:** 0.0050 (0.50 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. SFA Designs (Santa Barbara, CA) - 70.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "SFA Design"

**Score Difference:** 0.0028 (0.28 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 99. Grupo Duracell Ene 2025

**Query:** `Grupo Duracell Ene 2025` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** GRUPO MAZDA FEB 2025 • **Score:** 66.7%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity (Normalized) | 0.5586 | 30% | 0.1676 |
| Semantic Similarity (Raw) | 2.8738 | - | - |
| **Base Score** | **0.6634** | - | - |
| **FINAL SCORE** | **0.6675** | - | **66.7%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.6675
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
2 word(s) match exactly between your search and this company.

**Matching Words:**
• 2025, grupo

**Your Search Also Includes:**
• duracell, ene

**Company Name Also Includes:**
• feb, mazda

**Match Strength:**
• 50% word overlap
• This is a MODERATE match - worth investigating
• Action: Check if this makes business sense

**Score Breakdown:**
• Lexical Similarity: 0.7083 (Weight: 70%)
• Semantic Similarity: 0.5586 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. GRUPO MAZDA FEB 2025 - 66.7% • *Hybrid match based on combined lexical and semantic features.*
2. Workshop 2025 - Grupo CCM - 64.9% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "GRUPO MAZDA FEB 2025"

**Score Difference:** 0.0185 (1.85 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5586 vs 0.4975 (Δ +0.0612)


</details>
3. Grupo Eñe de  Comunicación - 52.4% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "Workshop 2025 - Grupo CCM"

**Score Difference:** 0.1252 (12.52 percentage points)

**Key Differentiators:**
- String Similarity: 0.7083 vs 0.5000 (Δ +0.2083)
- Semantic Similarity: 0.4975 vs 0.5688 (Δ -0.0713)


</details>
4. GRUPO Convenciones Y Eventos - 51.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "Grupo Eñe de  Comunicación"

**Score Difference:** 0.0076 (0.76 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. Grupo Brasil DPE - 51.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "GRUPO Convenciones Y Eventos"

**Score Difference:** 0.0004 (0.04 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.5436 vs 0.6485 (Δ -0.1048)


</details>
6. GRUPO Grand De Mexico - 51.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "Grupo Brasil DPE"

**Score Difference:** 0.0007 (0.07 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6485 vs 0.5402 (Δ +0.1083)


</details>
7. DURACELL USA - 51.1% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "GRUPO Grand De Mexico"

**Score Difference:** 0.0039 (0.39 percentage points)

**Key Differentiators:**
- String Similarity: 0.5000 vs 0.4167 (Δ +0.0833)
- Semantic Similarity: 0.5402 vs 0.7218 (Δ -0.1817)


</details>
8. Evento Grupo La Norteñita - 51.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "DURACELL USA"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:**
- String Similarity: 0.4167 vs 0.5000 (Δ -0.0833)
- Semantic Similarity: 0.7218 vs 0.5240 (Δ +0.1978)


</details>
9. Grupo BG de eventos (Caracas, ) - 51.0% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Evento Grupo La Norteñita"

**Score Difference:** -0.0000 (-0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 100. World Association of Medical Law

**Query:** `World Association of Medical Law` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** World Association for Medical Law (Marceline, MO) • **Score:** 95.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity (Normalized) | 0.6155 | 30% | 0.1847 |
| Semantic Similarity (Raw) | 3.2480 | - | - |
| **Base Score** | **0.7892** | - | - |
| **FINAL SCORE** | **0.9592** | - | **95.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9592
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH

**What This Means:**
4 word(s) match exactly between your search and this company.

**Matching Words:**
• association, law, medical, world

**Your Search Also Includes:**
• of

**Company Name Also Includes:**
• for

**Match Strength:**
• 80% word overlap
• This is a STRONG match - likely the same company
• Action: Use this match with high confidence

**Score Breakdown:**
• Lexical Similarity: 0.8636 (Weight: 70%)
• Semantic Similarity: 0.6155 (Weight: 30%)

**Why This Happens:**
• Company names often have multiple words
• Some words are more important than others
• Business names can vary in how they're written

**Top 10 Matches:**
1. World Association for Medical Law (Marceline, MO) - 95.9% • *High word-for-word overlap.*
2. World Association For Medical Law - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "World Association for Medical Law"

**Score Difference:** 0.0034 (0.34 percentage points)

**Key Differentiators:**
- Semantic Similarity: 0.6155 vs 1.0000 (Δ -0.3845)


</details>
3. World Association for Medical Law (Chesterfield, MO) - 95.6% • *High word-for-word overlap.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "World Association For Medical Law"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- Semantic Similarity: 1.0000 vs 0.6155 (Δ +0.3845)


</details>
4. World Medical Association - 80.4% • *High word-for-word overlap.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "World Association for Medical Law"

**Score Difference:** 0.1513 (15.13 percentage points)

**Key Differentiators:**
- String Similarity: 0.8636 vs 0.7159 (Δ +0.1477)
- Semantic Similarity: 0.6155 vs 0.9949 (Δ -0.3794)


</details>
5. World Law Foundation - 74.6% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "World Medical Association"

**Score Difference:** 0.0590 (5.90 percentage points)

**Key Differentiators:**
- String Similarity: 0.7159 vs 0.7727 (Δ -0.0568)
- Semantic Similarity: 0.9949 vs 0.6670 (Δ +0.3279)


</details>
6. International Law Student Association - 73.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "World Law Foundation"

**Score Difference:** 0.0138 (1.38 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. International Law Association - 72.3% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "International Law Student Association"

**Score Difference:** 0.0089 (0.89 percentage points)

**Key Differentiators:**
- String Similarity: 0.7438 vs 0.6761 (Δ +0.0676)
- Semantic Similarity: 0.6887 vs 0.8171 (Δ -0.1284)


</details>
8. Pacific Medical Law - 72.2% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "International Law Association"

**Score Difference:** 0.0009 (0.09 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. World Korean Medical Organization - 71.5% • *Hybrid match based on combined lexical and semantic features.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "Pacific Medical Law"

**Score Difference:** 0.0073 (0.73 percentage points)

**Key Differentiators:**
- String Similarity: 0.6761 vs 0.7438 (Δ -0.0676)
- Semantic Similarity: 0.8140 vs 0.6319 (Δ +0.1821)


</details>

---

## 101. ABA

**Query:** `ABA` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** ABA • **Score:** 100.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.3790 | 30% | 0.1137 |
| Semantic Similarity (Raw) | 2.8337 | - | - |
| **Base Score** | **0.8137** | - | - |
| **FINAL SCORE** | **1.0049** | - | **100.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0049
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (FAIR):** Some meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. ABA - 100.5% • *Perfect character-for-character match.*
2. ABA (Washington, DC) - 100.5% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "ABA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. ABA (Tacoma, WA) - 100.5% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "ABA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. ABA (Sao Paulo, Sao Paulo) - 100.5% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "ABA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. ABA (New York, NY) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "ABA"

**Score Difference:** 0.0010 (0.10 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. ABA (Carollton, TX) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "ABA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. ABA (, IL) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "ABA"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. ABA (Phoenix, AZ) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "ABA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. ABA (Plantation, FL) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "ABA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

## 102. PDMA

**Query:** `PDMA` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** PDMA (New York, NY) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.4074 | 30% | 0.1222 |
| Semantic Similarity (Raw) | 2.8941 | - | - |
| **Base Score** | **0.8222** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (FAIR):** Some meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. PDMA (New York, NY) - 100.4% • *Perfect character-for-character match.*
2. PDMA (Indianapolis, IN) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "PDMA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. PDMA (Saint Paul, MN) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "PDMA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. PDMA (Naples, FL) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "PDMA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. PDMA - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "PDMA"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. PDMA (, FL) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "PDMA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. PDMA (Ridgefield, CT) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "PDMA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. PDMA (Mount Laurel, NJ) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "PDMA"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. PdMA Corporation (Tampa, FL) - 96.2% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "PDMA"

**Score Difference:** 0.0409 (4.09 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.9000 (Δ +0.1000)
- Semantic Similarity: 0.4074 vs 0.5471 (Δ -0.1397)


</details>

---

## 103. IBM

**Query:** `IBM` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** IBM (Greely, ) • **Score:** 100.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 7.3994 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0024** | - | **100.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0024
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. IBM (Greely, ) - 100.2% • *Perfect character-for-character match.*
2. IBM (Charlotte, NC) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "IBM"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. IBM Belgium SA (Brussels, ) - 96.6% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "IBM"

**Score Difference:** 0.0362 (3.62 percentage points)

**Key Differentiators:**
- String Similarity: 1.0000 vs 0.7500 (Δ +0.2500)
- Semantic Similarity: 1.0000 vs 0.3792 (Δ +0.6208)


</details>
4. International Business Machines IBM (New York, NY) - 96.5% • *Substring match (target contains query text).*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "IBM Belgium SA"

**Score Difference:** 0.0013 (0.13 percentage points)

**Key Differentiators:**
- String Similarity: 0.7500 vs 0.6923 (Δ +0.0577)
- Semantic Similarity: 0.3792 vs 0.5126 (Δ -0.1334)


</details>
5. International Business Machines IBM (Austin, TX) - 96.3% • *Substring match (target contains query text).*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "International Business Machines IBM"

**Score Difference:** 0.0015 (0.15 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. International Business Machines IBM (Atlanta, GA) - 96.3% • *Substring match (target contains query text).*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "International Business Machines IBM"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. IBM Corporation OLD (Austin, TX) - 96.3% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "International Business Machines IBM"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.6923 vs 0.8182 (Δ -0.1259)
- Semantic Similarity: 0.5126 vs 0.4519 (Δ +0.0606)


</details>
8. IBM Global Business Service (Dallas, TX) - 96.3% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "IBM Corporation OLD"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:**
- String Similarity: 0.8182 vs 0.6923 (Δ +0.1259)


</details>
9. IBM India (Bangalore, ) - 96.2% • *Direct prefix match (target contains extra trailing words).*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "IBM Global Business Service"

**Score Difference:** 0.0019 (0.19 percentage points)

**Key Differentiators:**
- String Similarity: 0.6923 vs 0.8182 (Δ -0.1259)
- Semantic Similarity: 0.4124 vs 0.5254 (Δ -0.1130)


</details>

---

## 104. GE

**Query:** `GE` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** GE (Chicago, IL) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 7.4052 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH

**What This Means:**
This is exactly the same company name you're looking for.

**Action Required:**
• Use this match - no further checking needed
• This is 100% the same company

**Why This Happens:**
• Someone entered the company name name exactly as it appears in your system
• This is the ideal scenario for data entry

**Top 10 Matches:**
1. GE (Chicago, IL) - 100.4% • *Perfect character-for-character match.*
2. GE (Orlando, FL) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #1?</i></summary>

## Relative Positioning Analysis (Rank #2)

### Why Ranked Below #1: "GE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
3. GE (Milwaukee, WI) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #2?</i></summary>

## Relative Positioning Analysis (Rank #3)

### Why Ranked Below #2: "GE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
4. GE (Naperville, IL) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #3?</i></summary>

## Relative Positioning Analysis (Rank #4)

### Why Ranked Below #3: "GE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
5. GE (West Des Moines, IA) - 100.4% • *Perfect character-for-character match.*
<details><summary><i>Why below #4?</i></summary>

## Relative Positioning Analysis (Rank #5)

### Why Ranked Below #4: "GE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
6. GE (Atlanta, GA) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #5?</i></summary>

## Relative Positioning Analysis (Rank #6)

### Why Ranked Below #5: "GE"

**Score Difference:** 0.0014 (0.14 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
7. GE (Weston, FL) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #6?</i></summary>

## Relative Positioning Analysis (Rank #7)

### Why Ranked Below #6: "GE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
8. GE (Littleton, CO) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #7?</i></summary>

## Relative Positioning Analysis (Rank #8)

### Why Ranked Below #7: "GE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>
9. GE (Houston, TX) - 100.2% • *Perfect character-for-character match.*
<details><summary><i>Why below #8?</i></summary>

## Relative Positioning Analysis (Rank #9)

### Why Ranked Below #8: "GE"

**Score Difference:** 0.0000 (0.00 percentage points)

**Key Differentiators:** Scores are very similar - minor differences across components


</details>

---

