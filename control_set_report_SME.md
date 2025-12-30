# Control Set Report - SME Review
**Generated:** 2025-12-29 19:22:26

## Matching Methodology Overview

This report uses a **hybrid semantic + lexical matching** approach:

| Component | Weight | Description |
|-----------|--------|-------------|
| **String Similarity** | 70% | Lexical comparison using Jaro-Winkler distance on normalized company names |
| **Semantic Similarity** | 30% | Neural embedding comparison using SentenceTransformers (paraphrase-MiniLM-L3-v2) |
| **Acronym Fidelity** | +15% max | Bonus for matching acronym expansions (e.g., IBM → International Business Machines) |
| **Location Boost** | +5% max | Bonus when query location matches company location |

### Score Interpretation

| Score Range | Confidence | Recommendation |
|-------------|------------|----------------|
| ≥ 95% | **Exact/Near-Exact** | Auto-accept match |
| 80-94% | **High** | Likely correct, quick verification |
| 60-79% | **Medium** | Requires human review |
| 40-59% | **Low** | Multiple candidates, careful selection needed |
| < 40% | **Very Low** | May need manual search |

---

## 1. Query: `PDMA Association`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | PDMA Association | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Association Headquarters-PDMA | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | PDMA Alliance | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 4 | PDMA ALLIANCE | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | PDMA ALLIANCE | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | PDMA ALLIANCE | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | PDMA Alliance Inc. | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | PDMA Alliance | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | PDMA Alliance | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | PDMA | **80.9%** | 🟢 HIGH **80.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: PDMA Association

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.6802 | 30% | 1.1041 |
| **Base Score** | - | - | **1.8041** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** association, pdma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.680)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Association Headquarters-PDMA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 2.8192 | 30% | 0.8458 |
| **Base Score** | - | - | **1.4503** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** association
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.819)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.0471 higher than #3

---

### Rank #3: PDMA Alliance

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.4177 | 30% | 1.0253 |
| **Base Score** | - | - | **1.6203** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.418)

**Ranking Justification:**

- Ranked **#3** - score is 0.0471 lower than #2
- Score is 0.0032 higher than #4

---

### Rank #4: PDMA ALLIANCE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 5.0835 | 30% | 1.5251 |
| **Base Score** | - | - | **2.1201** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.084)

**Ranking Justification:**

- Ranked **#4** - score is 0.0032 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: PDMA ALLIANCE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 4.3325 | 30% | 1.2997 |
| **Base Score** | - | - | **1.8947** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.332)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 2. Query: `Nicolas/Sanchez Wedding`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Nicolas/Sanchez Wedding | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Sanchez/Justin Wedding | **80.0%** | 🟢 HIGH **80.0%** | Semantic/meaning-based match |
| 3 | Sanchez Wedding | **79.3%** | 🟡 MEDIUM **79.3%** | Semantic/meaning-based match |
| 4 | Sanchez/Ramirez Wedding | **78.8%** | 🟡 MEDIUM **78.8%** | Semantic/meaning-based match |
| 5 | Garcia Sanchez Wedding | **77.8%** | 🟡 MEDIUM **77.8%** | Semantic/meaning-based match |
| 6 | Gibson Sanchez Wedding | **77.0%** | 🟡 MEDIUM **77.0%** | Semantic/meaning-based match |
| 7 | Sanchez/Puerto Wedding | **76.9%** | 🟡 MEDIUM **76.9%** | Semantic/meaning-based match |
| 8 | Sanchez/Cohen Wedding | **76.2%** | 🟡 MEDIUM **76.2%** | Semantic/meaning-based match |
| 9 | Sanchez/Naranjo Wedding | **75.9%** | 🟡 MEDIUM **75.9%** | Semantic/meaning-based match |
| 10 | Sanchez/Fuentes Wedding | **75.9%** | 🟡 MEDIUM **75.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Nicolas/Sanchez Wedding

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.6541 | 30% | 1.0962 |
| **Base Score** | - | - | **1.7962** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** nicolas/sanchez, wedding
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.654)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2022 (20.2%)

---

### Rank #2: Sanchez/Justin Wedding

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.6285 | 30% | 1.3886 |
| **Base Score** | - | - | **1.9092** |
| **Final Score** | - | - | **0.8002** (80.0%) |

**Why This Matched:**

- **Word Overlap:** wedding
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.629)

**Ranking Justification:**

- Ranked **#2** - score is 0.2022 lower than #1
- Score is 0.0077 higher than #3

---

### Rank #3: Sanchez Wedding

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7875 | 70% | 0.5512 |
| Semantic Similarity | 3.9845 | 30% | 1.1954 |
| **Base Score** | - | - | **1.7466** |
| **Final Score** | - | - | **0.7926** (79.3%) |

**Why This Matched:**

- **Word Overlap:** wedding
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.787)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.985)

**Ranking Justification:**

- Ranked **#3** - score is 0.0077 lower than #2
- Score is 0.0046 higher than #4

---

### Rank #4: Sanchez/Ramirez Wedding

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.4232 | 30% | 1.3270 |
| **Base Score** | - | - | **1.8476** |
| **Final Score** | - | - | **0.7880** (78.8%) |

**Why This Matched:**

- **Word Overlap:** wedding
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.423)

**Ranking Justification:**

- Ranked **#4** - score is 0.0046 lower than #3
- Score is 0.0100 higher than #5

---

### Rank #5: Garcia Sanchez Wedding

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 5.0535 | 30% | 1.5161 |
| **Base Score** | - | - | **1.9893** |
| **Final Score** | - | - | **0.7780** (77.8%) |

**Why This Matched:**

- **Word Overlap:** wedding
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.054)

**Ranking Justification:**

- Ranked **#5** - score is 0.0100 lower than #4
- Score is 0.0076 higher than #6

---

</details>
## 3. Query: `Kehilat Ariel Synagogue`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Kehilat Ariel Synagogue | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Kehilat Ariel Messianic Synagogue | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | Kehilat Ariel | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Kehilat Ariel Passover | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | KAS | **69.0%** | 🟡 MEDIUM **69.0%** | Near-exact text match |
| 6 | Ariel Healing Arts | **61.6%** | 🟡 MEDIUM **61.6%** | Semantic/meaning-based match |
| 7 | Synagogue 3000 Organization | **60.9%** | 🟡 MEDIUM **60.9%** | Semantic/meaning-based match |
| 8 | Jewish Synagogue | **60.6%** | 🟡 MEDIUM **60.6%** | Semantic/meaning-based match |
| 9 | Temple Sinai Synagogue | **59.6%** | 🟠 LOW **59.6%** | Semantic/meaning-based match |
| 10 | Ariel Foundation | **59.3%** | 🟠 LOW **59.3%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Kehilat Ariel Synagogue

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4039 | 30% | 1.3212 |
| **Base Score** | - | - | **2.0212** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** ariel, kehilat, synagogue
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.404)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Kehilat Ariel Messianic Synagogue

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 3.8747 | 30% | 1.1624 |
| **Base Score** | - | - | **1.7670** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** ariel, kehilat, synagogue
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.875)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.0503 higher than #3

---

### Rank #3: Kehilat Ariel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 2.7727 | 30% | 0.8318 |
| **Base Score** | - | - | **1.4045** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** ariel, kehilat
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.773)

**Ranking Justification:**

- Ranked **#3** - score is 0.0503 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Kehilat Ariel Passover

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 2.6183 | 30% | 0.7855 |
| **Base Score** | - | - | **1.4038** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** ariel, kehilat
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.883)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.618)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.2155 higher than #5

---

### Rank #5: KAS

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | - | - | **1.0000** |
| Acronym Fidelity | 0.7000 | +15% max | +0.1050 |
| **Final Score** | - | - | **0.6900** (69.0%) |

**Why This Matched:**

- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 1.000)
- **Acronym:** Possible acronym relationship (70% fidelity)

**Ranking Justification:**

- Ranked **#5** - score is 0.2155 lower than #4
- Score is 0.0745 higher than #6

---

</details>
## 4. Query: `Next Level Events`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Next Level Events | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 2 | Next Level Events | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 3 | Next Level Events | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 4 | Next Level Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | Next Level Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 6 | NEXT LEVEL EVENTS | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 7 | Next Level Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 8 | NEXT LEVEL EVENTS | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 9 | Next Level Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 10 | Next Level Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Next Level Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.5914 | 30% | 1.0774 |
| **Base Score** | - | - | **1.7774** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** events, level, next
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.591)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0049)
- Score gap to #2: 0.0010 (0.1%)

---

### Rank #2: Next Level Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9835 | 30% | 1.1950 |
| **Base Score** | - | - | **1.8950** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** events, level, next
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.983)

**Ranking Justification:**

- Ranked **#2** - score is 0.0010 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Next Level Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.5693 | 30% | 1.0708 |
| **Base Score** | - | - | **1.7708** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** events, level, next
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.569)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0014 higher than #4

---

### Rank #4: Next Level Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.0250 | 30% | 1.8075 |
| **Base Score** | - | - | **2.5075** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** events, level, next
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.025)

**Ranking Justification:**

- Ranked **#4** - score is 0.0014 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Next Level Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7902 | 30% | 1.1371 |
| **Base Score** | - | - | **1.8371** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** events, level, next
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.790)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 5. Query: `Site Foundation Golf Tournament`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Site Foundation Golf Tournament | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Fore County Golf Tournament | **77.1%** | 🟡 MEDIUM **77.1%** | Semantic/meaning-based match |
| 3 | House Victory Golf Tournament | **76.6%** | 🟡 MEDIUM **76.6%** | Semantic/meaning-based match |
| 4 | Women In Golf Foundation | **76.5%** | 🟡 MEDIUM **76.5%** | Semantic/meaning-based match |
| 5 | ANNIKA Foundation - Golf Tournament | **76.4%** | 🟡 MEDIUM **76.4%** | Semantic/meaning-based match |
| 6 | Golf Tournament | **76.4%** | 🟡 MEDIUM **76.4%** | Semantic/meaning-based match |
| 7 | Bunker To Bunker Golf Tournament | **76.2%** | 🟡 MEDIUM **76.2%** | Semantic/meaning-based match |
| 8 | National Youth Golf Foundation | **76.0%** | 🟡 MEDIUM **76.0%** | Semantic/meaning-based match |
| 9 | World Golf Foundation | **75.8%** | 🟡 MEDIUM **75.8%** | Semantic/meaning-based match |
| 10 | National Golf Foundation | **75.1%** | 🟡 MEDIUM **75.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Site Foundation Golf Tournament

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.4478 | 30% | 1.6343 |
| **Base Score** | - | - | **2.3343** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** foundation, golf, site, tournament
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.448)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2314 (23.1%)

---

### Rank #2: Fore County Golf Tournament

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.6740 | 30% | 1.4022 |
| **Base Score** | - | - | **1.9228** |
| **Final Score** | - | - | **0.7710** (77.1%) |

**Why This Matched:**

- **Word Overlap:** golf, tournament
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.674)

**Ranking Justification:**

- Ranked **#2** - score is 0.2314 lower than #1
- Score is 0.0047 higher than #3

---

### Rank #3: House Victory Golf Tournament

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5856 | 30% | 1.3757 |
| **Base Score** | - | - | **1.8963** |
| **Final Score** | - | - | **0.7664** (76.6%) |

**Why This Matched:**

- **Word Overlap:** golf, tournament
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.586)

**Ranking Justification:**

- Ranked **#3** - score is 0.0047 lower than #2
- Score is 0.0016 higher than #4

---

### Rank #4: Women In Golf Foundation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5549 | 30% | 1.3665 |
| **Base Score** | - | - | **1.8871** |
| **Final Score** | - | - | **0.7647** (76.5%) |

**Why This Matched:**

- **Word Overlap:** foundation, golf
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.555)

**Ranking Justification:**

- Ranked **#4** - score is 0.0016 lower than #3
- Score is 0.0005 higher than #5

---

### Rank #5: ANNIKA Foundation - Golf Tournament

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7875 | 70% | 0.5512 |
| Semantic Similarity | 3.9627 | 30% | 1.1888 |
| **Base Score** | - | - | **1.7401** |
| **Final Score** | - | - | **0.7642** (76.4%) |

**Why This Matched:**

- **Word Overlap:** foundation, golf, tournament
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.787)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.963)

**Ranking Justification:**

- Ranked **#5** - score is 0.0005 lower than #4
- Score is 0.0002 higher than #6

---

</details>
## 6. Query: `Interim WG Meeting - BIER`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Interim WG Meeting - BIER | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Interim Healthcare TEAMM Meeting | **69.3%** | 🟡 MEDIUM **69.3%** | Semantic/meaning-based match |
| 3 | AACP 2012 Interim Meeting | **69.1%** | 🟡 MEDIUM **69.1%** | Semantic/meaning-based match |
| 4 | Legislative Interim Meeting | **65.9%** | 🟡 MEDIUM **65.9%** | Semantic/meaning-based match |
| 5 | BI Meeting | **56.6%** | 🟠 LOW **56.6%** | Semantic/meaning-based match |
| 6 | Bi Annual Meeting | **56.0%** | 🟠 LOW **56.0%** | Semantic/meaning-based match |
| 7 | HRC Advisory Board meeting | **53.9%** | 🟠 LOW **53.9%** | Semantic/meaning-based match |
| 8 | Biz Library January Meeting | **53.2%** | 🟠 LOW **53.2%** | Semantic/meaning-based match |
| 9 | Bim Object Meeting | **52.7%** | 🟠 LOW **52.7%** | Semantic/meaning-based match |
| 10 | American Biz Meeting | **52.5%** | 🟠 LOW **52.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Interim WG Meeting - BIER

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7157 | 30% | 1.1147 |
| **Base Score** | - | - | **1.8147** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** -, bier, interim, meeting, wg
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.716)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.3098 (31.0%)

---

### Rank #2: Interim Healthcare TEAMM Meeting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 2.9424 | 30% | 0.8827 |
| **Base Score** | - | - | **1.3786** |
| **Final Score** | - | - | **0.6926** (69.3%) |

**Why This Matched:**

- **Word Overlap:** interim, meeting
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.708)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.942)

**Ranking Justification:**

- Ranked **#2** - score is 0.3098 lower than #1
- Score is 0.0014 higher than #3

---

### Rank #3: AACP 2012 Interim Meeting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 2.9214 | 30% | 0.8764 |
| **Base Score** | - | - | **1.3723** |
| **Final Score** | - | - | **0.6912** (69.1%) |

**Why This Matched:**

- **Word Overlap:** interim, meeting
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.708)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.921)

**Ranking Justification:**

- Ranked **#3** - score is 0.0014 lower than #2
- Score is 0.0322 higher than #4

---

### Rank #4: Legislative Interim Meeting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6439 | 70% | 0.4508 |
| Semantic Similarity | 3.1204 | 30% | 0.9361 |
| **Base Score** | - | - | **1.3869** |
| **Final Score** | - | - | **0.6590** (65.9%) |

**Why This Matched:**

- **Word Overlap:** interim, meeting
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.644)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.120)

**Ranking Justification:**

- Ranked **#4** - score is 0.0322 lower than #3
- Score is 0.0931 higher than #5

---

### Rank #5: BI Meeting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3750 | 70% | 0.2625 |
| Semantic Similarity | 4.5833 | 30% | 1.3750 |
| **Base Score** | - | - | **1.6375** |
| **Final Score** | - | - | **0.5659** (56.6%) |

**Why This Matched:**

- **Word Overlap:** meeting
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.375) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.583)

**Ranking Justification:**

- Ranked **#5** - score is 0.0931 lower than #4
- Score is 0.0060 higher than #6

---

</details>
## 7. Query: `DermaQuest Inc`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | DermaQuest Inc | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Dermaquest Skin Care | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | Dermaquest, Incorporated | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 4 | Dermaquest Skin Therapy | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | Dermapen | **49.9%** | 🟠 LOW **49.9%** | Semantic/meaning-based match |
| 6 | DERMA E | **49.6%** | 🟠 LOW **49.6%** | Semantic/meaning-based match |
| 7 | Mapquest | **48.6%** | 🟠 LOW **48.6%** | Semantic/meaning-based match |
| 8 | Perquest | **48.0%** | 🟠 LOW **48.0%** | Semantic/meaning-based match |
| 9 | Interquest | **47.3%** | 🟠 LOW **47.3%** | Semantic/meaning-based match |
| 10 | RamQuest | **46.1%** | 🟠 LOW **46.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: DermaQuest Inc

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7958 | 30% | 1.1387 |
| **Base Score** | - | - | **1.8387** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** dermaquest, inc
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.796)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Dermaquest Skin Care

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 4.3077 | 30% | 1.2923 |
| **Base Score** | - | - | **1.8173** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** dermaquest
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.308)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Dermaquest, Incorporated

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.1187 | 30% | 0.9356 |
| **Base Score** | - | - | **1.6356** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.119)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Dermaquest Skin Therapy

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 2.7499 | 30% | 0.8250 |
| **Base Score** | - | - | **1.3500** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** dermaquest
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.750)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.4569 higher than #5

---

### Rank #5: Dermapen

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3000 | 70% | 0.2100 |
| Semantic Similarity | 4.0795 | 30% | 1.2239 |
| **Base Score** | - | - | **1.4339** |
| **Final Score** | - | - | **0.4989** (49.9%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.300) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.080)

**Ranking Justification:**

- Ranked **#5** - score is 0.4569 lower than #4
- Score is 0.0025 higher than #6

---

</details>
## 8. Query: `Ellwood Group Inc`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Ellwood Group Inc | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Ellwood Associates | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 3 | Ellwood TX Forge Houston | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 4 | Ellwood Community Church | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | Ellwood Rose Machine | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 6 | Ellwood TX Forge | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | Ellwood TX Forge Houston | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 8 | Ellwood Closed Die Group | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 9 | Ellwood Specialty Steel | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 10 | Ellwood City Area School District (inc) | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Ellwood Group Inc

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.2426 | 30% | 1.5728 |
| **Base Score** | - | - | **2.2728** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** ellwood, group, inc
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.243)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0433 (4.3%)

---

### Rank #2: Ellwood Associates

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 3.7930 | 30% | 1.1379 |
| **Base Score** | - | - | **1.7679** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** ellwood
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.793)

**Ranking Justification:**

- Ranked **#2** - score is 0.0433 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Ellwood TX Forge Houston

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6923 | 70% | 0.4846 |
| Semantic Similarity | 3.4103 | 30% | 1.0231 |
| **Base Score** | - | - | **1.5077** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** ellwood
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.692)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.410)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0034 higher than #4

---

### Rank #4: Ellwood Community Church

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 4.3296 | 30% | 1.2989 |
| **Base Score** | - | - | **1.8239** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** ellwood
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.330)

**Ranking Justification:**

- Ranked **#4** - score is 0.0034 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Ellwood Rose Machine

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 3.6845 | 30% | 1.1053 |
| **Base Score** | - | - | **1.6303** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** ellwood
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.684)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 9. Query: `American Miniature Horse Registry`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | American Miniature Horse Registry | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | American Miniature Horse Association | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | American Miniature Horse Association Headquarters | **77.0%** | 🟡 MEDIUM **77.0%** | Semantic/meaning-based match |
| 4 | American Saddle Horse Association | **69.6%** | 🟡 MEDIUM **69.6%** | Semantic/meaning-based match |
| 5 | American Horse Defense Fund | **69.1%** | 🟡 MEDIUM **69.1%** | Semantic/meaning-based match |
| 6 | Miniature Horse & Pony Show | **68.8%** | 🟡 MEDIUM **68.8%** | Semantic/meaning-based match |
| 7 | American Youth & Horse Council | **68.6%** | 🟡 MEDIUM **68.6%** | Semantic/meaning-based match |
| 8 | American Youth Horse Council | **68.3%** | 🟡 MEDIUM **68.3%** | Semantic/meaning-based match |
| 9 | American Miniature Hores Association | **68.2%** | 🟡 MEDIUM **68.2%** | Semantic/meaning-based match |
| 10 | American Youth Horse Council | **67.9%** | 🟡 MEDIUM **67.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: American Miniature Horse Registry

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.2780 | 30% | 1.8834 |
| **Base Score** | - | - | **2.5834** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** american, horse, miniature, registry
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.278)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: American Miniature Horse Association

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 3.7158 | 30% | 1.1148 |
| **Base Score** | - | - | **1.7054** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** american, horse, miniature
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.716)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.1356 higher than #3

---

### Rank #3: American Miniature Horse Association Headquarters

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7670 | 70% | 0.5369 |
| Semantic Similarity | 4.7764 | 30% | 1.4329 |
| **Base Score** | - | - | **1.9698** |
| **Final Score** | - | - | **0.7698** (77.0%) |

**Why This Matched:**

- **Word Overlap:** american, horse, miniature
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.767)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.776)

**Ranking Justification:**

- Ranked **#3** - score is 0.1356 lower than #2
- Score is 0.0735 higher than #4

---

### Rank #4: American Saddle Horse Association

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.5888 | 30% | 1.0766 |
| **Base Score** | - | - | **1.5973** |
| **Final Score** | - | - | **0.6963** (69.6%) |

**Why This Matched:**

- **Word Overlap:** american, horse
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.589)

**Ranking Justification:**

- Ranked **#4** - score is 0.0735 lower than #3
- Score is 0.0049 higher than #5

---

### Rank #5: American Horse Defense Fund

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.4866 | 30% | 1.0460 |
| **Base Score** | - | - | **1.5666** |
| **Final Score** | - | - | **0.6914** (69.1%) |

**Why This Matched:**

- **Word Overlap:** american, horse
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.487)

**Ranking Justification:**

- Ranked **#5** - score is 0.0049 lower than #4
- Score is 0.0037 higher than #6

---

</details>
## 10. Query: `YADA ENTERPRISES, INC`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | YADA ENTERPRISES, INC | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Yada Yada | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 3 | Yasuda Corporation Limited | **53.4%** | 🟠 LOW **53.4%** | Semantic/meaning-based match |
| 4 | Yama | **52.6%** | 🟠 LOW **52.6%** | Semantic/meaning-based match |
| 5 | Yama Group | **51.6%** | 🟠 LOW **51.6%** | Semantic/meaning-based match |
| 6 | Ya | **51.3%** | 🟠 LOW **51.3%** | Semantic/meaning-based match |
| 7 | Yara | **51.0%** | 🟠 LOW **51.0%** | Semantic/meaning-based match |
| 8 | YATA | **50.4%** | 🟠 LOW **50.4%** | Semantic/meaning-based match |
| 9 | Yama Enterprises | **50.3%** | 🟠 LOW **50.3%** | Semantic/meaning-based match |
| 10 | Yamas | **50.2%** | 🟠 LOW **50.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: YADA ENTERPRISES, INC

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7979 | 30% | 1.1394 |
| **Base Score** | - | - | **1.8394** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** enterprises,, inc, yada
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.798)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Yada Yada

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 3.3808 | 30% | 1.0142 |
| **Base Score** | - | - | **1.6442** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** yada
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.381)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.4217 higher than #3

---

### Rank #3: Yasuda Corporation Limited

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3600 | 70% | 0.2520 |
| Semantic Similarity | 3.9965 | 30% | 1.1990 |
| **Base Score** | - | - | **1.4510** |
| **Final Score** | - | - | **0.5341** (53.4%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.360) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.997)

**Ranking Justification:**

- Ranked **#3** - score is 0.4217 lower than #2
- Score is 0.0083 higher than #4

---

### Rank #4: Yama

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3375 | 70% | 0.2362 |
| Semantic Similarity | 4.1037 | 30% | 1.2311 |
| **Base Score** | - | - | **1.4673** |
| **Final Score** | - | - | **0.5257** (52.6%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.337) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.104)

**Ranking Justification:**

- Ranked **#4** - score is 0.0083 lower than #3
- Score is 0.0096 higher than #5

---

### Rank #5: Yama Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3375 | 70% | 0.2362 |
| Semantic Similarity | 3.9667 | 30% | 1.1900 |
| **Base Score** | - | - | **1.4263** |
| **Final Score** | - | - | **0.5161** (51.6%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.337) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.967)

**Ranking Justification:**

- Ranked **#5** - score is 0.0096 lower than #4
- Score is 0.0030 higher than #6

---

</details>
## 11. Query: `Seafood Nutrition Partnership`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Seafood Nutrition Partnership | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 2 | Seafood Nutrition Partnership | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Seafood Nutrition Partnership | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | Seafood Nutrition Partnership | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | Sustainable Seafood Partnership | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | SEAFOOD NUTRITION | **72.0%** | 🟡 MEDIUM **72.0%** | Semantic/meaning-based match |
| 7 | Seafood Choices Alliance | **59.8%** | 🟠 LOW **59.8%** | Semantic/meaning-based match |
| 8 | Pet Nutrition Alliance | **56.3%** | 🟠 LOW **56.3%** | Semantic/meaning-based match |
| 9 | East Coast Seafood | **55.8%** | 🟠 LOW **55.8%** | Semantic/meaning-based match |
| 10 | International Boston Seafood | **55.7%** | 🟠 LOW **55.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Seafood Nutrition Partnership

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.7158 | 30% | 1.4147 |
| **Base Score** | - | - | **2.1147** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** nutrition, partnership, seafood
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.716)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0049)
- Score gap to #2: 0.0024 (0.2%)

---

### Rank #2: Seafood Nutrition Partnership

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.7515 | 30% | 2.0255 |
| **Base Score** | - | - | **2.7255** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** nutrition, partnership, seafood
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.752)

**Ranking Justification:**

- Ranked **#2** - score is 0.0024 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Seafood Nutrition Partnership

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.7336 | 30% | 1.4201 |
| **Base Score** | - | - | **2.1201** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** nutrition, partnership, seafood
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.734)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Seafood Nutrition Partnership

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4852 | 30% | 1.3456 |
| **Base Score** | - | - | **2.0456** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** nutrition, partnership, seafood
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.485)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0970 higher than #5

---

### Rank #5: Sustainable Seafood Partnership

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.8853 | 30% | 1.1656 |
| **Base Score** | - | - | **1.7324** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** partnership, seafood
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.885)

**Ranking Justification:**

- Ranked **#5** - score is 0.0970 lower than #4
- Score is 0.1853 higher than #6

---

</details>
## 12. Query: `AVIAKOMPANIYA SIBIR, PAO`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | AVIAKOMPANIYA SIBIR, PAO | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | AVIAKOMPANIYA MIZHNARODNI AVIA | **59.3%** | 🟠 LOW **59.3%** | Semantic/meaning-based match |
| 3 | AVIAKOMPANIYA MIZHNARODNI AVIA | **58.2%** | 🟠 LOW **58.2%** | Semantic/meaning-based match |
| 4 | AVIAKOMPANIYA AEROSVIT, PRYVAT | **54.1%** | 🟠 LOW **54.1%** | Semantic/meaning-based match |
| 5 | AVIAKOMPANIYA AEROSVIT, PRYVATNE AT | **49.2%** | 🟠 LOW **49.2%** | Semantic/meaning-based match |
| 6 | Shibir  Desai | **34.7%** | 🔴 WEAK **34.7%** | Semantic/meaning-based match |
| 7 | AVIPAM Sao Paulo | **33.8%** | 🔴 WEAK **33.8%** | Semantic/meaning-based match |
| 8 | Avyaya Integrated | **33.6%** | 🔴 WEAK **33.6%** | Semantic/meaning-based match |
| 9 | Salaha Kabir | **33.5%** | 🔴 WEAK **33.5%** | Semantic/meaning-based match |
| 10 | AMANDA MAHABIR | **33.4%** | 🔴 WEAK **33.4%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: AVIAKOMPANIYA SIBIR, PAO

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4580 | 30% | 1.3374 |
| **Base Score** | - | - | **2.0374** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** aviakompaniya, pao, sibir,
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.458)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.4092 (40.9%)

---

### Rank #2: AVIAKOMPANIYA MIZHNARODNI AVIA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 3.2724 | 30% | 0.9817 |
| **Base Score** | - | - | **1.3512** |
| **Final Score** | - | - | **0.5932** (59.3%) |

**Why This Matched:**

- **Word Overlap:** aviakompaniya
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.528)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.272)

**Ranking Justification:**

- Ranked **#2** - score is 0.4092 lower than #1
- Score is 0.0109 higher than #3

---

### Rank #3: AVIAKOMPANIYA MIZHNARODNI AVIA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 3.0817 | 30% | 0.9245 |
| **Base Score** | - | - | **1.2940** |
| **Final Score** | - | - | **0.5824** (58.2%) |

**Why This Matched:**

- **Word Overlap:** aviakompaniya
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.528)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.082)

**Ranking Justification:**

- Ranked **#3** - score is 0.0109 lower than #2
- Score is 0.0409 higher than #4

---

### Rank #4: AVIAKOMPANIYA AEROSVIT, PRYVAT

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 2.5077 | 30% | 0.7523 |
| **Base Score** | - | - | **1.1218** |
| **Final Score** | - | - | **0.5415** (54.1%) |

**Why This Matched:**

- **Word Overlap:** aviakompaniya
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.528)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.508)

**Ranking Justification:**

- Ranked **#4** - score is 0.0409 lower than #3
- Score is 0.0498 higher than #5

---

### Rank #5: AVIAKOMPANIYA AEROSVIT, PRYVATNE AT

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4798 | 70% | 0.3359 |
| Semantic Similarity | 2.2715 | 30% | 0.6814 |
| **Base Score** | - | - | **1.0173** |
| **Final Score** | - | - | **0.4917** (49.2%) |

**Why This Matched:**

- **Word Overlap:** aviakompaniya
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.480) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.271)

**Ranking Justification:**

- Ranked **#5** - score is 0.0498 lower than #4
- Score is 0.1445 higher than #6

---

</details>
## 13. Query: `Hartford Hospital School of Nursing`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Hartford Hospital School of Nursing | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | East Hartford Middle School | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 3 | Hartford Hospital Offices USA | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Hartford Public High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Hartford Public High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | West Hartford Public School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | New Hartford High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Hartford Union High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | East Hartford Middle School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Hartford Magnet Middle School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Hartford Hospital School of Nursing

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4633 | 30% | 1.3390 |
| **Base Score** | - | - | **2.0390** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** hartford, hospital, nursing, of, school
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.463)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0938 (9.4%)

---

### Rank #2: East Hartford Middle School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 4.0442 | 30% | 1.2132 |
| **Base Score** | - | - | **1.8082** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** hartford, school
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.044)

**Ranking Justification:**

- Ranked **#2** - score is 0.0938 lower than #1
- Score is 0.0032 higher than #3

---

### Rank #3: Hartford Hospital Offices USA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 5.1271 | 30% | 1.5381 |
| **Base Score** | - | - | **2.1331** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** hartford, hospital
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.127)

**Ranking Justification:**

- Ranked **#3** - score is 0.0032 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Hartford Public High School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 4.5093 | 30% | 1.3528 |
| **Base Score** | - | - | **1.9478** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** hartford, school
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.509)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Hartford Public High School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 4.3148 | 30% | 1.2944 |
| **Base Score** | - | - | **1.8894** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** hartford, school
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.315)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 14. Query: `Internal J&J Meeting and Breakfast`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Internal J&J Meeting and Breakfast | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Internal J&J Meeting and Breakfast | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | ASCO Internal Pre Meeting | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | IT Management Internal Meeting | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Atea Internal Meeting | **74.7%** | 🟡 MEDIUM **74.7%** | Semantic/meaning-based match |
| 6 | Nov. Internal Meeting | **71.0%** | 🟡 MEDIUM **71.0%** | Semantic/meaning-based match |
| 7 | Internal Meeting | **68.6%** | 🟡 MEDIUM **68.6%** | Semantic/meaning-based match |
| 8 | Greg Tolliver Breakfast Meeting | **65.4%** | 🟡 MEDIUM **65.4%** | Semantic/meaning-based match |
| 9 | GMCVB Breakfast & Meeting | **65.3%** | 🟡 MEDIUM **65.3%** | Semantic/meaning-based match |
| 10 | DXC Technology Breakfast Meeting | **65.1%** | 🟡 MEDIUM **65.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Internal J&J Meeting and Breakfast

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9747 | 30% | 1.1924 |
| **Base Score** | - | - | **1.8924** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** and, breakfast, internal, j&j, meeting
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.975)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: Internal J&J Meeting and Breakfast

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7422 | 30% | 1.1227 |
| **Base Score** | - | - | **1.8227** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** and, breakfast, internal, j&j, meeting
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.742)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0970 higher than #3

---

### Rank #3: ASCO Internal Pre Meeting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 2.6800 | 30% | 0.8040 |
| **Base Score** | - | - | **1.3990** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** internal, meeting
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.680)

**Ranking Justification:**

- Ranked **#3** - score is 0.0970 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: IT Management Internal Meeting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 2.5241 | 30% | 0.7572 |
| **Base Score** | - | - | **1.3522** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** internal, meeting
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.524)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.1588 higher than #5

---

### Rank #5: Atea Internal Meeting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7727 | 70% | 0.5409 |
| Semantic Similarity | 3.0320 | 30% | 0.9096 |
| **Base Score** | - | - | **1.4505** |
| **Final Score** | - | - | **0.7467** (74.7%) |

**Why This Matched:**

- **Word Overlap:** internal, meeting
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.032)

**Ranking Justification:**

- Ranked **#5** - score is 0.1588 lower than #4
- Score is 0.0367 higher than #6

---

</details>
## 15. Query: `Spina Bifida Coalition of Cincinnati`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Spina Bifida Coalition of Cincinnati | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Spina Bifida Association of Cincinnati, Inc. | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | Spina Bifida Association of Kentucky | **75.7%** | 🟡 MEDIUM **75.7%** | Semantic/meaning-based match |
| 4 | SBCC | **75.0%** | 🟡 MEDIUM **75.0%** | Near-exact text match |
| 5 | SPINA BIFIDA ASSOCIATION OF ALABAMA | **74.1%** | 🟡 MEDIUM **74.1%** | Semantic/meaning-based match |
| 6 | SPINA BIFIDA ASSN AM | **73.1%** | 🟡 MEDIUM **73.1%** | Semantic/meaning-based match |
| 7 | Spina Bifida Association of Massachusetts | **72.7%** | 🟡 MEDIUM **72.7%** | Semantic/meaning-based match |
| 8 | Spina Bifida Association of Michigan | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 9 | Spina Bifida Resource Network | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 10 | Illinois Spina Bifida Association | **72.2%** | 🟡 MEDIUM **72.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Spina Bifida Coalition of Cincinnati

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.0489 | 30% | 1.5147 |
| **Base Score** | - | - | **2.2147** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** bifida, cincinnati, coalition, of, spina
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.049)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: Spina Bifida Association of Cincinnati, Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 4.0340 | 30% | 1.2102 |
| **Base Score** | - | - | **1.8008** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** bifida, of, spina
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.034)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.1485 higher than #3

---

### Rank #3: Spina Bifida Association of Kentucky

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.8564 | 30% | 1.1569 |
| **Base Score** | - | - | **1.6775** |
| **Final Score** | - | - | **0.7570** (75.7%) |

**Why This Matched:**

- **Word Overlap:** bifida, of, spina
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.856)

**Ranking Justification:**

- Ranked **#3** - score is 0.1485 lower than #2
- Score is 0.0070 higher than #4

---

### Rank #4: SBCC

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | - | - | **1.0000** |
| Acronym Fidelity | 1.0000 | +15% max | +0.1500 |
| **Final Score** | - | - | **0.7500** (75.0%) |

**Why This Matched:**

- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 1.000)
- **Acronym:** Query appears to be acronym of this company (100% fidelity)

**Ranking Justification:**

- Ranked **#4** - score is 0.0070 lower than #3
- Score is 0.0094 higher than #5

---

### Rank #5: SPINA BIFIDA ASSOCIATION OF ALABAMA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6274 | 30% | 1.0882 |
| **Base Score** | - | - | **1.6089** |
| **Final Score** | - | - | **0.7406** (74.1%) |

**Why This Matched:**

- **Word Overlap:** bifida, of, spina
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.627)

**Ranking Justification:**

- Ranked **#5** - score is 0.0094 lower than #4
- Score is 0.0099 higher than #6

---

</details>
## 16. Query: `THE SOCA GROUP ORGANIZATION`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | THE SOCA GROUP ORGANIZATION | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Team SOCA | **79.4%** | 🟡 MEDIUM **79.4%** | Semantic/meaning-based match |
| 3 | Soca Society | **78.5%** | 🟡 MEDIUM **78.5%** | Semantic/meaning-based match |
| 4 | Organization Management Group | **74.5%** | 🟡 MEDIUM **74.5%** | Semantic/meaning-based match |
| 5 | Four Organization | **73.7%** | 🟡 MEDIUM **73.7%** | Semantic/meaning-based match |
| 6 | System Organization | **73.5%** | 🟡 MEDIUM **73.5%** | Semantic/meaning-based match |
| 7 | Organization Meeting | **73.0%** | 🟡 MEDIUM **73.0%** | Semantic/meaning-based match |
| 8 | International organization | **72.7%** | 🟡 MEDIUM **72.7%** | Semantic/meaning-based match |
| 9 | social organization | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 10 | Organization Management | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: THE SOCA GROUP ORGANIZATION

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.0391 | 30% | 1.8117 |
| **Base Score** | - | - | **2.5117** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** group, organization, soca, the
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.039)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2088 (20.9%)

---

### Rank #2: Team SOCA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.3999 | 30% | 1.6200 |
| **Base Score** | - | - | **2.1406** |
| **Final Score** | - | - | **0.7937** (79.4%) |

**Why This Matched:**

- **Word Overlap:** soca
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.400)

**Ranking Justification:**

- Ranked **#2** - score is 0.2088 lower than #1
- Score is 0.0091 higher than #3

---

### Rank #3: Soca Society

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.2170 | 30% | 1.5651 |
| **Base Score** | - | - | **2.0857** |
| **Final Score** | - | - | **0.7845** (78.5%) |

**Why This Matched:**

- **Word Overlap:** soca
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.217)

**Ranking Justification:**

- Ranked **#3** - score is 0.0091 lower than #2
- Score is 0.0396 higher than #4

---

### Rank #4: Organization Management Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.4240 | 30% | 1.3272 |
| **Base Score** | - | - | **1.8478** |
| **Final Score** | - | - | **0.7449** (74.5%) |

**Why This Matched:**

- **Word Overlap:** group, organization
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.424)

**Ranking Justification:**

- Ranked **#4** - score is 0.0396 lower than #3
- Score is 0.0081 higher than #5

---

### Rank #5: Four Organization

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.2610 | 30% | 1.2783 |
| **Base Score** | - | - | **1.7989** |
| **Final Score** | - | - | **0.7368** (73.7%) |

**Why This Matched:**

- **Word Overlap:** organization
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.261)

**Ranking Justification:**

- Ranked **#5** - score is 0.0081 lower than #4
- Score is 0.0022 higher than #6

---

</details>
## 17. Query: `Shiroyama Junior High School`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Shiroyama Junior High School | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Brooks Junior High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | Kenmore Junior High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Sargent Junior High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Greenspun Junior High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | Junior High School #275 | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Nimitz Junior High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | CARROLL JUNIOR HIGH SCHOOL | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Junior High School 45 | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Hardin Junior High School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Shiroyama Junior High School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.5787 | 30% | 1.3736 |
| **Base Score** | - | - | **2.0736** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** high, junior, school, shiroyama
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.579)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: Brooks Junior High School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.7100 | 30% | 1.1130 |
| **Base Score** | - | - | **1.6905** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** high, junior, school
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.710)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Kenmore Junior High School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.4977 | 30% | 1.0493 |
| **Base Score** | - | - | **1.6268** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** high, junior, school
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.498)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Sargent Junior High School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.4300 | 30% | 1.0290 |
| **Base Score** | - | - | **1.6065** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** high, junior, school
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.430)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Greenspun Junior High School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.2351 | 30% | 0.9705 |
| **Base Score** | - | - | **1.5480** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** high, junior, school
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.235)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 18. Query: `National Home Health`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | National Home Health | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | National Home Health | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | National Home Health Care | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 4 | National Home Health Care | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | National Home Health Holiday Party | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 6 | National Association of Home Health Care Providers | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | National Home Health Care | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 8 | National Home Health Care | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 9 | National Home Health Care Expo | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 10 | National Home Health Care | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: National Home Health

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7280 | 30% | 1.1184 |
| **Base Score** | - | - | **1.8184** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** health, home, national
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.728)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0014 (0.1%)

---

### Rank #2: National Home Health

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.3145 | 30% | 1.8944 |
| **Base Score** | - | - | **2.5944** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** health, home, national
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.315)

**Ranking Justification:**

- Ranked **#2** - score is 0.0014 lower than #1
- Score is 0.0433 higher than #3

---

### Rank #3: National Home Health Care

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8225 | 70% | 0.5758 |
| Semantic Similarity | 4.6971 | 30% | 1.4091 |
| **Base Score** | - | - | **1.9849** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** health, home, national
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.823)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.697)

**Ranking Justification:**

- Ranked **#3** - score is 0.0433 lower than #2
- Score is 0.0034 higher than #4

---

### Rank #4: National Home Health Care

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8225 | 70% | 0.5758 |
| Semantic Similarity | 5.6346 | 30% | 1.6904 |
| **Base Score** | - | - | **2.2661** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** health, home, national
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.823)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.635)

**Ranking Justification:**

- Ranked **#4** - score is 0.0034 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: National Home Health Holiday Party

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 4.7352 | 30% | 1.4206 |
| **Base Score** | - | - | **1.9456** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** health, home, national
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.735)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 19. Query: `American News Women's Club`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | American News Women's Club | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Danish-American Women's Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | American Slavic Women's Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | American Women's Club | **80.3%** | 🟢 HIGH **80.3%** | Semantic/meaning-based match |
| 5 | American Women Club | **77.8%** | 🟡 MEDIUM **77.8%** | Semantic/meaning-based match |
| 6 | Indo American Press Club | **72.5%** | 🟡 MEDIUM **72.5%** | Semantic/meaning-based match |
| 7 | Thousand Oaks Women's Club | **72.3%** | 🟡 MEDIUM **72.3%** | Semantic/meaning-based match |
| 8 | Women's Club Board Meeting | **72.0%** | 🟡 MEDIUM **72.0%** | Semantic/meaning-based match |
| 9 | San Jose Women's Club | **71.8%** | 🟡 MEDIUM **71.8%** | Semantic/meaning-based match |
| 10 | Los Prados Women's Club | **71.8%** | 🟡 MEDIUM **71.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: American News Women's Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.0290 | 30% | 1.5087 |
| **Base Score** | - | - | **2.2087** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** american, club, news, women's
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.029)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: Danish-American Women's Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 2.9119 | 30% | 0.8736 |
| **Base Score** | - | - | **1.4642** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** club, women's
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.912)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: American Slavic Women's Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 2.8887 | 30% | 0.8666 |
| **Base Score** | - | - | **1.4572** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** american, club, women's
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.889)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.1023 higher than #4

---

### Rank #4: American Women's Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7670 | 70% | 0.5369 |
| Semantic Similarity | 4.5962 | 30% | 1.3788 |
| **Base Score** | - | - | **1.9158** |
| **Final Score** | - | - | **0.8032** (80.3%) |

**Why This Matched:**

- **Word Overlap:** american, club, women's
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.767)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.596)

**Ranking Justification:**

- Ranked **#4** - score is 0.1023 lower than #3
- Score is 0.0252 higher than #5

---

### Rank #5: American Women Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 5.2751 | 30% | 1.5825 |
| **Base Score** | - | - | **2.0558** |
| **Final Score** | - | - | **0.7780** (77.8%) |

**Why This Matched:**

- **Word Overlap:** american, club
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.275)

**Ranking Justification:**

- Ranked **#5** - score is 0.0252 lower than #4
- Score is 0.0526 higher than #6

---

</details>
## 20. Query: `Denise Roberge`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Denise Roberge | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Denise Abril | **81.6%** | 🟢 HIGH **81.6%** | Semantic/meaning-based match |
| 3 | Denise Beard | **81.5%** | 🟢 HIGH **81.5%** | Semantic/meaning-based match |
| 4 | Denise White | **81.4%** | 🟢 HIGH **81.4%** | Semantic/meaning-based match |
| 5 | Charmaine Denise | **81.0%** | 🟢 HIGH **81.0%** | Semantic/meaning-based match |
| 6 | Denise Wallack | **80.7%** | 🟢 HIGH **80.7%** | Semantic/meaning-based match |
| 7 | CeCi Denise | **80.7%** | 🟢 HIGH **80.7%** | Semantic/meaning-based match |
| 8 | Denise Ivy | **80.6%** | 🟢 HIGH **80.6%** | Semantic/meaning-based match |
| 9 | Denise Martin | **79.4%** | 🟡 MEDIUM **79.4%** | Semantic/meaning-based match |
| 10 | sylvia denise | **79.2%** | 🟡 MEDIUM **79.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Denise Roberge

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.1802 | 30% | 1.2541 |
| **Base Score** | - | - | **1.9541** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** denise, roberge
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.180)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.1865 (18.6%)

---

### Rank #2: Denise Abril

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.0800 | 30% | 1.2240 |
| **Base Score** | - | - | **1.7446** |
| **Final Score** | - | - | **0.8160** (81.6%) |

**Why This Matched:**

- **Word Overlap:** denise
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.080)

**Ranking Justification:**

- Ranked **#2** - score is 0.1865 lower than #1
- Score is 0.0006 higher than #3

---

### Rank #3: Denise Beard

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.0720 | 30% | 1.2216 |
| **Base Score** | - | - | **1.7422** |
| **Final Score** | - | - | **0.8154** (81.5%) |

**Why This Matched:**

- **Word Overlap:** denise
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.072)

**Ranking Justification:**

- Ranked **#3** - score is 0.0006 lower than #2
- Score is 0.0011 higher than #4

---

### Rank #4: Denise White

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.0569 | 30% | 1.2171 |
| **Base Score** | - | - | **1.7377** |
| **Final Score** | - | - | **0.8143** (81.4%) |

**Why This Matched:**

- **Word Overlap:** denise
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.057)

**Ranking Justification:**

- Ranked **#4** - score is 0.0011 lower than #3
- Score is 0.0040 higher than #5

---

### Rank #5: Charmaine Denise

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.0015 | 30% | 1.2004 |
| **Base Score** | - | - | **1.7211** |
| **Final Score** | - | - | **0.8103** (81.0%) |

**Why This Matched:**

- **Word Overlap:** denise
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.001)

**Ranking Justification:**

- Ranked **#5** - score is 0.0040 lower than #4
- Score is 0.0028 higher than #6

---

</details>
## 21. Query: `Synergy Soccer Club`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Synergy Soccer Club | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Nordic Soccer Club | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 3 | Alliance Soccer Club | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 4 | Club Ohio Soccer | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 5 | Synergy Volleyball Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | Club Soccer Event | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Sting Soccer Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Magic Soccer Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | International Soccer Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Classic Soccer Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Synergy Soccer Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4116 | 30% | 1.3235 |
| **Base Score** | - | - | **2.0235** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** club, soccer, synergy
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.412)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0938 (9.4%)

---

### Rank #2: Nordic Soccer Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.0237 | 30% | 0.9071 |
| **Base Score** | - | - | **1.4739** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** club, soccer
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.024)

**Ranking Justification:**

- Ranked **#2** - score is 0.0938 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Alliance Soccer Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 2.9272 | 30% | 0.8782 |
| **Base Score** | - | - | **1.4450** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** club, soccer
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.927)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Club Ohio Soccer

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 2.8585 | 30% | 0.8575 |
| **Base Score** | - | - | **1.4243** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** club, soccer
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.858)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0032 higher than #5

---

### Rank #5: Synergy Volleyball Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 4.2080 | 30% | 1.2624 |
| **Base Score** | - | - | **1.8292** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** club, synergy
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.208)

**Ranking Justification:**

- Ranked **#5** - score is 0.0032 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 22. Query: `NFC Forum`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | NFC FORUM | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 2 | NFC Forum | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 3 | NFC Forum | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | NFC Forum | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | NFC Forum | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 6 | NFC Forum | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 7 | NFC Forum         . | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 8 | NFC Forum Members | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 9 | NFC Forum         . | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 10 | NFC Consulting | **73.7%** | 🟡 MEDIUM **73.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: NFC FORUM

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3671 | 30% | 1.3101 |
| **Base Score** | - | - | **2.0101** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** forum, nfc
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.367)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0049)
- Score gap to #2: 0.0010 (0.1%)

---

### Rank #2: NFC Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.8315 | 30% | 1.4495 |
| **Base Score** | - | - | **2.1495** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** forum, nfc
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.832)

**Ranking Justification:**

- Ranked **#2** - score is 0.0010 lower than #1
- Score is 0.0014 higher than #3

---

### Rank #3: NFC Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 7.1179 | 30% | 2.1354 |
| **Base Score** | - | - | **2.8354** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** forum, nfc
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 7.118)

**Ranking Justification:**

- Ranked **#3** - score is 0.0014 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: NFC Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.1446 | 30% | 1.5434 |
| **Base Score** | - | - | **2.2434** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** forum, nfc
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.145)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: NFC Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4690 | 30% | 1.3407 |
| **Base Score** | - | - | **2.0407** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** forum, nfc
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.469)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 23. Query: `A Better Choice Limousine & Concierge`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | A Better Choice Limousine & Concierge | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | First Choice Limousine Services | **72.0%** | 🟡 MEDIUM **72.0%** | Semantic/meaning-based match |
| 3 | Better Choice Travel | **65.3%** | 🟡 MEDIUM **65.3%** | Semantic/meaning-based match |
| 4 | Better Choice Travel | **63.2%** | 🟡 MEDIUM **63.2%** | Semantic/meaning-based match |
| 5 | Executive Limousine | **56.7%** | 🟠 LOW **56.7%** | Semantic/meaning-based match |
| 6 | Chicago Limousine Transportation | **55.8%** | 🟠 LOW **55.8%** | Semantic/meaning-based match |
| 7 | Journey Limousine | **55.6%** | 🟠 LOW **55.6%** | Semantic/meaning-based match |
| 8 | Metropolitan Limousine | **55.4%** | 🟠 LOW **55.4%** | Semantic/meaning-based match |
| 9 | Greater Atlanta Limousine | **55.4%** | 🟠 LOW **55.4%** | Semantic/meaning-based match |
| 10 | Alliance Limousine | **55.4%** | 🟠 LOW **55.4%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: A Better Choice Limousine & Concierge

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.7189 | 30% | 1.7157 |
| **Base Score** | - | - | **2.4157** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, a, better, choice, concierge, limousine
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.719)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2823 (28.2%)

---

### Rank #2: First Choice Limousine Services

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.7201 | 30% | 1.1160 |
| **Base Score** | - | - | **1.6367** |
| **Final Score** | - | - | **0.7201** (72.0%) |

**Why This Matched:**

- **Word Overlap:** choice, limousine
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.720)

**Ranking Justification:**

- Ranked **#2** - score is 0.2823 lower than #1
- Score is 0.0673 higher than #3

---

### Rank #3: Better Choice Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 3.3465 | 30% | 1.0040 |
| **Base Score** | - | - | **1.4772** |
| **Final Score** | - | - | **0.6528** (65.3%) |

**Why This Matched:**

- **Word Overlap:** better, choice
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.347)

**Ranking Justification:**

- Ranked **#3** - score is 0.0673 lower than #2
- Score is 0.0210 higher than #4

---

### Rank #4: Better Choice Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 2.9073 | 30% | 0.8722 |
| **Base Score** | - | - | **1.3455** |
| **Final Score** | - | - | **0.6318** (63.2%) |

**Why This Matched:**

- **Word Overlap:** better, choice
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.907)

**Ranking Justification:**

- Ranked **#4** - score is 0.0210 lower than #3
- Score is 0.0650 higher than #5

---

### Rank #5: Executive Limousine

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4062 | 70% | 0.2844 |
| Semantic Similarity | 5.3185 | 30% | 1.5956 |
| **Base Score** | - | - | **1.8799** |
| **Final Score** | - | - | **0.5668** (56.7%) |

**Why This Matched:**

- **Word Overlap:** limousine
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.406) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.319)

**Ranking Justification:**

- Ranked **#5** - score is 0.0650 lower than #4
- Score is 0.0091 higher than #6

---

</details>
## 24. Query: `Danish Sisterhood of America`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Danish Sisterhood of America | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Danish Sisterhood and Brotherhood of America | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 3 | The Danish Sisterhood of America | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 4 | DANISH BROTHERHOOD AND DANISH SISTERHOOD OF AMERICA | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | Danish Brotherhood & Danish Sisterhood of America | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 6 | Danish Sisterhood of America National Board Mtg | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | Danish Sisterhood of the Americas | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | The Dansih Sisterhood of America | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Danish Sisterhood of Amercia | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | The Dansih Sisterhood of America | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Danish Sisterhood of America

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.6578 | 30% | 1.6974 |
| **Base Score** | - | - | **2.3974** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** america, danish, of, sisterhood
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.658)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0433 (4.3%)

---

### Rank #2: Danish Sisterhood and Brotherhood of America

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 3.6582 | 30% | 1.0975 |
| **Base Score** | - | - | **1.7020** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** america, danish, of, sisterhood
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.658)

**Ranking Justification:**

- Ranked **#2** - score is 0.0433 lower than #1
- Score is 0.0034 higher than #3

---

### Rank #3: The Danish Sisterhood of America

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7513 | 30% | 1.1254 |
| **Base Score** | - | - | **1.8254** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** america, danish, of, sisterhood
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.751)

**Ranking Justification:**

- Ranked **#3** - score is 0.0034 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: DANISH BROTHERHOOD AND DANISH SISTERHOOD OF AMERICA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 3.6521 | 30% | 1.0956 |
| **Base Score** | - | - | **1.6683** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** america, danish, of, sisterhood
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.652)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Danish Brotherhood & Danish Sisterhood of America

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 3.2356 | 30% | 0.9707 |
| **Base Score** | - | - | **1.5434** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** america, danish, of, sisterhood
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.236)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 25. Query: `Brooklyn Comics Club`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Brooklyn Comics Club | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Cathedral Club of Brooklyn | **91.4%** | 🟢 HIGH **91.4%** | Semantic/meaning-based match |
| 3 | Brooklyn Conversation Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Brooklyn Football Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Brooklyn Barbell Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | Brooklyn Barbell Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Brooklyn Wallyball Club | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Rotary Club of Brooklyn | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Cathedral Club of Brooklyn | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Brooklyn NY Film Club | **75.1%** | 🟡 MEDIUM **75.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Brooklyn Comics Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.6312 | 30% | 1.9894 |
| **Base Score** | - | - | **2.6894** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** brooklyn, club, comics
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.631)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0883 (8.8%)

---

### Rank #2: Cathedral Club of Brooklyn

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 4.4084 | 30% | 1.3225 |
| **Base Score** | - | - | **1.8893** |
| **Final Score** | - | - | **0.9142** (91.4%) |

**Why This Matched:**

- **Word Overlap:** brooklyn, club
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.408)

**Ranking Justification:**

- Ranked **#2** - score is 0.0883 lower than #1
- Score is 0.0087 higher than #3

---

### Rank #3: Brooklyn Conversation Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 4.9121 | 30% | 1.4736 |
| **Base Score** | - | - | **2.0404** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** brooklyn, club
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.912)

**Ranking Justification:**

- Ranked **#3** - score is 0.0087 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Brooklyn Football Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 4.8992 | 30% | 1.4698 |
| **Base Score** | - | - | **2.0366** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** brooklyn, club
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.899)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Brooklyn Barbell Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 4.7898 | 30% | 1.4369 |
| **Base Score** | - | - | **2.0037** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** brooklyn, club
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.790)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 26. Query: `Global Interagency Security Forum`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Global Interagency Security Forum | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Infrastructure Security and Resilience Forum | **81.8%** | 🟢 HIGH **81.8%** | Semantic/meaning-based match |
| 3 | NY Cyber Security Forum | **81.1%** | 🟢 HIGH **81.1%** | Semantic/meaning-based match |
| 4 | BITS Security Forum | **80.1%** | 🟢 HIGH **80.1%** | Semantic/meaning-based match |
| 5 | Privacy + Security Forum | **79.9%** | 🟡 MEDIUM **79.9%** | Semantic/meaning-based match |
| 6 | Halifax International Security Forum | **79.5%** | 🟡 MEDIUM **79.5%** | Semantic/meaning-based match |
| 7 | Information Security Leadership Forum | **79.1%** | 🟡 MEDIUM **79.1%** | Semantic/meaning-based match |
| 8 | Information Security Forum | **79.0%** | 🟡 MEDIUM **79.0%** | Semantic/meaning-based match |
| 9 | The Cyber Security Forum Initiative | **78.2%** | 🟡 MEDIUM **78.2%** | Semantic/meaning-based match |
| 10 | The Learning Forum Security Council | **78.1%** | 🟡 MEDIUM **78.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Global Interagency Security Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0566 | 30% | 1.2170 |
| **Base Score** | - | - | **1.9170** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** forum, global, interagency, security
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.057)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.1845 (18.5%)

---

### Rank #2: Infrastructure Security and Resilience Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.0410 | 30% | 1.2123 |
| **Base Score** | - | - | **1.7577** |
| **Final Score** | - | - | **0.8179** (81.8%) |

**Why This Matched:**

- **Word Overlap:** forum, security
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.041)

**Ranking Justification:**

- Ranked **#2** - score is 0.1845 lower than #1
- Score is 0.0065 higher than #3

---

### Rank #3: NY Cyber Security Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 3.9438 | 30% | 1.1831 |
| **Base Score** | - | - | **1.7286** |
| **Final Score** | - | - | **0.8114** (81.1%) |

**Why This Matched:**

- **Word Overlap:** forum, security
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.944)

**Ranking Justification:**

- Ranked **#3** - score is 0.0065 lower than #2
- Score is 0.0108 higher than #4

---

### Rank #4: BITS Security Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 4.5310 | 30% | 1.3593 |
| **Base Score** | - | - | **1.8551** |
| **Final Score** | - | - | **0.8007** (80.1%) |

**Why This Matched:**

- **Word Overlap:** forum, security
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.708)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.531)

**Ranking Justification:**

- Ranked **#4** - score is 0.0108 lower than #3
- Score is 0.0013 higher than #5

---

### Rank #5: Privacy + Security Forum

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 3.7622 | 30% | 1.1287 |
| **Base Score** | - | - | **1.6741** |
| **Final Score** | - | - | **0.7993** (79.9%) |

**Why This Matched:**

- **Word Overlap:** forum, security
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.762)

**Ranking Justification:**

- Ranked **#5** - score is 0.0013 lower than #4
- Score is 0.0040 higher than #6

---

</details>
## 27. Query: `Lancet Software`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Lancet Software | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Lancet Technology | **77.8%** | 🟡 MEDIUM **77.8%** | Semantic/meaning-based match |
| 3 | Quest Software | **76.8%** | 🟡 MEDIUM **76.8%** | Semantic/meaning-based match |
| 4 | Tech Software | **76.7%** | 🟡 MEDIUM **76.7%** | Semantic/meaning-based match |
| 5 | FRS Software | **76.5%** | 🟡 MEDIUM **76.5%** | Semantic/meaning-based match |
| 6 | ET Software | **76.4%** | 🟡 MEDIUM **76.4%** | Semantic/meaning-based match |
| 7 | Software Professionals | **76.1%** | 🟡 MEDIUM **76.1%** | Semantic/meaning-based match |
| 8 | Jaguar Software | **75.9%** | 🟡 MEDIUM **75.9%** | Semantic/meaning-based match |
| 9 | CDT Software | **75.7%** | 🟡 MEDIUM **75.7%** | Semantic/meaning-based match |
| 10 | Riptide Software | **75.7%** | 🟡 MEDIUM **75.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Lancet Software

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.0885 | 30% | 0.9265 |
| **Base Score** | - | - | **1.6265** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** lancet, software
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.088)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2240 (22.4%)

---

### Rank #2: Lancet Technology

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.7700 | 30% | 1.1310 |
| **Base Score** | - | - | **1.6516** |
| **Final Score** | - | - | **0.7785** (77.8%) |

**Why This Matched:**

- **Word Overlap:** lancet
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.770)

**Ranking Justification:**

- Ranked **#2** - score is 0.2240 lower than #1
- Score is 0.0106 higher than #3

---

### Rank #3: Quest Software

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6136 | 30% | 1.0841 |
| **Base Score** | - | - | **1.6047** |
| **Final Score** | - | - | **0.7679** (76.8%) |

**Why This Matched:**

- **Word Overlap:** software
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.614)

**Ranking Justification:**

- Ranked **#3** - score is 0.0106 lower than #2
- Score is 0.0007 higher than #4

---

### Rank #4: Tech Software

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6029 | 30% | 1.0809 |
| **Base Score** | - | - | **1.6015** |
| **Final Score** | - | - | **0.7672** (76.7%) |

**Why This Matched:**

- **Word Overlap:** software
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.603)

**Ranking Justification:**

- Ranked **#4** - score is 0.0007 lower than #3
- Score is 0.0024 higher than #5

---

### Rank #5: FRS Software

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.5675 | 30% | 1.0703 |
| **Base Score** | - | - | **1.5909** |
| **Final Score** | - | - | **0.7648** (76.5%) |

**Why This Matched:**

- **Word Overlap:** software
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.568)

**Ranking Justification:**

- Ranked **#5** - score is 0.0024 lower than #4
- Score is 0.0006 higher than #6

---

</details>
## 28. Query: `Our Lady of the Lakes Catholic Church and School`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Our Lady of the Lakes Catholic Church and School | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Our Lady of the Lakes Catholic Church | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 3 | Our Lady of the Lakes Catholic School | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 4 | Our Lady of the Lakes Catholic School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Our Lady of The Lakes Catholic Church | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | OUR LADY OF MOUNT CARMEL CATHOLIC SCHOOL | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Our Lady of the Lakes Catholic School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Our Lady of the Lakes Catholic Church | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Our Lady of the Lakes Catholic School | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Our Lady of The Holy Rosary Catholic Church | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Our Lady of the Lakes Catholic Church and School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3262 | 30% | 1.2978 |
| **Base Score** | - | - | **1.9978** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** and, catholic, church, lady, lakes, of, our, school, the
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.326)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0938 (9.4%)

---

### Rank #2: Our Lady of the Lakes Catholic Church

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8498 | 70% | 0.5949 |
| Semantic Similarity | 4.4298 | 30% | 1.3289 |
| **Base Score** | - | - | **1.9238** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** catholic, church, lady, lakes, of, our, the
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.430)

**Ranking Justification:**

- Ranked **#2** - score is 0.0938 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Our Lady of the Lakes Catholic School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8498 | 70% | 0.5949 |
| Semantic Similarity | 4.1277 | 30% | 1.2383 |
| **Base Score** | - | - | **1.8332** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** catholic, lady, lakes, of, our, school, the
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.128)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0032 higher than #4

---

### Rank #4: Our Lady of the Lakes Catholic School

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8498 | 70% | 0.5949 |
| Semantic Similarity | 4.9446 | 30% | 1.4834 |
| **Base Score** | - | - | **2.0782** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** catholic, lady, lakes, of, our, school, the
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.945)

**Ranking Justification:**

- Ranked **#4** - score is 0.0032 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Our Lady of The Lakes Catholic Church

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8498 | 70% | 0.5949 |
| Semantic Similarity | 4.2479 | 30% | 1.2744 |
| **Base Score** | - | - | **1.8692** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** catholic, church, lady, lakes, of, our, the
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.248)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 29. Query: `Broadway Bound International`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Broadway Bound International | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | Broadway Bound International | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Broadway Bound Kids | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Broadway Bound Kidz | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Broadway Bound | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | Broadway Bound | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Broadway Bound West | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Broadway Bound Childrens Theatre | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Broadway Bound Dance | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | BROADWAY BOUND DANCE CENTRE | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Broadway Bound International

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.1892 | 30% | 1.2568 |
| **Base Score** | - | - | **1.9568** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** bound, broadway, international
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.189)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0014 (0.1%)

---

### Rank #2: Broadway Bound International

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.0455 | 30% | 1.8136 |
| **Base Score** | - | - | **2.5136** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** bound, broadway, international
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.045)

**Ranking Justification:**

- Ranked **#2** - score is 0.0014 lower than #1
- Score is 0.0970 higher than #3

---

### Rank #3: Broadway Bound Kids

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 4.8216 | 30% | 1.4465 |
| **Base Score** | - | - | **2.0648** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** bound, broadway
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.883)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.822)

**Ranking Justification:**

- Ranked **#3** - score is 0.0970 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Broadway Bound Kidz

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 4.1980 | 30% | 1.2594 |
| **Base Score** | - | - | **1.8777** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** bound, broadway
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.883)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.198)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Broadway Bound

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 3.8762 | 30% | 1.1629 |
| **Base Score** | - | - | **1.7356** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** bound, broadway
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.876)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 30. Query: `E. H. Wachs`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | E. H. Wachs | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | E. H. Wachs | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | EHW | **75.0%** | 🟡 MEDIUM **75.0%** | Near-exact text match |
| 4 | E H Smith | **66.5%** | 🟡 MEDIUM **66.5%** | Semantic/meaning-based match |
| 5 | Wachs Services | **62.4%** | 🟡 MEDIUM **62.4%** | Semantic/meaning-based match |
| 6 | Elen Wachs | **62.2%** | 🟡 MEDIUM **62.2%** | Semantic/meaning-based match |
| 7 | H-E Parts | **61.6%** | 🟡 MEDIUM **61.6%** | Semantic/meaning-based match |
| 8 | E.H. Wachs | **60.6%** | 🟡 MEDIUM **60.6%** | Semantic/meaning-based match |
| 9 | Wachs Water Services | **53.4%** | 🟠 LOW **53.4%** | Semantic/meaning-based match |
| 10 | Wachs Wedding Room Block | **53.2%** | 🟠 LOW **53.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: E. H. Wachs

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3398 | 30% | 1.3019 |
| **Base Score** | - | - | **2.0019** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** e., h., wachs
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.340)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: E. H. Wachs

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0334 | 30% | 1.2100 |
| **Base Score** | - | - | **1.9100** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** e., h., wachs
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.033)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.2524 higher than #3

---

### Rank #3: EHW

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | - | - | **1.0000** |
| Acronym Fidelity | 1.0000 | +15% max | +0.1500 |
| **Final Score** | - | - | **0.7500** (75.0%) |

**Why This Matched:**

- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 1.000)
- **Acronym:** Query appears to be acronym of this company (100% fidelity)

**Ranking Justification:**

- Ranked **#3** - score is 0.2524 lower than #2
- Score is 0.0853 higher than #4

---

### Rank #4: E H Smith

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6625 | 70% | 0.4637 |
| Semantic Similarity | 3.3196 | 30% | 0.9959 |
| **Base Score** | - | - | **1.4596** |
| **Final Score** | - | - | **0.6647** (66.5%) |

**Why This Matched:**

- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.662)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.320)

**Ranking Justification:**

- Ranked **#4** - score is 0.0853 lower than #3
- Score is 0.0412 higher than #5

---

### Rank #5: Wachs Services

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4798 | 70% | 0.3359 |
| Semantic Similarity | 4.7856 | 30% | 1.4357 |
| **Base Score** | - | - | **1.7716** |
| **Final Score** | - | - | **0.6235** (62.4%) |

**Why This Matched:**

- **Word Overlap:** wachs
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.480) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.786)

**Ranking Justification:**

- Ranked **#5** - score is 0.0412 lower than #4
- Score is 0.0012 higher than #6

---

</details>
## 31. Query: `Marine Corps Fox 2/5`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Marine Corps Fox 2/5 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | MARINE CORPS BASE CAMP | **80.8%** | 🟢 HIGH **80.8%** | Semantic/meaning-based match |
| 3 | MARINE CORPS COMMUNITY SERVICE | **80.7%** | 🟢 HIGH **80.7%** | Semantic/meaning-based match |
| 4 | Navy Marine Corps Ball | **80.3%** | 🟢 HIGH **80.3%** | Semantic/meaning-based match |
| 5 | MARINE CORPS LOGISTICS BASE | **80.0%** | 🟢 HIGH **80.0%** | Semantic/meaning-based match |
| 6 | 25th US Marine Corps | **80.0%** | 🟢 HIGH **80.0%** | Semantic/meaning-based match |
| 7 | MARINE CORPS MILITARY REUNION | **79.4%** | 🟡 MEDIUM **79.4%** | Semantic/meaning-based match |
| 8 | Marine Corps Air Transport | **79.4%** | 🟡 MEDIUM **79.4%** | Semantic/meaning-based match |
| 9 | MARINE CORPS AIR GROUND | **79.4%** | 🟡 MEDIUM **79.4%** | Semantic/meaning-based match |
| 10 | US Marine Corps Training | **79.1%** | 🟡 MEDIUM **79.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Marine Corps Fox 2/5

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4066 | 30% | 1.3220 |
| **Base Score** | - | - | **2.0220** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 2/5, corps, fox, marine
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.407)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.1941 (19.4%)

---

### Rank #2: MARINE CORPS BASE CAMP

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.4003 | 30% | 1.3201 |
| **Base Score** | - | - | **1.8655** |
| **Final Score** | - | - | **0.8084** (80.8%) |

**Why This Matched:**

- **Word Overlap:** corps, marine
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.400)

**Ranking Justification:**

- Ranked **#2** - score is 0.1941 lower than #1
- Score is 0.0011 higher than #3

---

### Rank #3: MARINE CORPS COMMUNITY SERVICE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.3824 | 30% | 1.3147 |
| **Base Score** | - | - | **1.8601** |
| **Final Score** | - | - | **0.8073** (80.7%) |

**Why This Matched:**

- **Word Overlap:** corps, marine
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.382)

**Ranking Justification:**

- Ranked **#3** - score is 0.0011 lower than #2
- Score is 0.0043 higher than #4

---

### Rank #4: Navy Marine Corps Ball

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.3096 | 30% | 1.2929 |
| **Base Score** | - | - | **1.8383** |
| **Final Score** | - | - | **0.8030** (80.3%) |

**Why This Matched:**

- **Word Overlap:** corps, marine
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.310)

**Ranking Justification:**

- Ranked **#4** - score is 0.0043 lower than #3
- Score is 0.0027 higher than #5

---

### Rank #5: MARINE CORPS LOGISTICS BASE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.2643 | 30% | 1.2793 |
| **Base Score** | - | - | **1.8247** |
| **Final Score** | - | - | **0.8004** (80.0%) |

**Why This Matched:**

- **Word Overlap:** corps, marine
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.264)

**Ranking Justification:**

- Ranked **#5** - score is 0.0027 lower than #4
- Score is 0.0001 higher than #6

---

</details>
## 32. Query: `Fantasia Turistica`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Fantasia Turistica | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Fantasia | **74.6%** | 🟡 MEDIUM **74.6%** | Semantic/meaning-based match |
| 3 | Fantasia Travels | **72.8%** | 🟡 MEDIUM **72.8%** | Semantic/meaning-based match |
| 4 | Noreen Fantasia | **72.4%** | 🟡 MEDIUM **72.4%** | Semantic/meaning-based match |
| 5 | Fantasia Travel | **71.4%** | 🟡 MEDIUM **71.4%** | Semantic/meaning-based match |
| 6 | Fantasia Accessry | **71.2%** | 🟡 MEDIUM **71.2%** | Semantic/meaning-based match |
| 7 | Ferrari Fantasia | **70.7%** | 🟡 MEDIUM **70.7%** | Semantic/meaning-based match |
| 8 | Fantasia Travels | **70.6%** | 🟡 MEDIUM **70.6%** | Semantic/meaning-based match |
| 9 | Fantasia Veneziana | **69.8%** | 🟡 MEDIUM **69.8%** | Semantic/meaning-based match |
| 10 | Operadora Turistica | **68.8%** | 🟡 MEDIUM **68.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Fantasia Turistica

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.6719 | 30% | 1.4016 |
| **Base Score** | - | - | **2.1016** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** fantasia, turistica
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.672)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2569 (25.7%)

---

### Rank #2: Fantasia

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6300 | 70% | 0.4410 |
| Semantic Similarity | 6.6617 | 30% | 1.9985 |
| **Base Score** | - | - | **2.4395** |
| **Final Score** | - | - | **0.7455** (74.6%) |

**Why This Matched:**

- **Word Overlap:** fantasia
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.630)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.662)

**Ranking Justification:**

- Ranked **#2** - score is 0.2569 lower than #1
- Score is 0.0170 higher than #3

---

### Rank #3: Fantasia Travels

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5181 | 30% | 1.3554 |
| **Base Score** | - | - | **1.8761** |
| **Final Score** | - | - | **0.7285** (72.8%) |

**Why This Matched:**

- **Word Overlap:** fantasia
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.518)

**Ranking Justification:**

- Ranked **#3** - score is 0.0170 lower than #2
- Score is 0.0044 higher than #4

---

### Rank #4: Noreen Fantasia

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.4217 | 30% | 1.3265 |
| **Base Score** | - | - | **1.8471** |
| **Final Score** | - | - | **0.7241** (72.4%) |

**Why This Matched:**

- **Word Overlap:** fantasia
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.422)

**Ranking Justification:**

- Ranked **#4** - score is 0.0044 lower than #3
- Score is 0.0104 higher than #5

---

### Rank #5: Fantasia Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.1914 | 30% | 1.2574 |
| **Base Score** | - | - | **1.7780** |
| **Final Score** | - | - | **0.7137** (71.4%) |

**Why This Matched:**

- **Word Overlap:** fantasia
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.191)

**Ranking Justification:**

- Ranked **#5** - score is 0.0104 lower than #4
- Score is 0.0018 higher than #6

---

</details>
## 33. Query: `Esoterix`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Esoterix | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Esoterix Headquarters | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | Esoterix Integrated Genetics | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 4 | Esoterix Integrated Genetics | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | Esoterix Genetic Laboratories, LLC | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 6 | Esoterix Clinical Trials Services | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | ESOTERIX GENETC LABORATORIES, LLC | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 8 | Centrix | **42.8%** | 🟠 LOW **42.8%** | Semantic/meaning-based match |
| 9 | Verix | **42.2%** | 🟠 LOW **42.2%** | Semantic/meaning-based match |
| 10 | Netrix | **42.2%** | 🟠 LOW **42.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Esoterix

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.2666 | 30% | 1.8800 |
| **Base Score** | - | - | **2.5800** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** esoterix
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.267)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Esoterix Headquarters

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 4.7814 | 30% | 1.4344 |
| **Base Score** | - | - | **2.0071** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** esoterix
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.781)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Esoterix Integrated Genetics

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 3.3540 | 30% | 1.0062 |
| **Base Score** | - | - | **1.5312** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** esoterix
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.354)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Esoterix Integrated Genetics

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 2.8812 | 30% | 0.8644 |
| **Base Score** | - | - | **1.3894** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** esoterix
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.881)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Esoterix Genetic Laboratories, LLC

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 2.4528 | 30% | 0.7358 |
| **Base Score** | - | - | **1.2608** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** esoterix
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.453)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 34. Query: `Coker Group`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Coker Group | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 2 | Coker Group | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Coker Group | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | Coker Group | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | Coker Group | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 6 | Coker Group | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 7 | Coker Capital | **96.2%** | 🟢 EXACT **96.2%** | Semantic/meaning-based match |
| 8 | Jackson & Coker | **96.2%** | 🟢 EXACT **96.2%** | Semantic/meaning-based match |
| 9 | Coker Capital | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 10 | Kristen Coker | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Coker Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4074 | 30% | 1.3222 |
| **Base Score** | - | - | **2.0222** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** coker, group
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.407)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0049)
- Score gap to #2: 0.0024 (0.2%)

---

### Rank #2: Coker Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.3617 | 30% | 1.9085 |
| **Base Score** | - | - | **2.6085** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** coker, group
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.362)

**Ranking Justification:**

- Ranked **#2** - score is 0.0024 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Coker Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.6470 | 30% | 1.3941 |
| **Base Score** | - | - | **2.0941** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** coker, group
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.647)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Coker Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.1951 | 30% | 1.2585 |
| **Base Score** | - | - | **1.9585** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** coker, group
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.195)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Coker Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0915 | 30% | 1.2274 |
| **Base Score** | - | - | **1.9274** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** coker, group
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.091)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 35. Query: `GILEAD IT`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | GILEAD IT | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Gilead Productions | **77.7%** | 🟡 MEDIUM **77.7%** | Semantic/meaning-based match |
| 3 | Gilead Sciences | **77.5%** | 🟡 MEDIUM **77.5%** | Semantic/meaning-based match |
| 4 | Gilead 1N | **77.2%** | 🟡 MEDIUM **77.2%** | Semantic/meaning-based match |
| 5 | Gilead Services | **76.6%** | 🟡 MEDIUM **76.6%** | Semantic/meaning-based match |
| 6 | GILEAD MEDICAL | **76.5%** | 🟡 MEDIUM **76.5%** | Semantic/meaning-based match |
| 7 | Gilead Media | **75.7%** | 🟡 MEDIUM **75.7%** | Semantic/meaning-based match |
| 8 | Gilead Finance | **75.5%** | 🟡 MEDIUM **75.5%** | Semantic/meaning-based match |
| 9 | Baltimore Gilead | **75.0%** | 🟡 MEDIUM **75.0%** | Semantic/meaning-based match |
| 10 | Gilead | **75.0%** | 🟡 MEDIUM **75.0%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: GILEAD IT

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.6454 | 30% | 1.0936 |
| **Base Score** | - | - | **1.7936** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** gilead, it
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.645)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2249 (22.5%)

---

### Rank #2: Gilead Productions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.7787 | 30% | 1.7336 |
| **Base Score** | - | - | **2.2542** |
| **Final Score** | - | - | **0.7775** (77.7%) |

**Why This Matched:**

- **Word Overlap:** gilead
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.779)

**Ranking Justification:**

- Ranked **#2** - score is 0.2249 lower than #1
- Score is 0.0029 higher than #3

---

### Rank #3: Gilead Sciences

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.7124 | 30% | 1.7137 |
| **Base Score** | - | - | **2.2343** |
| **Final Score** | - | - | **0.7746** (77.5%) |

**Why This Matched:**

- **Word Overlap:** gilead
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.712)

**Ranking Justification:**

- Ranked **#3** - score is 0.0029 lower than #2
- Score is 0.0022 higher than #4

---

### Rank #4: Gilead 1N

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.6612 | 30% | 1.6984 |
| **Base Score** | - | - | **2.2190** |
| **Final Score** | - | - | **0.7723** (77.2%) |

**Why This Matched:**

- **Word Overlap:** gilead
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.661)

**Ranking Justification:**

- Ranked **#4** - score is 0.0022 lower than #3
- Score is 0.0059 higher than #5

---

### Rank #5: Gilead Services

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.5275 | 30% | 1.6582 |
| **Base Score** | - | - | **2.1789** |
| **Final Score** | - | - | **0.7665** (76.6%) |

**Why This Matched:**

- **Word Overlap:** gilead
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.527)

**Ranking Justification:**

- Ranked **#5** - score is 0.0059 lower than #4
- Score is 0.0013 higher than #6

---

</details>
## 36. Query: `4143 Affiliate INDA 2016`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | 4143 Affiliate INDA 2016 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | 1528 Affiliate INDA 2016 | **90.9%** | 🟢 HIGH **90.9%** | Near-exact text match |
| 3 | 1528 Affiliate INDA 2016 | **90.5%** | 🟢 HIGH **90.5%** | Near-exact text match |
| 4 | 1528 Affiliate INDA 2016 | **90.5%** | 🟢 HIGH **90.5%** | Near-exact text match |
| 5 | 1528 Affiliate AFA 2016 | **73.7%** | 🟡 MEDIUM **73.7%** | Semantic/meaning-based match |
| 6 | 2016 Google Affiliate | **73.3%** | 🟡 MEDIUM **73.3%** | Semantic/meaning-based match |
| 7 | 1035 Affiliate CASE 2016 | **73.1%** | 🟡 MEDIUM **73.1%** | Semantic/meaning-based match |
| 8 | 1035 JPMorgan Affiliate 2016 | **73.0%** | 🟡 MEDIUM **73.0%** | Semantic/meaning-based match |
| 9 | AACR AFFILIATE 2016 | **72.9%** | 🟡 MEDIUM **72.9%** | Semantic/meaning-based match |
| 10 | 1035 Affiliate CASE 2016 | **72.7%** | 🟡 MEDIUM **72.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: 4143 Affiliate INDA 2016

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.8132 | 30% | 1.4440 |
| **Base Score** | - | - | **2.1440** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 2016, 4143, affiliate, inda
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.813)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0938 (9.4%)

---

### Rank #2: 1528 Affiliate INDA 2016

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 4.0353 | 30% | 1.2106 |
| **Base Score** | - | - | **1.8406** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** 2016, affiliate, inda
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.035)

**Ranking Justification:**

- Ranked **#2** - score is 0.0938 lower than #1
- Score is 0.0032 higher than #3

---

### Rank #3: 1528 Affiliate INDA 2016

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 3.4318 | 30% | 1.0295 |
| **Base Score** | - | - | **1.6595** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** 2016, affiliate, inda
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.432)

**Ranking Justification:**

- Ranked **#3** - score is 0.0032 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: 1528 Affiliate INDA 2016

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 2.9808 | 30% | 0.8942 |
| **Base Score** | - | - | **1.5242** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** 2016, affiliate, inda
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.981)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.1683 higher than #5

---

### Rank #5: 1528 Affiliate AFA 2016

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.4024 | 30% | 1.0207 |
| **Base Score** | - | - | **1.5413** |
| **Final Score** | - | - | **0.7372** (73.7%) |

**Why This Matched:**

- **Word Overlap:** 2016, affiliate
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.402)

**Ranking Justification:**

- Ranked **#5** - score is 0.1683 lower than #4
- Score is 0.0039 higher than #6

---

</details>
## 37. Query: `Pipe and Plant Solutions`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Pipe and Plant Solutions | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Pipe & Plant | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | Advanced Pipe Solutions | **75.3%** | 🟡 MEDIUM **75.3%** | Semantic/meaning-based match |
| 4 | Infra Pipe Solutions | **72.1%** | 🟡 MEDIUM **72.1%** | Semantic/meaning-based match |
| 5 | Plant Solutions Limited | **70.2%** | 🟡 MEDIUM **70.2%** | Semantic/meaning-based match |
| 6 | PPS | **69.0%** | 🟡 MEDIUM **69.0%** | Near-exact text match |
| 7 | TV Pipe Solutions | **68.7%** | 🟡 MEDIUM **68.7%** | Semantic/meaning-based match |
| 8 | Plant Operations | **60.7%** | 🟡 MEDIUM **60.7%** | Semantic/meaning-based match |
| 9 | Cal Pipe Industries | **60.6%** | 🟡 MEDIUM **60.6%** | Semantic/meaning-based match |
| 10 | Independent Concrete Pipe | **60.5%** | 🟡 MEDIUM **60.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Pipe and Plant Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3529 | 30% | 1.3059 |
| **Base Score** | - | - | **2.0059** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** and, pipe, plant, solutions
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.353)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: Pipe & Plant

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8264 | 70% | 0.5785 |
| Semantic Similarity | 5.3401 | 30% | 1.6020 |
| **Base Score** | - | - | **2.1806** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** pipe, plant
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.826)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.340)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.1521 higher than #3

---

### Rank #3: Advanced Pipe Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 3.6990 | 30% | 1.1097 |
| **Base Score** | - | - | **1.6508** |
| **Final Score** | - | - | **0.7534** (75.3%) |

**Why This Matched:**

- **Word Overlap:** pipe, solutions
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.699)

**Ranking Justification:**

- Ranked **#3** - score is 0.1521 lower than #2
- Score is 0.0324 higher than #4

---

### Rank #4: Infra Pipe Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 3.1260 | 30% | 0.9378 |
| **Base Score** | - | - | **1.4788** |
| **Final Score** | - | - | **0.7210** (72.1%) |

**Why This Matched:**

- **Word Overlap:** pipe, solutions
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.126)

**Ranking Justification:**

- Ranked **#4** - score is 0.0324 lower than #3
- Score is 0.0191 higher than #5

---

### Rank #5: Plant Solutions Limited

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7159 | 70% | 0.5011 |
| Semantic Similarity | 3.4978 | 30% | 1.0494 |
| **Base Score** | - | - | **1.5505** |
| **Final Score** | - | - | **0.7019** (70.2%) |

**Why This Matched:**

- **Word Overlap:** plant, solutions
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.716)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.498)

**Ranking Justification:**

- Ranked **#5** - score is 0.0191 lower than #4
- Score is 0.0119 higher than #6

---

</details>
## 38. Query: `Stephen Rourke`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Stephen Rourke | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Rourke Publishing | **75.9%** | 🟡 MEDIUM **75.9%** | Semantic/meaning-based match |
| 3 | Stephen Scott | **75.1%** | 🟡 MEDIUM **75.1%** | Semantic/meaning-based match |
| 4 | Stephen Oh | **73.8%** | 🟡 MEDIUM **73.8%** | Semantic/meaning-based match |
| 5 | Stephen Lacy | **73.5%** | 🟡 MEDIUM **73.5%** | Semantic/meaning-based match |
| 6 | Stephen Michael | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 7 | Stephen Madden | **71.7%** | 🟡 MEDIUM **71.7%** | Semantic/meaning-based match |
| 8 | Stephen McConnell | **71.6%** | 🟡 MEDIUM **71.6%** | Semantic/meaning-based match |
| 9 | Eliza Stephen | **71.6%** | 🟡 MEDIUM **71.6%** | Semantic/meaning-based match |
| 10 | Stephen Kent | **71.2%** | 🟡 MEDIUM **71.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Stephen Rourke

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.6713 | 30% | 1.4014 |
| **Base Score** | - | - | **2.1014** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** rourke, stephen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.671)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2431 (24.3%)

---

### Rank #2: Rourke Publishing

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6038 | 30% | 1.0811 |
| **Base Score** | - | - | **1.6018** |
| **Final Score** | - | - | **0.7593** (75.9%) |

**Why This Matched:**

- **Word Overlap:** rourke
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.604)

**Ranking Justification:**

- Ranked **#2** - score is 0.2431 lower than #1
- Score is 0.0080 higher than #3

---

### Rank #3: Stephen Scott

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.5208 | 30% | 1.0562 |
| **Base Score** | - | - | **1.5769** |
| **Final Score** | - | - | **0.7513** (75.1%) |

**Why This Matched:**

- **Word Overlap:** stephen
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.521)

**Ranking Justification:**

- Ranked **#3** - score is 0.0080 lower than #2
- Score is 0.0131 higher than #4

---

### Rank #4: Stephen Oh

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.3179 | 30% | 0.9954 |
| **Base Score** | - | - | **1.5160** |
| **Final Score** | - | - | **0.7382** (73.8%) |

**Why This Matched:**

- **Word Overlap:** stephen
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.318)

**Ranking Justification:**

- Ranked **#4** - score is 0.0131 lower than #3
- Score is 0.0033 higher than #5

---

### Rank #5: Stephen Lacy

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.2673 | 30% | 0.9802 |
| **Base Score** | - | - | **1.5008** |
| **Final Score** | - | - | **0.7349** (73.5%) |

**Why This Matched:**

- **Word Overlap:** stephen
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.267)

**Ranking Justification:**

- Ranked **#5** - score is 0.0033 lower than #4
- Score is 0.0089 higher than #6

---

</details>
## 39. Query: `MIT Initiative on the Digital Economy`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | MIT Initiative on the Digital Economy | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | MIT Office of Digital Learning | **59.3%** | 🟠 LOW **59.3%** | Semantic/meaning-based match |
| 3 | MIT Energy Initiative | **58.4%** | 🟠 LOW **58.4%** | Semantic/meaning-based match |
| 4 | MIT Office of Digital Learning | **53.0%** | 🟠 LOW **53.0%** | Semantic/meaning-based match |
| 5 | MIT Office of Digital Learning | **53.0%** | 🟠 LOW **53.0%** | Semantic/meaning-based match |
| 6 | New Economy Initiative | **52.5%** | 🟠 LOW **52.5%** | Semantic/meaning-based match |
| 7 | MIT Office of Digital Learning (formerly OEIT) | **52.1%** | 🟠 LOW **52.1%** | Semantic/meaning-based match |
| 8 | Global Digital Health Initiative | **52.0%** | 🟠 LOW **52.0%** | Semantic/meaning-based match |
| 9 | MIT Lean Advancement Initiative | **51.8%** | 🟠 LOW **51.8%** | Semantic/meaning-based match |
| 10 | MIT Energy Initiative | **51.8%** | 🟠 LOW **51.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: MIT Initiative on the Digital Economy

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.2696 | 30% | 1.5809 |
| **Base Score** | - | - | **2.2809** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** digital, economy, initiative, mit, on, the
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.270)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.4090 (40.9%)

---

### Rank #2: MIT Office of Digital Learning

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5250 | 70% | 0.3675 |
| Semantic Similarity | 3.9060 | 30% | 1.1718 |
| **Base Score** | - | - | **1.5393** |
| **Final Score** | - | - | **0.5935** (59.3%) |

**Why This Matched:**

- **Word Overlap:** digital, mit
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.525)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.906)

**Ranking Justification:**

- Ranked **#2** - score is 0.4090 lower than #1
- Score is 0.0093 higher than #3

---

### Rank #3: MIT Energy Initiative

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4813 | 70% | 0.3369 |
| Semantic Similarity | 4.2817 | 30% | 1.2845 |
| **Base Score** | - | - | **1.6214** |
| **Final Score** | - | - | **0.5842** (58.4%) |

**Why This Matched:**

- **Word Overlap:** initiative, mit
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.481) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.282)

**Ranking Justification:**

- Ranked **#3** - score is 0.0093 lower than #2
- Score is 0.0538 higher than #4

---

### Rank #4: MIT Office of Digital Learning

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5250 | 70% | 0.3675 |
| Semantic Similarity | 2.8043 | 30% | 0.8413 |
| **Base Score** | - | - | **1.2088** |
| **Final Score** | - | - | **0.5304** (53.0%) |

**Why This Matched:**

- **Word Overlap:** digital, mit
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.525)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.804)

**Ranking Justification:**

- Ranked **#4** - score is 0.0538 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: MIT Office of Digital Learning

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5250 | 70% | 0.3675 |
| Semantic Similarity | 2.8040 | 30% | 0.8412 |
| **Base Score** | - | - | **1.2087** |
| **Final Score** | - | - | **0.5303** (53.0%) |

**Why This Matched:**

- **Word Overlap:** digital, mit
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.525)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.804)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0052 higher than #6

---

</details>
## 40. Query: `Urx Community USA`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Urx Community USA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Community Alliance USA | **71.4%** | 🟡 MEDIUM **71.4%** | Semantic/meaning-based match |
| 3 | Community Brands USA | **71.4%** | 🟡 MEDIUM **71.4%** | Semantic/meaning-based match |
| 4 | UiPath Community USA | **71.0%** | 🟡 MEDIUM **71.0%** | Semantic/meaning-based match |
| 5 | Community Events LLC USA | **69.6%** | 🟡 MEDIUM **69.6%** | Semantic/meaning-based match |
| 6 | Community Bridges Inc USA | **69.4%** | 🟡 MEDIUM **69.4%** | Semantic/meaning-based match |
| 7 | Umoja Community USA | **69.2%** | 🟡 MEDIUM **69.2%** | Semantic/meaning-based match |
| 8 | Community Development Society USA | **67.3%** | 🟡 MEDIUM **67.3%** | Semantic/meaning-based match |
| 9 | Community Information Exchange USA | **67.1%** | 🟡 MEDIUM **67.1%** | Semantic/meaning-based match |
| 10 | Oregon Community Trees USA | **65.1%** | 🟡 MEDIUM **65.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Urx Community USA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.0693 | 30% | 1.8208 |
| **Base Score** | - | - | **2.5208** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** community, urx, usa
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.069)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2882 (28.8%)

---

### Rank #2: Community Alliance USA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 3.4157 | 30% | 1.0247 |
| **Base Score** | - | - | **1.5657** |
| **Final Score** | - | - | **0.7142** (71.4%) |

**Why This Matched:**

- **Word Overlap:** community, usa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.416)

**Ranking Justification:**

- Ranked **#2** - score is 0.2882 lower than #1
- Score is 0.0003 higher than #3

---

### Rank #3: Community Brands USA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 3.4097 | 30% | 1.0229 |
| **Base Score** | - | - | **1.5639** |
| **Final Score** | - | - | **0.7139** (71.4%) |

**Why This Matched:**

- **Word Overlap:** community, usa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.410)

**Ranking Justification:**

- Ranked **#3** - score is 0.0003 lower than #2
- Score is 0.0041 higher than #4

---

### Rank #4: UiPath Community USA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 3.3276 | 30% | 0.9983 |
| **Base Score** | - | - | **1.5393** |
| **Final Score** | - | - | **0.7098** (71.0%) |

**Why This Matched:**

- **Word Overlap:** community, usa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.328)

**Ranking Justification:**

- Ranked **#4** - score is 0.0041 lower than #3
- Score is 0.0138 higher than #5

---

### Rank #5: Community Events LLC USA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 3.0511 | 30% | 0.9153 |
| **Base Score** | - | - | **1.4564** |
| **Final Score** | - | - | **0.6961** (69.6%) |

**Why This Matched:**

- **Word Overlap:** community, usa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.051)

**Ranking Justification:**

- Ranked **#5** - score is 0.0138 lower than #4
- Score is 0.0020 higher than #6

---

</details>
## 41. Query: `Spredfast Engage`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Spredfast Engage | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | Spredfast Product | **79.7%** | 🟡 MEDIUM **79.7%** | Semantic/meaning-based match |
| 3 | Spredfast Events | **72.9%** | 🟡 MEDIUM **72.9%** | Semantic/meaning-based match |
| 4 | Spredfast Events | **72.9%** | 🟡 MEDIUM **72.9%** | Semantic/meaning-based match |
| 5 | 8 Engage | **69.8%** | 🟡 MEDIUM **69.8%** | Semantic/meaning-based match |
| 6 | Engage Point | **68.9%** | 🟡 MEDIUM **68.9%** | Semantic/meaning-based match |
| 7 | engage fi | **67.9%** | 🟡 MEDIUM **67.9%** | Semantic/meaning-based match |
| 8 | CU Engage | **67.0%** | 🟡 MEDIUM **67.0%** | Semantic/meaning-based match |
| 9 | Engage R+D | **66.3%** | 🟡 MEDIUM **66.3%** | Semantic/meaning-based match |
| 10 | Engage 360 | **66.2%** | 🟡 MEDIUM **66.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Spredfast Engage

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.3675 | 30% | 1.0102 |
| **Base Score** | - | - | **1.7102** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** engage, spredfast
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.367)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.2073 (20.7%)

---

### Rank #2: Spredfast Product

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.1403 | 30% | 1.2421 |
| **Base Score** | - | - | **1.7627** |
| **Final Score** | - | - | **0.7965** (79.7%) |

**Why This Matched:**

- **Word Overlap:** spredfast
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.140)

**Ranking Justification:**

- Ranked **#2** - score is 0.2073 lower than #1
- Score is 0.0672 higher than #3

---

### Rank #3: Spredfast Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.1203 | 30% | 0.9361 |
| **Base Score** | - | - | **1.4567** |
| **Final Score** | - | - | **0.7293** (72.9%) |

**Why This Matched:**

- **Word Overlap:** spredfast
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.120)

**Ranking Justification:**

- Ranked **#3** - score is 0.0672 lower than #2
- Score is 0.0007 higher than #4

---

### Rank #4: Spredfast Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.1093 | 30% | 0.9328 |
| **Base Score** | - | - | **1.4534** |
| **Final Score** | - | - | **0.7286** (72.9%) |

**Why This Matched:**

- **Word Overlap:** spredfast
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.109)

**Ranking Justification:**

- Ranked **#4** - score is 0.0007 lower than #3
- Score is 0.0302 higher than #5

---

### Rank #5: 8 Engage

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 2.6510 | 30% | 0.7953 |
| **Base Score** | - | - | **1.3159** |
| **Final Score** | - | - | **0.6984** (69.8%) |

**Why This Matched:**

- **Word Overlap:** engage
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.651)

**Ranking Justification:**

- Ranked **#5** - score is 0.0302 lower than #4
- Score is 0.0091 higher than #6

---

</details>
## 42. Query: `City of Dallas-Parks & Recreation`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | City of Dallas-Parks & Recreation | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Dallas Parks and Recreation Dept | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | Dallas Parks and Recreation Department | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | City of Dallas Park & Recreation | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Baltimore City Recreation and Parks | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | City of Miami Parks & Recreation | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Midwest City Parks and Recreation | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | City of Irving - Parks and Recreation | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | City of Atlanta Parks and Recreation | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Forest City Parks and Recreation | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: City of Dallas-Parks & Recreation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.2195 | 30% | 1.5658 |
| **Base Score** | - | - | **2.2658** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, city, dallas-parks, of, recreation
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.219)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: Dallas Parks and Recreation Dept

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 5.2681 | 30% | 1.5804 |
| **Base Score** | - | - | **2.1711** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** recreation
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.268)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Dallas Parks and Recreation Department

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 5.2315 | 30% | 1.5694 |
| **Base Score** | - | - | **2.1601** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** recreation
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.231)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: City of Dallas Park & Recreation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 5.2091 | 30% | 1.5627 |
| **Base Score** | - | - | **2.1533** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** &, city, of, recreation
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.209)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Baltimore City Recreation and Parks

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 4.8669 | 30% | 1.4601 |
| **Base Score** | - | - | **2.0507** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** city, recreation
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.867)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 43. Query: `Kai Pono Builders, Inc.`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Kai Pono Builders, Inc. | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Kai Pono Builders | **96.2%** | 🟢 EXACT **96.2%** | Near-exact text match |
| 3 | Pono Kai Resort | **74.7%** | 🟡 MEDIUM **74.7%** | Semantic/meaning-based match |
| 4 | Pono Kai | **72.0%** | 🟡 MEDIUM **72.0%** | Semantic/meaning-based match |
| 5 | Pono Kai Holiday Party | **69.5%** | 🟡 MEDIUM **69.5%** | Semantic/meaning-based match |
| 6 | S Kai | **61.3%** | 🟡 MEDIUM **61.3%** | Semantic/meaning-based match |
| 7 | Kai Kai Communications | **59.3%** | 🟠 LOW **59.3%** | Semantic/meaning-based match |
| 8 | KAI Distribution Centre | **59.2%** | 🟠 LOW **59.2%** | Semantic/meaning-based match |
| 9 | Ian Kai | **58.2%** | 🟠 LOW **58.2%** | Semantic/meaning-based match |
| 10 | 2016 Kona Kai | **58.0%** | 🟠 LOW **58.0%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Kai Pono Builders, Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0814 | 30% | 1.2244 |
| **Base Score** | - | - | **1.9244** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** builders,, inc., kai, pono
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.081)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0404 (4.0%)

---

### Rank #2: Kai Pono Builders

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9375 | 70% | 0.6562 |
| Semantic Similarity | 4.7334 | 30% | 1.4200 |
| **Base Score** | - | - | **2.0763** |
| **Final Score** | - | - | **0.9621** (96.2%) |

**Why This Matched:**

- **Word Overlap:** kai, pono
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.938)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.733)

**Ranking Justification:**

- Ranked **#2** - score is 0.0404 lower than #1
- Score is 0.2148 higher than #3

---

### Rank #3: Pono Kai Resort

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 3.1830 | 30% | 0.9549 |
| **Base Score** | - | - | **1.4959** |
| **Final Score** | - | - | **0.7473** (74.7%) |

**Why This Matched:**

- **Word Overlap:** kai, pono
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.773)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.183)

**Ranking Justification:**

- Ranked **#3** - score is 0.2148 lower than #2
- Score is 0.0276 higher than #4

---

### Rank #4: Pono Kai

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7027 | 70% | 0.4919 |
| Semantic Similarity | 3.5256 | 30% | 1.0577 |
| **Base Score** | - | - | **1.5496** |
| **Final Score** | - | - | **0.7197** (72.0%) |

**Why This Matched:**

- **Word Overlap:** kai, pono
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.703)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.526)

**Ranking Justification:**

- Ranked **#4** - score is 0.0276 lower than #3
- Score is 0.0243 higher than #5

---

### Rank #5: Pono Kai Holiday Party

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7027 | 70% | 0.4919 |
| Semantic Similarity | 3.1445 | 30% | 0.9433 |
| **Base Score** | - | - | **1.4352** |
| **Final Score** | - | - | **0.6954** (69.5%) |

**Why This Matched:**

- **Word Overlap:** kai, pono
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.703)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.144)

**Ranking Justification:**

- Ranked **#5** - score is 0.0243 lower than #4
- Score is 0.0824 higher than #6

---

</details>
## 44. Query: `MUSICFIRST COALITION`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | MUSICFIRST COALITION | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Wind Coalition | **73.0%** | 🟡 MEDIUM **73.0%** | Semantic/meaning-based match |
| 3 | Unconventional Coalition | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 4 | Human Coalition | **71.4%** | 🟡 MEDIUM **71.4%** | Semantic/meaning-based match |
| 5 | Founders Coalition | **71.0%** | 🟡 MEDIUM **71.0%** | Semantic/meaning-based match |
| 6 | Tech Coalition | **70.4%** | 🟡 MEDIUM **70.4%** | Semantic/meaning-based match |
| 7 | Union Coalition | **70.3%** | 🟡 MEDIUM **70.3%** | Semantic/meaning-based match |
| 8 | Data Coalition | **70.2%** | 🟡 MEDIUM **70.2%** | Semantic/meaning-based match |
| 9 | District New Music Coalition | **69.9%** | 🟡 MEDIUM **69.9%** | Semantic/meaning-based match |
| 10 | Coalition International | **69.9%** | 🟡 MEDIUM **69.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: MUSICFIRST COALITION

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.2104 | 30% | 1.5631 |
| **Base Score** | - | - | **2.2631** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** coalition, musicfirst
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.210)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2724 (27.2%)

---

### Rank #2: Wind Coalition

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.5596 | 30% | 1.0679 |
| **Base Score** | - | - | **1.5885** |
| **Final Score** | - | - | **0.7300** (73.0%) |

**Why This Matched:**

- **Word Overlap:** coalition
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.560)

**Ranking Justification:**

- Ranked **#2** - score is 0.2724 lower than #1
- Score is 0.0035 higher than #3

---

### Rank #3: Unconventional Coalition

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.4988 | 30% | 1.0497 |
| **Base Score** | - | - | **1.5703** |
| **Final Score** | - | - | **0.7265** (72.6%) |

**Why This Matched:**

- **Word Overlap:** coalition
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.499)

**Ranking Justification:**

- Ranked **#3** - score is 0.0035 lower than #2
- Score is 0.0123 higher than #4

---

### Rank #4: Human Coalition

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.2865 | 30% | 0.9860 |
| **Base Score** | - | - | **1.5066** |
| **Final Score** | - | - | **0.7142** (71.4%) |

**Why This Matched:**

- **Word Overlap:** coalition
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.287)

**Ranking Justification:**

- Ranked **#4** - score is 0.0123 lower than #3
- Score is 0.0043 higher than #5

---

### Rank #5: Founders Coalition

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.2126 | 30% | 0.9638 |
| **Base Score** | - | - | **1.4844** |
| **Final Score** | - | - | **0.7099** (71.0%) |

**Why This Matched:**

- **Word Overlap:** coalition
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.213)

**Ranking Justification:**

- Ranked **#5** - score is 0.0043 lower than #4
- Score is 0.0055 higher than #6

---

</details>
## 45. Query: `Frontier Power Products`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Frontier Power Products | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Worldwide Power Products | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | Power Management Products | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Western Power Products Inc | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Zenith Power Products LLC | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | Frontier Business Products | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Advanced Power Products | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Residential and Power Products | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Pacific Power Products | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Frontier Natural Products | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Frontier Power Products

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.1672 | 30% | 1.5502 |
| **Base Score** | - | - | **2.2502** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** frontier, power, products
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.167)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: Worldwide Power Products

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 4.4807 | 30% | 1.3442 |
| **Base Score** | - | - | **1.9110** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** power, products
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.481)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Power Management Products

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.9040 | 30% | 1.1712 |
| **Base Score** | - | - | **1.7380** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** power, products
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.904)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Western Power Products Inc

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.8839 | 30% | 1.1652 |
| **Base Score** | - | - | **1.7320** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** power, products
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.884)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Zenith Power Products LLC

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.6173 | 30% | 1.0852 |
| **Base Score** | - | - | **1.6520** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** power, products
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.617)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 46. Query: `1960`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | 1960 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | 1960 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | 1960 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | 1960 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | 1960 Family Practice | **96.2%** | 🟢 EXACT **96.2%** | Semantic/meaning-based match |
| 6 | District 1960 | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | PAGE CLASS OF 1960 | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 8 | Playhouse 1960 | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 9 | Miami Beach Class 1960 | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 10 | 1960 Hope Center | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: 1960

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.9894 | 30% | 2.0968 |
| **Base Score** | - | - | **2.7968** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 1960
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.989)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: 1960

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 2.8370 | 30% | 0.8511 |
| **Base Score** | - | - | **1.5511** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 1960
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.837)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: 1960

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 2.6907 | 30% | 0.8072 |
| **Base Score** | - | - | **1.5072** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 1960
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.691)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: 1960

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 2.5350 | 30% | 0.7605 |
| **Base Score** | - | - | **1.4605** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 1960
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.535)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0409 higher than #5

---

### Rank #5: 1960 Family Practice

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 3.0767 | 30% | 0.9230 |
| **Base Score** | - | - | **1.4480** |
| **Final Score** | - | - | **0.9616** (96.2%) |

**Why This Matched:**

- **Word Overlap:** 1960
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.077)

**Ranking Justification:**

- Ranked **#5** - score is 0.0409 lower than #4
- Score is 0.0058 higher than #6

---

</details>
## 47. Query: `Pacific Northwest Diabetes Research Inst`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Pacific Northwest Diabetes Research Inst | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Pacific Northwest Diabetes Research | **84.9%** | 🟢 HIGH **84.9%** | Semantic/meaning-based match |
| 3 | Diabetes Research | **71.6%** | 🟡 MEDIUM **71.6%** | Semantic/meaning-based match |
| 4 | Pacific Northwest Research Foundation | **70.7%** | 🟡 MEDIUM **70.7%** | Semantic/meaning-based match |
| 5 | DIABETES RESEARCH INST FND | **69.1%** | 🟡 MEDIUM **69.1%** | Semantic/meaning-based match |
| 6 | Pacific Northwest Research Institite | **68.1%** | 🟡 MEDIUM **68.1%** | Semantic/meaning-based match |
| 7 | Pacific Northwest Research Institute | **67.5%** | 🟡 MEDIUM **67.5%** | Semantic/meaning-based match |
| 8 | GHNS62F3GFQ, Diabetes Pacific Northwest District Meeting | **67.4%** | 🟡 MEDIUM **67.4%** | Semantic/meaning-based match |
| 9 | Diabetes Research | **67.3%** | 🟡 MEDIUM **67.3%** | Semantic/meaning-based match |
| 10 | Diabetes Research Wellness Foundation | **62.7%** | 🟡 MEDIUM **62.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Pacific Northwest Diabetes Research Inst

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.8257 | 30% | 1.4477 |
| **Base Score** | - | - | **2.1477** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** diabetes, inst, northwest, pacific, research
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.826)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.1533 (15.3%)

---

### Rank #2: Pacific Northwest Diabetes Research

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7773 | 70% | 0.5441 |
| Semantic Similarity | 4.9265 | 30% | 1.4779 |
| **Base Score** | - | - | **2.0220** |
| **Final Score** | - | - | **0.8491** (84.9%) |

**Why This Matched:**

- **Word Overlap:** diabetes, northwest, pacific, research
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.777)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.926)

**Ranking Justification:**

- Ranked **#2** - score is 0.1533 lower than #1
- Score is 0.1329 higher than #3

---

### Rank #3: Diabetes Research

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5885 | 70% | 0.4119 |
| Semantic Similarity | 4.9279 | 30% | 1.4784 |
| **Base Score** | - | - | **1.8903** |
| **Final Score** | - | - | **0.7163** (71.6%) |

**Why This Matched:**

- **Word Overlap:** diabetes, research
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.588)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.928)

**Ranking Justification:**

- Ranked **#3** - score is 0.1329 lower than #2
- Score is 0.0088 higher than #4

---

### Rank #4: Pacific Northwest Research Foundation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7118 | 70% | 0.4983 |
| Semantic Similarity | 3.3651 | 30% | 1.0095 |
| **Base Score** | - | - | **1.5078** |
| **Final Score** | - | - | **0.7074** (70.7%) |

**Why This Matched:**

- **Word Overlap:** northwest, pacific, research
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.712)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.365)

**Ranking Justification:**

- Ranked **#4** - score is 0.0088 lower than #3
- Score is 0.0165 higher than #5

---

### Rank #5: DIABETES RESEARCH INST FND

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7118 | 70% | 0.4983 |
| Semantic Similarity | 3.0950 | 30% | 0.9285 |
| **Base Score** | - | - | **1.4268** |
| **Final Score** | - | - | **0.6909** (69.1%) |

**Why This Matched:**

- **Word Overlap:** diabetes, inst, research
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.712)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.095)

**Ranking Justification:**

- Ranked **#5** - score is 0.0165 lower than #4
- Score is 0.0096 higher than #6

---

</details>
## 48. Query: `Mentors & Mentees`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Mentors & Mentees | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | TRUE Mentors | **76.1%** | 🟡 MEDIUM **76.1%** | Semantic/meaning-based match |
| 3 | 3 Mentors | **75.7%** | 🟡 MEDIUM **75.7%** | Semantic/meaning-based match |
| 4 | SCORE Mentors | **74.2%** | 🟡 MEDIUM **74.2%** | Semantic/meaning-based match |
| 5 | Green Mentors | **73.6%** | 🟡 MEDIUM **73.6%** | Semantic/meaning-based match |
| 6 | Oregon Mentors | **73.4%** | 🟡 MEDIUM **73.4%** | Semantic/meaning-based match |
| 7 | 3-Mentors | **73.1%** | 🟡 MEDIUM **73.1%** | Semantic/meaning-based match |
| 8 | 3-Mentors, Inc. | **69.6%** | 🟡 MEDIUM **69.6%** | Semantic/meaning-based match |
| 9 | The Marketing Mentors | **69.4%** | 🟡 MEDIUM **69.4%** | Semantic/meaning-based match |
| 10 | CDL Mentors | **69.4%** | 🟡 MEDIUM **69.4%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Mentors & Mentees

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.8988 | 30% | 1.7696 |
| **Base Score** | - | - | **2.4696** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, mentees, mentors
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.899)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2418 (24.2%)

---

### Rank #2: TRUE Mentors

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.6289 | 30% | 1.3887 |
| **Base Score** | - | - | **1.9093** |
| **Final Score** | - | - | **0.7606** (76.1%) |

**Why This Matched:**

- **Word Overlap:** mentors
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.629)

**Ranking Justification:**

- Ranked **#2** - score is 0.2418 lower than #1
- Score is 0.0041 higher than #3

---

### Rank #3: 3 Mentors

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5487 | 30% | 1.3646 |
| **Base Score** | - | - | **1.8852** |
| **Final Score** | - | - | **0.7565** (75.7%) |

**Why This Matched:**

- **Word Overlap:** mentors
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.549)

**Ranking Justification:**

- Ranked **#3** - score is 0.0041 lower than #2
- Score is 0.0144 higher than #4

---

### Rank #4: SCORE Mentors

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.2668 | 30% | 1.2800 |
| **Base Score** | - | - | **1.8007** |
| **Final Score** | - | - | **0.7421** (74.2%) |

**Why This Matched:**

- **Word Overlap:** mentors
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.267)

**Ranking Justification:**

- Ranked **#4** - score is 0.0144 lower than #3
- Score is 0.0061 higher than #5

---

### Rank #5: Green Mentors

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.1468 | 30% | 1.2440 |
| **Base Score** | - | - | **1.7647** |
| **Final Score** | - | - | **0.7360** (73.6%) |

**Why This Matched:**

- **Word Overlap:** mentors
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.147)

**Ranking Justification:**

- Ranked **#5** - score is 0.0061 lower than #4
- Score is 0.0023 higher than #6

---

</details>
## 49. Query: `NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06 | **69.8%** | 🟡 MEDIUM **69.8%** | Semantic/meaning-based match |
| 3 | NaLA 2023 fall conference M01674661235470 01-25-23 10:40:54 | **67.6%** | 🟡 MEDIUM **67.6%** | Semantic/meaning-based match |
| 4 | 2024 CPF Conference - CRDF Global M01712246836200 04-04-24 12:07:23 | **52.2%** | 🟠 LOW **52.2%** | Semantic/meaning-based match |
| 5 | TCN Worldwide 2025 Fall Conference M01712007730091 04-01-24 17:42:23 | **51.9%** | 🟠 LOW **51.9%** | Semantic/meaning-based match |
| 6 | AE Events Corporate Conference M01724936557111 08-29-24 09:02:41 | **51.0%** | 🟠 LOW **51.0%** | Semantic/meaning-based match |
| 7 | General Contractors 2024 Conference M01712000754964 04-01-24 15:46:00 | **51.0%** | 🟠 LOW **51.0%** | Semantic/meaning-based match |
| 8 | Request for Availability of a Conference Room M01733155625999 12-02-24 11:07:12 | **50.2%** | 🟠 LOW **50.2%** | Semantic/meaning-based match |
| 9 | Fall Exchange 2024 M01704922221245 01-10-24 16:30:36 | **49.6%** | 🟠 LOW **49.6%** | Semantic/meaning-based match |
| 10 | Region II Conference M01706895307768 02-02-24 12:35:11 | **48.4%** | 🟠 LOW **48.4%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.6847 | 30% | 1.4054 |
| **Base Score** | - | - | **2.1054** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 02-29-24, 12:03:46, 2024, conference, fall, m01709226216947, nala
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.685)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.3040 (30.4%)

---

### Rank #2: NaLA 2023 fall conference M01674569043113 01-24-23 09:04:06

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5931 | 70% | 0.4151 |
| Semantic Similarity | 4.3577 | 30% | 1.3073 |
| **Base Score** | - | - | **1.7224** |
| **Final Score** | - | - | **0.6984** (69.8%) |

**Why This Matched:**

- **Word Overlap:** conference, fall, nala
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.593)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.358)

**Ranking Justification:**

- Ranked **#2** - score is 0.3040 lower than #1
- Score is 0.0225 higher than #3

---

### Rank #3: NaLA 2023 fall conference M01674661235470 01-25-23 10:40:54

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5542 | 70% | 0.3879 |
| Semantic Similarity | 4.4336 | 30% | 1.3301 |
| **Base Score** | - | - | **1.7180** |
| **Final Score** | - | - | **0.6759** (67.6%) |

**Why This Matched:**

- **Word Overlap:** conference, fall, nala
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.554)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.434)

**Ranking Justification:**

- Ranked **#3** - score is 0.0225 lower than #2
- Score is 0.1537 higher than #4

---

### Rank #4: 2024 CPF Conference - CRDF Global M01712246836200 04-04-24 12:07:23

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4750 | 70% | 0.3325 |
| Semantic Similarity | 2.9131 | 30% | 0.8739 |
| **Base Score** | - | - | **1.2064** |
| **Final Score** | - | - | **0.5222** (52.2%) |

**Why This Matched:**

- **Word Overlap:** 2024, conference
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.475) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.913)

**Ranking Justification:**

- Ranked **#4** - score is 0.1537 lower than #3
- Score is 0.0037 higher than #5

---

### Rank #5: TCN Worldwide 2025 Fall Conference M01712007730091 04-01-24 17:42:23

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4318 | 70% | 0.3023 |
| Semantic Similarity | 3.3276 | 30% | 0.9983 |
| **Base Score** | - | - | **1.3006** |
| **Final Score** | - | - | **0.5185** (51.9%) |

**Why This Matched:**

- **Word Overlap:** conference, fall
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.432) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.328)

**Ranking Justification:**

- Ranked **#5** - score is 0.0037 lower than #4
- Score is 0.0088 higher than #6

---

</details>
## 50. Query: `Donnelley Work Session`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Donnelley Work Session | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | SMDS Work Session | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | DWS | **75.0%** | 🟡 MEDIUM **75.0%** | Near-exact text match |
| 4 | Donnelley Financial Solutions | **54.9%** | 🟠 LOW **54.9%** | Semantic/meaning-based match |
| 5 | Professional Development Session | **54.2%** | 🟠 LOW **54.2%** | Semantic/meaning-based match |
| 6 | RH Donnelley Headquarters | **53.7%** | 🟠 LOW **53.7%** | Semantic/meaning-based match |
| 7 | Session One | **53.6%** | 🟠 LOW **53.6%** | Semantic/meaning-based match |
| 8 | Executive Breakfast Session | **53.2%** | 🟠 LOW **53.2%** | Semantic/meaning-based match |
| 9 | Session M | **53.1%** | 🟠 LOW **53.1%** | Semantic/meaning-based match |
| 10 | Strategy Session | **52.8%** | 🟠 LOW **52.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Donnelley Work Session

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.0390 | 30% | 1.8117 |
| **Base Score** | - | - | **2.5117** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** donnelley, session, work
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.039)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: SMDS Work Session

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.2162 | 30% | 0.9649 |
| **Base Score** | - | - | **1.5317** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** session, work
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.216)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.1555 higher than #3

---

### Rank #3: DWS

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | - | - | **1.0000** |
| Acronym Fidelity | 1.0000 | +15% max | +0.1500 |
| **Final Score** | - | - | **0.7500** (75.0%) |

**Why This Matched:**

- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 1.000)
- **Acronym:** Query appears to be acronym of this company (100% fidelity)

**Ranking Justification:**

- Ranked **#3** - score is 0.1555 lower than #2
- Score is 0.2009 higher than #4

---

### Rank #4: Donnelley Financial Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 3.5503 | 30% | 1.0651 |
| **Base Score** | - | - | **1.4345** |
| **Final Score** | - | - | **0.5491** (54.9%) |

**Why This Matched:**

- **Word Overlap:** donnelley
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.528)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.550)

**Ranking Justification:**

- Ranked **#4** - score is 0.2009 lower than #3
- Score is 0.0070 higher than #5

---

### Rank #5: Professional Development Session

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 3.4101 | 30% | 1.0230 |
| **Base Score** | - | - | **1.3925** |
| **Final Score** | - | - | **0.5421** (54.2%) |

**Why This Matched:**

- **Word Overlap:** session
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.528)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.410)

**Ranking Justification:**

- Ranked **#5** - score is 0.0070 lower than #4
- Score is 0.0049 higher than #6

---

</details>
## 51. Query: `North Shore Senior Center`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | North Shore Senior Center | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | North Shore Senior Center | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | South Shore Cultural Center | **91.4%** | 🟢 HIGH **91.4%** | Semantic/meaning-based match |
| 4 | North Shore Elder Services | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 5 | National Institute Senior Center | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 6 | North Carolina Solar Center | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 7 | Coastal North Town Center | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | North SHore Community College | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | North Shore Community Bank | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Glen Cove Senior Center | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: North Shore Senior Center

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.2566 | 30% | 1.2770 |
| **Base Score** | - | - | **1.9770** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** center, north, senior, shore
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.257)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0014 (0.1%)

---

### Rank #2: North Shore Senior Center

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.1147 | 30% | 1.8344 |
| **Base Score** | - | - | **2.5344** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** center, north, senior, shore
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.115)

**Ranking Justification:**

- Ranked **#2** - score is 0.0014 lower than #1
- Score is 0.0883 higher than #3

---

### Rank #3: South Shore Cultural Center

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.0615 | 30% | 0.9185 |
| **Base Score** | - | - | **1.5135** |
| **Final Score** | - | - | **0.9142** (91.4%) |

**Why This Matched:**

- **Word Overlap:** center, shore
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.062)

**Ranking Justification:**

- Ranked **#3** - score is 0.0883 lower than #2
- Score is 0.0055 higher than #4

---

### Rank #4: North Shore Elder Services

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.3958 | 30% | 1.0187 |
| **Base Score** | - | - | **1.6137** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** north, shore
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.396)

**Ranking Justification:**

- Ranked **#4** - score is 0.0055 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: National Institute Senior Center

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.1123 | 30% | 0.9337 |
| **Base Score** | - | - | **1.5287** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** center, senior
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.112)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 52. Query: `Singles Who Like Food & Fun`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Singles Who Like Food & Fun | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Fun Asian Singles | **61.0%** | 🟡 MEDIUM **61.0%** | Semantic/meaning-based match |
| 3 | American Singles Who Love Asian | **58.9%** | 🟠 LOW **58.9%** | Semantic/meaning-based match |
| 4 | Fun Social Singles 35+ | **58.1%** | 🟠 LOW **58.1%** | Semantic/meaning-based match |
| 5 | LGBT Food N' Fun Social Group | **57.7%** | 🟠 LOW **57.7%** | Semantic/meaning-based match |
| 6 | Fun and Awesome Adventures for Singles and Couples | **56.3%** | 🟠 LOW **56.3%** | Semantic/meaning-based match |
| 7 | Lesbians who love literature & food | **56.1%** | 🟠 LOW **56.1%** | Semantic/meaning-based match |
| 8 | NYC SINGLES FUN EVENTS | **56.0%** | 🟠 LOW **56.0%** | Semantic/meaning-based match |
| 9 | Food Fun & Fellowship | **55.9%** | 🟠 LOW **55.9%** | Semantic/meaning-based match |
| 10 | Singles Who Dance | **55.8%** | 🟠 LOW **55.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Singles Who Like Food & Fun

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0636 | 30% | 1.2191 |
| **Base Score** | - | - | **1.9191** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, food, fun, like, singles, who
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.064)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.3924 (39.2%)

---

### Rank #2: Fun Asian Singles

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4675 | 70% | 0.3273 |
| Semantic Similarity | 4.3279 | 30% | 1.2984 |
| **Base Score** | - | - | **1.6256** |
| **Final Score** | - | - | **0.6100** (61.0%) |

**Why This Matched:**

- **Word Overlap:** fun, singles
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.468) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.328)

**Ranking Justification:**

- Ranked **#2** - score is 0.3924 lower than #1
- Score is 0.0210 higher than #3

---

### Rank #3: American Singles Who Love Asian

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5610 | 70% | 0.3927 |
| Semantic Similarity | 2.9891 | 30% | 0.8967 |
| **Base Score** | - | - | **1.2894** |
| **Final Score** | - | - | **0.5890** (58.9%) |

**Why This Matched:**

- **Word Overlap:** singles, who
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.561)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.989)

**Ranking Justification:**

- Ranked **#3** - score is 0.0210 lower than #2
- Score is 0.0076 higher than #4

---

### Rank #4: Fun Social Singles 35+

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5100 | 70% | 0.3570 |
| Semantic Similarity | 3.4261 | 30% | 1.0278 |
| **Base Score** | - | - | **1.3848** |
| **Final Score** | - | - | **0.5814** (58.1%) |

**Why This Matched:**

- **Word Overlap:** fun, singles
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.510)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.426)

**Ranking Justification:**

- Ranked **#4** - score is 0.0076 lower than #3
- Score is 0.0046 higher than #5

---

### Rank #5: LGBT Food N' Fun Social Group

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5610 | 70% | 0.3927 |
| Semantic Similarity | 2.8015 | 30% | 0.8404 |
| **Base Score** | - | - | **1.2331** |
| **Final Score** | - | - | **0.5768** (57.7%) |

**Why This Matched:**

- **Word Overlap:** food, fun
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.561)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.801)

**Ranking Justification:**

- Ranked **#5** - score is 0.0046 lower than #4
- Score is 0.0141 higher than #6

---

</details>
## 53. Query: `Zen Meetings & Events`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Zen Meetings & Events | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | Zen Meetings & Events | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 3 | Zen Meetings & Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | Zen Meetings & Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | Zen Meetings & Events | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 6 | Zen Events México | **78.0%** | 🟡 MEDIUM **78.0%** | Semantic/meaning-based match |
| 7 | UHG Meetings & Events | **73.8%** | 🟡 MEDIUM **73.8%** | Semantic/meaning-based match |
| 8 | Exclusive Meetings Events | **73.6%** | 🟡 MEDIUM **73.6%** | Semantic/meaning-based match |
| 9 | Elements Meetings Events | **73.5%** | 🟡 MEDIUM **73.5%** | Semantic/meaning-based match |
| 10 | Strategic Meetings & Events | **73.3%** | 🟡 MEDIUM **73.3%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Zen Meetings & Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.2728 | 30% | 1.5818 |
| **Base Score** | - | - | **2.2818** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** &, events, meetings, zen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.273)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: Zen Meetings & Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0420 | 30% | 1.2126 |
| **Base Score** | - | - | **1.9126** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** &, events, meetings, zen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.042)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0014 higher than #3

---

### Rank #3: Zen Meetings & Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.0922 | 30% | 1.8277 |
| **Base Score** | - | - | **2.5277** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, events, meetings, zen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.092)

**Ranking Justification:**

- Ranked **#3** - score is 0.0014 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Zen Meetings & Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9254 | 30% | 1.1776 |
| **Base Score** | - | - | **1.8776** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, events, meetings, zen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.925)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Zen Meetings & Events

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.8687 | 30% | 1.1606 |
| **Base Score** | - | - | **1.8606** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, events, meetings, zen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.869)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.2223 higher than #6

---

</details>
## 54. Query: `Chicago South Swim Club`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Chicago South Swim Club | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | South Carolina Swim Club | **91.1%** | 🟢 HIGH **91.1%** | Semantic/meaning-based match |
| 3 | University of Michigan Swim Club | **78.6%** | 🟡 MEDIUM **78.6%** | Semantic/meaning-based match |
| 4 | National Capital Swim Club | **77.7%** | 🟡 MEDIUM **77.7%** | Semantic/meaning-based match |
| 5 | Baltimore City Swim Club | **77.5%** | 🟡 MEDIUM **77.5%** | Semantic/meaning-based match |
| 6 | Detroit Recreation Swim Club | **77.3%** | 🟡 MEDIUM **77.3%** | Semantic/meaning-based match |
| 7 | Carolina Aquatics Swim Club | **77.2%** | 🟡 MEDIUM **77.2%** | Semantic/meaning-based match |
| 8 | Team Carolina Swim Club | **76.8%** | 🟡 MEDIUM **76.8%** | Semantic/meaning-based match |
| 9 | N Shore Swim Club | **76.6%** | 🟡 MEDIUM **76.6%** | Semantic/meaning-based match |
| 10 | Shore Club Chicago | **76.5%** | 🟡 MEDIUM **76.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Chicago South Swim Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.6479 | 30% | 1.6944 |
| **Base Score** | - | - | **2.3944** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** chicago, club, south, swim
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.648)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0915 (9.1%)

---

### Rank #2: South Carolina Swim Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 4.1390 | 30% | 1.2417 |
| **Base Score** | - | - | **1.8192** |
| **Final Score** | - | - | **0.9110** (91.1%) |

**Why This Matched:**

- **Word Overlap:** club, south, swim
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.139)

**Ranking Justification:**

- Ranked **#2** - score is 0.0915 lower than #1
- Score is 0.1251 higher than #3

---

### Rank #3: University of Michigan Swim Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.4377 | 30% | 1.3313 |
| **Base Score** | - | - | **1.8767** |
| **Final Score** | - | - | **0.7859** (78.6%) |

**Why This Matched:**

- **Word Overlap:** club, swim
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.438)

**Ranking Justification:**

- Ranked **#3** - score is 0.1251 lower than #2
- Score is 0.0086 higher than #4

---

### Rank #4: National Capital Swim Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.2775 | 30% | 1.2832 |
| **Base Score** | - | - | **1.8287** |
| **Final Score** | - | - | **0.7773** (77.7%) |

**Why This Matched:**

- **Word Overlap:** club, swim
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.277)

**Ranking Justification:**

- Ranked **#4** - score is 0.0086 lower than #3
- Score is 0.0027 higher than #5

---

### Rank #5: Baltimore City Swim Club

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 4.2273 | 30% | 1.2682 |
| **Base Score** | - | - | **1.8136** |
| **Final Score** | - | - | **0.7746** (77.5%) |

**Why This Matched:**

- **Word Overlap:** club, swim
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.779)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.227)

**Ranking Justification:**

- Ranked **#5** - score is 0.0027 lower than #4
- Score is 0.0012 higher than #6

---

</details>
## 55. Query: `Edna, Dabra@SAP.IO`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Edna, Dabra@SAP.IO | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Edna Rose | **79.4%** | 🟡 MEDIUM **79.4%** | Semantic/meaning-based match |
| 3 | Edna ISD | **78.3%** | 🟡 MEDIUM **78.3%** | Semantic/meaning-based match |
| 4 | Edna ISD | **77.4%** | 🟡 MEDIUM **77.4%** | Semantic/meaning-based match |
| 5 | EDNA LUMBER COMPANY | **76.5%** | 🟡 MEDIUM **76.5%** | Semantic/meaning-based match |
| 6 | City of Edna | **76.2%** | 🟡 MEDIUM **76.2%** | Semantic/meaning-based match |
| 7 | Edna Owusu | **74.8%** | 🟡 MEDIUM **74.8%** | Semantic/meaning-based match |
| 8 | Edna Sawyer | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 9 | Edna 80th Celebration | **68.6%** | 🟡 MEDIUM **68.6%** | Semantic/meaning-based match |
| 10 | Viajes Edna S.A. | **67.2%** | 🟡 MEDIUM **67.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Edna, Dabra@SAP.IO

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.2455 | 30% | 1.2736 |
| **Base Score** | - | - | **1.9736** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** dabra@sap.io, edna,
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.245)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2083 (20.8%)

---

### Rank #2: Edna Rose

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.8702 | 30% | 1.1611 |
| **Base Score** | - | - | **1.6817** |
| **Final Score** | - | - | **0.7941** (79.4%) |

**Why This Matched:**

- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.870)

**Ranking Justification:**

- Ranked **#2** - score is 0.2083 lower than #1
- Score is 0.0112 higher than #3

---

### Rank #3: Edna ISD

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.7103 | 30% | 1.1131 |
| **Base Score** | - | - | **1.6337** |
| **Final Score** | - | - | **0.7829** (78.3%) |

**Why This Matched:**

- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.710)

**Ranking Justification:**

- Ranked **#3** - score is 0.0112 lower than #2
- Score is 0.0091 higher than #4

---

### Rank #4: Edna ISD

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.5412 | 30% | 1.0624 |
| **Base Score** | - | - | **1.5830** |
| **Final Score** | - | - | **0.7738** (77.4%) |

**Why This Matched:**

- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.541)

**Ranking Justification:**

- Ranked **#4** - score is 0.0091 lower than #3
- Score is 0.0092 higher than #5

---

### Rank #5: EDNA LUMBER COMPANY

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.4487 | 30% | 1.0346 |
| **Base Score** | - | - | **1.5552** |
| **Final Score** | - | - | **0.7646** (76.5%) |

**Why This Matched:**

- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.449)

**Ranking Justification:**

- Ranked **#5** - score is 0.0092 lower than #4
- Score is 0.0028 higher than #6

---

</details>
## 56. Query: `Boys and Girls Club of Dawson Community Centre`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Boys and Girls Club of Dawson Community Centre | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Boys and Girls Club Services of Greater Victoria | **72.9%** | 🟡 MEDIUM **72.9%** | Semantic/meaning-based match |
| 3 | Boys & Girls Club Metro Phoenix Area | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 4 | Boys & Girls Club in Orange County | **71.4%** | 🟡 MEDIUM **71.4%** | Semantic/meaning-based match |
| 5 | Boys & Girls Club of Hilton Head Island | **71.3%** | 🟡 MEDIUM **71.3%** | Semantic/meaning-based match |
| 6 | Boys & Girls Club of Greater High Point | **70.5%** | 🟡 MEDIUM **70.5%** | Semantic/meaning-based match |
| 7 | Pacific Youth Foundation Boys & Girls Club | **70.5%** | 🟡 MEDIUM **70.5%** | Semantic/meaning-based match |
| 8 | North Omaha Boys & Girls Club | **69.2%** | 🟡 MEDIUM **69.2%** | Semantic/meaning-based match |
| 9 | Boys and Girls Club of Green Bay | **69.1%** | 🟡 MEDIUM **69.1%** | Semantic/meaning-based match |
| 10 | Boys & Girls Club of Washington DC | **68.8%** | 🟡 MEDIUM **68.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Boys and Girls Club of Dawson Community Centre

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7830 | 30% | 1.1349 |
| **Base Score** | - | - | **1.8349** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** and, boys, centre, club, community, dawson, girls, of
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.783)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2736 (27.4%)

---

### Rank #2: Boys and Girls Club Services of Greater Victoria

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7650 | 70% | 0.5355 |
| Semantic Similarity | 2.9247 | 30% | 0.8774 |
| **Base Score** | - | - | **1.4129** |
| **Final Score** | - | - | **0.7289** (72.9%) |

**Why This Matched:**

- **Word Overlap:** and, boys, club, girls, of
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.765)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.925)

**Ranking Justification:**

- Ranked **#2** - score is 0.2736 lower than #1
- Score is 0.0030 higher than #3

---

### Rank #3: Boys & Girls Club Metro Phoenix Area

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7650 | 70% | 0.5355 |
| Semantic Similarity | 2.8789 | 30% | 0.8637 |
| **Base Score** | - | - | **1.3992** |
| **Final Score** | - | - | **0.7259** (72.6%) |

**Why This Matched:**

- **Word Overlap:** boys, club, girls
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.765)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.879)

**Ranking Justification:**

- Ranked **#3** - score is 0.0030 lower than #2
- Score is 0.0114 higher than #4

---

### Rank #4: Boys & Girls Club in Orange County

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7650 | 70% | 0.5355 |
| Semantic Similarity | 2.7028 | 30% | 0.8108 |
| **Base Score** | - | - | **1.3463** |
| **Final Score** | - | - | **0.7144** (71.4%) |

**Why This Matched:**

- **Word Overlap:** boys, club, girls
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.765)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.703)

**Ranking Justification:**

- Ranked **#4** - score is 0.0114 lower than #3
- Score is 0.0018 higher than #5

---

### Rank #5: Boys & Girls Club of Hilton Head Island

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7650 | 70% | 0.5355 |
| Semantic Similarity | 2.6746 | 30% | 0.8024 |
| **Base Score** | - | - | **1.3379** |
| **Final Score** | - | - | **0.7126** (71.3%) |

**Why This Matched:**

- **Word Overlap:** boys, club, girls, of
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.765)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.675)

**Ranking Justification:**

- Ranked **#5** - score is 0.0018 lower than #4
- Score is 0.0075 higher than #6

---

</details>
## 57. Query: `Beissbarth`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Beissbarth | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Beissbarth GmbH | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | Bitbar | **47.0%** | 🟠 LOW **47.0%** | Semantic/meaning-based match |
| 4 | Beiss Barth | **44.0%** | 🟠 LOW **44.0%** | Semantic/meaning-based match |
| 5 | Ziebart | **43.8%** | 🟠 LOW **43.8%** | Semantic/meaning-based match |
| 6 | ISOBAR | **41.9%** | 🟠 LOW **41.9%** | Semantic/meaning-based match |
| 7 | SideBar | **40.8%** | 🟠 LOW **40.8%** | Semantic/meaning-based match |
| 8 | MakerBar | **40.6%** | 🟠 LOW **40.6%** | Semantic/meaning-based match |
| 9 | backbar | **40.2%** | 🟠 LOW **40.2%** | Semantic/meaning-based match |
| 10 | Agbar | **39.8%** | 🔴 WEAK **39.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Beissbarth

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.0452 | 30% | 0.9136 |
| **Base Score** | - | - | **1.6136** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** beissbarth
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.045)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Beissbarth GmbH

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 4.6947 | 30% | 1.4084 |
| **Base Score** | - | - | **1.9811** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** beissbarth
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.695)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.4855 higher than #3

---

### Rank #3: Bitbar

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.2812 | 70% | 0.1969 |
| Semantic Similarity | 4.2335 | 30% | 1.2701 |
| **Base Score** | - | - | **1.4669** |
| **Final Score** | - | - | **0.4702** (47.0%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.281) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.234)

**Ranking Justification:**

- Ranked **#3** - score is 0.4855 lower than #2
- Score is 0.0299 higher than #4

---

### Rank #4: Beiss Barth

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3409 | 70% | 0.2386 |
| Semantic Similarity | 3.1152 | 30% | 0.9345 |
| **Base Score** | - | - | **1.1732** |
| **Final Score** | - | - | **0.4404** (44.0%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.341) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.115)

**Ranking Justification:**

- Ranked **#4** - score is 0.0299 lower than #3
- Score is 0.0019 higher than #5

---

### Rank #5: Ziebart

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.2647 | 70% | 0.1853 |
| Semantic Similarity | 3.9196 | 30% | 1.1759 |
| **Base Score** | - | - | **1.3612** |
| **Final Score** | - | - | **0.4384** (43.8%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.265) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.920)

**Ranking Justification:**

- Ranked **#5** - score is 0.0019 lower than #4
- Score is 0.0194 higher than #6

---

</details>
## 58. Query: `US Night Vision`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | US Night Vision | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | Association of U.S. Night Vision Manufacturers | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | WORLD VISION US | **91.4%** | 🟢 HIGH **91.4%** | Semantic/meaning-based match |
| 4 | Night Vision Entertainment | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 5 | WORLD VISION US | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 6 | World Vision US | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 7 | Night Vision Entertainment | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | WORLD VISION US | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | ITT Night Vision | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | Night Vision Systems, LLC | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: US Night Vision

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.1016 | 30% | 1.2305 |
| **Base Score** | - | - | **1.9305** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** night, us, vision
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.102)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0481 (4.8%)

---

### Rank #2: Association of U.S. Night Vision Manufacturers

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 3.0268 | 30% | 0.9080 |
| **Base Score** | - | - | **1.4330** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** night, vision
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.027)

**Ranking Justification:**

- Ranked **#2** - score is 0.0481 lower than #1
- Score is 0.0416 higher than #3

---

### Rank #3: WORLD VISION US

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.5576 | 30% | 1.0673 |
| **Base Score** | - | - | **1.6341** |
| **Final Score** | - | - | **0.9142** (91.4%) |

**Why This Matched:**

- **Word Overlap:** us, vision
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.558)

**Ranking Justification:**

- Ranked **#3** - score is 0.0416 lower than #2
- Score is 0.0055 higher than #4

---

### Rank #4: Night Vision Entertainment

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.9711 | 30% | 1.1913 |
| **Base Score** | - | - | **1.7581** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** night, vision
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.971)

**Ranking Justification:**

- Ranked **#4** - score is 0.0055 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: WORLD VISION US

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.9270 | 30% | 1.1781 |
| **Base Score** | - | - | **1.7449** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** us, vision
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.927)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 59. Query: `Amedysis, Incorporated`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Amedysis, Inc. | **100.6%** | 🟢 EXACT **100.6%** | Near-exact text match |
| 2 | Amedysis, Incorporated | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Amedysis, Incorporated | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | Amedysis Home Health | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | Dialysis Corporation | **47.2%** | 🟠 LOW **47.2%** | Semantic/meaning-based match |
| 6 | Personalysis Corporation | **44.9%** | 🟠 LOW **44.9%** | Semantic/meaning-based match |
| 7 | Medysis | **44.2%** | 🟠 LOW **44.2%** | Semantic/meaning-based match |
| 8 | Cardialysis | **42.4%** | 🟠 LOW **42.4%** | Semantic/meaning-based match |
| 9 | Avysis | **42.1%** | 🟠 LOW **42.1%** | Semantic/meaning-based match |
| 10 | Dialysis Centers Incorporated | **40.7%** | 🟠 LOW **40.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Amedysis, Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.7399 | 30% | 1.4220 |
| **Base Score** | - | - | **2.1220** |
| **Final Score** | - | - | **1.0061** (100.6%) |

**Why This Matched:**

- **Word Overlap:** amedysis,
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.740)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0061)
- Score gap to #2: 0.0037 (0.4%)

---

### Rank #2: Amedysis, Incorporated

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.2293 | 30% | 1.2688 |
| **Base Score** | - | - | **1.9688** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** amedysis,, incorporated
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.229)

**Ranking Justification:**

- Ranked **#2** - score is 0.0037 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Amedysis, Incorporated

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9352 | 30% | 1.1806 |
| **Base Score** | - | - | **1.8806** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** amedysis,, incorporated
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.935)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0467 higher than #4

---

### Rank #4: Amedysis Home Health

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 3.4784 | 30% | 1.0435 |
| **Base Score** | - | - | **1.5685** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.478)

**Ranking Justification:**

- Ranked **#4** - score is 0.0467 lower than #3
- Score is 0.4839 higher than #5

---

### Rank #5: Dialysis Corporation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.2812 | 70% | 0.1969 |
| Semantic Similarity | 4.2999 | 30% | 1.2900 |
| **Base Score** | - | - | **1.4869** |
| **Final Score** | - | - | **0.4719** (47.2%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.281) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.300)

**Ranking Justification:**

- Ranked **#5** - score is 0.4839 lower than #4
- Score is 0.0228 higher than #6

---

</details>
## 60. Query: `Taiyo Air Service Co.,Ltd`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Taiyo Air Service Co.,Ltd | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Taiyo Air Services Co. | **81.2%** | 🟢 HIGH **81.2%** | Semantic/meaning-based match |
| 3 | Fuyo Air Service Co. Ltd. | **74.4%** | 🟡 MEDIUM **74.4%** | Semantic/meaning-based match |
| 4 | CITS Taikoo Air Service Ltd | **73.2%** | 🟡 MEDIUM **73.2%** | Semantic/meaning-based match |
| 5 | Tec Air Service Co. Ltd. (tokyo) | **73.2%** | 🟡 MEDIUM **73.2%** | Semantic/meaning-based match |
| 6 | GUANGZHOU GZL AIR SERVICE CO., LTD. | **72.1%** | 🟡 MEDIUM **72.1%** | Semantic/meaning-based match |
| 7 | WORLD-AIR SEA SERVICE CO., LTD. | **71.8%** | 🟡 MEDIUM **71.8%** | Semantic/meaning-based match |
| 8 | Union Air Service Co. Ltd. (Japan) | **71.7%** | 🟡 MEDIUM **71.7%** | Semantic/meaning-based match |
| 9 | Yumen Air & Sea Service Co., Ltd. | **71.3%** | 🟡 MEDIUM **71.3%** | Semantic/meaning-based match |
| 10 | Ryowa Diamond Air Service Co., Ltd. | **71.2%** | 🟡 MEDIUM **71.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Taiyo Air Service Co.,Ltd

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.0738 | 30% | 1.5221 |
| **Base Score** | - | - | **2.2221** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** air, co.,ltd, service, taiyo
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.074)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.1904 (19.0%)

---

### Rank #2: Taiyo Air Services Co.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7244 | 70% | 0.5071 |
| Semantic Similarity | 5.2796 | 30% | 1.5839 |
| **Base Score** | - | - | **2.0910** |
| **Final Score** | - | - | **0.8120** (81.2%) |

**Why This Matched:**

- **Word Overlap:** air, taiyo
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.724)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.280)

**Ranking Justification:**

- Ranked **#2** - score is 0.1904 lower than #1
- Score is 0.0676 higher than #3

---

### Rank #3: Fuyo Air Service Co. Ltd.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7244 | 70% | 0.5071 |
| Semantic Similarity | 4.0969 | 30% | 1.2291 |
| **Base Score** | - | - | **1.7362** |
| **Final Score** | - | - | **0.7444** (74.4%) |

**Why This Matched:**

- **Word Overlap:** air, service
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.724)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.097)

**Ranking Justification:**

- Ranked **#3** - score is 0.0676 lower than #2
- Score is 0.0123 higher than #4

---

### Rank #4: CITS Taikoo Air Service Ltd

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6444 | 30% | 1.0933 |
| **Base Score** | - | - | **1.6139** |
| **Final Score** | - | - | **0.7321** (73.2%) |

**Why This Matched:**

- **Word Overlap:** air, service
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.644)

**Ranking Justification:**

- Ranked **#4** - score is 0.0123 lower than #3
- Score is 0.0001 higher than #5

---

### Rank #5: Tec Air Service Co. Ltd. (tokyo)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7969 | 70% | 0.5578 |
| Semantic Similarity | 2.9880 | 30% | 0.8964 |
| **Base Score** | - | - | **1.4542** |
| **Final Score** | - | - | **0.7320** (73.2%) |

**Why This Matched:**

- **Word Overlap:** air, service
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.797)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.988)

**Ranking Justification:**

- Ranked **#5** - score is 0.0001 lower than #4
- Score is 0.0114 higher than #6

---

</details>
## 61. Query: `National Conference on Race & Ethnicity in American Higher E`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | National Conference on Race & Ethnicity in American Higher E | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | National Conference On Race & Ethnicity In America Higher Ed | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 3 | National Conference on Race & Ethnicity in Am. Higher Educ | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 4 | NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | NCORE NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER EDUCATION | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | National Conference on Race & Ethnicity in AM Higher Education | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | NATIONAL CONF. ON RACE & ETHNICITY IN AMERICAN HIGHER EDUC. | **90.5%** | 🟢 HIGH **90.5%** | Near-exact text match |
| 8 | NATL CONF ON RACE & ETHNICITY IN AMERICAN HIGHER EDUCATION | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | National Conference on Race & Ethnicity in Higher Education | **76.1%** | 🟡 MEDIUM **76.1%** | Semantic/meaning-based match |
| 10 | National Conference on Race & Ethnicity in America in Higher Education (NCORE) | **74.2%** | 🟡 MEDIUM **74.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: National Conference on Race & Ethnicity in American Higher E

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9576 | 30% | 1.1873 |
| **Base Score** | - | - | **1.8873** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, american, conference, e, ethnicity, higher, in, national, on, race
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.958)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0938 (9.4%)

---

### Rank #2: National Conference On Race & Ethnicity In America Higher Ed

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8490 | 70% | 0.5943 |
| Semantic Similarity | 3.7917 | 30% | 1.1375 |
| **Base Score** | - | - | **1.7318** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** &, conference, ethnicity, higher, in, national, on, race
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.849)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.792)

**Ranking Justification:**

- Ranked **#2** - score is 0.0938 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: National Conference on Race & Ethnicity in Am. Higher Educ

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8490 | 70% | 0.5943 |
| Semantic Similarity | 3.4401 | 30% | 1.0320 |
| **Base Score** | - | - | **1.6263** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** &, conference, ethnicity, higher, in, national, on, race
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.849)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.440)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0032 higher than #4

---

### Rank #4: NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8421 | 70% | 0.5895 |
| Semantic Similarity | 3.7860 | 30% | 1.1358 |
| **Base Score** | - | - | **1.7253** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** american, conference, ethnicity, higher, in, national, on, race
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.842)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.786)

**Ranking Justification:**

- Ranked **#4** - score is 0.0032 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: NCORE NATIONAL CONFERENCE ON RACE AND ETHNICITY IN AMERICAN HIGHER EDUCATION

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 3.7776 | 30% | 1.1333 |
| **Base Score** | - | - | **1.7060** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** american, conference, ethnicity, higher, in, national, on, race
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.778)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 62. Query: `Reminger Law Firm`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Reminger Law Firm | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | Reminger & Reminger Law Firm | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 3 | Lanier Law Firm | **91.3%** | 🟢 HIGH **91.3%** | Semantic/meaning-based match |
| 4 | McNair Law Firm | **91.3%** | 🟢 HIGH **91.3%** | Semantic/meaning-based match |
| 5 | Levin Law Firm | **91.3%** | 🟢 HIGH **91.3%** | Semantic/meaning-based match |
| 6 | Hood Law Firm | **91.1%** | 🟢 HIGH **91.1%** | Semantic/meaning-based match |
| 7 | Lanier Law Firm | **91.1%** | 🟢 HIGH **91.1%** | Semantic/meaning-based match |
| 8 | Cordell Law Firm | **91.1%** | 🟢 HIGH **91.1%** | Semantic/meaning-based match |
| 9 | Ackerman Law Firm | **91.1%** | 🟢 HIGH **91.1%** | Semantic/meaning-based match |
| 10 | Lauro Law Firm | **91.1%** | 🟢 HIGH **91.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Reminger Law Firm

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0927 | 30% | 1.2278 |
| **Base Score** | - | - | **1.9278** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** firm, law, reminger
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.093)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0481 (4.8%)

---

### Rank #2: Reminger & Reminger Law Firm

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.8080 | 30% | 1.1424 |
| **Base Score** | - | - | **1.8424** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** firm, law, reminger
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.808)

**Ranking Justification:**

- Ranked **#2** - score is 0.0481 lower than #1
- Score is 0.0431 higher than #3

---

### Rank #3: Lanier Law Firm

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.1033 | 30% | 0.9310 |
| **Base Score** | - | - | **1.4978** |
| **Final Score** | - | - | **0.9127** (91.3%) |

**Why This Matched:**

- **Word Overlap:** firm, law
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.103)

**Ranking Justification:**

- Ranked **#3** - score is 0.0431 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: McNair Law Firm

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 2.8944 | 30% | 0.8683 |
| **Base Score** | - | - | **1.4351** |
| **Final Score** | - | - | **0.9127** (91.3%) |

**Why This Matched:**

- **Word Overlap:** firm, law
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.894)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Levin Law Firm

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 2.8739 | 30% | 0.8622 |
| **Base Score** | - | - | **1.4290** |
| **Final Score** | - | - | **0.9127** (91.3%) |

**Why This Matched:**

- **Word Overlap:** firm, law
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.874)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0018 higher than #6

---

</details>
## 63. Query: `SEMMOA BOD`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | SEMMOA BOD | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | World Bod | **75.6%** | 🟡 MEDIUM **75.6%** | Semantic/meaning-based match |
| 3 | BOD HD | **75.1%** | 🟡 MEDIUM **75.1%** | Semantic/meaning-based match |
| 4 | BOD | **74.6%** | 🟡 MEDIUM **74.6%** | Semantic/meaning-based match |
| 5 | SEMMOA AACM | **73.6%** | 🟡 MEDIUM **73.6%** | Semantic/meaning-based match |
| 6 | Regions BOD | **73.5%** | 🟡 MEDIUM **73.5%** | Semantic/meaning-based match |
| 7 | BOD Consulting | **72.6%** | 🟡 MEDIUM **72.6%** | Semantic/meaning-based match |
| 8 | BOD Meeding | **72.4%** | 🟡 MEDIUM **72.4%** | Semantic/meaning-based match |
| 9 | CSG BOD | **71.7%** | 🟡 MEDIUM **71.7%** | Semantic/meaning-based match |
| 10 | BOD Meeting | **71.7%** | 🟡 MEDIUM **71.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: SEMMOA BOD

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0002 | 30% | 1.2001 |
| **Base Score** | - | - | **1.9001** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** bod, semmoa
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.000)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2463 (24.6%)

---

### Rank #2: World Bod

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.3018 | 30% | 0.9905 |
| **Base Score** | - | - | **1.5112** |
| **Final Score** | - | - | **0.7561** (75.6%) |

**Why This Matched:**

- **Word Overlap:** bod
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.302)

**Ranking Justification:**

- Ranked **#2** - score is 0.2463 lower than #1
- Score is 0.0047 higher than #3

---

### Rank #3: BOD HD

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.2344 | 30% | 0.9703 |
| **Base Score** | - | - | **1.4910** |
| **Final Score** | - | - | **0.7514** (75.1%) |

**Why This Matched:**

- **Word Overlap:** bod
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.234)

**Ranking Justification:**

- Ranked **#3** - score is 0.0047 lower than #2
- Score is 0.0059 higher than #4

---

### Rank #4: BOD

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6300 | 70% | 0.4410 |
| Semantic Similarity | 4.2891 | 30% | 1.2867 |
| **Base Score** | - | - | **1.7277** |
| **Final Score** | - | - | **0.7455** (74.6%) |

**Why This Matched:**

- **Word Overlap:** bod
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.630)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.289)

**Ranking Justification:**

- Ranked **#4** - score is 0.0059 lower than #3
- Score is 0.0092 higher than #5

---

### Rank #5: SEMMOA AACM

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.0206 | 30% | 0.9062 |
| **Base Score** | - | - | **1.4268** |
| **Final Score** | - | - | **0.7364** (73.6%) |

**Why This Matched:**

- **Word Overlap:** semmoa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.021)

**Ranking Justification:**

- Ranked **#5** - score is 0.0092 lower than #4
- Score is 0.0014 higher than #6

---

</details>
## 64. Query: `Telefonica Global Solutions`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Telefonica Global Solutions | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Telefonica Global Solutions | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Telefonica Global Solutions USA Inc. | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 4 | Telefonica Multinational Solutions | **90.7%** | 🟢 HIGH **90.7%** | Semantic/meaning-based match |
| 5 | Telefonica Multinational Solutions | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | TGS | **75.0%** | 🟡 MEDIUM **75.0%** | Near-exact text match |
| 7 | 02 Telefonica | **67.1%** | 🟡 MEDIUM **67.1%** | Semantic/meaning-based match |
| 8 | Telefonica Del Peru | **66.4%** | 🟡 MEDIUM **66.4%** | Semantic/meaning-based match |
| 9 | Telefonica España | **66.0%** | 🟡 MEDIUM **66.0%** | Semantic/meaning-based match |
| 10 | TELEFONICA INTERNATIONAL USA | **65.9%** | 🟡 MEDIUM **65.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Telefonica Global Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.4480 | 30% | 1.6344 |
| **Base Score** | - | - | **2.3344** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** global, solutions, telefonica
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.448)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: Telefonica Global Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9407 | 30% | 1.1822 |
| **Base Score** | - | - | **1.8822** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** global, solutions, telefonica
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.941)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0467 higher than #3

---

### Rank #3: Telefonica Global Solutions USA Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8464 | 70% | 0.5925 |
| Semantic Similarity | 3.4763 | 30% | 1.0429 |
| **Base Score** | - | - | **1.6354** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** global, solutions, telefonica
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.846)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.476)

**Ranking Justification:**

- Ranked **#3** - score is 0.0467 lower than #2
- Score is 0.0492 higher than #4

---

### Rank #4: Telefonica Multinational Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 5.1342 | 30% | 1.5403 |
| **Base Score** | - | - | **2.1586** |
| **Final Score** | - | - | **0.9065** (90.7%) |

**Why This Matched:**

- **Word Overlap:** solutions, telefonica
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.883)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.134)

**Ranking Justification:**

- Ranked **#4** - score is 0.0492 lower than #3
- Score is 0.0011 higher than #5

---

### Rank #5: Telefonica Multinational Solutions

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 4.0495 | 30% | 1.2149 |
| **Base Score** | - | - | **1.8332** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** solutions, telefonica
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.883)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.050)

**Ranking Justification:**

- Ranked **#5** - score is 0.0011 lower than #4
- Score is 0.1555 higher than #6

---

</details>
## 65. Query: `Travel Leaders - Dube Travel`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Travel Leaders - Dube Travel | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Dube Travel Leaders | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 3 | Dube Travel / Travel Leaders | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 4 | Dube Travel Leaders | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 5 | dube Travel leaders | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 6 | Dube / Travel Leaders | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | Travel Leaders 365 | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 8 | Travel Leaders UK | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 9 | Travel Leaders Network | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 10 | Advent Travel Leaders | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Travel Leaders - Dube Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.2634 | 30% | 1.2790 |
| **Base Score** | - | - | **1.9790** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** -, dube, leaders, travel
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.263)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0433 (4.3%)

---

### Rank #2: Dube Travel Leaders

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.5068 | 30% | 1.3521 |
| **Base Score** | - | - | **2.0521** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** dube, leaders, travel
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.507)

**Ranking Justification:**

- Ranked **#2** - score is 0.0433 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Dube Travel / Travel Leaders

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 3.8292 | 30% | 1.1488 |
| **Base Score** | - | - | **1.7533** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** dube, leaders, travel
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.829)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0034 higher than #4

---

### Rank #4: Dube Travel Leaders

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4854 | 30% | 1.3456 |
| **Base Score** | - | - | **2.0456** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** dube, leaders, travel
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.485)

**Ranking Justification:**

- Ranked **#4** - score is 0.0034 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: dube Travel leaders

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3955 | 30% | 1.3186 |
| **Base Score** | - | - | **2.0186** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** dube, leaders, travel
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.395)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 66. Query: `Hi- Tours`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Hi- Tours | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Hi tours | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 3 | Hi-Tours | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 4 | Hi Life Tours | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | R&C HI Tours | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 6 | Hi Tour | **82.2%** | 🟢 HIGH **82.2%** | Semantic/meaning-based match |
| 7 | Nice Tours | **77.2%** | 🟡 MEDIUM **77.2%** | Semantic/meaning-based match |
| 8 | Sky Tours | **77.0%** | 🟡 MEDIUM **77.0%** | Semantic/meaning-based match |
| 9 | Destination Tours | **76.9%** | 🟡 MEDIUM **76.9%** | Semantic/meaning-based match |
| 10 | Fun Tours | **76.4%** | 🟡 MEDIUM **76.4%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Hi- Tours

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | - | - | **1.0000** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** hi-, tours
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 1.000)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Hi tours

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 6.5987 | 30% | 1.9796 |
| **Base Score** | - | - | **2.6096** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** tours
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.599)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Hi-Tours

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 5.8173 | 30% | 1.7452 |
| **Base Score** | - | - | **2.3752** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.817)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Hi Life Tours

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 5.5240 | 30% | 1.6572 |
| **Base Score** | - | - | **2.2617** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** tours
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.524)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: R&C HI Tours

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 4.4120 | 30% | 1.3236 |
| **Base Score** | - | - | **1.8963** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** tours
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.412)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.1342 higher than #6

---

</details>
## 67. Query: `Volkswagen Group China`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Volkswagen Group China | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | VOLKSWAGEN GROUP CHINA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Volkswagen China | **100.0%** | 🟢 EXACT **100.0%** | Near-exact text match |
| 4 | FAW Volkswagen Automotive Co., Ltd. South China Branch | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 5 | VOLKSWAGEN CHINA INVESTMENT COMPANY LTD | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 6 | Volkswagen China Investment Company.Ltd | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | FAW Volkswagen Automotive Co., Ltd. South China Branch | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 8 | FAW Volkswagen Automotive Co., Ltd. South China Branch | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 9 | Volkswagen Group Japan | **77.0%** | 🟡 MEDIUM **77.0%** | Semantic/meaning-based match |
| 10 | Volkswagen Group Australia | **76.2%** | 🟡 MEDIUM **76.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Volkswagen Group China

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.8691 | 30% | 1.4607 |
| **Base Score** | - | - | **2.1607** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** china, group, volkswagen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.869)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0014 (0.1%)

---

### Rank #2: VOLKSWAGEN GROUP CHINA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.6367 | 30% | 1.9910 |
| **Base Score** | - | - | **2.6910** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** china, group, volkswagen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.637)

**Ranking Justification:**

- Ranked **#2** - score is 0.0014 lower than #1
- Score is 0.0025 higher than #3

---

### Rank #3: Volkswagen China

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.4249 | 30% | 1.9275 |
| **Base Score** | - | - | **2.6275** |
| **Final Score** | - | - | **1.0000** (100.0%) |

**Why This Matched:**

- **Word Overlap:** china, volkswagen
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.425)

**Ranking Justification:**

- Ranked **#3** - score is 0.0025 lower than #2
- Score is 0.0408 higher than #4

---

### Rank #4: FAW Volkswagen Automotive Co., Ltd. South China Branch

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6786 | 70% | 0.4750 |
| Semantic Similarity | 3.3559 | 30% | 1.0068 |
| **Base Score** | - | - | **1.4818** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** china, volkswagen
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.679)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.356)

**Ranking Justification:**

- Ranked **#4** - score is 0.0408 lower than #3
- Score is 0.0034 higher than #5

---

### Rank #5: VOLKSWAGEN CHINA INVESTMENT COMPANY LTD

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 4.7534 | 30% | 1.4260 |
| **Base Score** | - | - | **1.9987** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** china, volkswagen
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.753)

**Ranking Justification:**

- Ranked **#5** - score is 0.0034 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 68. Query: `Sun Tx`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Sun Tx | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Sun Coast | **80.0%** | 🟢 HIGH **80.0%** | Semantic/meaning-based match |
| 3 | SUN TRAVEL | **79.7%** | 🟡 MEDIUM **79.7%** | Semantic/meaning-based match |
| 4 | Mod Sun | **78.7%** | 🟡 MEDIUM **78.7%** | Semantic/meaning-based match |
| 5 | Sun Travel | **77.6%** | 🟡 MEDIUM **77.6%** | Semantic/meaning-based match |
| 6 | Sun City | **77.5%** | 🟡 MEDIUM **77.5%** | Semantic/meaning-based match |
| 7 | sun coast | **77.3%** | 🟡 MEDIUM **77.3%** | Semantic/meaning-based match |
| 8 | Rising Sun | **77.2%** | 🟡 MEDIUM **77.2%** | Semantic/meaning-based match |
| 9 | Sun City Texas | **77.1%** | 🟡 MEDIUM **77.1%** | Semantic/meaning-based match |
| 10 | Sun Resorts | **76.6%** | 🟡 MEDIUM **76.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Sun Tx

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.6643 | 30% | 1.6993 |
| **Base Score** | - | - | **2.3993** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** sun, tx
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.664)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2021 (20.2%)

---

### Rank #2: Sun Coast

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.1374 | 30% | 1.5412 |
| **Base Score** | - | - | **2.0619** |
| **Final Score** | - | - | **0.8004** (80.0%) |

**Why This Matched:**

- **Word Overlap:** sun
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.137)

**Ranking Justification:**

- Ranked **#2** - score is 0.2021 lower than #1
- Score is 0.0032 higher than #3

---

### Rank #3: SUN TRAVEL

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.1303 | 30% | 1.5391 |
| **Base Score** | - | - | **2.0597** |
| **Final Score** | - | - | **0.7972** (79.7%) |

**Why This Matched:**

- **Word Overlap:** sun
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.130)

**Ranking Justification:**

- Ranked **#3** - score is 0.0032 lower than #2
- Score is 0.0105 higher than #4

---

### Rank #4: Mod Sun

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.9325 | 30% | 1.4798 |
| **Base Score** | - | - | **2.0004** |
| **Final Score** | - | - | **0.7866** (78.7%) |

**Why This Matched:**

- **Word Overlap:** sun
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.933)

**Ranking Justification:**

- Ranked **#4** - score is 0.0105 lower than #3
- Score is 0.0105 higher than #5

---

### Rank #5: Sun Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.6844 | 30% | 1.4053 |
| **Base Score** | - | - | **1.9259** |
| **Final Score** | - | - | **0.7761** (77.6%) |

**Why This Matched:**

- **Word Overlap:** sun
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.684)

**Ranking Justification:**

- Ranked **#5** - score is 0.0105 lower than #4
- Score is 0.0010 higher than #6

---

</details>
## 69. Query: `Southern Vermont Deerfield Valley Chamber of commerce`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Southern Vermont Deerfield Valley Chamber of commerce | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Regional Black Chamber of Commerce Southern California | **69.8%** | 🟡 MEDIUM **69.8%** | Semantic/meaning-based match |
| 3 | Deerfield Beach Chamber of Commerce | **67.6%** | 🟡 MEDIUM **67.6%** | Semantic/meaning-based match |
| 4 | Deerfield Bannockburn Riverwoods Chamber of Commerce | **66.9%** | 🟡 MEDIUM **66.9%** | Semantic/meaning-based match |
| 5 | West Valley Warner Center Chamber of Commerce | **66.6%** | 🟡 MEDIUM **66.6%** | Semantic/meaning-based match |
| 6 | Mill Valley Chamber of Commerce & Visitor Center | **65.8%** | 🟡 MEDIUM **65.8%** | Semantic/meaning-based match |
| 7 | Southern California Black Chamber of Commerce | **65.1%** | 🟡 MEDIUM **65.1%** | Semantic/meaning-based match |
| 8 | Southern Colorado Women's Chamber of Commerce | **64.5%** | 🟡 MEDIUM **64.5%** | Semantic/meaning-based match |
| 9 | DEERFIELD BEACH CHAMBER OF COMMERCE | **64.4%** | 🟡 MEDIUM **64.4%** | Semantic/meaning-based match |
| 10 | GREATER ORO VALLEY CHAMBER OF COMMERCE | **64.3%** | 🟡 MEDIUM **64.3%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Southern Vermont Deerfield Valley Chamber of commerce

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.6302 | 30% | 1.6891 |
| **Base Score** | - | - | **2.3891** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** chamber, commerce, deerfield, of, southern, valley, vermont
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.630)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.3047 (30.5%)

---

### Rank #2: Regional Black Chamber of Commerce Southern California

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.2439 | 30% | 0.9732 |
| **Base Score** | - | - | **1.4938** |
| **Final Score** | - | - | **0.6977** (69.8%) |

**Why This Matched:**

- **Word Overlap:** chamber, commerce, of, southern
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.244)

**Ranking Justification:**

- Ranked **#2** - score is 0.3047 lower than #1
- Score is 0.0217 higher than #3

---

### Rank #3: Deerfield Beach Chamber of Commerce

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6198 | 70% | 0.4339 |
| Semantic Similarity | 4.4668 | 30% | 1.3401 |
| **Base Score** | - | - | **1.7739** |
| **Final Score** | - | - | **0.6760** (67.6%) |

**Why This Matched:**

- **Word Overlap:** chamber, commerce, deerfield, of
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.620)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.467)

**Ranking Justification:**

- Ranked **#3** - score is 0.0217 lower than #2
- Score is 0.0069 higher than #4

---

### Rank #4: Deerfield Bannockburn Riverwoods Chamber of Commerce

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 3.5970 | 30% | 1.0791 |
| **Base Score** | - | - | **1.5524** |
| **Final Score** | - | - | **0.6690** (66.9%) |

**Why This Matched:**

- **Word Overlap:** chamber, commerce, deerfield, of
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.597)

**Ranking Justification:**

- Ranked **#4** - score is 0.0069 lower than #3
- Score is 0.0033 higher than #5

---

### Rank #5: West Valley Warner Center Chamber of Commerce

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6906 | 70% | 0.4834 |
| Semantic Similarity | 3.3450 | 30% | 1.0035 |
| **Base Score** | - | - | **1.4869** |
| **Final Score** | - | - | **0.6657** (66.6%) |

**Why This Matched:**

- **Word Overlap:** chamber, commerce, of, valley
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.691)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.345)

**Ranking Justification:**

- Ranked **#5** - score is 0.0033 lower than #4
- Score is 0.0076 higher than #6

---

</details>
## 70. Query: `DGR Ministries`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | DGR Ministries | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Power Ministries | **81.1%** | 🟢 HIGH **81.1%** | Semantic/meaning-based match |
| 3 | DG Ministries | **80.1%** | 🟢 HIGH **80.1%** | Semantic/meaning-based match |
| 4 | Empowered Ministries | **79.3%** | 🟡 MEDIUM **79.3%** | Semantic/meaning-based match |
| 5 | Legacy Ministries | **78.9%** | 🟡 MEDIUM **78.9%** | Semantic/meaning-based match |
| 6 | Sure ministries | **78.4%** | 🟡 MEDIUM **78.4%** | Semantic/meaning-based match |
| 7 | CV Ministries | **78.3%** | 🟡 MEDIUM **78.3%** | Semantic/meaning-based match |
| 8 | SB Ministries | **78.2%** | 🟡 MEDIUM **78.2%** | Semantic/meaning-based match |
| 9 | Special Ministries | **78.2%** | 🟡 MEDIUM **78.2%** | Semantic/meaning-based match |
| 10 | AG Ministries | **78.1%** | 🟡 MEDIUM **78.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: DGR Ministries

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.2341 | 30% | 1.5702 |
| **Base Score** | - | - | **2.2702** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** dgr, ministries
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.234)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.1911 (19.1%)

---

### Rank #2: Power Ministries

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.9856 | 30% | 1.4957 |
| **Base Score** | - | - | **2.0163** |
| **Final Score** | - | - | **0.8113** (81.1%) |

**Why This Matched:**

- **Word Overlap:** ministries
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.986)

**Ranking Justification:**

- Ranked **#2** - score is 0.1911 lower than #1
- Score is 0.0105 higher than #3

---

### Rank #3: DG Ministries

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.8043 | 30% | 1.4413 |
| **Base Score** | - | - | **1.9619** |
| **Final Score** | - | - | **0.8008** (80.1%) |

**Why This Matched:**

- **Word Overlap:** ministries
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.804)

**Ranking Justification:**

- Ranked **#3** - score is 0.0105 lower than #2
- Score is 0.0080 higher than #4

---

### Rank #4: Empowered Ministries

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.6663 | 30% | 1.3999 |
| **Base Score** | - | - | **1.9205** |
| **Final Score** | - | - | **0.7929** (79.3%) |

**Why This Matched:**

- **Word Overlap:** ministries
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.666)

**Ranking Justification:**

- Ranked **#4** - score is 0.0080 lower than #3
- Score is 0.0040 higher than #5

---

### Rank #5: Legacy Ministries

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5969 | 30% | 1.3791 |
| **Base Score** | - | - | **1.8997** |
| **Final Score** | - | - | **0.7889** (78.9%) |

**Why This Matched:**

- **Word Overlap:** ministries
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.597)

**Ranking Justification:**

- Ranked **#5** - score is 0.0040 lower than #4
- Score is 0.0050 higher than #6

---

</details>
## 71. Query: `Impacto 6`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Impacto 6 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Impacto Vital | **91.1%** | 🟢 HIGH **91.1%** | Semantic/meaning-based match |
| 3 | Impacto 52 | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Impacto EDL | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Impacto Tactico | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | Impacto YOUTH | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Impacto Ejecutivo | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Kiin Impacto | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Impacto Strategies | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | IMPACTO Youth | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Impacto 6

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.1650 | 30% | 0.9495 |
| **Base Score** | - | - | **1.6495** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 6, impacto
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.165)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0915 (9.1%)

---

### Rank #2: Impacto Vital

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 2.5618 | 30% | 0.7685 |
| **Base Score** | - | - | **1.3635** |
| **Final Score** | - | - | **0.9110** (91.1%) |

**Why This Matched:**

- **Word Overlap:** impacto
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.562)

**Ranking Justification:**

- Ranked **#2** - score is 0.0915 lower than #1
- Score is 0.0055 higher than #3

---

### Rank #3: Impacto 52

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 4.9085 | 30% | 1.4726 |
| **Base Score** | - | - | **2.0676** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** impacto
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.909)

**Ranking Justification:**

- Ranked **#3** - score is 0.0055 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Impacto EDL

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 4.1588 | 30% | 1.2477 |
| **Base Score** | - | - | **1.8427** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** impacto
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.159)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Impacto Tactico

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.9702 | 30% | 1.1911 |
| **Base Score** | - | - | **1.7861** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** impacto
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.970)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 72. Query: `Neos Therapeutics, Inc.`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Neos Therapeutics, Inc. | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Neos Therapeutics, Inc. | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Neos Therapeutics | **96.3%** | 🟢 EXACT **96.3%** | Near-exact text match |
| 4 | Neos Therapeutics | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 5 | Neos Therapeutics | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 6 | Neos Therapeutics | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 7 | Neos Therapeutics | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 8 | Neos Therapeutics | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 9 | Neos Therapeutics | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 10 | Neos Therapeutics LP | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Neos Therapeutics, Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.8159 | 30% | 1.1448 |
| **Base Score** | - | - | **1.8448** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** inc., neos, therapeutics,
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.816)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: Neos Therapeutics, Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.6107 | 30% | 1.0832 |
| **Base Score** | - | - | **1.7832** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** inc., neos, therapeutics,
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.611)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0390 higher than #3

---

### Rank #3: Neos Therapeutics

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 3.7917 | 30% | 1.1375 |
| **Base Score** | - | - | **1.7792** |
| **Final Score** | - | - | **0.9634** (96.3%) |

**Why This Matched:**

- **Word Overlap:** neos
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.917)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.792)

**Ranking Justification:**

- Ranked **#3** - score is 0.0390 lower than #2
- Score is 0.0043 higher than #4

---

### Rank #4: Neos Therapeutics

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 3.5994 | 30% | 1.0798 |
| **Base Score** | - | - | **1.7215** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** neos
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.917)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.599)

**Ranking Justification:**

- Ranked **#4** - score is 0.0043 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Neos Therapeutics

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 3.5082 | 30% | 1.0525 |
| **Base Score** | - | - | **1.6941** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** neos
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.917)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.508)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 73. Query: `International Tax Institute`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | International Tax Institute | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | International Tax Institute | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | International Property Tax Institute | **96.2%** | 🟢 EXACT **96.2%** | Semantic/meaning-based match |
| 4 | International Property Tax Institute | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 5 | INTERNATIONAL TAX INSTITUTE INC | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 6 | International Property Tax Institute | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | International Tax & Auditing Institute | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 8 | Federal Tax Institute | **92.4%** | 🟢 HIGH **92.4%** | Semantic/meaning-based match |
| 9 | International Tax Form | **92.2%** | 🟢 HIGH **92.2%** | Semantic/meaning-based match |
| 10 | Sales Tax Institute | **91.4%** | 🟢 HIGH **91.4%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: International Tax Institute

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.1791 | 30% | 1.2537 |
| **Base Score** | - | - | **1.9537** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** institute, international, tax
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.179)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: International Tax Institute

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.6195 | 30% | 1.0859 |
| **Base Score** | - | - | **1.7859** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** institute, international, tax
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.620)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0409 higher than #3

---

### Rank #3: International Property Tax Institute

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 4.1382 | 30% | 1.2415 |
| **Base Score** | - | - | **1.8460** |
| **Final Score** | - | - | **0.9616** (96.2%) |

**Why This Matched:**

- **Word Overlap:** institute, international, tax
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.138)

**Ranking Justification:**

- Ranked **#3** - score is 0.0409 lower than #2
- Score is 0.0024 higher than #4

---

### Rank #4: International Property Tax Institute

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 3.4557 | 30% | 1.0367 |
| **Base Score** | - | - | **1.6413** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** institute, international, tax
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.456)

**Ranking Justification:**

- Ranked **#4** - score is 0.0024 lower than #3
- Score is 0.0034 higher than #5

---

### Rank #5: INTERNATIONAL TAX INSTITUTE INC

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0977 | 30% | 1.2293 |
| **Base Score** | - | - | **1.9293** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** institute, international, tax
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.098)

**Ranking Justification:**

- Ranked **#5** - score is 0.0034 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 74. Query: `Mitsubishi M501G`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Mitsubishi M501G | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Mitsubishi Power | **78.9%** | 🟡 MEDIUM **78.9%** | Semantic/meaning-based match |
| 3 | Mitsubishi Motors | **78.8%** | 🟡 MEDIUM **78.8%** | Semantic/meaning-based match |
| 4 | Mitsubishi Securities | **77.1%** | 🟡 MEDIUM **77.1%** | Semantic/meaning-based match |
| 5 | Mitsubishi Germany | **76.9%** | 🟡 MEDIUM **76.9%** | Semantic/meaning-based match |
| 6 | Mitsubishi Digital | **76.8%** | 🟡 MEDIUM **76.8%** | Semantic/meaning-based match |
| 7 | Mitsubishi Motor Company | **76.2%** | 🟡 MEDIUM **76.2%** | Semantic/meaning-based match |
| 8 | Mitsubishi Paper | **76.1%** | 🟡 MEDIUM **76.1%** | Semantic/meaning-based match |
| 9 | Mitsubishi Meeting | **75.8%** | 🟡 MEDIUM **75.8%** | Semantic/meaning-based match |
| 10 | Mitsubishi Estate | **75.6%** | 🟡 MEDIUM **75.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Mitsubishi M501G

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.5248 | 30% | 1.3574 |
| **Base Score** | - | - | **2.0574** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** m501g, mitsubishi
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.525)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2138 (21.4%)

---

### Rank #2: Mitsubishi Power

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.5224 | 30% | 1.6567 |
| **Base Score** | - | - | **2.1773** |
| **Final Score** | - | - | **0.7886** (78.9%) |

**Why This Matched:**

- **Word Overlap:** mitsubishi
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.522)

**Ranking Justification:**

- Ranked **#2** - score is 0.2138 lower than #1
- Score is 0.0001 higher than #3

---

### Rank #3: Mitsubishi Motors

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.5194 | 30% | 1.6558 |
| **Base Score** | - | - | **2.1764** |
| **Final Score** | - | - | **0.7885** (78.8%) |

**Why This Matched:**

- **Word Overlap:** mitsubishi
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.519)

**Ranking Justification:**

- Ranked **#3** - score is 0.0001 lower than #2
- Score is 0.0174 higher than #4

---

### Rank #4: Mitsubishi Securities

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.1569 | 30% | 1.5471 |
| **Base Score** | - | - | **2.0677** |
| **Final Score** | - | - | **0.7711** (77.1%) |

**Why This Matched:**

- **Word Overlap:** mitsubishi
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.157)

**Ranking Justification:**

- Ranked **#4** - score is 0.0174 lower than #3
- Score is 0.0025 higher than #5

---

### Rank #5: Mitsubishi Germany

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.1051 | 30% | 1.5315 |
| **Base Score** | - | - | **2.0521** |
| **Final Score** | - | - | **0.7686** (76.9%) |

**Why This Matched:**

- **Word Overlap:** mitsubishi
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.105)

**Ranking Justification:**

- Ranked **#5** - score is 0.0025 lower than #4
- Score is 0.0008 higher than #6

---

</details>
## 75. Query: `Huskies Sports`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Huskies Sports | **100.7%** | 🟢 EXACT **100.7%** | Near-exact text match |
| 2 | Huskies Sports | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 3 | Huskies Basketball | **73.1%** | 🟡 MEDIUM **73.1%** | Semantic/meaning-based match |
| 4 | Empire State Huskies | **72.3%** | 🟡 MEDIUM **72.3%** | Semantic/meaning-based match |
| 5 | Miami Huskies | **72.2%** | 🟡 MEDIUM **72.2%** | Semantic/meaning-based match |
| 6 | Wolverines Sports | **71.5%** | 🟡 MEDIUM **71.5%** | Semantic/meaning-based match |
| 7 | Mid Huron Huskies | **71.5%** | 🟡 MEDIUM **71.5%** | Semantic/meaning-based match |
| 8 | NH Huskies | **71.1%** | 🟡 MEDIUM **71.1%** | Semantic/meaning-based match |
| 9 | Sports Collegiate | **71.0%** | 🟡 MEDIUM **71.0%** | Semantic/meaning-based match |
| 10 | Cheer Sports | **70.8%** | 🟡 MEDIUM **70.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Huskies Sports

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.2045 | 30% | 1.8614 |
| **Base Score** | - | - | **2.5614** |
| **Final Score** | - | - | **1.0068** (100.7%) |

**Why This Matched:**

- **Word Overlap:** huskies, sports
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.205)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0068)
- Score gap to #2: 0.0030 (0.3%)

---

### Rank #2: Huskies Sports

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.6607 | 30% | 1.6982 |
| **Base Score** | - | - | **2.3982** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** huskies, sports
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.661)

**Ranking Justification:**

- Ranked **#2** - score is 0.0030 lower than #1
- Score is 0.2729 higher than #3

---

### Rank #3: Huskies Basketball

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.2586 | 30% | 1.2776 |
| **Base Score** | - | - | **1.7982** |
| **Final Score** | - | - | **0.7310** (73.1%) |

**Why This Matched:**

- **Word Overlap:** huskies
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.259)

**Ranking Justification:**

- Ranked **#3** - score is 0.2729 lower than #2
- Score is 0.0076 higher than #4

---

### Rank #4: Empire State Huskies

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 5.0809 | 30% | 1.5243 |
| **Base Score** | - | - | **1.9976** |
| **Final Score** | - | - | **0.7233** (72.3%) |

**Why This Matched:**

- **Word Overlap:** huskies
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.081)

**Ranking Justification:**

- Ranked **#4** - score is 0.0076 lower than #3
- Score is 0.0010 higher than #5

---

### Rank #5: Miami Huskies

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.0815 | 30% | 1.2245 |
| **Base Score** | - | - | **1.7451** |
| **Final Score** | - | - | **0.7223** (72.2%) |

**Why This Matched:**

- **Word Overlap:** huskies
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.082)

**Ranking Justification:**

- Ranked **#5** - score is 0.0010 lower than #4
- Score is 0.0070 higher than #6

---

</details>
## 76. Query: `Acacia Pharma Group Inc.`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Acacia Pharma Group Inc. | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Acacia Pharma | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 3 | Acacia Pharma Ltd | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 4 | ACACIA PHARMA, Inc. | **95.9%** | 🟢 EXACT **95.9%** | Near-exact text match |
| 5 | Acacia Pharma Ltd | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 6 | Acacia pharma | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 7 | Acacia Pharma | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 8 | Acacia Pharma | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 9 | Acacia Pharma | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 10 | Acacia Pharma | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Acacia Pharma Group Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.8416 | 30% | 1.7525 |
| **Base Score** | - | - | **2.4525** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** acacia, group, inc., pharma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.842)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0433 (4.3%)

---

### Rank #2: Acacia Pharma

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 4.6050 | 30% | 1.3815 |
| **Base Score** | - | - | **2.0232** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** acacia, pharma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.917)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.605)

**Ranking Justification:**

- Ranked **#2** - score is 0.0433 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Acacia Pharma Ltd

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 4.4296 | 30% | 1.3289 |
| **Base Score** | - | - | **1.9705** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** acacia, pharma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.917)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.430)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: ACACIA PHARMA, Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0755 | 30% | 1.2226 |
| **Base Score** | - | - | **1.9226** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** acacia, inc.
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.075)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0034 higher than #5

---

### Rank #5: Acacia Pharma Ltd

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 5.7689 | 30% | 1.7307 |
| **Base Score** | - | - | **2.3723** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** acacia, pharma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.917)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.769)

**Ranking Justification:**

- Ranked **#5** - score is 0.0034 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 77. Query: `Acumatica Summit 2017 Z7NWPDKS625`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Acumatica Summit 2017 Z7NWPDKS625 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | 2017 MISMO Fall Summit | **65.2%** | 🟡 MEDIUM **65.2%** | Semantic/meaning-based match |
| 3 | Acumatica User Group Southeast | **57.1%** | 🟠 LOW **57.1%** | Semantic/meaning-based match |
| 4 | Acumatica Asia | **54.3%** | 🟠 LOW **54.3%** | Semantic/meaning-based match |
| 5 | Contact Center Compliance Summit | **54.1%** | 🟠 LOW **54.1%** | Semantic/meaning-based match |
| 6 | 2024 Lung Summit KQN5K6PH2MW | **53.6%** | 🟠 LOW **53.6%** | Semantic/meaning-based match |
| 7 | ACUMATICA PRESIDENTS CLUB | **53.5%** | 🟠 LOW **53.5%** | Semantic/meaning-based match |
| 8 | Acumatica The Cloud ERP | **53.0%** | 🟠 LOW **53.0%** | Semantic/meaning-based match |
| 9 | Acumatica Asia | **53.0%** | 🟠 LOW **53.0%** | Semantic/meaning-based match |
| 10 | Healthcare IT Connect Summit | **52.1%** | 🟠 LOW **52.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Acumatica Summit 2017 Z7NWPDKS625

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.4077 | 30% | 1.3223 |
| **Base Score** | - | - | **2.0223** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 2017, acumatica, summit, z7nwpdks625
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.408)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.3500 (35.0%)

---

### Rank #2: 2017 MISMO Fall Summit

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 2.2677 | 30% | 0.6803 |
| **Base Score** | - | - | **1.1761** |
| **Final Score** | - | - | **0.6524** (65.2%) |

**Why This Matched:**

- **Word Overlap:** 2017, summit
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.708)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.268)

**Ranking Justification:**

- Ranked **#2** - score is 0.3500 lower than #1
- Score is 0.0810 higher than #3

---

### Rank #3: Acumatica User Group Southeast

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4545 | 70% | 0.3182 |
| Semantic Similarity | 3.7104 | 30% | 1.1131 |
| **Base Score** | - | - | **1.4313** |
| **Final Score** | - | - | **0.5714** (57.1%) |

**Why This Matched:**

- **Word Overlap:** acumatica
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.455) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.710)

**Ranking Justification:**

- Ranked **#3** - score is 0.0810 lower than #2
- Score is 0.0284 higher than #4

---

### Rank #4: Acumatica Asia

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4167 | 70% | 0.2917 |
| Semantic Similarity | 3.6855 | 30% | 1.1056 |
| **Base Score** | - | - | **1.3973** |
| **Final Score** | - | - | **0.5430** (54.3%) |

**Why This Matched:**

- **Word Overlap:** acumatica
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.417) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.685)

**Ranking Justification:**

- Ranked **#4** - score is 0.0284 lower than #3
- Score is 0.0018 higher than #5

---

### Rank #5: Contact Center Compliance Summit

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 2.7922 | 30% | 0.8377 |
| **Base Score** | - | - | **1.1877** |
| **Final Score** | - | - | **0.5412** (54.1%) |

**Why This Matched:**

- **Word Overlap:** summit
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.500)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.792)

**Ranking Justification:**

- Ranked **#5** - score is 0.0018 lower than #4
- Score is 0.0052 higher than #6

---

</details>
## 78. Query: `Linklaters CIS`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Linklaters CIS | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Cis GmbH | **76.3%** | 🟡 MEDIUM **76.3%** | Semantic/meaning-based match |
| 3 | Cis 22 | **75.7%** | 🟡 MEDIUM **75.7%** | Semantic/meaning-based match |
| 4 | Cis Technologies | **74.5%** | 🟡 MEDIUM **74.5%** | Semantic/meaning-based match |
| 5 | CIS Conference | **73.5%** | 🟡 MEDIUM **73.5%** | Semantic/meaning-based match |
| 6 | STEELE CIS | **73.4%** | 🟡 MEDIUM **73.4%** | Semantic/meaning-based match |
| 7 | IEEE CIS | **71.7%** | 🟡 MEDIUM **71.7%** | Semantic/meaning-based match |
| 8 | CIS ASIA | **71.2%** | 🟡 MEDIUM **71.2%** | Semantic/meaning-based match |
| 9 | CIS Travel | **71.2%** | 🟡 MEDIUM **71.2%** | Semantic/meaning-based match |
| 10 | Deloitte CIS | **71.0%** | 🟡 MEDIUM **71.0%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Linklaters CIS

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.8690 | 30% | 1.7607 |
| **Base Score** | - | - | **2.4607** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** cis, linklaters
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.869)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2398 (24.0%)

---

### Rank #2: Cis GmbH

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.6453 | 30% | 1.3936 |
| **Base Score** | - | - | **1.9142** |
| **Final Score** | - | - | **0.7627** (76.3%) |

**Why This Matched:**

- **Word Overlap:** cis
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.645)

**Ranking Justification:**

- Ranked **#2** - score is 0.2398 lower than #1
- Score is 0.0057 higher than #3

---

### Rank #3: Cis 22

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5349 | 30% | 1.3605 |
| **Base Score** | - | - | **1.8811** |
| **Final Score** | - | - | **0.7570** (75.7%) |

**Why This Matched:**

- **Word Overlap:** cis
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.535)

**Ranking Justification:**

- Ranked **#3** - score is 0.0057 lower than #2
- Score is 0.0117 higher than #4

---

### Rank #4: Cis Technologies

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.3078 | 30% | 1.2923 |
| **Base Score** | - | - | **1.8130** |
| **Final Score** | - | - | **0.7453** (74.5%) |

**Why This Matched:**

- **Word Overlap:** cis
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.308)

**Ranking Justification:**

- Ranked **#4** - score is 0.0117 lower than #3
- Score is 0.0106 higher than #5

---

### Rank #5: CIS Conference

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.1023 | 30% | 1.2307 |
| **Base Score** | - | - | **1.7513** |
| **Final Score** | - | - | **0.7348** (73.5%) |

**Why This Matched:**

- **Word Overlap:** cis
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.102)

**Ranking Justification:**

- Ranked **#5** - score is 0.0106 lower than #4
- Score is 0.0009 higher than #6

---

</details>
## 79. Query: `Christian Girls Family Ministry`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Christian Girls Family Ministry | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Christian Girls Family Ministry Training | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | Christian Womens Ministry | **76.8%** | 🟡 MEDIUM **76.8%** | Semantic/meaning-based match |
| 4 | Kingdom Life Christian Ministry | **76.5%** | 🟡 MEDIUM **76.5%** | Semantic/meaning-based match |
| 5 | Christian Family Worship Center | **76.3%** | 🟡 MEDIUM **76.3%** | Semantic/meaning-based match |
| 6 | Christian Marriage Ministry | **76.0%** | 🟡 MEDIUM **76.0%** | Semantic/meaning-based match |
| 7 | Christian Community Ministry | **75.9%** | 🟡 MEDIUM **75.9%** | Semantic/meaning-based match |
| 8 | National Christian Ministry Association | **75.8%** | 🟡 MEDIUM **75.8%** | Semantic/meaning-based match |
| 9 | NEW FAMILY CHRISTIAN CHURCH | **75.8%** | 🟡 MEDIUM **75.8%** | Semantic/meaning-based match |
| 10 | Christian Family Home Educators | **75.2%** | 🟡 MEDIUM **75.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Christian Girls Family Ministry

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.8472 | 30% | 1.4542 |
| **Base Score** | - | - | **2.1542** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** christian, family, girls, ministry
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.847)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Christian Girls Family Ministry Training

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 5.2208 | 30% | 1.5662 |
| **Base Score** | - | - | **2.1390** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** christian, family, girls, ministry
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.221)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.1882 higher than #3

---

### Rank #3: Christian Womens Ministry

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 5.0405 | 30% | 1.5121 |
| **Base Score** | - | - | **1.9854** |
| **Final Score** | - | - | **0.7676** (76.8%) |

**Why This Matched:**

- **Word Overlap:** christian, ministry
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.040)

**Ranking Justification:**

- Ranked **#3** - score is 0.1882 lower than #2
- Score is 0.0027 higher than #4

---

### Rank #4: Kingdom Life Christian Ministry

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.1698 | 30% | 1.2510 |
| **Base Score** | - | - | **1.7716** |
| **Final Score** | - | - | **0.7649** (76.5%) |

**Why This Matched:**

- **Word Overlap:** christian, ministry
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.170)

**Ranking Justification:**

- Ranked **#4** - score is 0.0027 lower than #3
- Score is 0.0021 higher than #5

---

### Rank #5: Christian Family Worship Center

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.1343 | 30% | 1.2403 |
| **Base Score** | - | - | **1.7609** |
| **Final Score** | - | - | **0.7628** (76.3%) |

**Why This Matched:**

- **Word Overlap:** christian, family
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.134)

**Ranking Justification:**

- Ranked **#5** - score is 0.0021 lower than #4
- Score is 0.0023 higher than #6

---

</details>
## 80. Query: `Alosa Foundation`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | ALOSA FOUNDATION | **100.6%** | 🟢 EXACT **100.6%** | Near-exact text match |
| 2 | Alosa Foundation | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Foundation Workshop | **66.1%** | 🟡 MEDIUM **66.1%** | Semantic/meaning-based match |
| 4 | Alwan Foundation | **66.0%** | 🟡 MEDIUM **66.0%** | Semantic/meaning-based match |
| 5 | NEST Foundation | **65.8%** | 🟡 MEDIUM **65.8%** | Semantic/meaning-based match |
| 6 | Llosa's foundation | **65.8%** | 🟡 MEDIUM **65.8%** | Semantic/meaning-based match |
| 7 | Foundation Building | **65.7%** | 🟡 MEDIUM **65.7%** | Semantic/meaning-based match |
| 8 | SEE Foundation | **65.7%** | 🟡 MEDIUM **65.7%** | Semantic/meaning-based match |
| 9 | Ivy Foundation | **65.5%** | 🟡 MEDIUM **65.5%** | Semantic/meaning-based match |
| 10 | Help Foundation | **65.3%** | 🟡 MEDIUM **65.3%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: ALOSA FOUNDATION

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3286 | 30% | 1.2986 |
| **Base Score** | - | - | **1.9986** |
| **Final Score** | - | - | **1.0057** (100.6%) |

**Why This Matched:**

- **Word Overlap:** alosa, foundation
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.329)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0057)
- Score gap to #2: 0.0032 (0.3%)

---

### Rank #2: Alosa Foundation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.0562 | 30% | 1.8169 |
| **Base Score** | - | - | **2.5169** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** alosa, foundation
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.056)

**Ranking Justification:**

- Ranked **#2** - score is 0.0032 lower than #1
- Score is 0.3416 higher than #3

---

### Rank #3: Foundation Workshop

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6375 | 70% | 0.4462 |
| Semantic Similarity | 4.2512 | 30% | 1.2754 |
| **Base Score** | - | - | **1.7216** |
| **Final Score** | - | - | **0.6608** (66.1%) |

**Why This Matched:**

- **Word Overlap:** foundation
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.637)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.251)

**Ranking Justification:**

- Ranked **#3** - score is 0.3416 lower than #2
- Score is 0.0009 higher than #4

---

### Rank #4: Alwan Foundation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6375 | 70% | 0.4462 |
| Semantic Similarity | 4.2324 | 30% | 1.2697 |
| **Base Score** | - | - | **1.7160** |
| **Final Score** | - | - | **0.6599** (66.0%) |

**Why This Matched:**

- **Word Overlap:** foundation
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.637)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.232)

**Ranking Justification:**

- Ranked **#4** - score is 0.0009 lower than #3
- Score is 0.0021 higher than #5

---

### Rank #5: NEST Foundation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6375 | 70% | 0.4462 |
| Semantic Similarity | 4.1903 | 30% | 1.2571 |
| **Base Score** | - | - | **1.7033** |
| **Final Score** | - | - | **0.6578** (65.8%) |

**Why This Matched:**

- **Word Overlap:** foundation
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.637)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.190)

**Ranking Justification:**

- Ranked **#5** - score is 0.0021 lower than #4
- Score is 0.0001 higher than #6

---

</details>
## 81. Query: `La Chaine des Rotisseurs Wine Club of Newport Beach`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | La Chaine des Rotisseurs Wine Club of Newport Beach | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | La Chaine des Rotisseurs Bailliage de Newport Beach | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | La Chaine des Rotisseurs Bailliage de Newport Beach | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | La Chaine des Rotisseurs Wine Club of Ne | **81.3%** | 🟢 HIGH **81.3%** | Semantic/meaning-based match |
| 5 | La Confrerie de la Chaine des Rotisseurs Region | **69.7%** | 🟡 MEDIUM **69.7%** | Semantic/meaning-based match |
| 6 | La Confrerie de la Chaine des Rotisseurs Region | **67.9%** | 🟡 MEDIUM **67.9%** | Semantic/meaning-based match |
| 7 | Confrerie de la Chaine des Rotisseurs | **64.8%** | 🟡 MEDIUM **64.8%** | Semantic/meaning-based match |
| 8 | Confrerie Del La Chaine des Rotisseurs | **64.1%** | 🟡 MEDIUM **64.1%** | Semantic/meaning-based match |
| 9 | La Chaine Des Rotisseurs Hillsborough Chapter | **64.0%** | 🟡 MEDIUM **64.0%** | Semantic/meaning-based match |
| 10 | Confrerie des la Chaine des Rotisseurs | **60.7%** | 🟡 MEDIUM **60.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: La Chaine des Rotisseurs Wine Club of Newport Beach

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.5622 | 30% | 1.3687 |
| **Base Score** | - | - | **2.0687** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** beach, chaine, club, des, la, newport, of, rotisseurs, wine
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.562)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: La Chaine des Rotisseurs Bailliage de Newport Beach

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8357 | 70% | 0.5850 |
| Semantic Similarity | 3.5137 | 30% | 1.0541 |
| **Base Score** | - | - | **1.6391** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** beach, chaine, des, la, newport, rotisseurs
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.836)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.514)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: La Chaine des Rotisseurs Bailliage de Newport Beach

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8357 | 70% | 0.5850 |
| Semantic Similarity | 3.1226 | 30% | 0.9368 |
| **Base Score** | - | - | **1.5218** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** beach, chaine, des, la, newport, rotisseurs
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.836)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.123)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0928 higher than #4

---

### Rank #4: La Chaine des Rotisseurs Wine Club of Ne

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7597 | 70% | 0.5318 |
| Semantic Similarity | 4.1964 | 30% | 1.2589 |
| **Base Score** | - | - | **1.7907** |
| **Final Score** | - | - | **0.8127** (81.3%) |

**Why This Matched:**

- **Word Overlap:** chaine, club, des, la, of, rotisseurs, wine
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.760)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.196)

**Ranking Justification:**

- Ranked **#4** - score is 0.0928 lower than #3
- Score is 0.1158 higher than #5

---

### Rank #5: La Confrerie de la Chaine des Rotisseurs Region

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6623 | 70% | 0.4636 |
| Semantic Similarity | 3.4827 | 30% | 1.0448 |
| **Base Score** | - | - | **1.5084** |
| **Final Score** | - | - | **0.6969** (69.7%) |

**Why This Matched:**

- **Word Overlap:** chaine, des, la, rotisseurs
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.662)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.483)

**Ranking Justification:**

- Ranked **#5** - score is 0.1158 lower than #4
- Score is 0.0174 higher than #6

---

</details>
## 82. Query: `Sumner & Ryan, LLC`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Sumner & Ryan, LLC | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Ryan McCall | **79.7%** | 🟡 MEDIUM **79.7%** | Semantic/meaning-based match |
| 3 | Sumner 360 | **79.2%** | 🟡 MEDIUM **79.2%** | Semantic/meaning-based match |
| 4 | Ryan Consulting | **78.1%** | 🟡 MEDIUM **78.1%** | Semantic/meaning-based match |
| 5 | Ryan Mitchell Associates, LLC | **78.1%** | 🟡 MEDIUM **78.1%** | Semantic/meaning-based match |
| 6 | Brianna Sumner | **77.9%** | 🟡 MEDIUM **77.9%** | Semantic/meaning-based match |
| 7 | Sumner Baseball | **77.1%** | 🟡 MEDIUM **77.1%** | Semantic/meaning-based match |
| 8 | Ryan Henderson | **76.7%** | 🟡 MEDIUM **76.7%** | Semantic/meaning-based match |
| 9 | Ryan Prospects | **76.7%** | 🟡 MEDIUM **76.7%** | Semantic/meaning-based match |
| 10 | Ryan Staker | **76.6%** | 🟡 MEDIUM **76.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Sumner & Ryan, LLC

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.4430 | 30% | 1.0329 |
| **Base Score** | - | - | **1.7329** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** &, llc, ryan,, sumner
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.443)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2058 (20.6%)

---

### Rank #2: Ryan McCall

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.8435 | 30% | 1.1531 |
| **Base Score** | - | - | **1.6737** |
| **Final Score** | - | - | **0.7967** (79.7%) |

**Why This Matched:**

- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.844)

**Ranking Justification:**

- Ranked **#2** - score is 0.2058 lower than #1
- Score is 0.0048 higher than #3

---

### Rank #3: Sumner 360

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.7762 | 30% | 1.1329 |
| **Base Score** | - | - | **1.6535** |
| **Final Score** | - | - | **0.7919** (79.2%) |

**Why This Matched:**

- **Word Overlap:** sumner
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.776)

**Ranking Justification:**

- Ranked **#3** - score is 0.0048 lower than #2
- Score is 0.0107 higher than #4

---

### Rank #4: Ryan Consulting

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6252 | 30% | 1.0876 |
| **Base Score** | - | - | **1.6082** |
| **Final Score** | - | - | **0.7812** (78.1%) |

**Why This Matched:**

- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.625)

**Ranking Justification:**

- Ranked **#4** - score is 0.0107 lower than #3
- Score is 0.0005 higher than #5

---

### Rank #5: Ryan Mitchell Associates, LLC

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6188 | 30% | 1.0856 |
| **Base Score** | - | - | **1.6063** |
| **Final Score** | - | - | **0.7807** (78.1%) |

**Why This Matched:**

- **Word Overlap:** llc
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.619)

**Ranking Justification:**

- Ranked **#5** - score is 0.0005 lower than #4
- Score is 0.0015 higher than #6

---

</details>
## 83. Query: `Tilt Creative & Production`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Tilt Creative & Production | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 2 | Tilt Creative + Production | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | Creative Production Incentives | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Tilt Production | **74.4%** | 🟡 MEDIUM **74.4%** | Semantic/meaning-based match |
| 5 | Absolute Creative Design & Production | **71.5%** | 🟡 MEDIUM **71.5%** | Semantic/meaning-based match |
| 6 | Bam Creative Production Pte Ltd | **71.0%** | 🟡 MEDIUM **71.0%** | Semantic/meaning-based match |
| 7 | Shiloh Creative Production Studios Inc | **70.8%** | 🟡 MEDIUM **70.8%** | Semantic/meaning-based match |
| 8 | Upper Room Creative Production | **70.8%** | 🟡 MEDIUM **70.8%** | Semantic/meaning-based match |
| 9 | Full Tilt Marketing | **64.2%** | 🟡 MEDIUM **64.2%** | Semantic/meaning-based match |
| 10 | CREATIVE THINKING PRODUCTIONS | **62.5%** | 🟡 MEDIUM **62.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Tilt Creative & Production

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0351 | 30% | 1.2105 |
| **Base Score** | - | - | **1.9105** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** &, creative, production, tilt
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.035)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0049)
- Score gap to #2: 0.0491 (4.9%)

---

### Rank #2: Tilt Creative + Production

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8727 | 70% | 0.6109 |
| Semantic Similarity | 3.7950 | 30% | 1.1385 |
| **Base Score** | - | - | **1.7494** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** creative, production, tilt
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.873)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.795)

**Ranking Justification:**

- Ranked **#2** - score is 0.0491 lower than #1
- Score is 0.0503 higher than #3

---

### Rank #3: Creative Production Incentives

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 3.9419 | 30% | 1.1826 |
| **Base Score** | - | - | **1.7494** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** creative, production
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.810)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.942)

**Ranking Justification:**

- Ranked **#3** - score is 0.0503 lower than #2
- Score is 0.1615 higher than #4

---

### Rank #4: Tilt Production

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7361 | 70% | 0.5153 |
| Semantic Similarity | 3.2630 | 30% | 0.9789 |
| **Base Score** | - | - | **1.4942** |
| **Final Score** | - | - | **0.7440** (74.4%) |

**Why This Matched:**

- **Word Overlap:** production, tilt
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.736)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.263)

**Ranking Justification:**

- Ranked **#4** - score is 0.1615 lower than #3
- Score is 0.0289 higher than #5

---

### Rank #5: Absolute Creative Design & Production

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7361 | 70% | 0.5153 |
| Semantic Similarity | 2.8790 | 30% | 0.8637 |
| **Base Score** | - | - | **1.3790** |
| **Final Score** | - | - | **0.7151** (71.5%) |

**Why This Matched:**

- **Word Overlap:** &, creative, production
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.736)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.879)

**Ranking Justification:**

- Ranked **#5** - score is 0.0289 lower than #4
- Score is 0.0046 higher than #6

---

</details>
## 84. Query: `Cerberus Capital`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Cerberus Capital | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Cerberus Capital | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | YP-Cerberus Capital Mgmt | **96.2%** | 🟢 EXACT **96.2%** | Semantic/meaning-based match |
| 4 | Cerberus Capital Management LP | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | Cerberus Capital Management | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 6 | Cerberus Capital Management L | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | YP-Cerberus Capital Mgmt | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 8 | CEC Capital | **74.8%** | 🟡 MEDIUM **74.8%** | Semantic/meaning-based match |
| 9 | *Cerberus Capital | **74.0%** | 🟡 MEDIUM **74.0%** | Semantic/meaning-based match |
| 10 | Cerberus | **71.6%** | 🟡 MEDIUM **71.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Cerberus Capital

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.8753 | 30% | 1.7626 |
| **Base Score** | - | - | **2.4626** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** capital, cerberus
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.875)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: Cerberus Capital

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.1385 | 30% | 1.2416 |
| **Base Score** | - | - | **1.9416** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** capital, cerberus
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.139)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0409 higher than #3

---

### Rank #3: YP-Cerberus Capital Mgmt

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 2.7256 | 30% | 0.8177 |
| **Base Score** | - | - | **1.3427** |
| **Final Score** | - | - | **0.9616** (96.2%) |

**Why This Matched:**

- **Word Overlap:** capital
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.726)

**Ranking Justification:**

- Ranked **#3** - score is 0.0409 lower than #2
- Score is 0.0058 higher than #4

---

### Rank #4: Cerberus Capital Management LP

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 4.7774 | 30% | 1.4332 |
| **Base Score** | - | - | **1.9582** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** capital, cerberus
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.777)

**Ranking Justification:**

- Ranked **#4** - score is 0.0058 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Cerberus Capital Management

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 4.0115 | 30% | 1.2034 |
| **Base Score** | - | - | **1.7762** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** capital, cerberus
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.011)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 85. Query: `Institute of Health Technology Transformation`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Institute of Health Technology Transformation | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | Institute for Health Technology Transformation | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 4 | Technology Health Experience | **77.8%** | 🟡 MEDIUM **77.8%** | Semantic/meaning-based match |
| 5 | Health Technology Assessment International | **77.7%** | 🟡 MEDIUM **77.7%** | Semantic/meaning-based match |
| 6 | HEALTH TECHNOLOGY ASSESSMENT INT | **77.0%** | 🟡 MEDIUM **77.0%** | Semantic/meaning-based match |
| 7 | Health Technology Association | **76.5%** | 🟡 MEDIUM **76.5%** | Semantic/meaning-based match |
| 8 | Health Technology Assesment International | **76.2%** | 🟡 MEDIUM **76.2%** | Semantic/meaning-based match |
| 9 | Health Technology Assessment | **76.0%** | 🟡 MEDIUM **76.0%** | Semantic/meaning-based match |
| 10 | Health Technology Exchange | **75.7%** | 🟡 MEDIUM **75.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Institute of Health Technology Transformation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.5446 | 30% | 1.0634 |
| **Base Score** | - | - | **1.7634** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** health, institute, of, technology, transformation
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.545)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: INSTITUTE FOR HEALTH & TECHNOLOGY TRANSFORMATION

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8678 | 70% | 0.6074 |
| Semantic Similarity | 4.7241 | 30% | 1.4172 |
| **Base Score** | - | - | **2.0247** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** health, institute, technology, transformation
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.868)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.724)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Institute for Health Technology Transformation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8678 | 70% | 0.6074 |
| Semantic Similarity | 3.5356 | 30% | 1.0607 |
| **Base Score** | - | - | **1.6681** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** health, institute, technology, transformation
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.868)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.536)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.1778 higher than #4

---

### Rank #4: Technology Health Experience

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 5.0444 | 30% | 1.5133 |
| **Base Score** | - | - | **1.9866** |
| **Final Score** | - | - | **0.7780** (77.8%) |

**Why This Matched:**

- **Word Overlap:** health, technology
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.044)

**Ranking Justification:**

- Ranked **#4** - score is 0.1778 lower than #3
- Score is 0.0013 higher than #5

---

### Rank #5: Health Technology Assessment International

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.1812 | 30% | 1.2544 |
| **Base Score** | - | - | **1.7750** |
| **Final Score** | - | - | **0.7767** (77.7%) |

**Why This Matched:**

- **Word Overlap:** health, technology
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.181)

**Ranking Justification:**

- Ranked **#5** - score is 0.0013 lower than #4
- Score is 0.0070 higher than #6

---

</details>
## 86. Query: `The Jones Assembly`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | The Jones Assembly | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | The Jones Assembly Presents | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 3 | General Assembly | **74.8%** | 🟡 MEDIUM **74.8%** | Semantic/meaning-based match |
| 4 | First Assembly | **74.3%** | 🟡 MEDIUM **74.3%** | Semantic/meaning-based match |
| 5 | Jones Ag | **74.1%** | 🟡 MEDIUM **74.1%** | Semantic/meaning-based match |
| 6 | Jones Companies | **73.9%** | 🟡 MEDIUM **73.9%** | Semantic/meaning-based match |
| 7 | Jones Capital | **73.4%** | 🟡 MEDIUM **73.4%** | Semantic/meaning-based match |
| 8 | State Assembly | **73.3%** | 🟡 MEDIUM **73.3%** | Semantic/meaning-based match |
| 9 | MC Assembly | **73.2%** | 🟡 MEDIUM **73.2%** | Semantic/meaning-based match |
| 10 | John Jones | **72.8%** | 🟡 MEDIUM **72.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: The Jones Assembly

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.2758 | 30% | 1.8827 |
| **Base Score** | - | - | **2.5827** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** assembly, jones, the
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.276)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: The Jones Assembly Presents

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 3.5947 | 30% | 1.0784 |
| **Base Score** | - | - | **1.6511** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** assembly, jones, the
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.818)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.595)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.2077 higher than #3

---

### Rank #3: General Assembly

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.6631 | 30% | 1.3989 |
| **Base Score** | - | - | **1.9195** |
| **Final Score** | - | - | **0.7481** (74.8%) |

**Why This Matched:**

- **Word Overlap:** assembly
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.663)

**Ranking Justification:**

- Ranked **#3** - score is 0.2077 lower than #2
- Score is 0.0052 higher than #4

---

### Rank #4: First Assembly

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5544 | 30% | 1.3663 |
| **Base Score** | - | - | **1.8870** |
| **Final Score** | - | - | **0.7428** (74.3%) |

**Why This Matched:**

- **Word Overlap:** assembly
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.554)

**Ranking Justification:**

- Ranked **#4** - score is 0.0052 lower than #3
- Score is 0.0022 higher than #5

---

### Rank #5: Jones Ag

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5079 | 30% | 1.3524 |
| **Base Score** | - | - | **1.8730** |
| **Final Score** | - | - | **0.7406** (74.1%) |

**Why This Matched:**

- **Word Overlap:** jones
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.508)

**Ranking Justification:**

- Ranked **#5** - score is 0.0022 lower than #4
- Score is 0.0018 higher than #6

---

</details>
## 87. Query: `American Black Film Insitutute`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | American Black Film Insitutute | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | American Black Film Festival | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | International Black Film Festival | **76.8%** | 🟡 MEDIUM **76.8%** | Semantic/meaning-based match |
| 4 | Black Women Film Preservation | **76.4%** | 🟡 MEDIUM **76.4%** | Semantic/meaning-based match |
| 5 | Black Film Initiative | **76.3%** | 🟡 MEDIUM **76.3%** | Semantic/meaning-based match |
| 6 | American Black Film Festival Ventures | **75.2%** | 🟡 MEDIUM **75.2%** | Semantic/meaning-based match |
| 7 | Black Star Film Festival | **74.8%** | 🟡 MEDIUM **74.8%** | Semantic/meaning-based match |
| 8 | American Film Works | **73.1%** | 🟡 MEDIUM **73.1%** | Semantic/meaning-based match |
| 9 | Black Women Film Network | **72.9%** | 🟡 MEDIUM **72.9%** | Semantic/meaning-based match |
| 10 | American Black Film Festival Honors | **72.8%** | 🟡 MEDIUM **72.8%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: American Black Film Insitutute

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.2302 | 30% | 1.2691 |
| **Base Score** | - | - | **1.9691** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** american, black, film, insitutute
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.230)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: American Black Film Festival

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 3.2801 | 30% | 0.9840 |
| **Base Score** | - | - | **1.5747** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** american, black, film
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.280)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.1370 higher than #3

---

### Rank #3: International Black Film Festival

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.6193 | 30% | 1.0858 |
| **Base Score** | - | - | **1.6064** |
| **Final Score** | - | - | **0.7685** (76.8%) |

**Why This Matched:**

- **Word Overlap:** black, film
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.619)

**Ranking Justification:**

- Ranked **#3** - score is 0.1370 lower than #2
- Score is 0.0045 higher than #4

---

### Rank #4: Black Women Film Preservation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.5524 | 30% | 1.0657 |
| **Base Score** | - | - | **1.5863** |
| **Final Score** | - | - | **0.7639** (76.4%) |

**Why This Matched:**

- **Word Overlap:** black, film
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.552)

**Ranking Justification:**

- Ranked **#4** - score is 0.0045 lower than #3
- Score is 0.0013 higher than #5

---

### Rank #5: Black Film Initiative

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 4.2373 | 30% | 1.2712 |
| **Base Score** | - | - | **1.7445** |
| **Final Score** | - | - | **0.7626** (76.3%) |

**Why This Matched:**

- **Word Overlap:** black, film
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.237)

**Ranking Justification:**

- Ranked **#5** - score is 0.0013 lower than #4
- Score is 0.0109 higher than #6

---

</details>
## 88. Query: `Berk Tek`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Berk Tek | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Berk Tek | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | Berk Tek | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | Berk Tek | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | Berk-Tek | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 6 | Berk Tek / Leviton | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 7 | Berk-Tek,a Nexans Company | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Berk Tck | **75.0%** | 🟡 MEDIUM **75.0%** | Semantic/meaning-based match |
| 9 | Berk Technologies | **73.7%** | 🟡 MEDIUM **73.7%** | Semantic/meaning-based match |
| 10 | TEK Source | **73.7%** | 🟡 MEDIUM **73.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Berk Tek

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.7643 | 30% | 1.7293 |
| **Base Score** | - | - | **2.4293** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** berk, tek
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.764)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: Berk Tek

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7208 | 30% | 1.1162 |
| **Base Score** | - | - | **1.8162** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** berk, tek
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.721)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Berk Tek

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.5091 | 30% | 1.0527 |
| **Base Score** | - | - | **1.7527** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** berk, tek
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.509)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Berk Tek

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.3621 | 30% | 1.0086 |
| **Base Score** | - | - | **1.7086** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** berk, tek
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.362)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0467 higher than #5

---

### Rank #5: Berk-Tek

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity | 3.0571 | 30% | 0.9171 |
| **Base Score** | - | - | **1.5471** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Lexical Match:** Nearly identical text (Jaro-Winkler: 0.900)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.057)

**Ranking Justification:**

- Ranked **#5** - score is 0.0467 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 89. Query: `Northbridge Travel`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Northbridge Travel | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | Bridge Travel | **78.7%** | 🟡 MEDIUM **78.7%** | Semantic/meaning-based match |
| 3 | Skybridge Travel | **77.8%** | 🟡 MEDIUM **77.8%** | Semantic/meaning-based match |
| 4 | Northbridge Insurance | **77.1%** | 🟡 MEDIUM **77.1%** | Semantic/meaning-based match |
| 5 | KingsBridge Travel | **76.9%** | 🟡 MEDIUM **76.9%** | Semantic/meaning-based match |
| 6 | Travel Bridge | **75.9%** | 🟡 MEDIUM **75.9%** | Semantic/meaning-based match |
| 7 | Northbridge | **74.6%** | 🟡 MEDIUM **74.6%** | Semantic/meaning-based match |
| 8 | Northbridge Financial Corp | **74.0%** | 🟡 MEDIUM **74.0%** | Semantic/meaning-based match |
| 9 | Northbridge Insurance | **73.8%** | 🟡 MEDIUM **73.8%** | Semantic/meaning-based match |
| 10 | Northbridge Environmental | **73.7%** | 🟡 MEDIUM **73.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Northbridge Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.9155 | 30% | 1.7747 |
| **Base Score** | - | - | **2.4747** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** northbridge, travel
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.916)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.2165 (21.7%)

---

### Rank #2: Bridge Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7875 | 70% | 0.5512 |
| Semantic Similarity | 5.0702 | 30% | 1.5210 |
| **Base Score** | - | - | **2.0723** |
| **Final Score** | - | - | **0.7873** (78.7%) |

**Why This Matched:**

- **Word Overlap:** travel
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.787)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.070)

**Ranking Justification:**

- Ranked **#2** - score is 0.2165 lower than #1
- Score is 0.0096 higher than #3

---

### Rank #3: Skybridge Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.5322 | 30% | 1.6596 |
| **Base Score** | - | - | **2.1803** |
| **Final Score** | - | - | **0.7777** (77.8%) |

**Why This Matched:**

- **Word Overlap:** travel
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.532)

**Ranking Justification:**

- Ranked **#3** - score is 0.0096 lower than #2
- Score is 0.0066 higher than #4

---

### Rank #4: Northbridge Insurance

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.3887 | 30% | 1.6166 |
| **Base Score** | - | - | **2.1372** |
| **Final Score** | - | - | **0.7712** (77.1%) |

**Why This Matched:**

- **Word Overlap:** northbridge
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.389)

**Ranking Justification:**

- Ranked **#4** - score is 0.0066 lower than #3
- Score is 0.0018 higher than #5

---

### Rank #5: KingsBridge Travel

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 5.3500 | 30% | 1.6050 |
| **Base Score** | - | - | **2.1256** |
| **Final Score** | - | - | **0.7694** (76.9%) |

**Why This Matched:**

- **Word Overlap:** travel
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.350)

**Ranking Justification:**

- Ranked **#5** - score is 0.0018 lower than #4
- Score is 0.0100 higher than #6

---

</details>
## 90. Query: `Kohler 2024`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Kohler 2024 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Kohler Company Sales | **92.1%** | 🟢 HIGH **92.1%** | Semantic/meaning-based match |
| 3 | Kohler Company Marketing | **91.9%** | 🟢 HIGH **91.9%** | Semantic/meaning-based match |
| 4 | Kohler Communications | **91.7%** | 🟢 HIGH **91.7%** | Semantic/meaning-based match |
| 5 | Kohler Engines | **91.7%** | 🟢 HIGH **91.7%** | Semantic/meaning-based match |
| 6 | Kohler Company Finance | **91.5%** | 🟢 HIGH **91.5%** | Semantic/meaning-based match |
| 7 | Kohler Schools | **91.5%** | 🟢 HIGH **91.5%** | Semantic/meaning-based match |
| 8 | Kohler Energy | **91.4%** | 🟢 HIGH **91.4%** | Semantic/meaning-based match |
| 9 | Kohler Company Accounting | **91.4%** | 🟢 HIGH **91.4%** | Semantic/meaning-based match |
| 10 | Kohler Fixtures | **91.3%** | 🟢 HIGH **91.3%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Kohler 2024

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.8853 | 30% | 1.1656 |
| **Base Score** | - | - | **1.8656** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 2024, kohler
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.885)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0816 (8.2%)

---

### Rank #2: Kohler Company Sales

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.9060 | 30% | 1.1718 |
| **Base Score** | - | - | **1.7668** |
| **Final Score** | - | - | **0.9208** (92.1%) |

**Why This Matched:**

- **Word Overlap:** kohler
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.906)

**Ranking Justification:**

- Ranked **#2** - score is 0.0816 lower than #1
- Score is 0.0019 higher than #3

---

### Rank #3: Kohler Company Marketing

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.8350 | 30% | 1.1505 |
| **Base Score** | - | - | **1.7455** |
| **Final Score** | - | - | **0.9189** (91.9%) |

**Why This Matched:**

- **Word Overlap:** kohler
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.835)

**Ranking Justification:**

- Ranked **#3** - score is 0.0019 lower than #2
- Score is 0.0016 higher than #4

---

### Rank #4: Kohler Communications

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.9640 | 30% | 1.1892 |
| **Base Score** | - | - | **1.7842** |
| **Final Score** | - | - | **0.9174** (91.7%) |

**Why This Matched:**

- **Word Overlap:** kohler
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.964)

**Ranking Justification:**

- Ranked **#4** - score is 0.0016 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Kohler Engines

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 3.8351 | 30% | 1.1505 |
| **Base Score** | - | - | **1.7455** |
| **Final Score** | - | - | **0.9174** (91.7%) |

**Why This Matched:**

- **Word Overlap:** kohler
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.850)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.835)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0020 higher than #6

---

</details>
## 91. Query: `Louisiana State University Swim`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Louisiana State University Swim | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Louisiana State University System | **92.2%** | 🟢 HIGH **92.2%** | Semantic/meaning-based match |
| 3 | Louisiana State University Foundation | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 4 | Louisiana State University System | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 5 | Louisiana State University CCT | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 6 | Louisiana State University Foundation | **90.9%** | 🟢 HIGH **90.9%** | Semantic/meaning-based match |
| 7 | Louisiana State University USA | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Louisiana State University Health | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Louisiana State Univ Swim | **90.5%** | 🟢 HIGH **90.5%** | Near-exact text match |
| 10 | Louisiana State University System | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Louisiana State University Swim

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 6.4700 | 30% | 1.9410 |
| **Base Score** | - | - | **2.6410** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** louisiana, state, swim, university
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 6.470)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0801 (8.0%)

---

### Rank #2: Louisiana State University System

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.7722 | 30% | 1.1317 |
| **Base Score** | - | - | **1.7092** |
| **Final Score** | - | - | **0.9224** (92.2%) |

**Why This Matched:**

- **Word Overlap:** louisiana, state, university
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.772)

**Ranking Justification:**

- Ranked **#2** - score is 0.0801 lower than #1
- Score is 0.0137 higher than #3

---

### Rank #3: Louisiana State University Foundation

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.5609 | 30% | 1.0683 |
| **Base Score** | - | - | **1.6458** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** louisiana, state, university
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.561)

**Ranking Justification:**

- Ranked **#3** - score is 0.0137 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: Louisiana State University System

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.5568 | 30% | 1.0671 |
| **Base Score** | - | - | **1.6446** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** louisiana, state, university
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.557)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Louisiana State University CCT

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.4620 | 30% | 1.0386 |
| **Base Score** | - | - | **1.6161** |
| **Final Score** | - | - | **0.9087** (90.9%) |

**Why This Matched:**

- **Word Overlap:** louisiana, state, university
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.462)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 92. Query: `X DO NOT USE - FRANCIS PARKER SCHOOL`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | X DO NOT USE - FRANCIS PARKER SCHOOL | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | X DO NOT USE - FRANCIS PARKER SCHOOL | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | X DO NOT USE - FRANCIS PARKER SCHOOL | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | X DO NOT USE - FRANCIS PARKER SCHOOL | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 5 | X DO NOT USE - FRANCIS PARKER SCHOOL | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 6 | X DO NOT USE - FRANCIS PARKER SCHOOL | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 7 | X DO NOT USE - FRANCIS PARKER SCHOOL | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 8 | X DO NOT USE - CLAIREMONT HIGH SCHOOL | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Duke Nicholas School of the Environment - Do Not Use | **68.8%** | 🟡 MEDIUM **68.8%** | Semantic/meaning-based match |
| 10 | zAmerican Association of School Personnel- do not use | **68.1%** | 🟡 MEDIUM **68.1%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: X DO NOT USE - FRANCIS PARKER SCHOOL

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9538 | 30% | 1.1861 |
| **Base Score** | - | - | **1.8861** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** -, do, francis, not, parker, school, use, x
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.954)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0014 (0.1%)

---

### Rank #2: X DO NOT USE - FRANCIS PARKER SCHOOL

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.8480 | 30% | 1.4544 |
| **Base Score** | - | - | **2.1544** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** -, do, francis, not, parker, school, use, x
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.848)

**Ranking Justification:**

- Ranked **#2** - score is 0.0014 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: X DO NOT USE - FRANCIS PARKER SCHOOL

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0197 | 30% | 1.2059 |
| **Base Score** | - | - | **1.9059** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** -, do, francis, not, parker, school, use, x
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.020)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: X DO NOT USE - FRANCIS PARKER SCHOOL

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9476 | 30% | 1.1843 |
| **Base Score** | - | - | **1.8843** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** -, do, francis, not, parker, school, use, x
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.948)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: X DO NOT USE - FRANCIS PARKER SCHOOL

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.8314 | 30% | 1.1494 |
| **Base Score** | - | - | **1.8494** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** -, do, francis, not, parker, school, use, x
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.831)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 93. Query: `Mitsubishi Motor Sales of America, Incorporated`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Mitsubishi Motor Sales of America, Incorporated | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Mitsubishi Motor Sales of America | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 3 | Mitsubishi Motor Sales of America, Inc. | **95.6%** | 🟢 EXACT **95.6%** | Near-exact text match |
| 4 | Mitsubishi Electronic Sales America | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 5 | Mitsubishi Motors Sales of America | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 6 | Mitsubishi Electric Sales of America | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 7 | Mitsubishi Motor Sales Of Amer | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 8 | Mitsubishi Motor Sales of Canada, Incorporated | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 9 | Mitsubishi Motor Sales of Caribbean | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 10 | MITSUBISHI MOTOR NORTH AMERICA, INC | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Mitsubishi Motor Sales of America, Incorporated

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3971 | 30% | 1.3191 |
| **Base Score** | - | - | **2.0191** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** america,, incorporated, mitsubishi, motor, of, sales
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.397)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0467 (4.7%)

---

### Rank #2: Mitsubishi Motor Sales of America

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.9950 | 30% | 1.4985 |
| **Base Score** | - | - | **2.1985** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** mitsubishi, motor, of, sales
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.995)

**Ranking Justification:**

- Ranked **#2** - score is 0.0467 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Mitsubishi Motor Sales of America, Inc.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0428 | 30% | 1.2128 |
| **Base Score** | - | - | **1.9128** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** america,, mitsubishi, motor, of, sales
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.043)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0503 higher than #4

---

### Rank #4: Mitsubishi Electronic Sales America

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 5.5413 | 30% | 1.6624 |
| **Base Score** | - | - | **2.2530** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** mitsubishi, sales
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.541)

**Ranking Justification:**

- Ranked **#4** - score is 0.0503 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: Mitsubishi Motors Sales of America

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity | 4.7459 | 30% | 1.4238 |
| **Base Score** | - | - | **2.0144** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** mitsubishi, of, sales
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.844)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.746)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
## 94. Query: `Energy Distribution Partners Holdings'`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Energy Distribution Partners Holdings' | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Energy Distribution Partners (EDP) | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 3 | Energy Distribution Partners Holdings L.P. | **90.5%** | 🟢 HIGH **90.5%** | Semantic/meaning-based match |
| 4 | Energy Distribution Holdings | **78.6%** | 🟡 MEDIUM **78.6%** | Semantic/meaning-based match |
| 5 | Energy Distribution Partners | **75.6%** | 🟡 MEDIUM **75.6%** | Semantic/meaning-based match |
| 6 | Energy Transfer Partners LP | **73.1%** | 🟡 MEDIUM **73.1%** | Semantic/meaning-based match |
| 7 | Energy Power Partners | **72.1%** | 🟡 MEDIUM **72.1%** | Semantic/meaning-based match |
| 8 | EIG Global Energy Partners | **72.0%** | 🟡 MEDIUM **72.0%** | Semantic/meaning-based match |
| 9 | Energy Impact Partners | **71.0%** | 🟡 MEDIUM **71.0%** | Semantic/meaning-based match |
| 10 | Energy Transfer Partners LP | **70.7%** | 🟡 MEDIUM **70.7%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Energy Distribution Partners Holdings'

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.6703 | 30% | 1.7011 |
| **Base Score** | - | - | **2.4011** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** distribution, energy, holdings', partners
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.670)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0970 (9.7%)

---

### Rank #2: Energy Distribution Partners (EDP)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 4.1316 | 30% | 1.2395 |
| **Base Score** | - | - | **1.8170** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** distribution, energy, partners
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.132)

**Ranking Justification:**

- Ranked **#2** - score is 0.0970 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: Energy Distribution Partners Holdings L.P.

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 3.7996 | 30% | 1.1399 |
| **Base Score** | - | - | **1.7174** |
| **Final Score** | - | - | **0.9055** (90.5%) |

**Why This Matched:**

- **Word Overlap:** distribution, energy, partners
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.825)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.800)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.1195 higher than #4

---

### Rank #4: Energy Distribution Holdings

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6875 | 70% | 0.4812 |
| Semantic Similarity | 5.6955 | 30% | 1.7086 |
| **Base Score** | - | - | **2.1899** |
| **Final Score** | - | - | **0.7860** (78.6%) |

**Why This Matched:**

- **Word Overlap:** distribution, energy
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.688)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.695)

**Ranking Justification:**

- Ranked **#4** - score is 0.1195 lower than #3
- Score is 0.0295 higher than #5

---

### Rank #5: Energy Distribution Partners

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 4.1941 | 30% | 1.2582 |
| **Base Score** | - | - | **1.7832** |
| **Final Score** | - | - | **0.7565** (75.6%) |

**Why This Matched:**

- **Word Overlap:** distribution, energy, partners
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.194)

**Ranking Justification:**

- Ranked **#5** - score is 0.0295 lower than #4
- Score is 0.0251 higher than #6

---

</details>
## 95. Query: `ThinkAdvisor`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | ThinkAdvisor | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Planadvisor | **46.8%** | 🟠 LOW **46.8%** | Semantic/meaning-based match |
| 3 | TripAdvisor | **46.0%** | 🟠 LOW **46.0%** | Semantic/meaning-based match |
| 4 | Invisors | **43.6%** | 🟠 LOW **43.6%** | Semantic/meaning-based match |
| 5 | ChannelAdvisor | **42.2%** | 🟠 LOW **42.2%** | Semantic/meaning-based match |
| 6 | HomeAdvisor | **41.6%** | 🟠 LOW **41.6%** | Semantic/meaning-based match |
| 7 | Tripadvisor | **40.7%** | 🟠 LOW **40.7%** | Semantic/meaning-based match |
| 8 | Tripadvisor | **40.4%** | 🟠 LOW **40.4%** | Semantic/meaning-based match |
| 9 | TRIPADVISOR | **40.0%** | 🟠 LOW **40.0%** | Semantic/meaning-based match |
| 10 | ScoutAdvisor Corporation | **39.5%** | 🔴 WEAK **39.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: ThinkAdvisor

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.6218 | 30% | 1.6865 |
| **Base Score** | - | - | **2.3865** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** thinkadvisor
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.622)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.5348 (53.5%)

---

### Rank #2: Planadvisor

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3130 | 70% | 0.2191 |
| Semantic Similarity | 4.6040 | 30% | 1.3812 |
| **Base Score** | - | - | **1.6003** |
| **Final Score** | - | - | **0.4676** (46.8%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.313) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.604)

**Ranking Justification:**

- Ranked **#2** - score is 0.5348 lower than #1
- Score is 0.0079 higher than #3

---

### Rank #3: TripAdvisor

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3522 | 70% | 0.2465 |
| Semantic Similarity | 3.9435 | 30% | 1.1830 |
| **Base Score** | - | - | **1.4296** |
| **Final Score** | - | - | **0.4597** (46.0%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.352) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.943)

**Ranking Justification:**

- Ranked **#3** - score is 0.0079 lower than #2
- Score is 0.0238 higher than #4

---

### Rank #4: Invisors

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3316 | 70% | 0.2321 |
| Semantic Similarity | 3.7695 | 30% | 1.1308 |
| **Base Score** | - | - | **1.3630** |
| **Final Score** | - | - | **0.4359** (43.6%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.332) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.769)

**Ranking Justification:**

- Ranked **#4** - score is 0.0238 lower than #3
- Score is 0.0134 higher than #5

---

### Rank #5: ChannelAdvisor

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.3115 | 70% | 0.2181 |
| Semantic Similarity | 3.7822 | 30% | 1.1347 |
| **Base Score** | - | - | **1.3527** |
| **Final Score** | - | - | **0.4225** (42.2%) |

**Why This Matched:**

- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.312) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.782)

**Ranking Justification:**

- Ranked **#5** - score is 0.0134 lower than #4
- Score is 0.0067 higher than #6

---

</details>
## 96. Query: `Jump on it Outreach`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Jump on it Outreach | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Jump On It | **69.4%** | 🟡 MEDIUM **69.4%** | Semantic/meaning-based match |
| 3 | Evangelism on the Move Outreach Ministry | **62.0%** | 🟡 MEDIUM **62.0%** | Semantic/meaning-based match |
| 4 | Outreach | **61.9%** | 🟡 MEDIUM **61.9%** | Semantic/meaning-based match |
| 5 | Above N Beyond Outreach | **61.5%** | 🟡 MEDIUM **61.5%** | Semantic/meaning-based match |
| 6 | Community Connections Outreach Program | **59.2%** | 🟠 LOW **59.2%** | Semantic/meaning-based match |
| 7 | Outreach Strategies | **57.8%** | 🟠 LOW **57.8%** | Semantic/meaning-based match |
| 8 | Jump | **57.3%** | 🟠 LOW **57.3%** | Semantic/meaning-based match |
| 9 | Where are you? Outreach | **57.1%** | 🟠 LOW **57.1%** | Semantic/meaning-based match |
| 10 | Human Outreach Project | **56.9%** | 🟠 LOW **56.9%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Jump on it Outreach

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.3191 | 30% | 1.2957 |
| **Base Score** | - | - | **1.9957** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** it, jump, on, outreach
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.319)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.3081 (30.8%)

---

### Rank #2: Jump On It

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7159 | 70% | 0.5011 |
| Semantic Similarity | 3.6371 | 30% | 1.0911 |
| **Base Score** | - | - | **1.5923** |
| **Final Score** | - | - | **0.6944** (69.4%) |

**Why This Matched:**

- **Word Overlap:** it, jump, on
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.716)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.637)

**Ranking Justification:**

- Ranked **#2** - score is 0.3081 lower than #1
- Score is 0.0742 higher than #3

---

### Rank #3: Evangelism on the Move Outreach Ministry

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 2.7541 | 30% | 0.8262 |
| **Base Score** | - | - | **1.2995** |
| **Final Score** | - | - | **0.6202** (62.0%) |

**Why This Matched:**

- **Word Overlap:** on, outreach
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.754)

**Ranking Justification:**

- Ranked **#3** - score is 0.0742 lower than #2
- Score is 0.0014 higher than #4

---

### Rank #4: Outreach

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.4500 | 70% | 0.3150 |
| Semantic Similarity | 5.7720 | 30% | 1.7316 |
| **Base Score** | - | - | **2.0466** |
| **Final Score** | - | - | **0.6187** (61.9%) |

**Why This Matched:**

- **Word Overlap:** outreach
- **Lexical Match:** Weak word alignment (Jaro-Winkler: 0.450) - relies on semantic similarity
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.772)

**Ranking Justification:**

- Ranked **#4** - score is 0.0014 lower than #3
- Score is 0.0040 higher than #5

---

### Rank #5: Above N Beyond Outreach

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5250 | 70% | 0.3675 |
| Semantic Similarity | 4.6856 | 30% | 1.4057 |
| **Base Score** | - | - | **1.7732** |
| **Final Score** | - | - | **0.6148** (61.5%) |

**Why This Matched:**

- **Word Overlap:** outreach
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.525)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.686)

**Ranking Justification:**

- Ranked **#5** - score is 0.0040 lower than #4
- Score is 0.0232 higher than #6

---

</details>
## 97. Query: `The Association of Ringside Consultants (ARC)`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | The Association of Ringside Consultants (ARC) | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | Virginia Association of Legal Consultants | **71.8%** | 🟡 MEDIUM **71.8%** | Semantic/meaning-based match |
| 3 | Oklahoma Association of Personnel Consultants | **71.5%** | 🟡 MEDIUM **71.5%** | Semantic/meaning-based match |
| 4 | Association of Ringside Physicians | **71.2%** | 🟡 MEDIUM **71.2%** | Semantic/meaning-based match |
| 5 | Association of Charlotte Area Consultants | **70.8%** | 🟡 MEDIUM **70.8%** | Semantic/meaning-based match |
| 6 | Professional Consultants Association | **69.7%** | 🟡 MEDIUM **69.7%** | Semantic/meaning-based match |
| 7 | Investment Management Consultants Association | **69.5%** | 🟡 MEDIUM **69.5%** | Semantic/meaning-based match |
| 8 | American Association of Ringside Physicians | **69.4%** | 🟡 MEDIUM **69.4%** | Semantic/meaning-based match |
| 9 | Association of Professional Investment Consultants | **69.2%** | 🟡 MEDIUM **69.2%** | Semantic/meaning-based match |
| 10 | Association of Ringside Physicians (ARP) | **69.0%** | 🟡 MEDIUM **69.0%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: The Association of Ringside Consultants (ARC)

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.7512 | 30% | 1.4254 |
| **Base Score** | - | - | **2.1254** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** (arc), association, consultants, of, ringside, the
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.751)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2845 (28.4%)

---

### Rank #2: Virginia Association of Legal Consultants

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.0567 | 30% | 0.9170 |
| **Base Score** | - | - | **1.4376** |
| **Final Score** | - | - | **0.7180** (71.8%) |

**Why This Matched:**

- **Word Overlap:** association, consultants, of
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.057)

**Ranking Justification:**

- Ranked **#2** - score is 0.2845 lower than #1
- Score is 0.0031 higher than #3

---

### Rank #3: Oklahoma Association of Personnel Consultants

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 3.0074 | 30% | 0.9022 |
| **Base Score** | - | - | **1.4229** |
| **Final Score** | - | - | **0.7148** (71.5%) |

**Why This Matched:**

- **Word Overlap:** association, consultants, of
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.007)

**Ranking Justification:**

- Ranked **#3** - score is 0.0031 lower than #2
- Score is 0.0024 higher than #4

---

### Rank #4: Association of Ringside Physicians

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 3.7188 | 30% | 1.1156 |
| **Base Score** | - | - | **1.5889** |
| **Final Score** | - | - | **0.7124** (71.2%) |

**Why This Matched:**

- **Word Overlap:** association, of, ringside
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.676)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.719)

**Ranking Justification:**

- Ranked **#4** - score is 0.0024 lower than #3
- Score is 0.0045 higher than #5

---

### Rank #5: Association of Charlotte Area Consultants

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 2.8584 | 30% | 0.8575 |
| **Base Score** | - | - | **1.3782** |
| **Final Score** | - | - | **0.7079** (70.8%) |

**Why This Matched:**

- **Word Overlap:** association, consultants, of
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.858)

**Ranking Justification:**

- Ranked **#5** - score is 0.0045 lower than #4
- Score is 0.0109 higher than #6

---

</details>
## 98. Query: `SFA HASA`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | SFA HASA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | SFA Partners | **76.8%** | 🟡 MEDIUM **76.8%** | Semantic/meaning-based match |
| 3 | SFA Opportunity | **75.6%** | 🟡 MEDIUM **75.6%** | Semantic/meaning-based match |
| 4 | Sfa | **74.6%** | 🟡 MEDIUM **74.6%** | Semantic/meaning-based match |
| 5 | SFA Training | **74.5%** | 🟡 MEDIUM **74.5%** | Semantic/meaning-based match |
| 6 | Test Sfa | **73.4%** | 🟡 MEDIUM **73.4%** | Semantic/meaning-based match |
| 7 | SFA System Account | **71.5%** | 🟡 MEDIUM **71.5%** | Semantic/meaning-based match |
| 8 | SFA Saniflo | **71.4%** | 🟡 MEDIUM **71.4%** | Semantic/meaning-based match |
| 9 | SFA Design | **70.9%** | 🟡 MEDIUM **70.9%** | Semantic/meaning-based match |
| 10 | SFA Designs | **70.6%** | 🟡 MEDIUM **70.6%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: SFA HASA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7192 | 30% | 1.1158 |
| **Base Score** | - | - | **1.8158** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** hasa, sfa
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.719)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.2343 (23.4%)

---

### Rank #2: SFA Partners

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.7743 | 30% | 1.4323 |
| **Base Score** | - | - | **1.9529** |
| **Final Score** | - | - | **0.7681** (76.8%) |

**Why This Matched:**

- **Word Overlap:** sfa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.774)

**Ranking Justification:**

- Ranked **#2** - score is 0.2343 lower than #1
- Score is 0.0119 higher than #3

---

### Rank #3: SFA Opportunity

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.5421 | 30% | 1.3626 |
| **Base Score** | - | - | **1.8832** |
| **Final Score** | - | - | **0.7563** (75.6%) |

**Why This Matched:**

- **Word Overlap:** sfa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.542)

**Ranking Justification:**

- Ranked **#3** - score is 0.0119 lower than #2
- Score is 0.0107 higher than #4

---

### Rank #4: Sfa

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6300 | 70% | 0.4410 |
| Semantic Similarity | 5.8973 | 30% | 1.7692 |
| **Base Score** | - | - | **2.2102** |
| **Final Score** | - | - | **0.7455** (74.6%) |

**Why This Matched:**

- **Word Overlap:** sfa
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.630)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.897)

**Ranking Justification:**

- Ranked **#4** - score is 0.0107 lower than #3
- Score is 0.0010 higher than #5

---

### Rank #5: SFA Training

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 4.3127 | 30% | 1.2938 |
| **Base Score** | - | - | **1.8144** |
| **Final Score** | - | - | **0.7445** (74.5%) |

**Why This Matched:**

- **Word Overlap:** sfa
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.744)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.313)

**Ranking Justification:**

- Ranked **#5** - score is 0.0010 lower than #4
- Score is 0.0110 higher than #6

---

</details>
## 99. Query: `Grupo Duracell Ene 2025`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | Grupo Duracell Ene 2025 | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | GRUPO MAZDA FEB 2025 | **66.7%** | 🟡 MEDIUM **66.7%** | Semantic/meaning-based match |
| 3 | Workshop 2025 - Grupo CCM | **64.9%** | 🟡 MEDIUM **64.9%** | Semantic/meaning-based match |
| 4 | Grupo Eñe de  Comunicación | **52.4%** | 🟠 LOW **52.4%** | Semantic/meaning-based match |
| 5 | GRUPO Convenciones Y Eventos | **51.6%** | 🟠 LOW **51.6%** | Semantic/meaning-based match |
| 6 | Grupo Brasil DPE | **51.6%** | 🟠 LOW **51.6%** | Semantic/meaning-based match |
| 7 | GRUPO Grand De Mexico | **51.5%** | 🟠 LOW **51.5%** | Semantic/meaning-based match |
| 8 | DURACELL USA | **51.1%** | 🟠 LOW **51.1%** | Semantic/meaning-based match |
| 9 | Evento Grupo La Norteñita | **51.0%** | 🟠 LOW **51.0%** | Semantic/meaning-based match |
| 10 | Grupo BG de eventos | **51.0%** | 🟠 LOW **51.0%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: Grupo Duracell Ene 2025

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 5.1445 | 30% | 1.5434 |
| **Base Score** | - | - | **2.2434** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** 2025, duracell, ene, grupo
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.145)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.3350 (33.5%)

---

### Rank #2: GRUPO MAZDA FEB 2025

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 2.8738 | 30% | 0.8622 |
| **Base Score** | - | - | **1.3580** |
| **Final Score** | - | - | **0.6675** (66.7%) |

**Why This Matched:**

- **Word Overlap:** 2025, grupo
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.708)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.874)

**Ranking Justification:**

- Ranked **#2** - score is 0.3350 lower than #1
- Score is 0.0185 higher than #3

---

### Rank #3: Workshop 2025 - Grupo CCM

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 2.5592 | 30% | 0.7678 |
| **Base Score** | - | - | **1.2636** |
| **Final Score** | - | - | **0.6490** (64.9%) |

**Why This Matched:**

- **Word Overlap:** 2025, grupo
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.708)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.559)

**Ranking Justification:**

- Ranked **#3** - score is 0.0185 lower than #2
- Score is 0.1252 higher than #4

---

### Rank #4: Grupo Eñe de  Comunicación

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 2.9263 | 30% | 0.8779 |
| **Base Score** | - | - | **1.2279** |
| **Final Score** | - | - | **0.5238** (52.4%) |

**Why This Matched:**

- **Word Overlap:** grupo
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.500)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.926)

**Ranking Justification:**

- Ranked **#4** - score is 0.1252 lower than #3
- Score is 0.0076 higher than #5

---

### Rank #5: GRUPO Convenciones Y Eventos

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 2.7967 | 30% | 0.8390 |
| **Base Score** | - | - | **1.1890** |
| **Final Score** | - | - | **0.5162** (51.6%) |

**Why This Matched:**

- **Word Overlap:** grupo
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.500)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.797)

**Ranking Justification:**

- Ranked **#5** - score is 0.0076 lower than #4
- Score is 0.0004 higher than #6

---

</details>
## 100. Query: `World Association of Medical Law`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | World Association of Medical Law | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | World Association for Medical Law | **95.9%** | 🟢 EXACT **95.9%** | Semantic/meaning-based match |
| 3 | World Association For Medical Law | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 4 | World Association for Medical Law | **95.6%** | 🟢 EXACT **95.6%** | Semantic/meaning-based match |
| 5 | World Medical Association | **80.4%** | 🟢 HIGH **80.4%** | Semantic/meaning-based match |
| 6 | World Law Foundation | **74.6%** | 🟡 MEDIUM **74.6%** | Semantic/meaning-based match |
| 7 | International Law Student Association | **73.2%** | 🟡 MEDIUM **73.2%** | Semantic/meaning-based match |
| 8 | International Law Association | **72.3%** | 🟡 MEDIUM **72.3%** | Semantic/meaning-based match |
| 9 | Pacific Medical Law | **72.2%** | 🟡 MEDIUM **72.2%** | Semantic/meaning-based match |
| 10 | World Korean Medical Organization | **71.5%** | 🟡 MEDIUM **71.5%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: World Association of Medical Law

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.1509 | 30% | 0.9453 |
| **Base Score** | - | - | **1.6453** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** association, law, medical, of, world
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.151)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0433 (4.3%)

---

### Rank #2: World Association for Medical Law

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 3.2480 | 30% | 0.9744 |
| **Base Score** | - | - | **1.5789** |
| **Final Score** | - | - | **0.9592** (95.9%) |

**Why This Matched:**

- **Word Overlap:** association, law, medical, world
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.248)

**Ranking Justification:**

- Ranked **#2** - score is 0.0433 lower than #1
- Score is 0.0034 higher than #3

---

### Rank #3: World Association For Medical Law

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 5.2767 | 30% | 1.5830 |
| **Base Score** | - | - | **2.1876** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** association, law, medical, world
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.277)

**Ranking Justification:**

- Ranked **#3** - score is 0.0034 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: World Association for Medical Law

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity | 3.2211 | 30% | 0.9663 |
| **Base Score** | - | - | **1.5709** |
| **Final Score** | - | - | **0.9558** (95.6%) |

**Why This Matched:**

- **Word Overlap:** association, law, medical, world
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.864)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.221)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.1513 higher than #5

---

### Rank #5: World Medical Association

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7159 | 70% | 0.5011 |
| Semantic Similarity | 5.2498 | 30% | 1.5749 |
| **Base Score** | - | - | **2.0761** |
| **Final Score** | - | - | **0.8045** (80.4%) |

**Why This Matched:**

- **Word Overlap:** association, medical, world
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.716)
- **Semantic Match:** Very strong meaning-based connection (cosine: 5.250)

**Ranking Justification:**

- Ranked **#5** - score is 0.1513 lower than #4
- Score is 0.0590 higher than #6

---

</details>
## 101. Query: `ABA`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | ABA | **100.7%** | 🟢 EXACT **100.7%** | Near-exact text match |
| 2 | ABA | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 3 | ABA | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 4 | ABA | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 5 | ABA | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 6 | ABA | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 7 | ABA | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 8 | ABA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 9 | ABA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 10 | ABA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: ABA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 2.8337 | 30% | 0.8501 |
| **Base Score** | - | - | **1.5501** |
| **Final Score** | - | - | **1.0073** (100.7%) |

**Why This Matched:**

- **Word Overlap:** aba
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.834)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0073)
- Score gap to #2: 0.0024 (0.2%)

---

### Rank #2: ABA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 7.4766 | 30% | 2.2430 |
| **Base Score** | - | - | **2.9430** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** aba
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 7.477)

**Ranking Justification:**

- Ranked **#2** - score is 0.0024 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: ABA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.4199 | 30% | 1.0260 |
| **Base Score** | - | - | **1.7260** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** aba
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.420)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: ABA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 2.6837 | 30% | 0.8051 |
| **Base Score** | - | - | **1.5051** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** aba
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.684)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: ABA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | - | - | **1.0000** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** aba
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 1.000)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0010 higher than #6

---

</details>
## 102. Query: `PDMA`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | PDMA | **100.5%** | 🟢 EXACT **100.5%** | Near-exact text match |
| 2 | PDMA | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 3 | PDMA | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 4 | PDMA | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 5 | PDMA | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 6 | PDMA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 7 | PDMA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 8 | PDMA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 9 | PDMA | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 10 | PdMA Corporation | **96.2%** | 🟢 EXACT **96.2%** | Near-exact text match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: PDMA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 2.8941 | 30% | 0.8682 |
| **Base Score** | - | - | **1.5682** |
| **Final Score** | - | - | **1.0049** (100.5%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.894)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0049)
- Score gap to #2: 0.0010 (0.1%)

---

### Rank #2: PDMA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.1719 | 30% | 1.2516 |
| **Base Score** | - | - | **1.9516** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.172)

**Ranking Justification:**

- Ranked **#2** - score is 0.0010 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: PDMA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.8754 | 30% | 1.1626 |
| **Base Score** | - | - | **1.8626** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.875)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: PDMA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7191 | 30% | 1.1157 |
| **Base Score** | - | - | **1.8157** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.719)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: PDMA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.4054 | 30% | 1.0216 |
| **Base Score** | - | - | **1.7216** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** pdma
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.405)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0014 higher than #6

---

</details>
## 103. Query: `IBM`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | IBM | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 2 | IBM | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 3 | IBM | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 4 | IBM Belgium SA | **96.6%** | 🟢 EXACT **96.6%** | Semantic/meaning-based match |
| 5 | International Business Machines IBM | **96.5%** | 🟢 EXACT **96.5%** | Semantic/meaning-based match |
| 6 | International Business Machines IBM | **96.3%** | 🟢 EXACT **96.3%** | Semantic/meaning-based match |
| 7 | International Business Machines IBM | **96.3%** | 🟢 EXACT **96.3%** | Semantic/meaning-based match |
| 8 | IBM Corporation OLD | **96.3%** | 🟢 EXACT **96.3%** | Semantic/meaning-based match |
| 9 | IBM Global Business Service | **96.3%** | 🟢 EXACT **96.3%** | Semantic/meaning-based match |
| 10 | IBM India | **96.2%** | 🟢 EXACT **96.2%** | Semantic/meaning-based match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: IBM

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 7.3994 | 30% | 2.2198 |
| **Base Score** | - | - | **2.9198** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** ibm
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 7.399)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0024)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: IBM

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 4.0384 | 30% | 1.2115 |
| **Base Score** | - | - | **1.9115** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** ibm
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 4.038)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: IBM

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.8073 | 30% | 1.1422 |
| **Base Score** | - | - | **1.8422** |
| **Final Score** | - | - | **1.0024** (100.2%) |

**Why This Matched:**

- **Word Overlap:** ibm
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.807)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0362 higher than #4

---

### Rank #4: IBM Belgium SA

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 2.8059 | 30% | 0.8418 |
| **Base Score** | - | - | **1.3668** |
| **Final Score** | - | - | **0.9662** (96.6%) |

**Why This Matched:**

- **Word Overlap:** ibm
- **Lexical Match:** Strong word alignment (Jaro-Winkler: 0.750)
- **Semantic Match:** Very strong meaning-based connection (cosine: 2.806)

**Ranking Justification:**

- Ranked **#4** - score is 0.0362 lower than #3
- Score is 0.0013 higher than #5

---

### Rank #5: International Business Machines IBM

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 0.6923 | 70% | 0.4846 |
| Semantic Similarity | 3.7927 | 30% | 1.1378 |
| **Base Score** | - | - | **1.6224** |
| **Final Score** | - | - | **0.9649** (96.5%) |

**Why This Matched:**

- **Word Overlap:** ibm
- **Lexical Match:** Moderate word alignment (Jaro-Winkler: 0.692)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.793)

**Ranking Justification:**

- Ranked **#5** - score is 0.0013 lower than #4
- Score is 0.0015 higher than #6

---

</details>
## 104. Query: `GE`

| Rank | Company | Score | Summary |
|:----:|---------|:-----:|---------|
| 1 | GE | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 2 | GE | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 3 | GE | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 4 | GE | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 5 | GE | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 6 | GE | **100.4%** | 🟢 EXACT **100.4%** | Near-exact text match |
| 7 | GE | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 8 | GE | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 9 | GE | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |
| 10 | GE | **100.2%** | 🟢 EXACT **100.2%** | Near-exact text match |

<details>
<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>

### Rank #1: GE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 7.4052 | 30% | 2.2215 |
| **Base Score** | - | - | **2.9215** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** ge
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 7.405)

**Ranking Justification:**

- Ranked **#1** because it has the highest combined score (1.0039)
- Score gap to #2: 0.0000 (0.0%)

---

### Rank #2: GE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.9563 | 30% | 1.1869 |
| **Base Score** | - | - | **1.8869** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** ge
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.956)

**Ranking Justification:**

- Ranked **#2** - score is 0.0000 lower than #1
- Score is 0.0000 higher than #3

---

### Rank #3: GE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.7395 | 30% | 1.1218 |
| **Base Score** | - | - | **1.8218** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** ge
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.739)

**Ranking Justification:**

- Ranked **#3** - score is 0.0000 lower than #2
- Score is 0.0000 higher than #4

---

### Rank #4: GE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 3.5024 | 30% | 1.0507 |
| **Base Score** | - | - | **1.7507** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** ge
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 3.502)

**Ranking Justification:**

- Ranked **#4** - score is 0.0000 lower than #3
- Score is 0.0000 higher than #5

---

### Rank #5: GE

**Score Breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | - | - | **1.0000** |
| **Final Score** | - | - | **1.0039** (100.4%) |

**Why This Matched:**

- **Word Overlap:** ge
- **Lexical Match:** Nearly identical text (Jaro-Winkler: 1.000)
- **Semantic Match:** Very strong meaning-based connection (cosine: 1.000)

**Ranking Justification:**

- Ranked **#5** - score is 0.0000 lower than #4
- Score is 0.0000 higher than #6

---

</details>
