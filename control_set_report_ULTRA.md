# Company Matching Control Set Report (Location-Aware)

**Generated:** 2025-12-28 18:10:52

---

## Matching Scenarios Handled

This system is designed to handle the following real-world company matching challenges:

### 1. **Exact Matches**
Perfect text matching when query exactly equals company name.
- Example: `"IBM"` → `"IBM"` (100% match)

### 2. **Acronym Expansions**
Matching acronyms to their full company names.
- Example: `"IBM"` → `"International Business Machines"` (98%+ match)

### 3. **Location-Aware Matching (NEW)**
Differentiates identical names using geographic context.
- Example: `"Acme"` in `"Chicago"` → Matches `"Acme Corp (Chicago)"` higher than `"Acme Corp (Miami)"`

### 4. **Popularity/Frequency Bias (NEW)**
Uses occurrence counts to break ties and prioritize larger entities.
- Example: Frequent national brands rank higher than obscure single-occurrence entries.

---

## Advanced Scoring Formula

```
Base Score = (String Similarity × 70%) + (Semantic Similarity × 30%)
Fidelity Boost = Acronym Fidelity × 15%
Location Boost = Location Score × 5% (Implicit via Location Baking)
Final Score = Base Score + Fidelity Boost + Location Boost + Popularity Boost
```

---

## Control Set Results

### 1. PDMA Association

**Top Non-Self Match (Rank 2):** `PDMA Alliance` (York, SC) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 0.9555 | 30% | 0.2866 |
| **Base Score** | **0.8816** | - | **88.16%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. PDMA Alliance (York, SC) (90.0%)
3. PDMA (Windermere, FL) (72.5%)
4. Association Services Group (70.9%)
5. Association Forum (, IL) (69.7%)
6. IIB Association Group (69.3%)

---

### 2. Nicolas/Sanchez Wedding

**Top Non-Self Match (Rank 2):** `Castillo Sanchez Wedding` (Charlotte, NC) (77.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.7733** | - | **77.33%** |
| **Final Score** | **0.7730** | - | **77.30%** |

**Top 5 Non-Self Matches:**

2. Castillo Sanchez Wedding (Charlotte, NC) (77.3%)
3. Puebla Wedding (71.3%)
4. Rogers wedding Wedding (68.1%)
5. Smith/Rivera Wedding (68.0%)
6. Veale Wedding (68.0%)

---

### 3. Kehilat Ariel Synagogue

**Top Non-Self Match (Rank 2):** `Ohel Moshe Synagogue` (Los Angeles, CA) (59.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4750 | 70% | 0.3325 |
| Semantic Similarity | 0.8746 | 30% | 0.2624 |
| **Base Score** | **0.5949** | - | **59.49%** |
| **Final Score** | **0.5950** | - | **59.50%** |

**Top 5 Non-Self Matches:**

2. Ohel Moshe Synagogue (Los Angeles, CA) (59.5%)
3. Beth Ariel Fellowship (Canoga Park, CA) (59.1%)
4. Sim Shalom Jewish Universalist Synagogue (New York, NY) (52.7%)
5. Allah Temple Shriners (35.8%)
6. Holy Faith Temple (34.7%)

---

### 4. Next Level Events

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Next Level Performance` (Canton, MA) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 0.7523 | 30% | 0.2257 |
| **Base Score** | **0.8440** | - | **84.40%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Next Level Performance (Canton, MA) (90.0%)
3. Next Level Performance (New Brunswick, NJ) (90.0%)
4. Stage Events (60.2%)
5. Impact Events (59.7%)
6. Well Planned Events (58.8%)

---

### 5. Site Foundation Golf Tournament

**Top Non-Self Match (Rank 2):** `WORLD  GOLF FOUNDATION` (, IL) (77.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.7733** | - | **77.33%** |
| **Final Score** | **0.7730** | - | **77.30%** |

**Top 5 Non-Self Matches:**

2. WORLD  GOLF FOUNDATION (, IL) (77.3%)
3. IMG Golf Tournament (76.6%)
4. US OPEN GOLF TOURNAMENT (Pine Hurst, ) (75.3%)
5. Robert Brooks Charity Golf Tournament (74.8%)
6. World Golf Foundation (Jacksonville, FL) (74.6%)

---

### 6. Interim WG Meeting - BIER

**Top Non-Self Match (Rank 2):** `Executive Advisory Board Meeting` (60.9%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4500 | 70% | 0.3150 |
| Semantic Similarity | 0.9796 | 30% | 0.2939 |
| **Base Score** | **0.6089** | - | **60.89%** |
| **Final Score** | **0.6090** | - | **60.90%** |

**Top 5 Non-Self Matches:**

2. Executive Advisory Board Meeting (60.9%)
3. MEP Advisory Board meeting (58.9%)
4. Bimbo January Meeting (58.6%)
5. BTS Store Managers Meeting (58.5%)
6. Medical Meeting Systems (Bievre, ) (58.3%)

---

### 7. DermaQuest Inc

**Top Non-Self Match (Rank 2):** `Dermaquest Skin Care` (82.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8250** | - | **82.50%** |
| **Final Score** | **0.8250** | - | **82.50%** |

**Top 5 Non-Self Matches:**

2. Dermaquest Skin Care (82.5%)
3. GenQuest (44.4%)
4. Dermapure (Surrey, BC) (41.8%)
5. Learnquest (41.7%)
6. Saxquest (40.8%)

---

### 8. Ellwood Group Inc

**Top Non-Self Match (Rank 2):** `EGI` (75.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **1.0000** | - | **100.00%** |
| Acronym Fidelity Boost | 1.0000 | 15% max | +0.1500 |
| **Final Score** | **0.7500** | - | **75.00%** |

**Top 5 Non-Self Matches:**

2. EGI (75.0%)
3. Hillwood (54.7%)
4. Ashwood Group LLC (Portland, OR) (46.4%)
5. Festival at Lakewood (39.4%)
6. Friendswood Community Church (Friendswood, TX) (37.5%)

---

### 9. American Miniature Horse Registry

**Top Non-Self Match (Rank 2):** `American Miniature Horse Association Headquarters` (83.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7670 | 70% | 0.5369 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8369** | - | **83.69%** |
| **Final Score** | **0.8370** | - | **83.70%** |

**Top 5 Non-Self Matches:**

2. American Miniature Horse Association Headquarters (83.7%)
3. American Carbon Registry (Arlington, VA) (62.9%)
4. Palomino Horse Breeders Association (53.6%)
5. Arabian Horse Show (50.7%)
6. National Horse Show (Lexington, KY) (48.8%)

---

### 10. YADA ENTERPRISES, INC

**Top Non-Self Match (Rank 2):** `Yassaka` (46.9%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.2455 | 70% | 0.1718 |
| Semantic Similarity | 0.9920 | 30% | 0.2976 |
| **Base Score** | **0.4694** | - | **46.94%** |
| **Final Score** | **0.4690** | - | **46.90%** |

**Top 5 Non-Self Matches:**

2. Yassaka (46.9%)
3. Vanda (43.4%)
4. Yadabada Media Group (42.7%)
5. Alliance Enterprises (Lacey, WA) (35.0%)
6. WDR Enterprises company (34.9%)

---

### 11. Seafood Nutrition Partnership

**Top Non-Self Match (Rank 2):** `USDA Food Nutrition Service` (63.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4798 | 70% | 0.3359 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6359** | - | **63.59%** |
| **Final Score** | **0.6360** | - | **63.60%** |

**Top 5 Non-Self Matches:**

2. USDA Food Nutrition Service (63.6%)
3. Stronger U Nutrition (61.8%)
4. Integrating Nutrition and Nutrition Services (61.7%)
5. Livestock Nutrition Center (Hereford, TX) (61.0%)
6. Linéa Natural Nutrition (60.9%)

---

### 12. AVIAKOMPANIYA SIBIR, PAO

**Top Non-Self Match (Rank 2):** `AVIPAM Sao Paulo` (43.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.2538 | 70% | 0.1777 |
| Semantic Similarity | 0.8488 | 30% | 0.2546 |
| **Base Score** | **0.4323** | - | **43.23%** |
| **Final Score** | **0.4320** | - | **43.20%** |

**Top 5 Non-Self Matches:**

2. AVIPAM Sao Paulo (43.2%)
3. Kaiulani Kauahi (40.6%)
4. Kabira Technology (San Rafael, CA) (37.6%)
5. Liliana Ferpi (Milan, ) (36.7%)
6. Vishal Morjaria (36.3%)

---

### 13. Hartford Hospital School of Nursing

**Top Non-Self Match (Rank 2):** `Hartford Public Schools` (Hartford, CT) (68.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5455 | 70% | 0.3818 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6818** | - | **68.18%** |
| **Final Score** | **0.6820** | - | **68.20%** |

**Top 5 Non-Self Matches:**

2. Hartford Public Schools (Hartford, CT) (68.2%)
3. HARTFORD ATHLETIC (West Hartford, CT) (61.2%)
4. West Hartford Environmental Group (West Hartford, CT) (59.2%)
5. Charity Hospital School of Nursing Alumni Association (New Orleans, LA) (58.6%)
6. Heads Up! Hartford (57.4%)

---

### 14. Internal J&J Meeting and Breakfast

**Top Non-Self Match (Rank 2):** `Legislative Breakfast` (56.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.3750 | 70% | 0.2625 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.5625** | - | **56.25%** |
| **Final Score** | **0.5620** | - | **56.20%** |

**Top 5 Non-Self Matches:**

2. Legislative Breakfast (56.2%)
3. JMATE Meeting (53.2%)
4. Prosper Alexandria Networking Breakfast (52.3%)
5. SPICE MARKET BREAKFAST (51.4%)
6. Fisher Family Breakfast (51.3%)

---

### 15. Spina Bifida Coalition of Cincinnati

**Top Non-Self Match (Rank 2):** `Illinois Spina Bifida Association` (Lisle, IL) (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Illinois Spina Bifida Association (Lisle, IL) (82.1%)
3. Spina Bifida Natl Conf (Washington, DC) (81.5%)
4. Spina Bifida of Greater New Orleans (New Orleans, LA) (72.7%)
5. Cincinnati Realtist Association (Cincinnati, OH) (60.2%)
6. Cincinnati Development Fund (Cincinnati, OH) (59.6%)

---

### 16. THE SOCA GROUP ORGANIZATION

**Top Non-Self Match (Rank 2):** `POLITICAL ORGANIZATION` (79.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.9044 | 30% | 0.2713 |
| **Base Score** | **0.7919** | - | **79.19%** |
| **Final Score** | **0.7920** | - | **79.20%** |

**Top 5 Non-Self Matches:**

2. POLITICAL ORGANIZATION (79.2%)
3. Team SOCA (Toronto, ON) (78.9%)
4. 4 Life Organization (69.4%)
5. #SSS Secret Soca Society (69.3%)
6. National Organization of Coaching Association Directors (61.8%)

---

### 17. Shiroyama Junior High School

**Top Non-Self Match (Rank 2):** `Shizuoka Jonan High School` (76.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 0.8790 | 30% | 0.2637 |
| **Base Score** | **0.7595** | - | **75.95%** |
| **Final Score** | **0.7600** | - | **76.00%** |

**Top 5 Non-Self Matches:**

2. Shizuoka Jonan High School (76.0%)
3. Old Mill High School (75.6%)
4. Gulf Shores High School (75.4%)
5. Chico High School (75.1%)
6. Cabin John High School (73.1%)

---

### 18. National Home Health

**Top Non-Self Match (Rank 2):** `Haven Home Health` (91.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.9183** | - | **91.83%** |
| **Final Score** | **0.9180** | - | **91.80%** |

**Top 5 Non-Self Matches:**

2. Haven Home Health (91.8%)
3. CenterWell Home Health (Atlanta, GA) (90.0%)
4. NATIONAL HEALTH COUNCIL (Washington, DC) (68.8%)
5. National Association for Home Care and H (Washington, DC) (67.7%)
6. National Association for Home Care and H (washington, DC) (67.7%)

---

### 19. American News Women's Club

**Top Non-Self Match (Rank 2):** `OLPH Women's Club` (72.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.8229 | 30% | 0.2469 |
| **Base Score** | **0.7202** | - | **72.02%** |
| **Final Score** | **0.7200** | - | **72.00%** |

**Top 5 Non-Self Matches:**

2. OLPH Women's Club (72.0%)
3. Empower Women Club (61.0%)
4. Your Book Club for Women (58.7%)
5. American Univ Womens Basketball (Washington, DC) (58.4%)
6. FoCo Womens Book Club (57.1%)

---

### 20. Denise Roberge

**Top Non-Self Match (Rank 2):** `Denise Eventos` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Denise Eventos (82.1%)
3. Melanie Benjamin (38.2%)
4. Melissa Wedding (37.2%)
5. Vicki Tours (37.2%)
6. Michelle Shaw (37.0%)

---

### 21. Synergy Soccer Club

**Top Non-Self Match (Rank 2):** `Tennessee Soccer Club` (Columbus, OH) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.6092 | 30% | 0.1828 |
| **Base Score** | **0.7496** | - | **74.96%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Tennessee Soccer Club (Columbus, OH) (90.0%)
3. ASHWAUBENON SOCCER CLUB (90.0%)
4. Broken Arrow Soccer Club (74.6%)
5. Emerald Youth Soccer Club (73.2%)
6. ATI US Club Soccer (71.2%)

---

### 22. NFC Forum

**Top Non-Self Match (Rank 2):** `Forum USA` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Forum USA (82.1%)
3. Perspective Forum (80.7%)
4. Association Forum (, IL) (78.8%)
5. GroupRides Forum (77.8%)
6. UID Forum (Berea, OH) (77.7%)

---

### 23. A Better Choice Limousine & Concierge

**Top Non-Self Match (Rank 2):** `LCT Limousine & Chauffeur Transportation Show` (Torrance, CA) (58.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4432 | 70% | 0.3102 |
| Semantic Similarity | 0.9213 | 30% | 0.2764 |
| **Base Score** | **0.5866** | - | **58.66%** |
| **Final Score** | **0.5870** | - | **58.70%** |

**Top 5 Non-Self Matches:**

2. LCT Limousine & Chauffeur Transportation Show (Torrance, CA) (58.7%)
3. Lucky Limousine (Pearl City, HI) (58.4%)
4. Travel Concierge (Toronto, ON) (52.7%)
5. Concierge Detroit (Detroit, MI) (51.9%)
6. Corporate Concierge (Chicago, IL) (51.6%)

---

### 24. Danish Sisterhood of America

**Top Non-Self Match (Rank 2):** `Intentional Sisterhood Ministry` (62.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 0.8589 | 30% | 0.2577 |
| **Base Score** | **0.6271** | - | **62.71%** |
| **Final Score** | **0.6270** | - | **62.70%** |

**Top 5 Non-Self Matches:**

2. Intentional Sisterhood Ministry (62.7%)
3. Sisterhood of Congregation Beth Israel (West Hartford, CT) (55.0%)
4. Alliance America (53.6%)
5. Treat America (52.9%)
6. Echos of Sisterhood (Houston, TX) (52.8%)

---

### 25. Brooklyn Comics Club

**Top Non-Self Match (Rank 2):** `Cathedral Club of Brooklyn` (Brooklyn, NY) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.8122 | 30% | 0.2437 |
| **Base Score** | **0.8105** | - | **81.05%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Cathedral Club of Brooklyn (Brooklyn, NY) (90.0%)
3. Brooklyn Baseball Club (Yakima, WA) (90.0%)
4. South Brooklyn Running Club (Brooklyn, NY) (76.7%)
5. Brooklyn Arts (Brooklyn, NY) (63.6%)
6. Brooklyn Historical Society (Brooklyn, NY) (62.9%)

---

### 26. Global Interagency Security Forum

**Top Non-Self Match (Rank 2):** `Internet Security Systems` (61.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4545 | 70% | 0.3182 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6182** | - | **61.82%** |
| **Final Score** | **0.6180** | - | **61.80%** |

**Top 5 Non-Self Matches:**

2. Internet Security Systems (61.8%)
3. Network Processing Forum Headquarters (60.8%)
4. National Intergovernmental Audit Forum (59.6%)
5. Allied Universal Security Services (59.2%)
6. 360 Security Solutions (57.8%)

---

### 27. Lancet Software

**Top Non-Self Match (Rank 2):** `Lancet Technology` (Boston, MA) (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Lancet Technology (Boston, MA) (82.1%)
3. SOFTWARE DISTRIBUTOR (78.1%)
4. EPLAN Software (76.9%)
5. delivering software (76.7%)
6. Software Innovations (Barrie, ON) (75.5%)

---

### 28. Our Lady of the Lakes Catholic Church and School

**Top Non-Self Match (Rank 2):** `Our Lady Lake Church` (Sparta, NJ) (73.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6198 | 70% | 0.4339 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.7339** | - | **73.39%** |
| **Final Score** | **0.7340** | - | **73.40%** |

**Top 5 Non-Self Matches:**

2. Our Lady Lake Church (Sparta, NJ) (73.4%)
3. Our Lady of the Lake Church (Mandeville, LA) (66.7%)
4. St. Mark Catholic Church (Lake Mary, FL) (55.7%)
5. St. Martha Catholic School (53.9%)
6. Saint Augustine Catholic Church (Washington, DC) (53.0%)

---

### 29. Broadway Bound International

**Top Non-Self Match (Rank 2):** `Broadway Stages` (Brooklyn, NY) (65.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5038 | 70% | 0.3527 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6527** | - | **65.27%** |
| **Final Score** | **0.6530** | - | **65.30%** |

**Top 5 Non-Self Matches:**

2. Broadway Stages (Brooklyn, NY) (65.3%)
3. Broadway Inbound (Newark, Oh) (63.8%)
4. BROADWAY EN ESPANOL (West Hollywood, CA) (62.1%)
5. Broadway in Bronzeville (Chicago, IL) (62.1%)
6. Broadway Paranormal Society (60.0%)

---

### 30. E. H. Wachs

**Top Non-Self Match (Rank 2):** `Dog Show Production` (36.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.1333 | 70% | 0.0933 |
| Semantic Similarity | 0.9131 | 30% | 0.2739 |
| **Base Score** | **0.3673** | - | **36.73%** |
| **Final Score** | **0.3670** | - | **36.70%** |

**Top 5 Non-Self Matches:**

2. Dog Show Production (36.7%)
3. ORCHSE Strategies (Washington, DC) (36.6%)
4. Northwest Bearded Collie Club (Seattle, WA) (36.4%)
5. H.I.G. WhiteHorse (35.9%)
6. Wechsler 60th Anniversary (Weston, MA) (35.1%)

---

### 31. Marine Corps Fox 2/5

**Top Non-Self Match (Rank 2):** `US MARINE CORPS MOBILIZATION` (Chicago, IL) (79.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7792 | 70% | 0.5454 |
| Semantic Similarity | 0.8430 | 30% | 0.2529 |
| **Base Score** | **0.7983** | - | **79.83%** |
| **Final Score** | **0.7980** | - | **79.80%** |

**Top 5 Non-Self Matches:**

2. US MARINE CORPS MOBILIZATION (Chicago, IL) (79.8%)
3. Marine Corps Weatherman Reunion (79.7%)
4. Marine Corps Association (Washington, DC) (79.6%)
5. Marine Corps Heritage Foundation (Triangle, VA) (77.7%)
6. US Marine Corps Mustang Association (73.9%)

---

### 32. Fantasia Turistica

**Top Non-Self Match (Rank 2):** `Franquia Fantasia` (Sao Paulo, Sao Paulo) (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Franquia Fantasia (Sao Paulo, Sao Paulo) (82.1%)
3. Policia Zona Turistica (San Juan, PR) (70.1%)
4. Divani Turismo (43.3%)
5. ATLANTA TURISMO (40.1%)
6. Fernando Carranza (38.7%)

---

### 33. Esoterix

**Top Non-Self Match (Rank 2):** `Solutions Metrix` (47.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.2490 | 70% | 0.1743 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.4743** | - | **47.43%** |
| **Final Score** | **0.4740** | - | **47.40%** |

**Top 5 Non-Self Matches:**

2. Solutions Metrix (47.4%)
3. CONCENTRIX (Pleasanton, ) (40.2%)
4. Vectrix (Middletown, RI) (39.1%)
5. Nutral Metrix (37.3%)
6. Esse (36.3%)

---

### 34. Coker Group

**Top Non-Self Match (Rank 2):** `Coker Consultants` (Plano, TX) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.8104 | 30% | 0.2431 |
| **Base Score** | **0.8159** | - | **81.59%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Coker Consultants (Plano, TX) (90.0%)
3. Coker Tire (Chattanooga, TN) (90.0%)
4. Friends of Leslie Coker (Indian Head, MD) (74.3%)
5. COLAS Group (44.0%)
6. Cola (China, GA) (32.9%)

---

### 35. GILEAD IT

**Top Non-Self Match (Rank 2):** `Gilead Canada` (Westport, CT) (81.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.9843 | 30% | 0.2953 |
| **Base Score** | **0.8159** | - | **81.59%** |
| **Final Score** | **0.8160** | - | **81.60%** |

**Top 5 Non-Self Matches:**

2. Gilead Canada (Westport, CT) (81.6%)
3. Gilead Sciences Europe Ltd. (Vienna, ) (74.8%)
4. Gilead (Sao Paulo, SP) (74.1%)
5. GILEAD-HIV (JERSEY CITY, NJ) (73.8%)
6. Creadis (38.1%)

---

### 36. 4143 Affiliate INDA 2016

**Top Non-Self Match (Rank 2):** `1528 Affiliate AASHTO 2016` (New York, NY) (76.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8029 | 30% | 0.2409 |
| **Base Score** | **0.7615** | - | **76.15%** |
| **Final Score** | **0.7610** | - | **76.10%** |

**Top 5 Non-Self Matches:**

2. 1528 Affiliate AASHTO 2016 (New York, NY) (76.1%)
3. 4142 Affiliate AAN 2017 (MINNEAPOLIS, MN) (64.4%)
4. 01528 Affiliate Inta 2019 (New York, NY) (63.6%)
5. 1528 Affiliate ASCD 2018 (Dallas, TX) (63.0%)
6. 01528 Affiliate Inta 2019 (Omaha, NE) (62.8%)

---

### 37. Pipe and Plant Solutions

**Top Non-Self Match (Rank 2):** `Advanced Pipe Solutions` (Findlay, OH) (79.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 0.8283 | 30% | 0.2485 |
| **Base Score** | **0.7895** | - | **78.95%** |
| **Final Score** | **0.7900** | - | **79.00%** |

**Top 5 Non-Self Matches:**

2. Advanced Pipe Solutions (Findlay, OH) (79.0%)
3. Plant Operations (65.3%)
4. PLANT (62.3%)
5. Evergreen Tank Solutions (58.4%)
6. Texas Pipe and Supply (Houston, TX) (58.1%)

---

### 38. Stephen Rourke

**Top Non-Self Match (Rank 2):** `O'Rourke Darvin Wedding` (39.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.1327 | 70% | 0.0929 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.3929** | - | **39.29%** |
| **Final Score** | **0.3930** | - | **39.30%** |

**Top 5 Non-Self Matches:**

2. O'Rourke Darvin Wedding (39.3%)
3. John Prine (34.4%)
4. Terry McPherson (34.2%)
5. Nick Brooks (Austin, TX) (34.2%)
6. Reneé Perry (33.8%)

---

### 39. MIT Initiative on the Digital Economy

**Top Non-Self Match (Rank 2):** `Digital Health World Congress` (41.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.2078 | 70% | 0.1455 |
| Semantic Similarity | 0.9021 | 30% | 0.2706 |
| **Base Score** | **0.4161** | - | **41.61%** |
| **Final Score** | **0.4160** | - | **41.60%** |

**Top 5 Non-Self Matches:**

2. Digital Health World Congress (41.6%)
3. Boston Institute for Developing Economies (Washington, DC) (38.5%)
4. PW Mitra Technology (Milwaukee, WI) (38.1%)
5. MIT -MASSACHUSETTS INSTITUTE OF TECHNOLOGY (Cambridge, MA) (37.5%)
6. Liberty International GmbH (Berlin, ) (37.3%)

---

### 40. Urx Community USA

**Top Non-Self Match (Rank 2):** `Inxpress USA` (64.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5038 | 70% | 0.3527 |
| Semantic Similarity | 0.9811 | 30% | 0.2943 |
| **Base Score** | **0.6470** | - | **64.70%** |
| **Final Score** | **0.6470** | - | **64.70%** |

**Top 5 Non-Self Matches:**

2. Inxpress USA (64.7%)
3. Sea Mar Community Health Centers USA (64.6%)
4. Forum USA (64.1%)
5. United Community Center (62.2%)
6. USA Vacation (Flushing, NY) (61.2%)

---

### 41. Spredfast Engage

**Top Non-Self Match (Rank 2):** `Interfast B.V.` (44.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.2000 | 70% | 0.1400 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.4400** | - | **44.00%** |
| **Final Score** | **0.4400** | - | **44.00%** |

**Top 5 Non-Self Matches:**

2. Interfast B.V. (44.0%)
3. Fast Pace, Incorporated (33.8%)
4. Joy Speede (33.8%)
5. Accelerate Performance (33.0%)
6. Fastbraces (32.0%)

---

### 42. City of Dallas-Parks & Recreation

**Top Non-Self Match (Rank 2):** `Dallas Parks and Recreation` (Houston, TX) (83.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7670 | 70% | 0.5369 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8369** | - | **83.69%** |
| **Final Score** | **0.8370** | - | **83.70%** |

**Top 5 Non-Self Matches:**

2. Dallas Parks and Recreation (Houston, TX) (83.7%)
3. Marion County Parks & Recreation (73.0%)
4. Parks and Recreation Ontario Canada (73.0%)
5. City Year Dallas (Dallas, TX) (72.7%)
6. Skyline High School Dallas (Dallas, TX) (58.2%)

---

### 43. Kai Pono Builders, Inc.

**Top Non-Self Match (Rank 2):** `Kai Pono Builders` (Honolulu, HI) (95.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9375 | 70% | 0.6562 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.9563** | - | **95.62%** |
| **Final Score** | **0.9560** | - | **95.60%** |

**Top 5 Non-Self Matches:**

2. Kai Pono Builders (Honolulu, HI) (95.6%)
3. Construction Builders Association (54.6%)
4. Kingdom Builders US, Inc (53.8%)
5. Indiana Builders Association (Indianapolis, IN) (52.8%)
6. Credit Builders Alliance (Brea, ) (50.8%)

---

### 44. MUSICFIRST COALITION

**Top Non-Self Match (Rank 2):** `NYS CDFI Coalition` (72.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.8347 | 30% | 0.2504 |
| **Base Score** | **0.7237** | - | **72.37%** |
| **Final Score** | **0.7240** | - | **72.40%** |

**Top 5 Non-Self Matches:**

2. NYS CDFI Coalition (72.4%)
3. CAEAR Coalition (72.1%)
4. Coalition Security Group (71.7%)
5. Melodic Connections (43.4%)
6. UP Music (41.2%)

---

### 45. Frontier Power Products

**Top Non-Self Match (Rank 2):** `Power Up DC` (64.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 0.9095 | 30% | 0.2729 |
| **Base Score** | **0.6423** | - | **64.23%** |
| **Final Score** | **0.6420** | - | **64.20%** |

**Top 5 Non-Self Matches:**

2. Power Up DC (64.2%)
3. Golden Frontier (63.6%)
4. ADVANCED POWER TECHNOLOGIES LLC (62.8%)
5. POWER SUPPLIES UTILITIES (SCOTTSDALE, AZ) (62.1%)
6. Terra Gen Power (Reno, NV) (60.9%)

---

### 46. 1960

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `1973` (Baltimore, MD) (41.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.3000 | 70% | 0.2100 |
| Semantic Similarity | 0.6830 | 30% | 0.2049 |
| **Base Score** | **0.4149** | - | **41.49%** |
| **Final Score** | **0.4150** | - | **41.50%** |

**Top 5 Non-Self Matches:**

2. 1973 (Baltimore, MD) (41.5%)
3. Trend (30.0%)
4. Legacy (28.7%)
5. Annual 2011 (27.0%)
6. Generations Humanitarian (26.5%)

---

### 47. Pacific Northwest Diabetes Research Inst

**Top Non-Self Match (Rank 2):** `Sansum Diabetes Research Institute` (Santa Barbara, CA) (65.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5100 | 70% | 0.3570 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6570** | - | **65.70%** |
| **Final Score** | **0.6570** | - | **65.70%** |

**Top 5 Non-Self Matches:**

2. Sansum Diabetes Research Institute (Santa Barbara, CA) (65.7%)
3. The Diabetes Research Foundation (Aventura, FL) (61.7%)
4. Behavioral Diabetes Inst (San Diego, CA) (61.3%)
5. Foundation for Diabetes Research (Liviingston, NJ) (58.9%)
6. Pacific Northwest ISA (Surrey, BC) (57.1%)

---

### 48. Mentors & Mentees

**Top Non-Self Match (Rank 2):** `Mentors, Incorporated` (69.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6300 | 70% | 0.4410 |
| Semantic Similarity | 0.8566 | 30% | 0.2570 |
| **Base Score** | **0.6980** | - | **69.80%** |
| **Final Score** | **0.6980** | - | **69.80%** |

**Top 5 Non-Self Matches:**

2. Mentors, Incorporated (69.8%)
3. 3 Mentors Big Event (69.7%)
4. College Mentors team, LLC (Miami, FL) (66.4%)
5. Mentor Grp (49.5%)
6. MOBE Mentoring (46.5%)

---

### 49. NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46

**Top Non-Self Match (Rank 2):** `ESH Meeting Fall 2022 M01661780925905 08-29-22 09:48:51` (40.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.1882 | 70% | 0.1317 |
| Semantic Similarity | 0.9108 | 30% | 0.2732 |
| **Base Score** | **0.4049** | - | **40.49%** |
| **Final Score** | **0.4050** | - | **40.50%** |

**Top 5 Non-Self Matches:**

2. ESH Meeting Fall 2022 M01661780925905 08-29-22 09:48:51 (40.5%)
3. Best Practices EXPO & Conference M01719423071572 06-26-24 13:31:16 (40.3%)
4. TCOM Conference 2025 (37.5%)
5. Fall Annul Conference 2759585 (35.8%)
6. NACAA 2023 Fall Membership Meeting (35.2%)

---

### 50. Donnelley Work Session

**Top Non-Self Match (Rank 2):** `Planning Session` (63.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4798 | 70% | 0.3359 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6359** | - | **63.59%** |
| **Final Score** | **0.6360** | - | **63.60%** |

**Top 5 Non-Self Matches:**

2. Planning Session (63.6%)
3. Sales Leader Dialogue Session (55.3%)
4. R.R. Donnelley Logistics (Willowbrook, IL) (54.7%)
5. Pernell At Work Services (54.5%)
6. engagement session (Oldsmar, FL) (52.0%)

---

### 51. North Shore Senior Center

**Top Non-Self Match (Rank 2):** `North American Senior Benefits` (Rock HIll, ) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 0.7731 | 30% | 0.2319 |
| **Base Score** | **0.8269** | - | **82.69%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. North American Senior Benefits (Rock HIll, ) (90.0%)
3. North Beach Village Resorts (71.2%)
4. Senior Living Communities (Palm Harbor, FL) (67.3%)
5. LEADING AGE NORTH CAROLINA (Chapel Hill, NC) (65.2%)
6. Senior Care Centers (Dallas, TX) (64.9%)

---

### 52. Singles Who Like Food & Fun

**Top Non-Self Match (Rank 2):** `Singles Source` (45.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.2146 | 70% | 0.1502 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.4502** | - | **45.02%** |
| **Final Score** | **0.4500** | - | **45.00%** |

**Top 5 Non-Self Matches:**

2. Singles Source (45.0%)
3. Singapore Food Shows (40.1%)
4. Singles Ski (39.9%)
5. Catering Food for Thought (39.1%)
6. J Singles Florida Event (38.4%)

---

### 53. Zen Meetings & Events

**Top Non-Self Match (Rank 2):** `Global Events and Meetings` (83.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7729 | 70% | 0.5410 |
| Semantic Similarity | 0.9723 | 30% | 0.2917 |
| **Base Score** | **0.8327** | - | **83.27%** |
| **Final Score** | **0.8330** | - | **83.30%** |

**Top 5 Non-Self Matches:**

2. Global Events and Meetings (83.3%)
3. TVG Meetings & Events (81.8%)
4. FDG Meetings and Events (81.4%)
5. Meetings Plus Events (Narragansett, RI) (80.1%)
6. Meetings and Events Management (Humble, TX) (80.1%)

---

### 54. Chicago South Swim Club

**Top Non-Self Match (Rank 2):** `CHICAGO YACHT CLUB OF CHICAGO` (Birmingham, ) (79.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.7958** | - | **79.58%** |
| **Final Score** | **0.7960** | - | **79.60%** |

**Top 5 Non-Self Matches:**

2. CHICAGO YACHT CLUB OF CHICAGO (Birmingham, ) (79.6%)
3. Chicago NEW RICH CLUB (78.3%)
4. Chicago Film Club Meetup Group (Chicago, IL) (76.6%)
5. Maverick Swim Club (Naperville, IL) (76.5%)
6. Beachwood Bison Swim Club (Mansfield, OH) (76.1%)

---

### 55. Edna, Dabra@SAP.IO

**Top Non-Self Match (Rank 2):** `Viajes Edna S.A.` (70.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.7692 | 30% | 0.2308 |
| **Base Score** | **0.7041** | - | **70.41%** |
| **Final Score** | **0.7040** | - | **70.40%** |

**Top 5 Non-Self Matches:**

2. Viajes Edna S.A. (70.4%)
3. The Edna Lewis Foundation (67.5%)
4. Sapa AB (38.2%)
5. Red Sapiens (Canton, MA) (36.4%)
6. MEI Brazil (36.3%)

---

### 56. Boys and Girls Club of Dawson Community Centre

**Top Non-Self Match (Rank 2):** `Challengers Boys & Girls Club` (72.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6375 | 70% | 0.4463 |
| Semantic Similarity | 0.9403 | 30% | 0.2821 |
| **Base Score** | **0.7283** | - | **72.83%** |
| **Final Score** | **0.7280** | - | **72.80%** |

**Top 5 Non-Self Matches:**

2. Challengers Boys & Girls Club (72.8%)
3. Boys and Girls Club America (72.6%)
4. Westside Girls Gymnastics Parents Club (Tigard, OR) (58.5%)
5. Boyts & Girls Clubs of America (Nashville, TN) (41.1%)
6. APRIL DAWSON GROUP (WASHINGTON, DC) (40.2%)

---

### 57. Beissbarth

**Top Non-Self Match (Rank 2):** `50Barz` (41.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.1687 | 70% | 0.1181 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.4181** | - | **41.81%** |
| **Final Score** | **0.4180** | - | **41.80%** |

**Top 5 Non-Self Matches:**

2. 50Barz (41.8%)
3. Myers/Segebarth (Chicago, IL) (37.0%)
4. Yoni BarMitzvah (34.4%)
5. Cinnabar (Albany, NY) (33.3%)
6. Iess (32.4%)

---

### 58. US Night Vision

**Top Non-Self Match (Rank 2):** `Vision Source USA` (66.9%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6694** | - | **66.94%** |
| **Final Score** | **0.6690** | - | **66.90%** |

**Top 5 Non-Self Matches:**

2. Vision Source USA (66.9%)
3. VISION EVENTS INTERNATIONAL (64.3%)
4. World Vision USA (Federal Way, WA) (63.4%)
5. Vision Trends USA (Washington, DC) (63.3%)
6. Vision 33 (61.4%)

---

### 59. Amedysis, Incorporated

**Top Non-Self Match (Rank 2):** `TaxAnalysis` (Beaverton, OR) (36.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.2368 | 70% | 0.1658 |
| Semantic Similarity | 0.6675 | 30% | 0.2003 |
| **Base Score** | **0.3661** | - | **36.61%** |
| **Final Score** | **0.3660** | - | **36.60%** |

**Top 5 Non-Self Matches:**

2. TaxAnalysis (Beaverton, OR) (36.6%)
3. Amtech (36.1%)
4. IASED (36.1%)
5. Controversies In Dialysis Access (36.1%)
6. AMM (34.5%)

---

### 60. Taiyo Air Service Co.,Ltd

**Top Non-Self Match (Rank 2):** `Ryowa Diamond Air Service Co. Ltd(tokyo)` (Tokyo, ) (70.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7244 | 70% | 0.5071 |
| Semantic Similarity | 0.6441 | 30% | 0.1932 |
| **Base Score** | **0.7003** | - | **70.03%** |
| **Final Score** | **0.7000** | - | **70.00%** |

**Top 5 Non-Self Matches:**

2. Ryowa Diamond Air Service Co. Ltd(tokyo) (Tokyo, ) (70.0%)
3. TASC (69.0%)
4. Headquarters Pacific Air Force (Hickam Air Force Base, HI) (55.8%)
5. Medical Air Services Association (55.6%)
6. FLYING DRAGON TRAVEL SERVICE CO., LTD. (55.2%)

---

### 61. National Conference on Race & Ethnicity in American Higher E

**Top Non-Self Match (Rank 2):** `National Conference on Race Ethnicity` (Norman, OK) (69.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5625 | 70% | 0.3937 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6937** | - | **69.38%** |
| **Final Score** | **0.6940** | - | **69.40%** |

**Top 5 Non-Self Matches:**

2. National Conference on Race Ethnicity (Norman, OK) (69.4%)
3. National African American Caucus (Washington, DC) (33.2%)
4. Latin America Conference (29.6%)
5. African American Federal Executive Assoc (Washington, DC) (29.2%)
6. Amazing Race Toronto (27.6%)

---

### 62. Reminger Law Firm

**Top Non-Self Match (Rank 2):** `Weaver Law Firm` (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8668** | - | **86.68%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Weaver Law Firm (90.0%)
3. Cressman Law Firm (90.0%)
4. Lieben Law Firm (90.0%)
5. Trembly Law Firm (90.0%)
6. Smith Law Firm (Orange, TX) (90.0%)

---

### 63. SEMMOA BOD

**Top Non-Self Match (Rank 2):** `BODE` (41.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.1662 | 70% | 0.1163 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.4163** | - | **41.63%** |
| **Final Score** | **0.4160** | - | **41.60%** |

**Top 5 Non-Self Matches:**

2. BODE (41.6%)
3. Eroads (36.2%)
4. USS BORDELON (35.8%)
5. Verina Bols (35.4%)
6. SewPro (35.1%)

---

### 64. Telefonica Global Solutions

**Top Non-Self Match (Rank 2):** `TOUR HOUSE TELEFONICA` (70.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6333 | 70% | 0.4433 |
| Semantic Similarity | 0.8664 | 30% | 0.2599 |
| **Base Score** | **0.7033** | - | **70.33%** |
| **Final Score** | **0.7030** | - | **70.30%** |

**Top 5 Non-Self Matches:**

2. TOUR HOUSE TELEFONICA (70.3%)
3. Celulares Telefonica (68.8%)
4. Expert Global Solutions (67.5%)
5. Global Exchange (54.1%)
6. Trident Global Partners (52.6%)

---

### 65. Travel Leaders - Dube Travel

**Top Non-Self Match (Rank 2):** `Travel Leaders of Wausau` (Wausau, WI) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity | 0.6123 | 30% | 0.1837 |
| **Base Score** | **0.7505** | - | **75.05%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Travel Leaders of Wausau (Wausau, WI) (90.0%)
3. Travel Leaders - Spears Travel (Tulsa, OK) (90.0%)
4. Travel Leaders (82.5%)
5. Travel Leaders Group (Suwanee, GA) (74.1%)
6. Travel Leaders (Spokane, WA) (72.7%)

---

### 66. Hi- Tours

**Top Non-Self Match (Rank 2):** `HiFiveLive Tours` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. HiFiveLive Tours (82.1%)
3. Heavenly Tours (79.7%)
4. Grand Tours (79.6%)
5. Tal Tours (77.8%)
6. EA Tours (77.8%)

---

### 67. Volkswagen Group China

**Top Non-Self Match (Rank 2):** `China ARVO group` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. China ARVO group (82.1%)
3. China Tour Group (79.9%)
4. Bayer China (Beijing, Beijing) (78.9%)
5. GSO China (Guangzhou, Guangdong) (77.2%)
6. China Merchants Group Co, LTD (Shenzhen, Guangdong) (76.6%)

---

### 68. Sun Tx

**Top Non-Self Match (Rank 2):** `Sun Source` (79.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.9228 | 30% | 0.2768 |
| **Base Score** | **0.7975** | - | **79.75%** |
| **Final Score** | **0.7970** | - | **79.70%** |

**Top 5 Non-Self Matches:**

2. Sun Source (79.7%)
3. Sun Outdoors (Delaware, OH) (77.0%)
4. Sun Crowne (72.0%)
5. Sun Power Corporation (71.9%)
6. GLOBAL SUN AND SUN TRAVEL (Atlanta, GA) (65.6%)

---

### 69. Southern Vermont Deerfield Valley Chamber of commerce

**Top Non-Self Match (Rank 2):** `Greater Meadowlands Chamber of Commerce and CVB` (Rutherford, NJ) (57.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4678 | 70% | 0.3275 |
| Semantic Similarity | 0.8201 | 30% | 0.2460 |
| **Base Score** | **0.5735** | - | **57.35%** |
| **Final Score** | **0.5730** | - | **57.30%** |

**Top 5 Non-Self Matches:**

2. Greater Meadowlands Chamber of Commerce and CVB (Rutherford, NJ) (57.3%)
3. Washington Area Chamber of Commerce (57.2%)
4. Spruce Grove & District Chamber of Commerce (56.1%)
5. St Louis Chamber of Commerce (55.5%)
6. St. Louis Chamber Commerce (55.5%)

---

### 70. DGR Ministries

**Top Non-Self Match (Rank 2):** `Legacy Ministries` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Legacy Ministries (82.1%)
3. Arise Ministries (80.8%)
4. Able Ministries (80.5%)
5. ABSOLUTE Ministries (77.1%)
6. YFC Ministries (77.0%)

---

### 71. Impacto 6

**Top Non-Self Match (Rank 2):** `District 6` (59.9%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6375 | 70% | 0.4462 |
| Semantic Similarity | 0.5100 | 30% | 0.1530 |
| **Base Score** | **0.5993** | - | **59.93%** |
| **Final Score** | **0.5990** | - | **59.90%** |

**Top 5 Non-Self Matches:**

2. District 6 (59.9%)
3. Impact (50.2%)
4. Impact Events (45.5%)
5. Impact XM (Denver, CO) (41.9%)
6. Human Impact (40.2%)

---

### 72. Neos Therapeutics, Inc.

**Top Non-Self Match (Rank 2):** `Neos Therapeutics` (Trussville, AL) (90.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.9167 | 70% | 0.6417 |
| Semantic Similarity | 0.8890 | 30% | 0.2667 |
| **Base Score** | **0.9084** | - | **90.84%** |
| **Final Score** | **0.9080** | - | **90.80%** |

**Top 5 Non-Self Matches:**

2. Neos Therapeutics (Trussville, AL) (90.8%)
3. Iconic Therapeutics, Inc. (80.5%)
4. vTv Therapeutics Inc. (76.6%)
5. Practic Therapeutics (75.9%)
6. Seres Therapeutics, Inc. (Cambridge, MA) (73.5%)

---

### 73. International Tax Institute

**Top Non-Self Match (Rank 2):** `Tax Research Institute` (Angier, NC) (91.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8833 | 70% | 0.6183 |
| Semantic Similarity | 0.9729 | 30% | 0.2919 |
| **Base Score** | **0.9102** | - | **91.02%** |
| **Final Score** | **0.9100** | - | **91.00%** |

**Top 5 Non-Self Matches:**

2. Tax Research Institute (Angier, NC) (91.0%)
3. Sales Tax Institute (Chicago, IL) (90.0%)
4. National Tax Group (Greenacres, FL) (90.0%)
5. National Tax Lein (74.3%)
6. COUNCIL FOR INTERNATIONAL TAX EDUCATION CITE (White Plains, NY) (71.0%)

---

### 74. Mitsubishi M501G

**Top Non-Self Match (Rank 2):** `Mitsubishi Meeting` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Mitsubishi Meeting (82.1%)
3. Mitsubishi USA (Philadelphia, PA) (76.5%)
4. Mitsubishi Materials (Rolling Meadows, IL) (70.6%)
5. MITSUBISHI ELECTRIC CORPORATIO (Plymouth, MI) (69.2%)
6. Mitsubishi Digital Electronics (Irvine, CA) (67.8%)

---

### 75. Huskies Sports

**Top Non-Self Match (Rank 2):** `Sports One` (80.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.9415 | 30% | 0.2824 |
| **Base Score** | **0.8031** | - | **80.31%** |
| **Final Score** | **0.8030** | - | **80.30%** |

**Top 5 Non-Self Matches:**

2. Sports One (80.3%)
3. Sports Injuries (79.5%)
4. Hamilton Huskies 14U (Chandler, AZ) (77.3%)
5. Adventure Sports Inc (75.4%)
6. Cycling Sports Group (75.1%)

---

### 76. Acacia Pharma Group Inc.

**Top Non-Self Match (Rank 2):** `Quintilis Pharma Group` (79.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.7958** | - | **79.58%** |
| **Final Score** | **0.7960** | - | **79.60%** |

**Top 5 Non-Self Matches:**

2. Quintilis Pharma Group (79.6%)
3. RGR PHARMA LTD (78.6%)
4. Vanguard Pharma (77.7%)
5. Kurative Pharma (75.3%)
6. Gennium Pharma (Vancouver, BC) (74.9%)

---

### 77. Acumatica Summit 2017 Z7NWPDKS625

**Top Non-Self Match (Rank 2):** `Healthcare Summit` (49.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4167 | 70% | 0.2917 |
| Semantic Similarity | 0.6783 | 30% | 0.2035 |
| **Base Score** | **0.4951** | - | **49.51%** |
| **Final Score** | **0.4950** | - | **49.50%** |

**Top 5 Non-Self Matches:**

2. Healthcare Summit (49.5%)
3. Elite Dental Summit (49.3%)
4. Health IT Summit (48.8%)
5. Health Reform Summit (48.3%)
6. HCA Trauma Summit Aug2017 NPN8CY36RQ4 42856 (Alpharetta, GA) (46.1%)

---

### 78. Linklaters CIS

**Top Non-Self Match (Rank 2):** `Linklaters LLP` (Brussels, ) (74.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.7434 | 30% | 0.2230 |
| **Base Score** | **0.7436** | - | **74.36%** |
| **Final Score** | **0.7440** | - | **74.40%** |

**Top 5 Non-Self Matches:**

2. Linklaters LLP (Brussels, ) (74.4%)
3. CIS Group (Southlake, TX) (64.8%)
4. Linklaters (Saginaw, MI) (64.8%)
5. 2016 CIS Annual Conference 2196533 (59.5%)
6. Theresa Cisneros (42.9%)

---

### 79. Christian Girls Family Ministry

**Top Non-Self Match (Rank 2):** `Christian Family Fellowship Church` (Los Angeles, CA) (76.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8102 | 30% | 0.2431 |
| **Base Score** | **0.7637** | - | **76.37%** |
| **Final Score** | **0.7640** | - | **76.40%** |

**Top 5 Non-Self Matches:**

2. Christian Family Fellowship Church (Los Angeles, CA) (76.4%)
3. Family of Faith Christian Ministries (Beaumont, TX) (74.9%)
4. Family Christian Church (La Palma, CA) (70.1%)
5. Holy Family Middle School (62.0%)
6. Community Christian Church (61.0%)

---

### 80. Alosa Foundation

**Top Non-Self Match (Rank 2):** `Step Foundation` (74.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6375 | 70% | 0.4462 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.7462** | - | **74.62%** |
| **Final Score** | **0.7460** | - | **74.60%** |

**Top 5 Non-Self Matches:**

2. Step Foundation (74.6%)
3. ACC Foundation (74.1%)
4. SAM Foundation (72.4%)
5. Openstack Foundation (72.2%)
6. ESA Foundation (71.8%)

---

### 81. La Chaine des Rotisseurs Wine Club of Newport Beach

**Top Non-Self Match (Rank 2):** `CHAINE DES ROTISSEURS DINNER` (San Diego, CA) (52.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.3814 | 70% | 0.2670 |
| Semantic Similarity | 0.8529 | 30% | 0.2559 |
| **Base Score** | **0.5228** | - | **52.28%** |
| **Final Score** | **0.5230** | - | **52.30%** |

**Top 5 Non-Self Matches:**

2. CHAINE DES ROTISSEURS DINNER (San Diego, CA) (52.3%)
3. Queen Wine Club (46.5%)
4. Vero Beach Art Club (44.7%)
5. Johnathan Wine Club (41.7%)
6. Wine (32.0%)

---

### 82. Sumner & Ryan, LLC

**Top Non-Self Match (Rank 2):** `Sumner 360` (Washington, DC) (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Sumner 360 (Washington, DC) (82.1%)
3. Ryan Lei (75.5%)
4. Ryan Wallace (74.5%)
5. Ryan Ybanez (73.6%)
6. Ryan LLC (McKinney, TX) (70.6%)

---

### 83. Tilt Creative & Production

**Top Non-Self Match (Rank 2):** `Cap Creative` (63.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4798 | 70% | 0.3359 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.6359** | - | **63.59%** |
| **Final Score** | **0.6360** | - | **63.60%** |

**Top 5 Non-Self Matches:**

2. Cap Creative (63.6%)
3. Sew Creative Lounge (59.3%)
4. CREATIVE TRAVEL (59.0%)
5. Zap Creative (57.2%)
6. Vancouver Creative Fun (57.2%)

---

### 84. Cerberus Capital

**Top Non-Self Match (Rank 2):** `Timber Capital` (72.6%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.6830 | 30% | 0.2049 |
| **Base Score** | **0.7255** | - | **72.55%** |
| **Final Score** | **0.7260** | - | **72.60%** |

**Top 5 Non-Self Matches:**

2. Timber Capital (72.6%)
3. MERCK CAPITAL LIMITED (72.4%)
4. Hennessy Capital (71.8%)
5. C5 Capital (71.8%)
6. Janus Capital (Denver, CO) (71.6%)

---

### 85. Institute of Health Technology Transformation

**Top Non-Self Match (Rank 2):** `Health Technology Assessment International` (Edmonton, AB) (76.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8068 | 30% | 0.2420 |
| **Base Score** | **0.7627** | - | **76.27%** |
| **Final Score** | **0.7630** | - | **76.30%** |

**Top 5 Non-Self Matches:**

2. Health Technology Assessment International (Edmonton, AB) (76.3%)
3. Personal Transformation Institute (74.8%)
4. Integrated Health Services (58.6%)
5. Digital Health World Congress (58.2%)
6. Health Human Services Commission (57.1%)

---

### 86. The Jones Assembly

**Top Non-Self Match (Rank 2):** `Leslie Jones` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Leslie Jones (82.1%)
3. Jones Coaching (Washington, DC) (78.4%)
4. Dianne Jones (77.8%)
5. Cathie Jones (77.3%)
6. Jarell Jones (77.0%)

---

### 87. American Black Film Insitutute

**Top Non-Self Match (Rank 2):** `Black Maria Film Festival` (76.5%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 0.8148 | 30% | 0.2444 |
| **Base Score** | **0.7651** | - | **76.51%** |
| **Final Score** | **0.7650** | - | **76.50%** |

**Top 5 Non-Self Matches:**

2. Black Maria Film Festival (76.5%)
3. Black Women Film! Canada (Toronto, ON) (63.3%)
4. Black Portraiture (58.4%)
5. BLACK BUDDIES ENTERTAINMENT (56.4%)
6. Black boy art show (55.0%)

---

### 88. Berk Tek

**Top Non-Self Match (Rank 2):** `Paul Berk Travel` (Great Neck, NY) (71.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.8147 | 30% | 0.2444 |
| **Base Score** | **0.7177** | - | **71.77%** |
| **Final Score** | **0.7180** | - | **71.80%** |

**Top 5 Non-Self Matches:**

2. Paul Berk Travel (Great Neck, NY) (71.8%)
3. Andre Berkowitz (44.3%)
4. Berg Hansen (0105 Oslo, ) (38.3%)
5. Skytrak Travel Ltd (36.4%)
6. Berkowitz Bar Mitzvah (Dallas, TX) (35.7%)

---

### 89. Northbridge Travel

**Top Non-Self Match (Rank 2):** `Travel Bridge` (82.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity | 1.0000 | 30% | 0.3000 |
| **Base Score** | **0.8206** | - | **82.06%** |
| **Final Score** | **0.8210** | - | **82.10%** |

**Top 5 Non-Self Matches:**

2. Travel Bridge (82.1%)
3. Northbridge Communities (Burlington, MA) (76.1%)
4. Travel Connection (London, ) (72.8%)
5. Chambers Travel (London, ) (72.3%)
6. Form Travel (Dublin, Ireland) (72.2%)

---

### 90. Kohler 2024

**Top Non-Self Match (Rank 2):** `Kohler Company Accounting` (Kohler, WI) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8500 | 70% | 0.5950 |
| Semantic Similarity | 0.9940 | 30% | 0.2982 |
| **Base Score** | **0.8932** | - | **89.32%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Kohler Company Accounting (Kohler, WI) (90.0%)
3. Kohler Global Procurement (Kohler, WI) (82.5%)
4. Kohler Co. (Kohler, WI) (80.4%)
5. Kohler & Plumbers Supply (Kohler, WI) (80.0%)
6. Kohler Company Talent Sourcing (Kohler, WI) (79.7%)

---

### 91. Louisiana State University Swim

**Top Non-Self Match (Rank 2):** `Louisiana State University (LSU) Admissions` (Baton Rouge, LA) (76.3%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity | 0.7945 | 30% | 0.2384 |
| **Base Score** | **0.7634** | - | **76.34%** |
| **Final Score** | **0.7630** | - | **76.30%** |

**Top 5 Non-Self Matches:**

2. Louisiana State University (LSU) Admissions (Baton Rouge, LA) (76.3%)
3. Louisiana State Board of Regents (Baton Rouge, LA) (75.3%)
4. LOUISIANA STATE UNIV SYSTEM (Shreveport, LA) (73.8%)
5. Louisiana State Urological Society (NEW ORLEANS, LA) (73.6%)
6. Arkansas State University Alumni (State University, AR) (71.7%)

---

### 92. X DO NOT USE - FRANCIS PARKER SCHOOL

**Top Non-Self Match (Rank 2):** `St. Francis High School College Prep` (Clarksville, TN) (54.1%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.4468 | 70% | 0.3127 |
| Semantic Similarity | 0.7619 | 30% | 0.2286 |
| **Base Score** | **0.5413** | - | **54.13%** |
| **Final Score** | **0.5410** | - | **54.10%** |

**Top 5 Non-Self Matches:**

2. St. Francis High School College Prep (Clarksville, TN) (54.1%)
3. Saint Francis High School (Arlington, VA) (49.4%)
4. St Francis Xavier 6th Form College (47.0%)
5. ST FRANCIS XAVIER COLLEGE CHR (Saint Louis, MO) (38.5%)
6. Hyde School (38.3%)

---

### 93. Mitsubishi Motor Sales of America, Incorporated

**Top Non-Self Match (Rank 2):** `Balise Motor Sales` (68.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.6879 | 30% | 0.2064 |
| **Base Score** | **0.6797** | - | **67.97%** |
| **Final Score** | **0.6800** | - | **68.00%** |

**Top 5 Non-Self Matches:**

2. Balise Motor Sales (68.0%)
3. Mitsubishi Meeting (58.4%)
4. Mitsubishi USA (Philadelphia, PA) (57.0%)
5. Mitsubishi Electric UPS Intl (Cypress, CA) (56.9%)
6. Mitsubishi International Food Ingredients (Midland Park, NJ) (56.2%)

---

### 94. Energy Distribution Partners Holdings'

**Top Non-Self Match (Rank 2):** `Energy Distribution Partners Holdings L.P.` (Chicago, IL) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity | 0.7518 | 30% | 0.2255 |
| **Base Score** | **0.8030** | - | **80.30%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Energy Distribution Partners Holdings L.P. (Chicago, IL) (90.0%)
3. Energy Power Partners (75.1%)
4. Distribution Energy Financial Group (Bethesda, MD) (68.6%)
5. NGL Energy Partners (64.4%)
6. GulfTerra Energy Partners (Houston, TX) (62.5%)

---

### 95. ThinkAdvisor

**Top Non-Self Match (Rank 2):** `TripAdvisor` (Waltham, MA) (52.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.3522 | 70% | 0.2465 |
| Semantic Similarity | 0.9367 | 30% | 0.2810 |
| **Base Score** | **0.5275** | - | **52.75%** |
| **Final Score** | **0.5280** | - | **52.80%** |

**Top 5 Non-Self Matches:**

2. TripAdvisor (Waltham, MA) (52.8%)
3. ChannelAdvisor Corporation (Charlotte, NC) (49.1%)
4. Thinkrite (44.0%)
5. NileNevis Enterprises (41.2%)
6. Directtiva (38.6%)

---

### 96. Jump on it Outreach

**Top Non-Self Match (Rank 2):** `Santa Cruz Social Outreach` (64.7%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5250 | 70% | 0.3675 |
| Semantic Similarity | 0.9331 | 30% | 0.2799 |
| **Base Score** | **0.6474** | - | **64.74%** |
| **Final Score** | **0.6470** | - | **64.70%** |

**Top 5 Non-Self Matches:**

2. Santa Cruz Social Outreach (64.7%)
3. Solid Rock Outreach Ministry (61.7%)
4. Major Taylor Community Outreach (61.6%)
5. Outreach Ministries Group (Tucker, GA) (60.6%)
6. Colorado Outreach Exchange (Cherry Hills Village, CO) (60.0%)

---

### 97. The Association of Ringside Consultants (ARC)

**Top Non-Self Match (Rank 2):** `Association of Ringside Physicians` (Chicago, IL) (74.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6761 | 70% | 0.4733 |
| Semantic Similarity | 0.8941 | 30% | 0.2682 |
| **Base Score** | **0.7415** | - | **74.15%** |
| **Final Score** | **0.7420** | - | **74.20%** |

**Top 5 Non-Self Matches:**

2. Association of Ringside Physicians (Chicago, IL) (74.2%)
3. Financial Insurance Consultants (59.9%)
4. RBC CONSULTANTS (56.5%)
5. Well Group Consultants (55.1%)
6. Educational Consultants Consortium (54.9%)

---

### 98. SFA HASA

**Top Non-Self Match (Rank 2):** `SFS Group` (42.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.1964 | 70% | 0.1375 |
| Semantic Similarity | 0.9493 | 30% | 0.2848 |
| **Base Score** | **0.4222** | - | **42.22%** |
| **Final Score** | **0.4220** | - | **42.20%** |

**Top 5 Non-Self Matches:**

2. SFS Group (42.2%)
3. STS USA (San Francisco, CA) (38.8%)
4. STS USA (San Francisco, CA) (38.8%)
5. SF Environment (38.6%)
6. AAAA (San Francisco, CA) (38.4%)

---

### 99. Grupo Duracell Ene 2025

**Top Non-Self Match (Rank 2):** `GRUPO Convenciones Y Eventos` (59.4%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 0.8129 | 30% | 0.2439 |
| **Base Score** | **0.5939** | - | **59.39%** |
| **Final Score** | **0.5940** | - | **59.40%** |

**Top 5 Non-Self Matches:**

2. GRUPO Convenciones Y Eventos (59.4%)
3. Grupo Cer (56.9%)
4. Convene 2025 (56.2%)
5. Grupo MM Eventos (55.9%)
6. 2025 SWCA Marketing Summit (55.1%)

---

### 100. World Association of Medical Law

**Top Non-Self Match (Rank 2):** `Medical Air Services Association` (73.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.6375 | 70% | 0.4462 |
| Semantic Similarity | 0.9457 | 30% | 0.2837 |
| **Base Score** | **0.7300** | - | **73.00%** |
| **Final Score** | **0.7300** | - | **73.00%** |

**Top 5 Non-Self Matches:**

2. Medical Air Services Association (73.0%)
3. Physicians Medical Association (Vancouver, WA) (67.7%)
4. Patient Medical Association, LLC (Frederick, ) (65.6%)
5. British Islamic Medical Association - BIMA (64.0%)
6. Alberta Medical Association (Edmonton, AB) (63.9%)

---

### 101. ABA

**Top Non-Self Match (Rank 2):** `Aba Travel` (Barcelona, ) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.7517 | 30% | 0.2255 |
| **Base Score** | **0.7983** | - | **79.83%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. Aba Travel (Barcelona, ) (90.0%)
3. American Bonsai Association (86.4%)
4. ARIZONA BUSINESS AVIATION (85.7%)
5. American Beverage Association (84.8%)
6. American Booksellers Association (84.6%)

---

### 102. PDMA

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `PDMA Alliance` (York, SC) (90.0%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.8182 | 70% | 0.5727 |
| Semantic Similarity | 0.9467 | 30% | 0.2840 |
| **Base Score** | **0.8567** | - | **85.67%** |
| **Final Score** | **0.9000** | - | **90.00%** |

**Top 5 Non-Self Matches:**

2. PDMA Alliance (York, SC) (90.0%)
3. PDA (Bethesda, MD) (56.2%)
4. PDAC (New York, NY) (48.1%)
5. ACMA (44.2%)
6. PDAC (Miraflores, Lima) (44.0%)

---

### 103. IBM

**Top Non-Self Match (Rank 2):** `International Bluegrass Music` (91.2%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 0.1201 | 30% | 0.0360 |
| **Base Score** | **0.3860** | - | **38.60%** |
| Acronym Fidelity Boost | 1.0000 | 15% max | +0.1500 |
| **Final Score** | **0.9120** | - | **91.20%** |

**Top 5 Non-Self Matches:**

2. International Bluegrass Music (91.2%)
3. IBM Systems (New York, NY) (90.0%)
4. Simpler An IBM Company (Chicago, IL) (90.0%)
5. Compose An IBM Company (Tulsa, OK) (90.0%)
6. IBM Analytics (Somers, NY) (90.0%)

---

### 104. GE

✅ **Exact Match (Self) Found & Filtered**

**Top Non-Self Match (Rank 2):** `Grupo Erictel` (91.8%)

**Score Breakdown for Best Non-Self Match:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| String Similarity | 0.5000 | 70% | 0.3500 |
| Semantic Similarity | 0.1846 | 30% | 0.0554 |
| **Base Score** | **0.4054** | - | **40.54%** |
| Acronym Fidelity Boost | 1.0000 | 15% max | +0.1500 |
| **Final Score** | **0.9180** | - | **91.80%** |

**Top 5 Non-Self Matches:**

2. Grupo Erictel (91.8%)
3. Garg Engagement (91.8%)
4. Griffin Enterprise (91.7%)
5. GMT Exploration (91.7%)
6. Gordon Enteprises (91.4%)

---

