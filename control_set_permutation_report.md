# Control Set Permutation Test Report
Generated: 2026-03-14 20:27:48
Companies: 20
Total variant queries: 175

## Summary by Variation Type

| Type | Same Top-1 | Different | % Robust |
|------|------------|-----------|----------|
| casing_lower | 14 | 6 | 70.0% |
| casing_title | 19 | 0 | 100.0% |
| casing_upper | 20 | 0 | 100.0% |
| spacing_inner | 19 | 0 | 100.0% |
| spacing_outer | 19 | 0 | 100.0% |
| suffix | 1 | 1 | 50.0% |
| typo_drop | 11 | 27 | 28.9% |
| typo_swap | 16 | 22 | 42.1% |

## Sample Details (first 15 companies)

### 1. NIH
**Baseline top-1:** 2017 NIH Regional Seminar (NIH/OER) - Outbound

- `NIH` (casing_upper) → 2017 NIH Regional Seminar (NIH/OER) - Outbound [OK]
- `nih` (casing_lower) → 2017 NIH Regional Seminar (NIH/OER) - Outbound [OK]

### 2. Ohio University
**Baseline top-1:** A.R.M. of Ohio

- `OHIO UNIVERSITY` (casing_upper) → A.R.M. of Ohio [OK]
- `ohio university` (casing_lower) → A.R.M. of Ohio [OK]
- `Ohio University` (casing_title) → A.R.M. of Ohio [OK]
- `Ohio Univeristy` (typo_swap) → A.R.M. of Ohio [OK]
- `Oho University` (typo_drop) → AAMCO University [DIFF]
- `Oiho University` (typo_swap) → AAMCO University [DIFF]
- `Ohio Universty` (typo_drop) → A.R.M. of Ohio [OK]
- `  Ohio University  ` (spacing_outer) → A.R.M. of Ohio [OK]
- `Ohio  University` (spacing_inner) → A.R.M. of Ohio [OK]

### 3. Western University
**Baseline top-1:** 8U Western State Champions

- `WESTERN UNIVERSITY` (casing_upper) → 8U Western State Champions [OK]
- `western university` (casing_lower) → 1 Midwestern University [DIFF]
- `Western University` (casing_title) → 8U Western State Champions [OK]
- `Western Uinversity` (typo_swap) → AAF Western Region [DIFF]
- `Western niversity` (typo_drop) → AAF Western Region [DIFF]
- `Western nUiversity` (typo_swap) → AAF Western Region [DIFF]
- `Westen University` (typo_drop) → *West Virginia University [DIFF]
- `  Western University  ` (spacing_outer) → 8U Western State Champions [OK]
- `Western  University` (spacing_inner) → 8U Western State Champions [OK]

### 4. Kruger Products
**Baseline top-1:** 3S Products

- `KRUGER PRODUCTS` (casing_upper) → 3S Products [OK]
- `kruger products` (casing_lower) → 3S Products [OK]
- `Kruger Products` (casing_title) → 3S Products [OK]
- `Kruger Produtcs` (typo_swap) → A Produtora Produções [DIFF]
- `Krger Products` (typo_drop) → !!Kendro Products [DIFF]
- `Kruger Prodcuts` (typo_swap) → 2K Productions Group [DIFF]
- `Kruger Produts` (typo_drop) → A Produtora Produções [DIFF]
- `  Kruger Products  ` (spacing_outer) → 3S Products [OK]
- `Kruger  Products` (spacing_inner) → 3S Products [OK]

### 5. Vision America
**Baseline top-1:** Vision 2000

- `VISION AMERICA` (casing_upper) → Vision 2000 [OK]
- `vision america` (casing_lower) → Vision 2000 [OK]
- `Vision America` (casing_title) → Vision 2000 [OK]
- `Vision Amreica` (typo_swap) → A Vision Experience [DIFF]
- `Viion America` (typo_drop) → ***Epson America [DIFF]
- `Vision Ameirca` (typo_swap) → Vision 2000 [OK]
- `Vision merica` (typo_drop) → A Vision Experience [DIFF]
- `  Vision America  ` (spacing_outer) → Vision 2000 [OK]
- `Vision  America` (spacing_inner) → Vision 2000 [OK]

### 6. PDMA Association
**Baseline top-1:** *Unknown Association

- `PDMA ASSOCIATION` (casing_upper) → *Unknown Association [OK]
- `pdma association` (casing_lower) → *Unknown Association [OK]
- `Pdma Association` (casing_title) → *Unknown Association [OK]
- `PMDA Association` (typo_swap) → *CPASNET Association [DIFF]
- `PMA Association` (typo_drop) → 1569 ICW 2019 Produce Marketing Association PMA [DIFF]
- `PDAM Association` (typo_swap) → *Unknown Association [OK]
- `PDMAAssociation` (typo_drop) → 3PLAssociation [DIFF]
- `  PDMA Association  ` (spacing_outer) → *Unknown Association [OK]
- `PDMA  Association` (spacing_inner) → *Unknown Association [OK]

### 7. Nicolas/Sanchez Wedding
**Baseline top-1:** 1- Nicolas Wedding

- `NICOLAS/SANCHEZ WEDDING` (casing_upper) → 1- Nicolas Wedding [OK]
- `nicolas/sanchez wedding` (casing_lower) → A. Cruz Wedding [DIFF]
- `Nicolas/Sanchez Wedding` (casing_title) → 1- Nicolas Wedding [OK]
- `Nicolas/aSnchez Wedding` (typo_swap) → 1- Nicolas Wedding [OK]
- `Nicolas/Sanchez Wdding` (typo_drop) → (Local/International) Rafy Sanchez [DIFF]
- `Nicolas/Sanchez Weddnig` (typo_swap) → 1- Nicolas Wedding [OK]
- `Ncolas/Sanchez Wedding` (typo_drop) → A. Cruz Wedding [DIFF]
- `  Nicolas/Sanchez Wedding  ` (spacing_outer) → 1- Nicolas Wedding [OK]
- `Nicolas/Sanchez  Wedding` (spacing_inner) → 1- Nicolas Wedding [OK]

### 8. Kehilat Ariel Synagogue
**Baseline top-1:** AA Synagogue

- `KEHILAT ARIEL SYNAGOGUE` (casing_upper) → AA Synagogue [OK]
- `kehilat ariel synagogue` (casing_lower) → AA Synagogue [OK]
- `Kehilat Ariel Synagogue` (casing_title) → AA Synagogue [OK]
- `Kehilat Ariel Synaoggue` (typo_swap) → AAHA Servco [DIFF]
- `KehilatAriel Synagogue` (typo_drop) → AA Synagogue [OK]
- `Kehilat Ariel Synagogeu` (typo_swap) → AA NOGUEIRA [DIFF]
- `Kehilat Ariel Synaogue` (typo_drop) → AA NOGUEIRA [DIFF]
- `  Kehilat Ariel Synagogue  ` (spacing_outer) → AA Synagogue [OK]
- `Kehilat  Ariel Synagogue` (spacing_inner) → AA Synagogue [OK]

### 9. Next Level Events
**Baseline top-1:** A Step Ahead Events

- `NEXT LEVEL EVENTS` (casing_upper) → A Step Ahead Events [OK]
- `next level events` (casing_lower) → A Step Ahead Events [OK]
- `Next Level Events` (casing_title) → A Step Ahead Events [OK]
- `Next Leevl Events` (typo_swap) → 5b Events [DIFF]
- `Next Levl Events` (typo_drop) → A Step Ahead Events [OK]
- `Next Levle Events` (typo_swap) → A Step Ahead Events [OK]
- `Next Leve Events` (typo_drop) → 2e Events [DIFF]
- `  Next Level Events  ` (spacing_outer) → A Step Ahead Events [OK]
- `Next  Level Events` (spacing_inner) → A Step Ahead Events [OK]

### 10. Site Foundation Golf Tournament
**Baseline top-1:** 7 Eleven Golf Tournament

- `SITE FOUNDATION GOLF TOURNAMENT` (casing_upper) → 7 Eleven Golf Tournament [OK]
- `site foundation golf tournament` (casing_lower) → 7 Eleven Golf Tournament [OK]
- `Site Foundation Golf Tournament` (casing_title) → 7 Eleven Golf Tournament [OK]
- `Site Foundation Golf Tournmaent` (typo_swap) → 1st Tee World Golf Foundation [DIFF]
- `Site Foundation Golf Tournamnt` (typo_drop) → 1st Tee World Golf Foundation [DIFF]
- `Stie Foundation Golf Tournament` (typo_swap) → 7 Eleven Golf Tournament [OK]
- `Site Foundation Golf Tourament` (typo_drop) → 1st Tee World Golf Foundation [DIFF]
- `  Site Foundation Golf Tournament  ` (spacing_outer) → 7 Eleven Golf Tournament [OK]
- `Site  Foundation Golf Tournament` (spacing_inner) → 7 Eleven Golf Tournament [OK]

### 11. Interim WG Meeting - BIER
**Baseline top-1:** AACP 2012 Interim Meeting

- `INTERIM WG MEETING - BIER` (casing_upper) → AACP 2012 Interim Meeting [OK]
- `interim wg meeting - bier` (casing_lower) → AACP 2012 Interim Meeting [OK]
- `Interim Wg Meeting - Bier` (casing_title) → AACP 2012 Interim Meeting [OK]
- `Interi mWG Meeting - BIER` (typo_swap) → A Meeting by Design - MT [DIFF]
- `Interim WG Meeting - BIR` (typo_drop) → AACP 2012 Interim Meeting [OK]
- `Interim WG Meeitng - BIER` (typo_swap) → 2025 FCLB Annual Meeitng [DIFF]
- `Interim WG eeting - BIER` (typo_drop) → AACP 2012 Interim Meeting [OK]
- `  Interim WG Meeting - BIER  ` (spacing_outer) → AACP 2012 Interim Meeting [OK]
- `Interim  WG Meeting - BIER` (spacing_inner) → AACP 2012 Interim Meeting [OK]

### 12. DermaQuest Inc
**Baseline top-1:** A.N. Deringer Inc.

- `DERMAQUEST INC` (casing_upper) → A.N. Deringer Inc. [OK]
- `dermaquest inc` (casing_lower) → A & R Corporation, Inc. [DIFF]
- `Dermaquest Inc` (casing_title) → A.N. Deringer Inc. [OK]
- `DermauQest Inc` (typo_swap) → A.N. Deringer Inc. [OK]
- `DeraQuest Inc` (typo_drop) → A.N. Deringer Inc. [OK]
- `DermQauest Inc` (typo_swap) → A.N. Deringer Inc. [OK]
- `DermaQuest In` (typo_drop) → **Fun in the Sun [DIFF]
- `DermaQuest Incorporated` (suffix) → A.N. Deringer, Incorporated [DIFF]
- `  DermaQuest Inc  ` (spacing_outer) → A.N. Deringer Inc. [OK]
- `DermaQuest  Inc` (spacing_inner) → A.N. Deringer Inc. [OK]

### 13. Ellwood Group Inc
**Baseline top-1:** 9Wood, Inc

- `ELLWOOD GROUP INC` (casing_upper) → 9Wood, Inc [OK]
- `ellwood group inc` (casing_lower) → 9Wood, Inc [OK]
- `Ellwood Group Inc` (casing_title) → 9Wood, Inc [OK]
- `Ellwoo dGroup Inc` (typo_swap) → 2018 Corporate Groups [DIFF]
- `Ellwod Group Inc` (typo_drop) → 1LoD Ltd [DIFF]
- `Elwlood Group Inc` (typo_swap) → 9Wood, Inc [OK]
- `Ellwood GroupInc` (typo_drop) → Woodland Group [DIFF]
- `Ellwood Group Incorporated` (suffix) → 9Wood, Inc [OK]
- `  Ellwood Group Inc  ` (spacing_outer) → 9Wood, Inc [OK]
- `Ellwood  Group Inc` (spacing_inner) → 9Wood, Inc [OK]

### 14. American Miniature Horse Registry
**Baseline top-1:** 2024 American Horse Publications GKN4DCLWW68

- `AMERICAN MINIATURE HORSE REGISTRY` (casing_upper) → 2024 American Horse Publications GKN4DCLWW68 [OK]
- `american miniature horse registry` (casing_lower) → 2024 American Horse Publications GKN4DCLWW68 [OK]
- `American Miniature Horse Registry` (casing_title) → 2024 American Horse Publications GKN4DCLWW68 [OK]
- `Amercian Miniature Horse Registry` (typo_swap) → 2024 American Horse Publications GKN4DCLWW68 [OK]
- `American Miniature Hors Registry` (typo_drop) → AAA-American Automotive Association [DIFF]
- `American Miniature Horse Regsitry` (typo_swap) → 2024 American Horse Publications GKN4DCLWW68 [OK]
- `American Miniature Hors Registry` (typo_drop) → AAA-American Automotive Association [DIFF]
- `  American Miniature Horse Registry  ` (spacing_outer) → 2024 American Horse Publications GKN4DCLWW68 [OK]
- `American  Miniature Horse Registry` (spacing_inner) → 2024 American Horse Publications GKN4DCLWW68 [OK]

### 15. YADA ENTERPRISES, INC
**Baseline top-1:** AADAP, Inc.

- `YADA ENTERPRISES, INC` (casing_upper) → AADAP, Inc. [OK]
- `yada enterprises, inc` (casing_lower) → A & D Company Limited [DIFF]
- `Yada Enterprises, Inc` (casing_title) → AADAP, Inc. [OK]
- `YADA ENTEPRRISES, INC` (typo_swap) → ***JPS Emterprises, Inc. [DIFF]
- `YAA ENTERPRISES, INC` (typo_drop) → A1A, Inc. [DIFF]
- `YADA ENTERPRISE,S INC` (typo_swap) → AADAP, Inc. [OK]
- `YADA ENTERPRISES, NC` (typo_drop) → A Duda & Sons Inc [DIFF]
- `  YADA ENTERPRISES, INC  ` (spacing_outer) → AADAP, Inc. [OK]
- `YADA  ENTERPRISES, INC` (spacing_inner) → AADAP, Inc. [OK]
