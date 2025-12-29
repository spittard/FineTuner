Loading 1M sample...
Initializing CompanyMatcher with model: paraphrase-MiniLM-L3-v2...
      Loading embeddings from cache... [OK] (1.9s)
      Loading FAISS index from cache... [OK] (2.5s)
      Loading company names from cache... [OK] (2.5s)
      Verifying cache metadata... [OK] (0.0s)
Creating fast lookup sets for exact matching...
   Step 1/3: Creating lowercase lookup set...

   Lowercase set:   0%|                          | 0/1000000 [00:00<?, ?names/s]
   Lowercase set:  10%|8       | 102946/1000000 [00:00<00:00, 1029262.21names/s]
   Lowercase set:  21%|#6      | 205873/1000000 [00:00<00:00, 1023567.39names/s]
   Lowercase set:  32%|##5     | 317452/1000000 [00:00<00:00, 1065523.20names/s]
   Lowercase set:  42%|###3    | 424019/1000000 [00:00<00:00, 1019314.18names/s]
   Lowercase set:  53%|####2   | 533068/1000000 [00:00<00:00, 1044207.69names/s]
   Lowercase set:  64%|#####1  | 638047/1000000 [00:00<00:00, 1046047.20names/s]
   Lowercase set:  74%|#####9  | 742828/1000000 [00:00<00:00, 1031731.85names/s]
   Lowercase set:  85%|#######6 | 846143/1000000 [00:00<00:00, 918024.06names/s]
   Lowercase set:  95%|########5| 949033/1000000 [00:00<00:00, 949510.93names/s]
   Lowercase set: 100%|########| 1000000/1000000 [00:01<00:00, 994278.93names/s]
   Step 2/3: Creating reverse lookup dictionary...

   Reverse lookup:   0%|                         | 0/1000000 [00:00<?, ?names/s]
   Reverse lookup:   7%|6        | 67376/1000000 [00:00<00:01, 673720.49names/s]
   Reverse lookup:  13%|#       | 134749/1000000 [00:00<00:01, 659365.66names/s]
   Reverse lookup:  20%|#6      | 200707/1000000 [00:00<00:01, 607302.24names/s]
   Reverse lookup:  27%|##1     | 270276/1000000 [00:00<00:01, 640211.42names/s]
   Reverse lookup:  34%|##7     | 337931/1000000 [00:00<00:01, 652821.15names/s]
   Reverse lookup:  41%|###2    | 409121/1000000 [00:00<00:00, 672399.77names/s]
   Reverse lookup:  48%|###8    | 476631/1000000 [00:00<00:00, 583020.80names/s]
   Reverse lookup:  54%|####3   | 543550/1000000 [00:00<00:00, 607530.62names/s]
   Reverse lookup:  61%|####8   | 611281/1000000 [00:00<00:00, 627687.90names/s]
   Reverse lookup:  68%|#####4  | 680021/1000000 [00:01<00:00, 645128.70names/s]
   Reverse lookup:  75%|#####9  | 745544/1000000 [00:01<00:00, 647986.33names/s]
   Reverse lookup:  81%|######4 | 811055/1000000 [00:01<00:00, 646819.57names/s]
   Reverse lookup:  88%|####### | 876231/1000000 [00:01<00:00, 627376.10names/s]
   Reverse lookup:  94%|#######5| 939410/1000000 [00:01<00:00, 469947.18names/s]
   Step 3/3: Creating word-based lookup dictionary...
   Reverse lookup: 100%|#######| 1000000/1000000 [00:01<00:00, 592280.72names/s]

   Word lookup:   0%|                            | 0/1000000 [00:00<?, ?names/s]
   Word lookup:   0%|              | 4979/1000000 [00:00<00:19, 49786.49names/s]
   Word lookup:   2%|2            | 22425/1000000 [00:00<00:16, 60016.44names/s]
   Word lookup:   4%|5           | 43570/1000000 [00:00<00:09, 104019.01names/s]
   Word lookup:   6%|7           | 62109/1000000 [00:00<00:07, 127735.88names/s]
   Word lookup:   8%|#           | 83624/1000000 [00:00<00:05, 153524.73names/s]
   Word lookup:  11%|#1         | 105449/1000000 [00:00<00:05, 172714.09names/s]
   Word lookup:  13%|#3         | 126189/1000000 [00:00<00:04, 183031.64names/s]
   Word lookup:  15%|#6         | 147867/1000000 [00:00<00:04, 193094.53names/s]
   Word lookup:  17%|#8         | 168051/1000000 [00:01<00:04, 191043.71names/s]
   Word lookup:  19%|##         | 190116/1000000 [00:01<00:04, 199747.34names/s]
   Word lookup:  21%|##3        | 211348/1000000 [00:01<00:03, 203463.35names/s]
   Word lookup:  23%|##5        | 232266/1000000 [00:01<00:03, 205157.22names/s]
   Word lookup:  25%|###         | 253017/1000000 [00:02<00:09, 77483.03names/s]
   Word lookup:  27%|###2        | 274460/1000000 [00:02<00:07, 96392.35names/s]
   Word lookup:  30%|###2       | 296096/1000000 [00:02<00:06, 116157.49names/s]
   Word lookup:  32%|###4       | 317866/1000000 [00:02<00:05, 135502.12names/s]
   Word lookup:  34%|###7       | 338120/1000000 [00:02<00:04, 149425.86names/s]
   Word lookup:  36%|###9       | 358801/1000000 [00:02<00:03, 162683.87names/s]
   Word lookup:  38%|####1      | 379499/1000000 [00:02<00:03, 173730.67names/s]
   Word lookup:  40%|####3      | 399669/1000000 [00:02<00:03, 172734.17names/s]
   Word lookup:  42%|####6      | 418899/1000000 [00:02<00:03, 167060.53names/s]
   Word lookup:  44%|####8      | 437962/1000000 [00:03<00:03, 160073.36names/s]
   Word lookup:  46%|#####      | 455394/1000000 [00:03<00:03, 163727.35names/s]
   Word lookup:  47%|#####2     | 473481/1000000 [00:03<00:03, 168323.07names/s]
   Word lookup:  50%|#####4     | 495149/1000000 [00:03<00:02, 181717.67names/s]
   Word lookup:  51%|#####6     | 513846/1000000 [00:03<00:02, 180848.49names/s]
   Word lookup:  53%|#####8     | 532780/1000000 [00:03<00:02, 183269.34names/s]
   Word lookup:  55%|######     | 553155/1000000 [00:03<00:02, 189201.64names/s]
   Word lookup:  57%|######3    | 574140/1000000 [00:03<00:02, 195243.89names/s]
   Word lookup:  59%|######5    | 594466/1000000 [00:03<00:02, 197604.48names/s]
   Word lookup:  62%|######7    | 615868/1000000 [00:03<00:01, 202286.47names/s]
   Word lookup:  64%|#######    | 638419/1000000 [00:04<00:01, 208277.15names/s]
   Word lookup:  66%|#######2   | 659812/1000000 [00:04<00:01, 209952.98names/s]
   Word lookup:  68%|#######4   | 680851/1000000 [00:04<00:01, 209292.59names/s]
   Word lookup:  70%|#######7   | 702962/1000000 [00:04<00:01, 212808.50names/s]
   Word lookup:  72%|#######9   | 724267/1000000 [00:04<00:01, 208848.65names/s]
   Word lookup:  75%|########2  | 745958/1000000 [00:04<00:01, 211223.01names/s]
   Word lookup:  77%|########4  | 768156/1000000 [00:04<00:01, 213877.30names/s]
   Word lookup:  79%|########6  | 789564/1000000 [00:04<00:01, 210348.23names/s]
   Word lookup:  81%|########9  | 810919/1000000 [00:04<00:00, 210076.66names/s]
   Word lookup:  83%|#########1 | 831944/1000000 [00:04<00:00, 209499.91names/s]
   Word lookup:  85%|#########3 | 853577/1000000 [00:05<00:00, 211514.03names/s]
   Word lookup:  88%|#########6 | 876218/1000000 [00:05<00:00, 215103.14names/s]
   Word lookup:  90%|#########8 | 897937/1000000 [00:05<00:00, 215718.88names/s]
   Word lookup:  92%|##########1| 919515/1000000 [00:05<00:00, 213350.16names/s]
   Word lookup:  94%|##########3| 940859/1000000 [00:05<00:00, 207247.12names/s]
   Word lookup:  96%|##########5| 962706/1000000 [00:05<00:00, 210510.42names/s]
   Word lookup:  98%|##########8| 984723/1000000 [00:05<00:00, 213345.07names/s]
   Word lookup: 100%|##########| 1000000/1000000 [00:05<00:00, 174640.38names/s]
   [OK] Created fast lookup sets for 1,000,000 companies
   Cache loaded successfully: 8c2b0be73b9cd6b6... (total: 15.3s)
   Loaded 1,000,000 companies from cache
   Location data: Available (1,000,000 entries)
   [OK] Fast load successful! Loaded location-aware index from cache.
Traceback (most recent call last):
  File "E:\projects\FineTuner\FineTuner\tests\test_compact_format.py", line 21, in <module>
    from tests.generate_full_control_report import format_company_result
ModuleNotFoundError: No module named 'tests'
