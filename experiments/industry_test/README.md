# Industry matching experiment (shadow, non-destructive)

Does not modify `plugging_matches.json`, `companies_with_location.json`, or `company_matcher_cache/`.

## Phase 1 (offline counterfactual)

1. Refresh frozen copy (optional): copy `plugging_matches.json` to `plugging_matches_frozen.json`.
2. `python fetch_row_industry_map.py` — builds `row_industry_map.json` (requires SQL).
3. `python analyze_industry_rerank.py` — writes `phase1_sensitivity.csv`, `phase1_rank_swaps.json`, `phase1_per_entry_default.csv`.

## Phase 2 (shadow index, default 50k rows in plan; smaller for quick test)

1. `python export_companies_sample.py --rows 50000` — writes `companies_sample_plain.json` and `companies_sample_embed.json`.
2. `python build_industry_shadow_index.py --input companies_sample_plain.json --cache-name shadow_plain`
3. `python build_industry_shadow_index.py --input companies_sample_embed.json --cache-name shadow_embed`
4. `python compare_phase2_shadow.py --limit 200` — writes `phase2_compare.csv`

Use `--rows 800` for a fast smoke test (as in repo development).
