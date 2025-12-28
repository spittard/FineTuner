---
description: Debug a specific company match interactively
---

// turbo-all
1. Start the interactive matcher:
   ```powershell
   python company_matcher_interactive.py companies.json
   ```
2. Enter the company name you want to test.
3. Review the ranking, scores (String vs Semantic), and the matching rationale.
4. If the results are unexpected, check:
   - Acronym fidelity score.
   - Distinctive word penalties.
   - Category mismatch logic.
