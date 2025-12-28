---
description: Add a new test case to the control set
---

1. Open `companies_control_set.json`.
2. Add a new entry to the list:
   ```json
   {
     "Company Name": "The New Query Company",
     "Expected Match": "The Target Name in Database",
     "Scenario": "Brief description of what this tests (e.g., Typo, Acronym)"
   }
   ```
3. Run the verification:
   ```powershell
   python tests/verify_control_set.py
   ```
4. Check `companies_control_set_results.md` to ensure the new case behaves as expected.
