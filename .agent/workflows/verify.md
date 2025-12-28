---
description: Verify the current matching logic against the control set
---

// turbo-all
1. Run the control set verification:
   ```powershell
   python tests/verify_control_set.py
   ```
2. Verify that there are no regressions in the scores.
3. If you changed the scoring logic, generate the ultra report to see the detailed impact:
   ```powershell
   python tests/generate_ultra_report.py
   ```
4. Review `control_set_report_ULTRA.md` for any unexpected changes in rationales or penalties.
