# Paste into Claude (terminal) — handoff from Cursor agent

You are continuing work in the same repo as the Cursor agent. **Do not** ask the human to run the long pipeline unless RPC is down; **you** execute it when possible.

## Repo

`e:\projects\FineTuner\FineTuner` (or your local path — `cd` there first).

## Mandatory doc (plugging / matcher / report)

Read and follow **`docs/CLAUDE_PLUGGING_CLOSED_LOOP.md`** end-to-end.

## Cursor rules already in repo

- `.cursor/rules/plugging-closed-loop.mdc` — always-on plugging loop reminder  
- `.cursor/rules/agent-accountability.mdc` — no false “full SME reviewed 3k rows” claims  

## Quick command block (after RPC + cache load; restart RPC if `matcher.py` changed)

```powershell
Set-Location e:\projects\FineTuner\FineTuner
python scripts/assess_plugging_report_full.py
python scripts/export_plugging_egregious_cases.py --max-cases 120
$env:RUN_PLUGGING_REGRESSION="1"
python -m pytest tests/test_plugging_egregious_regression.py -q --tb=short
# Then rotate plugging_matches.json / reports, then:
python match_plugging_records.py --no-resume
python tests/generate_plugging_report.py
python scripts/assess_plugging_report_full.py
python audit_scoring.py
python tests/verify_control_set.py
```

## What to report back (in this chat or to the human)

Paths written, exit codes, before/after counts from `plugging_report_assessment_summary.md`, and explicit gaps—not “looks good.”

---

*This file exists so the human can copy the block above into a Claude Code terminal session; two agents cannot literally DM each other from here.*
