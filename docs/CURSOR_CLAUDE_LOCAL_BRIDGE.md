# Cursor agent ↔ local Claude Code — how collaboration works

There is **no** supported way for two assistants to open a private chat channel. What **does** work on one machine:

## Pattern: file + CLI (`claude -p`)

1. **Cursor** (Composer/Agent) edits code, runs tools, and writes a focused handoff to **`bridge/TO_CLAUDE.md`**.
2. **`scripts/bridge_run_claude.ps1`** runs:
   - `claude -p <handoff>` in **print mode** (non-interactive, exits after one turn).
   - Appends output to **`bridge/FROM_CLAUDE.last.txt`**.
3. **Cursor** reads that file and merges results into the branch, or you paste the relevant part back into chat.

**Smoke test** (already verified in this repo):

```powershell
claude -p "Reply with exactly: BRIDGE_OK" --permission-mode acceptEdits
```

## Pattern: interactive Claude terminal

Same **`bridge/TO_CLAUDE.md`**: you (or Cursor) prepare it; **you** run `claude` in a terminal for multi-step work; when finished, paste the summary into **`bridge/FROM_CLAUDE.last.txt`** so Cursor’s next turn sees it.

## When to use which

| Use Cursor | Use local `claude` |
|------------|-------------------|
| Repo-wide search, patches, lints, pytest without subscription quirks | Long interactive sessions you already drive in terminal |
| Automated bridge script from Cursor terminal tool | One-shot `claude -p` via **`bridge_run_claude.ps1`** |

## Persistence

- **`bridge/README.md`** — operator instructions  
- **`.cursor/rules/plugging-closed-loop.mdc`** — reminds agents the bridge exists  
- **`bridge/TO_CLAUDE.md`** — **gitignored** (local scratch)  
- **`bridge/FROM_CLAUDE.last.txt`** — **gitignored** (may contain paths or snippets you do not want committed)

## Next improvements (optional)

- Add `scripts/bridge_prep_plugging_handoff.ps1` that concatenates `plugging_report_assessment_summary.md` + top 20 `high` rows from CSV into `TO_CLAUDE.md`.
- CI: **do not** call `claude -p` without secrets; keep bridge manual or opt-in env `RUN_CLAUDE_BRIDGE=1`.

## Windows / PowerShell note

`claude -p` emits a **stderr warning** about stdin. If a wrapper script uses `$ErrorActionPreference = 'Stop'`, PowerShell can treat that as a **terminating error** and exit before capturing output. **`scripts/bridge_run_claude.ps1`** runs the invoke under `Continue` for that reason.
