# Cursor ↔ local Claude Code bridge

**Goal:** Cursor (this IDE agent) and **Claude Code in your terminal** collaborate on the same repo without magic cross-chat wiring.

## Roles

| Who | Best for |
|-----|-----------|
| **Cursor agent** | Broad edits, grep, multi-file refactors, running pytest, reading lints |
| **Claude Code (`claude`)** | Long autonomous terminal sessions, `claude -p` one-shot tasks, habits the user already runs in CLI |

## Protocol

1. **Cursor** writes the handoff into **`bridge/TO_CLAUDE.md`** (copy from `TO_CLAUDE.example.md` if missing).
2. Run **`scripts/bridge_run_claude.ps1`** from repo root (or Cursor runs it for you). That invokes:
   - `claude -p <contents>` with `--permission-mode acceptEdits`
   - Captures **stdout + stderr** to **`bridge/FROM_CLAUDE.last.txt`** with a timestamp header.
3. **Cursor** (or you) reads `FROM_CLAUDE.last.txt` and continues—paste summary back into Cursor chat if needed.

**Interactive Claude:** Instead of the script, open a terminal, `cd` to the repo, and paste the contents of `TO_CLAUDE.md` into `claude` (interactive). When done, paste Claude’s summary into `FROM_CLAUDE.last.txt` yourself so Cursor has a paper trail.

## Limits

- No live socket between two LLMs here—only **files + CLI**.
- `claude -p` uses your Anthropic quota; keep handoffs focused.
- Very long prompts: keep `TO_CLAUDE.md` under a few KB, or make it say “read `docs/CLAUDE_PLUGGING_CLOSED_LOOP.md` and execute section X” instead of pasting the whole doc.

## Files

| File | Tracked? | Purpose |
|------|----------|---------|
| `TO_CLAUDE.example.md` | yes | Template |
| `TO_CLAUDE.md` | gitignored | Current handoff |
| `FROM_CLAUDE.last.txt` | gitignored | Last `claude -p` capture |
