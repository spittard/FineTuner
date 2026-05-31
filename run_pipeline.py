#!/usr/bin/env python3
"""Mechanical pipeline runner for the scoring-audit iteration loop.

Runs in this fixed order:
  1. Rotate backups for the three SME artifacts (>=2 historical copies retained).
  2. Run the batch matcher against the live RPC index.
  3. Regenerate the SME markdown + CSV reports.
  4. Run audit_scoring.py (the deterministic pass/fail gate).
  5. Append a one-line summary to .audit_state.json and exit with the audit's
     exit code (0 = goal reached, non-zero = at least one gate failing).

Designed for a less-capable model to invoke as a single command:

    python run_pipeline.py             # full cycle
    python run_pipeline.py --skip-rematch   # just regen report + audit
    python run_pipeline.py --audit-only     # just audit on existing files

The script never deletes the most recent N backups (default N=3 -> at least 2
historical + current = 3 on disk). Older backups are pruned automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = [
    os.path.join(ROOT, "plugging_matches.json"),
    os.path.join(ROOT, "plugging_report.md"),
    os.path.join(ROOT, "plugging_report.csv"),
]
STATE_PATH = os.path.join(ROOT, ".audit_state.json")
LOG_PATH = os.path.join(ROOT, "pipeline_run.log")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _rotate_backups(path: str, keep: int) -> None:
    """Copy `path` to `path.bak_<utc_ts>` and prune older backups beyond `keep`.
    Always retains at least the most recent `keep` backups (default 3).
    """
    if not os.path.exists(path):
        _log(f"  rotate: skip (missing) {os.path.basename(path)}")
        return
    bak = f"{path}.bak_{_ts()}"
    shutil.copy2(path, bak)
    _log(f"  rotate: copied -> {os.path.basename(bak)}")
    base = os.path.basename(path)
    parent = os.path.dirname(path) or "."
    candidates = sorted(
        [
            os.path.join(parent, f)
            for f in os.listdir(parent)
            if f.startswith(base + ".bak_")
        ],
        reverse=True,
    )
    for old in candidates[keep:]:
        try:
            os.remove(old)
            _log(f"  rotate: pruned -> {os.path.basename(old)}")
        except Exception as exc:
            _log(f"  rotate: WARN could not prune {old}: {exc}")


def _run(cmd: list[str], step_name: str, timeout: int | None = None) -> int:
    _log(f"step: {step_name} -> {' '.join(cmd)}")
    started = time.time()
    try:
        cp = subprocess.run(cmd, cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        _log(f"step: {step_name} TIMEOUT after {timeout}s")
        return 124
    rc = cp.returncode
    _log(f"step: {step_name} exit={rc} elapsed={time.time() - started:.1f}s")
    return rc


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"iteration": 0, "history": []}


def _save_state(state: dict) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        _log(f"state: WARN could not save: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mechanical scoring-audit pipeline")
    parser.add_argument("--skip-rematch", action="store_true", help="skip match_plugging_records.py")
    parser.add_argument("--audit-only", action="store_true", help="only run audit_scoring.py on existing files")
    parser.add_argument("--keep", type=int, default=3, help="backups to retain per artifact (default 3 -> >=2 historical copies)")
    parser.add_argument("--rematch-timeout", type=int, default=14400, help="rematch timeout seconds (default 4h)")
    args = parser.parse_args()

    if args.keep < 3:
        _log(f"WARN --keep={args.keep} would not preserve 2 historical copies; clamping to 3")
        args.keep = 3

    state = _load_state()
    iteration = int(state.get("iteration", 0)) + 1
    started_at = datetime.now(timezone.utc).isoformat()
    _log(f"=== pipeline iteration {iteration} started at {started_at} ===")

    if not args.audit_only:
        _log("rotating backups...")
        for p in ARTIFACTS:
            _rotate_backups(p, keep=args.keep)

    if not args.skip_rematch and not args.audit_only:
        rc = _run(
            [sys.executable, "match_plugging_records.py", "--no-resume", "--checkpoint-every", "200"],
            "rematch",
            timeout=args.rematch_timeout,
        )
        if rc != 0:
            _log(f"FAIL rematch exit={rc}")
            state.setdefault("history", []).append({
                "iteration": iteration, "started_at": started_at,
                "stage": "rematch", "exit": rc, "audit_exit": None,
            })
            state["iteration"] = iteration
            _save_state(state)
            return rc

    if not args.audit_only:
        rc = _run(
            [sys.executable, os.path.join("tests", "generate_plugging_report.py")],
            "regen-report",
        )
        if rc != 0:
            _log(f"FAIL regen-report exit={rc}")
            state.setdefault("history", []).append({
                "iteration": iteration, "started_at": started_at,
                "stage": "regen", "exit": rc, "audit_exit": None,
            })
            state["iteration"] = iteration
            _save_state(state)
            return rc

    audit_rc = _run([sys.executable, "audit_scoring.py"], "audit")

    state.setdefault("history", []).append({
        "iteration": iteration,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "audit_exit": audit_rc,
    })
    state["iteration"] = iteration
    _save_state(state)

    _log(f"=== pipeline iteration {iteration} done; audit_exit={audit_rc} ===")
    return audit_rc


if __name__ == "__main__":
    sys.exit(main())
