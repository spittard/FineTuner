#!/usr/bin/env python3
"""
Shadow A/B for the narrowed city==US-state guard (_sanitize_candidate_loc).

Blast radius = plugging records whose BASELINE top-K contains a candidate whose city
literally echoes a US state (e.g. "UT, UT"). Only those can change. We re-run just those
queries through the live RPC (which now runs the fixed guard) and diff vs baseline.

NON-DESTRUCTIVE: reads plugging_matches.json (baseline) + RPC; writes one CSV.
Requires the cache RPC server running with the CURRENT matcher.py.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from finetuner.utils.text_preprocessor import TextPreprocessor as TP


def is_us_state_token(s: str) -> bool:
    n = TP.normalize_state(s) if s else ""
    return len(n) == 2 and n in TP.STATE_ABBREV


def is_echo(m: dict) -> bool:
    c = str(m.get("city") or "").strip()
    st = str(m.get("state") or "").strip()
    return bool(c) and bool(st) and c.casefold() == st.casefold() and is_us_state_token(c)


def nkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def main():
    from finetuner.core.cache_rpc import connect, is_server_running

    if not is_server_running():
        print("ERROR: needs the cache RPC server running with current matcher.py", file=sys.stderr)
        sys.exit(2)
    server = connect()

    data = json.load(open(os.path.join(ROOT, "plugging_matches.json"), encoding="utf-8"))
    affected = [r for r in data if any(is_echo(m) for m in (r.get("matches") or []))]
    print(f"Baseline records: {len(data):,}")
    print(f"Blast radius (top-K contains a US-state-echo candidate): {len(affected)}")

    out = os.path.join(ROOT, "guard_blast_radius_ab.csv")
    fields = ["row_id", "query", "geo", "base_top1", "base_top1_score",
              "new_top1", "new_top1_score", "top1_changed",
              "echo_base_ranks", "echo_cleaned_in_new", "echo_still_present_new"]
    top1_changed = echoes_cleaned = echo_still = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in affected:
            base = rec.get("matches") or []
            q = rec.get("query_company") or ""
            qc = (rec.get("query_city") or "").strip() or None
            qs = (rec.get("query_state") or "").strip() or None
            res = server.search(q, top_k=max(len(base), 10), city=qc, state=qs)
            new = res.get("results", []) if isinstance(res, dict) else []

            b1 = base[0] if base else {}
            n1 = new[0] if new else {}
            changed = nkey(b1.get("name")) != nkey(n1.get("name"))
            if changed:
                top1_changed += 1

            echo_ranks = [i + 1 for i, m in enumerate(base) if is_echo(m)]
            echo_ids = {int(m["id"]) for m in base if is_echo(m) and m.get("id") is not None}
            # In the new result, are those same ids now cleaned (city blanked) or still echoing?
            cleaned = still = False
            for m in new:
                mid = m.get("id")
                if mid is not None and int(mid) in echo_ids:
                    c = str(m.get("city") or "").strip()
                    st = str(m.get("state") or "").strip()
                    if not c:
                        cleaned = True
                    elif c.casefold() == st.casefold():
                        still = True
            if cleaned:
                echoes_cleaned += 1
            if still:
                echo_still += 1

            w.writerow({
                "row_id": rec.get("row_id", ""),
                "query": q[:80],
                "geo": f"{qc or ''},{qs or ''}",
                "base_top1": (b1.get("name") or "")[:60],
                "base_top1_score": f"{float(b1.get('score') or 0):.4f}",
                "new_top1": (n1.get("name") or "")[:60],
                "new_top1_score": f"{float(n1.get('score') or 0):.4f}",
                "top1_changed": int(changed),
                "echo_base_ranks": ";".join(map(str, echo_ranks)),
                "echo_cleaned_in_new": int(cleaned),
                "echo_still_present_new": int(still),
            })

    n = len(affected) or 1
    print(f"\nWrote {out}")
    print("=" * 60)
    print(f"Top-1 changed:                 {top1_changed}/{len(affected)}")
    print(f"Echo candidate CLEANED in new: {echoes_cleaned}/{len(affected)}")
    print(f"Echo candidate STILL echoing:  {echo_still}/{len(affected)}  (should be 0)")


if __name__ == "__main__":
    main()
