#!/usr/bin/env python3
"""
Build a small JSON fixture of the worst plugging rows for fast RPC regression
before a full rematch (Phase B in docs/CLAUDE_PLUGGING_CLOSED_LOOP.md).

Reads:
  plugging_report_assessment.csv  (from assess_plugging_report_full.py)
  plugging_records.json           (source queries; join on row_id == ID)

Writes:
  tests/fixtures/plugging_egregious_cases.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ASSESS = os.path.join(ROOT, "plugging_report_assessment.csv")
DEFAULT_RECORDS = os.path.join(ROOT, "plugging_records.json")
DEFAULT_OUT = os.path.join(ROOT, "tests", "fixtures", "plugging_egregious_cases.json")

PRIORITY_CODES = (
    "NO_QUERY_GEO_BARE_ROW_NOT_FIRST",
    "GATE_A_TOP5_TIE_CLUSTER",
    "HIGH_SCORE_NON_EXACT_NAME",
    "NO_QUERY_GEO_RANK1_HAS_LOCATION",
    "HYBRID_SEMANTIC_LED_AT_CEILING",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assessment", default=DEFAULT_ASSESS)
    ap.add_argument("--records", default=DEFAULT_RECORDS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-cases", type=int, default=120)
    args = ap.parse_args()

    if not os.path.isfile(args.assessment):
        print(f"ERROR: missing {args.assessment}; run assess_plugging_report_full.py first", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.records):
        print(f"ERROR: missing {args.records}", file=sys.stderr)
        sys.exit(1)

    with open(args.records, encoding="utf-8") as f:
        records = json.load(f)
    by_id = {str(r.get("ID")): r for r in records if r.get("ID") is not None}

    scored: list[tuple[int, dict]] = []

    with open(args.assessment, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sev = (row.get("severity") or "").lower()
            codes = row.get("issue_codes") or ""
            rid = str(row.get("row_id") or "").strip()
            if not rid or rid not in by_id:
                continue
            pri = sum(1 for c in PRIORITY_CODES if c in codes)
            score = 0
            if sev == "high":
                score += 1000
            elif sev == "med":
                score += 100
            score += pri * 50
            rec = by_id[rid]
            scored.append(
                (
                    score,
                    {
                        "row_id": int(rid) if rid.isdigit() else rid,
                        "query_company": rec.get("Company Name") or "",
                        "query_city": rec.get("City") or "",
                        "query_state": rec.get("State") or "",
                        "severity": row.get("severity") or "",
                        "issue_codes": codes,
                        "narrative": (row.get("narrative") or "")[:800],
                    },
                )
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    out_cases = [c for _, c in scored[: args.max_cases]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {
        "source_assessment": os.path.basename(args.assessment),
        "source_records": os.path.basename(args.records),
        "max_cases": args.max_cases,
        "exported": len(out_cases),
        "cases": out_cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.out} ({len(out_cases)} cases)")


if __name__ == "__main__":
    main()
