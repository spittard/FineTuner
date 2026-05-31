"""Report-only scan of plugging_matches.json: scenario buckets for SME review."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT = os.path.join(ROOT, "plugging_matches.json")
DEFAULT_SUMMARY = os.path.join(ROOT, "sme_plugging_scenario_summary.md")


def has_q_geo(rec: dict) -> bool:
    return bool(str(rec.get("query_city") or "").strip() or str(rec.get("query_state") or "").strip())


def cand_bare(m: dict) -> bool:
    return not str(m.get("city") or "").strip() and not str(m.get("state") or "").strip()


def tier(s) -> str:
    s = float(s or 0)
    if s >= 0.95:
        return "High"
    if s >= 0.85:
        return "Good"
    if s >= 0.70:
        return "Medium"
    return "Low"


def gate_a_row_ids(data: list[dict]) -> list:
    out = []
    for rec in data:
        ms = rec.get("matches") or []
        if len(ms) < 5:
            continue
        sc = [float(m.get("score") or 0) for m in ms[:5]]
        if max(sc) - min(sc) < 0.001:
            out.append(rec.get("row_id"))
    return out


def run_analysis(data: list[dict], print_out: bool = True):
    lines = []
    n = len(data)

    def out(s: str = ""):
        if print_out:
            print(s)
        lines.append(s)

    out(f"plugging_matches.json records: {n}\n")

    no_qgeo = [r for r in data if not has_q_geo(r)]
    out(f"1. Query has NO city/state: {len(no_qgeo)} ({100 * len(no_qgeo) / n:.1f}%)")

    top1_bare = 0
    top1_geo = 0
    for rec in no_qgeo:
        m0 = (rec.get("matches") or [{}])[0]
        if cand_bare(m0):
            top1_bare += 1
        else:
            top1_geo += 1
    out(f"   Top-1 candidate: WITH geo on file: {top1_geo} | BARE (no city/state): {top1_bare}")

    no_geo_bare_later = []
    for rec in no_qgeo:
        ms = rec.get("matches") or []
        if len(ms) < 2:
            continue
        bare_ranks = [i + 1 for i, m in enumerate(ms[:5]) if cand_bare(m)]
        if not bare_ranks:
            continue
        if cand_bare(ms[0]):
            continue
        no_geo_bare_later.append(
            {
                "row_id": rec.get("row_id"),
                "query": (rec.get("query_company") or "")[:70],
                "bare_ranks": bare_ranks,
                "rank1_score": round(float(ms[0].get("score") or 0), 4),
                "rank1": (ms[0].get("name") or "")[:55],
                "rank1_loc": f"{ms[0].get('city') or ''}, {ms[0].get('state') or ''}".strip(", "),
            }
        )
    out(
        f"\n2. Query NO geo, but a BARE candidate appears in top-5 while rank-1 HAS geo: "
        f"{len(no_geo_bare_later)} rows (SME: 'why is national/HQ row lower than an office?')"
    )
    for row in no_geo_bare_later[:15]:
        out(
            f"   row {row['row_id']}: bare at ranks {row['bare_ranks']} | "
            f"#1 {row['rank1']!r} @ {row['rank1_loc']}"
        )

    qgeo = [r for r in data if has_q_geo(r)]
    out(f"\n3. Query HAS city/state: {len(qgeo)} ({100 * len(qgeo) / n:.1f}%)")

    top1_type = Counter()
    for rec in data:
        m0 = (rec.get("matches") or [{}])[0]
        t = (m0.get("match_type") or "unknown").split()[0][:24]
        top1_type[t] += 1
    out(f"\n4. Top-1 match_type (first word): {top1_type.most_common(15)}")

    tier_ct = Counter()
    low_conf = 0
    for rec in data:
        m0 = (rec.get("matches") or [{}])[0]
        sc = float(m0.get("score") or 0)
        tier_ct[tier(sc)] += 1
        if sc < 0.55:
            low_conf += 1
    out(f"\n5. Top-1 tier (report bands): {dict(tier_ct)}")
    out(f"   Top-1 score < 0.55 (no-confident-match chip): {low_conf}")

    tie5 = 0
    gate_a_ids = []
    for rec in data:
        ms = rec.get("matches") or []
        if len(ms) < 5:
            continue
        sc = [float(m.get("score") or 0) for m in ms[:5]]
        if max(sc) - min(sc) < 0.001:
            tie5 += 1
            gate_a_ids.append(rec.get("row_id"))
    out(
        f"\n6. All top-5 scores within 0.001: {tie5} "
        f"(audit Gate A; hard to explain order to SME without tie rationale)"
    )
    out("\n--- Done ---")
    return "\n".join(lines), gate_a_ids, no_geo_bare_later


def main() -> int:
    ap = argparse.ArgumentParser(description="Bucket plugging_matches.json for SME report-only review")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="plugging_matches.json path")
    ap.add_argument(
        "--summary-out",
        default=DEFAULT_SUMMARY,
        help="Write markdown summary + Gate A row_id list (empty path to skip)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Overwrite summary-out before writing",
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Missing {args.input}", file=sys.stderr)
        return 2

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    text, gate_a_ids, no_geo_bare_later = run_analysis(data, print_out=True)

    if args.summary_out:
        if args.fresh and os.path.exists(args.summary_out):
            os.remove(args.summary_out)
        iso = datetime.now(timezone.utc).isoformat()
        md = [
            "# Plugging SME scenario scan\n",
            f"_UTC {iso}_ · source: `{os.path.basename(args.input)}`\n",
            "## Console output\n",
            "```\n",
            text,
            "\n```\n",
            "## Audit Gate A — prioritize SME review\n",
            "All `row_id` where **top-5 scores** span **less than 0.001** (`audit_scoring.py` Gate A).\n",
            f"**Count:** {len(gate_a_ids)}\n",
            "\n```\n",
            "\n".join(str(r) for r in gate_a_ids),
            "\n```\n",
            "\nSee also: `python scripts/sme_assess_plugging_matches.py --fresh` (flag `TOP5_TIE_CLUSTER`).\n",
        ]
        with open(args.summary_out, "w", encoding="utf-8") as sf:
            sf.write("".join(md))
        print(f"\nWrote {args.summary_out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
