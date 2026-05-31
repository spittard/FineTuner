#!/usr/bin/env python3
"""
Heuristic SME reasonability scan over plugging_matches.json.

Logs likely "would an SME push back?" rows in batches (append-only).
Re-run after full pipeline:  python scripts/sme_assess_plugging_matches.py

Uses rule-based flags only (no LLM). Extend flags as you learn patterns.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from finetuner.utils.text_preprocessor import TextPreprocessor


def _norm_legal(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).casefold().strip())


def _token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm_legal(s)) if len(t) > 1}


def assess_record(rec: dict) -> list[str]:
    flags: list[str] = []
    q = (rec.get("query_company") or "").strip()
    q_city = (rec.get("query_city") or "").strip()
    q_state = (rec.get("query_state") or "").strip()
    matches = rec.get("matches") or []
    if not matches:
        flags.append("NO_MATCHES")
        return flags
    top = matches[0]
    sc = float(top.get("score") or 0)
    mt = (top.get("match_type") or "").lower()
    ns = float(top.get("name_score") or 0)
    ls = float(top.get("location_score") or 0)
    t_state_raw = top.get("state") or ""
    t_city_raw = top.get("city") or ""
    q_st = TextPreprocessor.normalize_state(q_state) if q_state else ""
    t_st = TextPreprocessor.normalize_state(t_state_raw) if t_state_raw else ""

    if q_st and t_st and q_st != t_st:
        flags.append("WRONG_STATE_TOP1")

    if q_city and q_state and (not t_city_raw.strip()) and (not t_state_raw.strip()) and sc >= 0.55:
        flags.append("QUERY_HAS_GEO_TOP1_BARE_ROW")

    if q_st and ls < 0.01 and sc >= 0.58:
        flags.append("HIGH_SCORE_NEAR_ZERO_LOCATION")

    if sc < 0.55:
        flags.append("LOW_CONFIDENCE_TOP1")

    if len(q) <= 4 and sc >= 0.85 and ns < 0.05 and "acronym" in mt:
        flags.append("ACRONYM_SHORT_QUERY_HIGH_SCORE")

    if "low_conf" in mt or "low_conf" in (top.get("match_type") or ""):
        flags.append("MATCH_TYPE_LOW_CONF")

    if len(matches) >= 5:
        scores = [float(m.get("score") or 0) for m in matches[:5]]
        if max(scores) - min(scores) < 0.001:
            flags.append("TOP5_TIE_CLUSTER")

    # Distinctive tokens: query has a token not in top-1 name but score looks "Good"
    q_tokens = _token_set(q)
    n_tokens = _token_set(top.get("name") or "")
    if q_tokens and n_tokens:
        only_in_q = q_tokens - n_tokens
        generics = {"inc", "llc", "corp", "ltd", "dba", "the", "and", "of", "co", "for"}
        distinctive = {t for t in only_in_q if t not in generics and len(t) > 2}
        inter = len(q_tokens & n_tokens)
        union = len(q_tokens | n_tokens) or 1
        jaccard = inter / union
        if (
            len(distinctive) >= 2
            and sc >= 0.74
            and jaccard < 0.5
            and len(q_tokens) >= 3
        ):
            flags.append("POSSIBLE_ENTITY_TAIL_MISMATCH")

    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description="SME heuristic assessment of plugging matches")
    ap.add_argument(
        "--input",
        default=os.path.join(ROOT, "plugging_matches.json"),
        help="plugging_matches.json path",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Records per batch for incremental log sections",
    )
    ap.add_argument(
        "--md-out",
        default=os.path.join(ROOT, "sme_plugging_findings.md"),
        help="Append markdown findings here",
    )
    ap.add_argument(
        "--jsonl-out",
        default=os.path.join(ROOT, "sme_plugging_findings.jsonl"),
        help="Append one JSON object per flagged row",
    )
    ap.add_argument(
        "--run-label",
        default="",
        help="Optional label for this run (e.g. post-pipeline-20260501)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Overwrite md/jsonl outputs before this run (default: append)",
    )
    args = ap.parse_args()

    if args.fresh:
        for p in (args.md_out, args.jsonl_out):
            if os.path.exists(p):
                os.remove(p)
        with open(args.md_out, "w", encoding="utf-8") as md:
            md.write(
                "# SME plugging assessment (heuristic)\n\n"
                "Rule-based flags for rows where a **subject-matter expert might disagree** "
                "with the top match or score story. **Not** ground truth—use for triage.\n\n"
                "Regenerate after full `plugging_matches.json`:\n\n"
                "`python scripts/sme_assess_plugging_matches.py --fresh --run-label <label>`\n\n"
            )

    if not os.path.exists(args.input):
        print(f"Missing {args.input}", file=sys.stderr)
        return 2

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    label = args.run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    md_header = (
        f"\n\n---\n\n## Run `{label}` — {n} records, batch_size={args.batch_size}\n\n"
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_\n\n"
    )
    with open(args.md_out, "a", encoding="utf-8") as md:
        md.write(md_header)

    total_flagged = 0
    batch_idx = 0
    all_flagged_rows: list[dict] = []
    for start in range(0, n, args.batch_size):
        batch = data[start : start + args.batch_size]
        batch_rows: list[dict] = []
        for rec in batch:
            fl = assess_record(rec)
            if not fl:
                continue
            total_flagged += 1
            top = (rec.get("matches") or [{}])[0]
            row = {
                "row_id": rec.get("row_id"),
                "flags": fl,
                "top_score": top.get("score"),
                "query_company": rec.get("query_company"),
                "query_city": rec.get("query_city"),
                "query_state": rec.get("query_state"),
                "top_name": top.get("name"),
                "top_city": top.get("city"),
                "top_state": top.get("state"),
                "match_type": top.get("match_type"),
            }
            batch_rows.append(row)
            all_flagged_rows.append(row)

        batch_idx += 1
        s, e = start, min(start + args.batch_size, n)
        rate = 100.0 * len(batch_rows) / max(1, len(batch)) if batch else 0
        section = (
            f"### Batch {batch_idx} (records {s}..{e - 1})\n\n"
            f"- **In batch:** {len(batch)} | **Flagged:** {len(batch_rows)} | **Flag rate:** {rate:.1f}%\n\n"
        )
        if batch_rows:
            section += "| row_id | flags | score | query (trunc) | top-1 (trunc) |\n"
            section += "|--------|-------|-------|-----------------|-----------------|\n"
            for r in batch_rows:
                qc = (r["query_company"] or "")[:55]
                tn = (r["top_name"] or "")[:55]
                fl = ", ".join(r["flags"])
                sc = r.get("top_score")
                section += f"| {r['row_id']} | `{fl}` | {sc} | {qc} | {tn} |\n"
            section += "\n"

        with open(args.md_out, "a", encoding="utf-8") as md:
            md.write(section)
        with open(args.jsonl_out, "a", encoding="utf-8") as jl:
            for r in batch_rows:
                r["run_label"] = label
                r["batch_index"] = batch_idx
                jl.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(
            f"Batch {batch_idx} [{s}..{e - 1}]: flagged {len(batch_rows)}/{len(batch)} "
            f"({rate:.1f}%) -> {args.md_out}",
            flush=True,
        )

    summary = (
        f"\n**Run `{label}` summary:** {n} records, **{total_flagged}** flagged at least once "
        f"({100.0 * total_flagged / max(1, n):.1f}% of rows).\n"
    )
    gate_a_rows = [r for r in all_flagged_rows if "TOP5_TIE_CLUSTER" in r["flags"]]
    gate_a_rows.sort(key=lambda r: (r["row_id"] is None, r["row_id"]))
    if gate_a_rows:
        ga_lines = [
            "\n## Priority: audit Gate A (`TOP5_TIE_CLUSTER`)\n\n",
            "Review these first: all top-5 scores within **0.001** (same flag as `audit_scoring` Gate A).\n\n",
            "| row_id | flags | score | query (trunc) | top-1 (trunc) |\n",
            "|--------|-------|-------|-----------------|-----------------|\n",
        ]
        for r in gate_a_rows:
            qc = (r["query_company"] or "")[:55]
            tn = (r["top_name"] or "")[:55]
            fl = ", ".join(r["flags"])
            sc = r.get("top_score")
            ga_lines.append(f"| {r['row_id']} | `{fl}` | {sc} | {qc} | {tn} |\n")
        ga_lines.append("\n")
        summary = "".join(ga_lines) + summary

    with open(args.md_out, "a", encoding="utf-8") as md:
        md.write(summary)
    print(summary.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
