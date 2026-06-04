#!/usr/bin/env python3
"""
Per-record assessment for the full plugging corpus (plugging_matches.json).

Encodes the SME rubric used for manual screenshot review, as machine-detectable
flags + a short narrative per row. Use after each rematch + report regen.

Reads:  plugging_matches.json (default: repo root)
Writes: plugging_report_assessment.csv
         plugging_report_assessment_summary.md

Usage:
  python scripts/assess_plugging_report_full.py
  python scripts/assess_plugging_report_full.py --input path/to/plugging_matches.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from finetuner.utils.text_preprocessor import TextPreprocessor


def _norm_q(s: str) -> str:
    return TextPreprocessor.clean_company_name(str(s or "")).lower().strip()


def _name_key(s: str) -> str:
    """Strict same-entity key: casefold + punctuation-collapse, suffix RETAINED so
    'X Inc' != 'X Corp'. Used to tell a company's own offices apart from different firms."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _cand_bare(m: dict) -> bool:
    return not str(m.get("city") or "").strip() and not str(m.get("state") or "").strip()


def _has_q_geo(rec: dict) -> bool:
    return bool(str(rec.get("query_city") or "").strip() or str(rec.get("query_state") or "").strip())


def _is_us_state_token(s: str) -> bool:
    n = TextPreprocessor.normalize_state(s) if s else ""
    return len(n) == 2 and n in TextPreprocessor.STATE_ABBREV


def _suspect_geo_label(m: dict) -> str | None:
    c = str(m.get("city") or "").strip()
    st = str(m.get("state") or "").strip()
    if not c or not st:
        return None
    # Genuine defect only when the city LITERALLY echoes the state AND that token is a US
    # state (e.g. city="UT" state="UT"). A literal-echo gate keeps legit US pairs whose raw
    # strings differ (New York / NY); the US-token gate keeps legit non-US city==region
    # echoes (Beijing/Beijing, Dubai/Dubai, Sao Paulo/Sao Paulo, Saint Michael/Saint Michael).
    if c.casefold() == st.casefold() and _is_us_state_token(c):
        return "us_state_in_city_field"
    return None


def _name_mismatch_query_top1(rec: dict) -> tuple[bool, str]:
    q = _norm_q(rec.get("query_company") or "")
    ms = rec.get("matches") or []
    if not ms:
        return False, ""
    n = _norm_q(ms[0].get("name") or "")
    if not q or not n:
        return False, ""
    if q == n:
        return False, ""
    # Substring / token drop (e.g. "Group" missing)
    if n in q or q in n:
        return True, "query_and_top1_names_related_non_equal"
    qt = set(q.split())
    nt = set(n.split())
    if qt <= nt or nt <= qt:
        return True, "token_subset_mismatch"
    overlap = qt & nt
    if overlap and len(overlap) >= min(len(qt), len(nt)) * 0.7:
        return True, "heavy_token_overlap_non_equal"
    return True, "lexically_different"


def _dup_scores_top5(ms: list) -> tuple[bool, bool]:
    """Return (has_dup, diff_entity_dup).

    has_dup: any two of top-5 share a score to 4dp.
    diff_entity_dup: a tied score group contains 2+ DIFFERENT entities (the real concern).
    Same-name multi-office ties are expected and are NOT a ranking defect.
    """
    if len(ms) < 2:
        return False, False
    groups: dict[float, list[str]] = {}
    for m in ms[:5]:
        s = round(float(m.get("score") or 0), 4)
        groups.setdefault(s, []).append(_name_key(m.get("name") or ""))
    has_dup = any(len(v) > 1 for v in groups.values())
    diff_entity = any(len(v) > 1 and len(set(v)) > 1 for v in groups.values())
    return has_dup, diff_entity


def _gate_a_tie(ms: list) -> bool:
    if len(ms) < 5:
        return False
    sc = [float(m.get("score") or 0) for m in ms[:5]]
    return max(sc) - min(sc) < 0.001


def assess_record(rec: dict) -> tuple[list[str], str, str]:
    """Return (issue_codes, severity, narrative)."""
    issues: list[str] = []
    sev = "info"
    parts: list[str] = []

    row_id = rec.get("row_id", "")
    qcomp = (rec.get("query_company") or "").strip()
    ms = rec.get("matches") or []

    if _gate_a_tie(ms):
        issues.append("GATE_A_TOP5_TIE_CLUSTER")
        sev = "high"

    if not ms:
        issues.append("NO_MATCHES")
        return issues, "high", "No matches returned for this record."

    m0 = ms[0]
    s0 = float(m0.get("score") or 0)

    if not _has_q_geo(rec) and not _cand_bare(m0):
        # Genuine SME violation: a bare row that is the SAME entity (same strict name) as
        # the geo rank-1 AND tied within 4dp. On a geo-less query the bare/national row of
        # the *same company* should lead. A geo rank-1 that merely carries an address while
        # being the best (and not same-name-tied) match is NOT a defect (e.g. WAHUPA), so it
        # is info-only — this is what previously over-inflated the "high" count.
        top_name = _name_key(m0.get("name") or "")
        s0r = round(s0, 4)
        genuine_bare = [
            i + 1
            for i, m in enumerate(ms[:5])
            if _cand_bare(m)
            and _name_key(m.get("name") or "") == top_name
            and round(float(m.get("score") or 0), 4) == s0r
        ]
        if genuine_bare:
            issues.append("NO_QUERY_GEO_BARE_ROW_NOT_FIRST")
            sev = "high"
            parts.append(
                f"Same-entity bare row at ranks {genuine_bare} ties geo rank-1 ({s0:.4f}); "
                f"the bare/national row should lead when the query carries no geography."
            )
        else:
            issues.append("NO_QUERY_GEO_RANK1_HAS_LOCATION")
            parts.append(
                "Query omits city/state and rank-1 carries an address, but no same-name bare "
                "row ties it — acceptable when rank-1 is the best/only-named match (info only)."
            )

    mm, mm_code = _name_mismatch_query_top1(rec)
    if mm and s0 >= 0.998:
        issues.append("HIGH_SCORE_NON_EXACT_NAME")
        sev = "high"
        parts.append(
            f"Top-1 name is not the same cleaned string as the query ({mm_code}) but score is {s0:.4f}. "
            "Check display tier / rescale vs SME copy (99.9% implies exact name only when geo omitted)."
        )
    elif mm and s0 >= 0.95:
        issues.append("STRONG_SCORE_NON_EXACT_NAME")
        sev = "med" if sev == "info" else sev
        parts.append(
            f"Top-1 is a related but non-identical name ({mm_code}) at {s0:.4f}; rationale should justify suffix/token gaps."
        )

    has_dup, diff_entity_dup = _dup_scores_top5(ms)
    if has_dup:
        issues.append("DUPLICATE_ROUNDED_SCORES_IN_TOP5")
        if diff_entity_dup:
            if sev == "info":
                sev = "med"
            parts.append(
                "Two or more top-5 rows of DIFFERENT entities share the same score to 4 decimals; "
                "ordering looks arbitrary without exposed tie-breakers."
            )
        else:
            parts.append(
                "Top-5 score ties are all the same entity (multi-office); expected, not a ranking defect (info only)."
            )

    for i, m in enumerate(ms[:5], start=1):
        sg = _suspect_geo_label(m)
        if sg:
            issues.append(f"SUSPECT_GEO_RANK{i}_{sg.upper()}")
            sev = "high"
            parts.append(
                f"Rank {i} location fields look malformed ({sg}): city={m.get('city')!r} state={m.get('state')!r}."
            )

    mt0 = (m0.get("match_type") or "").lower()
    if "hybrid" in mt0 and s0 >= 0.95 and mm:
        issues.append("HYBRID_SEMANTIC_LED_AT_CEILING")
        parts.append(
            "Hybrid (semantic-led) at a ceiling score with non-exact name overlap — defensibility risk for SMEs."
        )

    narrative = " ".join(parts) if parts else "No automated issues flagged; still spot-check for domain edge cases."
    return issues, sev, narrative


def main():
    ap = argparse.ArgumentParser(description="Full plugging corpus assessment CSV + summary.")
    ap.add_argument(
        "--input",
        default=os.path.join(ROOT, "plugging_matches.json"),
        help="plugging_matches.json path",
    )
    ap.add_argument(
        "--out-csv",
        default=os.path.join(ROOT, "plugging_report_assessment.csv"),
    )
    ap.add_argument(
        "--out-md",
        default=os.path.join(ROOT, "plugging_report_assessment_summary.md"),
    )
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("ERROR: input must be a JSON array", file=sys.stderr)
        sys.exit(1)

    flag_counts: Counter[str] = Counter()
    sev_counts: Counter[str] = Counter()
    rows_out = []

    for rec in data:
        issues, sev, narrative = assess_record(rec)
        for code in issues:
            flag_counts[code] += 1
        sev_counts[sev] += 1
        rows_out.append(
            {
                "row_id": rec.get("row_id", ""),
                "query_company": (rec.get("query_company") or "")[:200],
                "query_city": rec.get("query_city") or "",
                "query_state": rec.get("query_state") or "",
                "severity": sev,
                "issue_codes": ";".join(issues) if issues else "",
                "narrative": narrative.replace("\n", " ").strip(),
            }
        )

    fieldnames = [
        "row_id",
        "query_company",
        "query_city",
        "query_state",
        "severity",
        "issue_codes",
        "narrative",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    lines = [
        "# Plugging report — automated assessment summary",
        "",
        f"Source: `{os.path.basename(args.input)}`  ",
        f"Records: **{len(data)}**  ",
        f"CSV: `{os.path.basename(args.out_csv)}` (one row per plugging record; use filters / pivot in Excel)",
        "",
        "## Severity counts",
        "",
        "| severity | count |",
        "|----------|------:|",
    ]
    for k in sorted(sev_counts.keys()):
        lines.append(f"| {k} | {sev_counts[k]} |")
    lines.extend(["", "## Issue code counts", "", "| code | count |", "|------|------:|"])
    for code, n in flag_counts.most_common():
        lines.append(f"| `{code}` | {n} |")
    lines.extend(
        [
            "",
            "## How to use with SMEs",
            "",
            "1. Sort CSV by `severity` (high → med → info) then `issue_codes`.",
            "2. For each row, read `narrative` as the same style of critique as a manual screenshot pass.",
            "3. Map `issue_codes` to fix tracks: matcher (rank/score), data (bad city/state), UI (display tiers), rationale (copy).",
            "4. Re-run after `match_plugging_records.py` + matcher changes so scores reflect new logic.",
            "",
            "## Rubric (manual pass — same depth as single-record review)",
            "",
            "- **Geo–query alignment**: If query has no locale, bare candidates should not rank below geo-only duplicates at the same score.",
            "- **Score vs name truth**: Near-100% implies exact legal string (or documented exception); hybrid semantic-led at ceiling needs rationale.",
            "- **Ties**: Identical scores with different specificity (geo vs bare) need visible tie-breakers or ordering rules.",
            "- **Data hygiene**: City/state duplicates, `UT, UT`, city==state token, etc.",
            "- **Trade-style drift**: Different suffixes (`Group` vs `IT Solutions`) at high confidence need explicit justification.",
            "",
        ]
    )
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out_csv} ({len(rows_out)} rows)")
    print(f"Wrote {args.out_md}")
    print(f"Distinct issue codes: {len(flag_counts)}")


if __name__ == "__main__":
    main()
