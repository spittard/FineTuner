#!/usr/bin/env python3
"""
Generate SME-style plugging records report.

Reads:   plugging_matches.json  (from match_plugging_records.py)
Writes:  plugging_report.md     (SME-ready markdown, same format as control_set_report_SME.md)
         plugging_report.csv    (flat, all fields for data handoff)

Usage:
    python tests/generate_plugging_report.py
    python tests/generate_plugging_report.py --input plugging_matches.json --top-k 5
"""

import csv
import json
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.web.services.rationale_service import RationaleService

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'plugging_matches.json')
OUTPUT_MD = os.path.join(PROJECT_ROOT, 'plugging_report.md')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'plugging_report.csv')
TIER_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'tier_config.json')

# Match quality tiers (0–1 scale); set by init_tier_from_file() from tier_config.json
# Four-tier scheme aligned with natural score breaks observed in the matcher:
#   High   >= 0.95  (exact name + good loc, or near-perfect with strong loc)
#   Good   >= 0.85  (exact name + state-only, or strong hybrid + loc)
#   Medium >= 0.70  (hybrid lexical-floor matches; SME review recommended)
#   Low    <  0.70  (weak signal; manual research likely needed)
# `NO_MATCH_THRESHOLD` adds a "no confident match" annotation chip when score < 0.55.
TIER_HIGH = 0.95
TIER_GOOD = 0.85
TIER_MED = 0.70
NO_MATCH_THRESHOLD = 0.55


def load_tier_thresholds() -> tuple[float, float, float, float]:
    """Read tier cutoffs as percent (1–100) from tier_config.json; return 0–1 floats.
    Backward-compatible: legacy {high, medium} files are still accepted.
    """
    h, g, m, nm = 95, 85, 70, 55
    if os.path.exists(TIER_CONFIG_PATH):
        try:
            with open(TIER_CONFIG_PATH, encoding='utf-8') as f:
                cfg = json.load(f)
            h = int(cfg.get('high', h))
            g = int(cfg.get('good', g))
            m = int(cfg.get('medium', m))
            nm = int(cfg.get('no_match', nm))
            # Validate ordering; fall back to defaults if malformed.
            if not (1 <= nm <= m <= g <= h <= 100):
                h, g, m, nm = 95, 85, 70, 55
        except Exception:
            pass
    return h / 100.0, g / 100.0, m / 100.0, nm / 100.0


def init_tier_from_file() -> None:
    global TIER_HIGH, TIER_GOOD, TIER_MED, NO_MATCH_THRESHOLD
    TIER_HIGH, TIER_GOOD, TIER_MED, NO_MATCH_THRESHOLD = load_tier_thresholds()


def tier_label(score: float) -> str:
    if score >= TIER_HIGH:
        return "High"
    if score >= TIER_GOOD:
        return "Good"
    if score >= TIER_MED:
        return "Medium"
    return "Low"


def score_icon(score: float) -> str:
    if score >= TIER_HIGH:
        return "🟢"
    if score >= TIER_GOOD:
        return "🟢"
    if score >= TIER_MED:
        return "🟡"
    if score >= NO_MATCH_THRESHOLD:
        return "🟠"
    return "🔴"


def confidence_chip(score: float) -> str:
    """Return a short suffix chip for borderline rows; empty otherwise."""
    if score < NO_MATCH_THRESHOLD:
        return " `[no confident match]`"
    return ""


def match_type_label(match: dict, query_city: str = "", query_state: str = "") -> str:
    """
    Type column: use matcher match_type + final score + location context — not string-only 'Exact'.
    """
    mt = (match.get('match_type') or '').lower()
    string_score = float(match.get('string_score', 0) or 0)
    # Use the NORMALIZED semantic score (0-1). The raw 'semantic_score' is a FAISS
    # inner-product sum (often ~6.0), so comparing it to string_score made the
    # "semantic-led" branch fire for nearly every hybrid row.
    semantic_score = float(
        match.get('normalized_semantic_score', match.get('semantic_score', 0)) or 0
    )
    acronym_fidelity = float(match.get('acronym_fidelity', 0) or 0)
    loc = float(match.get('location_score', 0) or 0)
    q_city = (query_city or '').strip()
    q_state = (query_state or '').strip()
    has_query_loc = bool(q_city or q_state)

    if mt.startswith('acronym') or (acronym_fidelity > 0.5 and string_score < 0.85):
        return "Acronym"

    # Only true Phase-3 / matcher "exact" rows get exact-style labels — high string_score
    # on hybrid substring matches must not read as "Exact name".
    name_like_exact = mt == 'exact'
    if name_like_exact:
        if not has_query_loc:
            return "Exact (name only)"
        cand_loc = bool(
            (str(match.get("city") or "").strip()) or (str(match.get("state") or "").strip())
        )
        if loc >= 0.65:
            return "Exact + location"
        if loc >= 0.35:
            return "Exact + partial geo"
        if not cand_loc:
            return "Exact name, verify address"
        if loc < 0.05:
            return "Exact name, wrong office"
        return "Exact name, weak geo"

    if mt == 'hybrid':
        if semantic_score > string_score and semantic_score >= 0.4:
            return "Hybrid (semantic-led)"
        return "Hybrid"

    if semantic_score > string_score and semantic_score >= 0.5:
        return "Semantic-led"
    return "Lexical"


def format_location(city: str, state: str) -> str:
    parts = [p for p in [city, state] if p]
    return ", ".join(parts) if parts else ""


def _escape_table_pipes(text: str) -> str:
    """Escape literal '|' for a GFM table cell (works inside inline code spans)."""
    return str(text or "").replace("|", "\\|")


def format_company_cell(name: str, city: str, state: str) -> str:
    loc = format_location(city, state)
    safe_name = _escape_table_pipes(name)
    if loc:
        return f"`{safe_name}` ({_escape_table_pipes(loc)})"
    return f"`{safe_name}`"


def _drop_self_match(matches: list, row_id) -> list:
    """Remove only the self-match (same row ID), preserving other exact matches."""
    result = []
    dropped = False
    row_id_str = str(row_id) if row_id is not None else ""
    for m in matches:
        match_id = m.get('id')
        if not dropped and match_id is not None and str(match_id) == row_id_str:
            dropped = True
            continue
        mcopy = dict(m)
        result.append(mcopy)
    for i, m in enumerate(result, 1):
        m['rank'] = i
    return result


def generate_report_header(total: int, tier_counts: dict) -> str:
    high = tier_counts.get('High', 0)
    good = tier_counts.get('Good', 0)
    med = tier_counts.get('Medium', 0)
    low = tier_counts.get('Low', 0)
    no_match = tier_counts.get('NoMatch', 0)

    return f"""# Plugging Records Match Report — SME Review
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Stat | Value |
|------|-------|
| Total Plugging Records | {total:,} |
| High Confidence (>= {TIER_HIGH:.0%}) | {high:,} |
| Good Confidence ({TIER_GOOD:.0%} – {TIER_HIGH:.0%}) | {good:,} |
| Medium Confidence ({TIER_MED:.0%} – {TIER_GOOD:.0%}) | {med:,} |
| Low Confidence (< {TIER_MED:.0%}) | {low:,} |
| No Matches | {no_match:,} |

## Tier bands (from `tier_config.json`)

| Tier | Rule (best match score) | SME action |
|------|-------------------------|------------|
| High | ≥ {TIER_HIGH:.0%} | Trust top-1; minimal verification |
| Good | {TIER_GOOD:.0%} – {TIER_HIGH:.0%} | Likely correct; brief check |
| Medium | {TIER_MED:.0%} – {TIER_GOOD:.0%} | Plausible; verify before use |
| Low | < {TIER_MED:.0%} | Weak signal; rows < {NO_MATCH_THRESHOLD:.0%} chip-flagged "no confident match" |

## Score semantics (SME)

- **Below top tier on exact name, query has no location:** Exact legal-name matches are **capped around 94%** until you add city/state that matches the record, so one geotagged row is not shown at ~99% when the same name may exist in many places. **100%** requires exact name **and** matching city **and** state on **both** sides.
- **Candidate location when the query has no location:** City/state in the **Company** column is **master-file reference**; it does not add a location “boost” to the score. Tie-break still prefers rows that list an office when scores tie.

Hybrid re-rank uses string + semantic + concept + location; see Rationale in each row.

---

"""


def format_plugging_entry(entry: dict, entry_num: int, top_k: int) -> str:
    row_id = entry.get('row_id', 'N/A')
    company = entry.get('query_company', '')
    city = entry.get('query_city', '')
    state = entry.get('query_state', '')
    matches = _drop_self_match(entry.get('matches', []), row_id)
    error = entry.get('error')

    loc = format_location(city, state)
    loc_str = f" — {loc}" if loc else ""
    # Add "no confident match" chip when the best-match score is below NO_MATCH_THRESHOLD.
    best_for_chip = matches[0].get('score', 0.0) if matches else 0.0
    chip = confidence_chip(best_for_chip) if matches else ""
    output = [f"## {entry_num}. Plugging Record: `{company}`{loc_str} (Row: {row_id}){chip}"]
    output.append("")

    if error:
        output.append(f"**ERROR:** `{error}`")
        output.append("")
        output.append("---")
        output.append("")
        return "\n".join(output)

    if not matches:
        output.append("*No matches found*")
        output.append("")
        output.append("---")
        output.append("")
        return "\n".join(output)

    # Table — identical structure to SME report, company cell adds location
    output.append("| # | Score | Company | Type | Rationale |")
    output.append("|:-:|:-----:|---------|------|-----------|")

    for match in matches[:top_k]:
        name = match.get('name', 'Unknown')
        match_city = match.get('city', '')
        match_state = match.get('state', '')
        score = match.get('score', 0)

        icon = score_icon(score)
        mtype = match_type_label(match, city, state)
        company_cell = format_company_cell(name, match_city, match_state)

        full_rationale = RationaleService.generate_match_rationale(
            query=company,
            company_name=name,
            explanation=match,
            score=score,
            query_city=city,
            query_state=state,
        )
        score_breakdown = RationaleService.generate_detailed_score_breakdown(
            match, company, city, state
        )
        combined = f"{full_rationale}\n\n---\n\n{score_breakdown}"
        rationale_html = combined.replace('\n', '<br>').replace('|', '&#124;')

        output.append(
            f"| {match.get('rank', '?')} | {icon} **{score:.1%}** | {company_cell} | {mtype} | "
            f"<details><summary>📊 View</summary><br>{rationale_html}</details> |"
        )

    output.append("")
    output.append("---")
    output.append("")
    return "\n".join(output)


def build_csv_rows(entries: list, top_k: int) -> list:
    rows = []
    for entry in entries:
        row_id = entry.get('row_id', '')
        company = entry.get('query_company', '')
        city = entry.get('query_city', '')
        state = entry.get('query_state', '')
        matches = _drop_self_match(entry.get('matches', []), row_id)

        if not matches:
            rows.append({
                'row_id': row_id,
                'query_company': company,
                'query_city': city,
                'query_state': state,
                'match_rank': '',
                'match_name': '',
                'match_city': '',
                'match_state': '',
                'match_row_id': '',
                'score': '',
                'tier': 'NoMatch',
                'match_type': '',
                'string_score': '',
                'semantic_score': '',
                'location_score': '',
            })
        else:
            for m in matches[:top_k]:
                score = m.get('score', 0)
                rows.append({
                    'row_id': row_id,
                    'query_company': company,
                    'query_city': city,
                    'query_state': state,
                    'match_rank': m.get('rank', ''),
                    'match_name': m.get('name', ''),
                    'match_city': m.get('city', ''),
                    'match_state': m.get('state', ''),
                    'match_row_id': m.get('id', ''),
                    'score': f"{score:.4f}",
                    'tier': tier_label(score),
                    'match_type': match_type_label(m, city, state),
                    'string_score': f"{m.get('string_score', 0):.4f}",
                    'semantic_score': f"{m.get('semantic_score', 0):.4f}",
                    'location_score': f"{m.get('location_score', 0):.4f}",
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Generate plugging records SME report')
    parser.add_argument('--input', default=INPUT_FILE, help='Input JSON file')
    parser.add_argument('--output-md', default=OUTPUT_MD, help='Output Markdown file')
    parser.add_argument('--output-csv', default=OUTPUT_CSV, help='Output CSV file')
    parser.add_argument('--top-k', type=int, default=5, help='Matches per record to show (default: 5)')
    args = parser.parse_args()

    init_tier_from_file()
    print(
        f"Tier thresholds: High >= {TIER_HIGH:.0%}, Good >= {TIER_GOOD:.0%}, "
        f"Medium >= {TIER_MED:.0%}, NoMatchChip < {NO_MATCH_THRESHOLD:.0%} "
        f"(from {TIER_CONFIG_PATH})"
    )

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("       Run match_plugging_records.py first.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    print(f"Loaded {len(entries):,} entries from {args.input}")

    # Compute tier counts (based on best match per record, excluding row-ID self-match)
    tier_counts = {'High': 0, 'Good': 0, 'Medium': 0, 'Low': 0, 'NoMatch': 0}
    for entry in entries:
        matches = _drop_self_match(entry.get('matches', []), entry.get('row_id', ''))
        if not matches:
            tier_counts['NoMatch'] += 1
        else:
            best = matches[0].get('score', 0)
            tier_counts[tier_label(best)] += 1

    # --- Generate Markdown ---
    print(f"Generating Markdown report: {args.output_md}")
    report_content = generate_report_header(len(entries), tier_counts)

    # Group sections by tier for navigation, then emit all entries
    for tier_name in ['High', 'Good', 'Medium', 'Low', 'NoMatch']:
        tier_entries = []
        for entry in entries:
            matches = _drop_self_match(entry.get('matches', []), entry.get('row_id', ''))
            if not matches:
                if tier_name == 'NoMatch':
                    tier_entries.append(entry)
            else:
                best = matches[0].get('score', 0)
                if tier_label(best) == tier_name and tier_name != 'NoMatch':
                    tier_entries.append(entry)

        if not tier_entries:
            continue

        tier_display = {
            'High': f'High Confidence (>= {TIER_HIGH:.0%})',
            'Good': f'Good Confidence ({TIER_GOOD:.0%} – {TIER_HIGH:.0%})',
            'Medium': f'Medium Confidence ({TIER_MED:.0%} – {TIER_GOOD:.0%})',
            'Low': f'Low Confidence (< {TIER_MED:.0%})',
            'NoMatch': 'No Matches',
        }[tier_name]

        report_content += f"# Tier: {tier_display} — {len(tier_entries):,} records\n\n---\n\n"

        for i, entry in enumerate(tier_entries, 1):
            print(f"  [{tier_name}] {i}/{len(tier_entries)}: {entry.get('query_company', '')} (Row: {entry.get('row_id', '')})")
            report_content += format_plugging_entry(entry, i, args.top_k)

            # Checkpoint every 50 entries
            if i % 50 == 0:
                with open(args.output_md, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"    [saved] checkpoint at {i} entries")

    with open(args.output_md, 'w', encoding='utf-8') as f:
        f.write(report_content)
    md_mb = os.path.getsize(args.output_md) / (1024 * 1024)

    # --- Generate CSV ---
    print(f"\nGenerating CSV: {args.output_csv}")
    csv_rows = build_csv_rows(entries, args.top_k)
    csv_fields = [
        'row_id', 'query_company', 'query_city', 'query_state',
        'match_rank', 'match_name', 'match_city', 'match_state', 'match_row_id',
        'score', 'tier', 'match_type', 'string_score', 'semantic_score', 'location_score',
    ]
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    csv_mb = os.path.getsize(args.output_csv) / (1024 * 1024)

    print()
    print("=" * 60)
    print("[DONE]")
    print(f"  Records processed:  {len(entries):,}")
    print(f"  High confidence:    {tier_counts['High']:,}")
    print(f"  Good confidence:    {tier_counts['Good']:,}")
    print(f"  Medium confidence:  {tier_counts['Medium']:,}")
    print(f"  Low confidence:     {tier_counts['Low']:,}")
    print(f"  No matches:         {tier_counts['NoMatch']:,}")
    print(f"  Markdown report:    {args.output_md} ({md_mb:.1f} MB)")
    print(f"  CSV report:         {args.output_csv} ({csv_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
