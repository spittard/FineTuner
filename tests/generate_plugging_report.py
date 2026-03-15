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

INPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'plugging_matches.json')
OUTPUT_MD = os.path.join(os.path.dirname(__file__), '..', 'plugging_report.md')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'plugging_report.csv')

# Match quality tiers
TIER_HIGH = 0.85
TIER_MED = 0.70


def tier_label(score: float) -> str:
    if score >= TIER_HIGH:
        return "High"
    if score >= TIER_MED:
        return "Medium"
    return "Low"


def score_icon(score: float) -> str:
    if score >= 0.80:
        return "🟢"
    if score >= 0.60:
        return "🟡"
    if score >= 0.40:
        return "🟠"
    return "🔴"


def match_type_label(match: dict) -> str:
    string_score = match.get('string_score', 0)
    semantic_score = match.get('semantic_score', 0)
    acronym_fidelity = match.get('acronym_fidelity', 0)
    if string_score >= 0.9:
        return "Exact"
    if acronym_fidelity > 0.5:
        return "Acronym"
    if semantic_score > string_score:
        return "Semantic"
    return "Lexical"


def format_location(city: str, state: str) -> str:
    parts = [p for p in [city, state] if p]
    return ", ".join(parts) if parts else ""


def format_company_cell(name: str, city: str, state: str) -> str:
    loc = format_location(city, state)
    if loc:
        return f"`{name}` ({loc})"
    return f"`{name}`"


def _drop_first_exact_match(matches: list) -> list:
    """Remove first exact match (self-match when plug record is in index)."""
    result = []
    dropped = False
    for m in matches:
        if not dropped and m.get('match_type') == 'exact':
            dropped = True
            continue
        mcopy = dict(m)
        result.append(mcopy)
    for i, m in enumerate(result, 1):
        m['rank'] = i
    return result


def generate_report_header(total: int, tier_counts: dict) -> str:
    high = tier_counts.get('High', 0)
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
| Medium Confidence ({TIER_MED:.0%} – {TIER_HIGH:.0%}) | {med:,} |
| Low Confidence (< {TIER_MED:.0%}) | {low:,} |
| No Matches | {no_match:,} |

## Matching Methodology

| Component | Weight | Description |
|-----------|--------|-------------|
| String Similarity | 70% | Jaro-Winkler lexical comparison |
| Semantic Similarity | 30% | Neural embedding (MiniLM) |
| Acronym Fidelity | +15% | Bonus for acronym expansion |
| Location Boost | +5% | Bonus for location match |

---

"""


def format_plugging_entry(entry: dict, entry_num: int, top_k: int) -> str:
    row_id = entry.get('row_id', 'N/A')
    company = entry.get('query_company', '')
    city = entry.get('query_city', '')
    state = entry.get('query_state', '')
    matches = _drop_first_exact_match(entry.get('matches', []))
    error = entry.get('error')

    loc = format_location(city, state)
    loc_str = f" — {loc}" if loc else ""
    output = [f"## {entry_num}. Plugging Record: `{company}`{loc_str} (Row: {row_id})"]
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
        mtype = match_type_label(match)
        company_cell = format_company_cell(name, match_city, match_state)

        full_rationale = RationaleService.generate_match_rationale(
            query=company,
            company_name=name,
            explanation=match,
            score=score
        )
        score_breakdown = RationaleService.generate_detailed_score_breakdown(match, company)
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
        matches = _drop_first_exact_match(entry.get('matches', []))

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
                    'match_type': match_type_label(m),
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

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("       Run match_plugging_records.py first.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    print(f"Loaded {len(entries):,} entries from {args.input}")

    # Compute tier counts (based on best match per record, excluding first exact self-match)
    tier_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'NoMatch': 0}
    for entry in entries:
        matches = _drop_first_exact_match(entry.get('matches', []))
        if not matches:
            tier_counts['NoMatch'] += 1
        else:
            best = matches[0].get('score', 0)
            tier_counts[tier_label(best)] += 1

    # --- Generate Markdown ---
    print(f"Generating Markdown report: {args.output_md}")
    report_content = generate_report_header(len(entries), tier_counts)

    # Group sections by tier for navigation, then emit all entries
    for tier_name in ['High', 'Medium', 'Low', 'NoMatch']:
        tier_entries = []
        for entry in entries:
            matches = _drop_first_exact_match(entry.get('matches', []))
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
            'Medium': f'Medium Confidence ({TIER_MED:.0%} – {TIER_HIGH:.0%})',
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
    print(f"  Medium confidence:  {tier_counts['Medium']:,}")
    print(f"  Low confidence:     {tier_counts['Low']:,}")
    print(f"  No matches:         {tier_counts['NoMatch']:,}")
    print(f"  Markdown report:    {args.output_md} ({md_mb:.1f} MB)")
    print(f"  CSV report:         {args.output_csv} ({csv_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
