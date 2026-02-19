#!/usr/bin/env python3
"""
Generate ultra-detailed control set report with 15 matches and ASCII diagrams.
Following the pattern in COMPREHENSIVE_MATCHING_SCENARIOS_CORRECTED.md.
"""

import json
import os
import sys
import time
from datetime import datetime

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.web.services.search_service import SearchService
from finetuner.web.services.rationale_service import RationaleService


def generate_report_header():
    """Generate the report header with scenario highlights and fidelity explanation"""
    header = []
    header.append("# Company Matching Control Set - Ultra Detailed Report\n\n")
    header.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    header.append("---\n\n")
    
    header.append("## Project Objectives & Scenarios\n\n")
    header.append("This project provides high-precision company matching across diverse scenarios:\n\n")
    
    scenarios = [
        ("Exact Match", "Perfect character-for-character matching.", "\"IBM\" → \"IBM\""),
        ("Acronym Expansion", "Literal mapping of initials to full name.", "\"IBM\" → \"International Business Machines\""),
        ("Acronym Reverse", "Mapping full name back to its acronym.", "\"International Business Machines\" → \"IBM\""),
        ("Subsequence Acronym", "Handling initials found within word starts.", "\"IBM\" → \"International Bureau of Management\""),
        ("Typo Handling", "Tolerance for missing/extra characters.", "\"Microsft\" → \"Microsoft\""),
        ("Abbreviation Sync", "Equating variations like Corp/Corporation.", "\"Acme Corp\" → \"Acme Corporation\""),
        ("Plural Handling", "Handling singular/plural variations.", "\"Machine\" ↔ \"Machines\""),
        ("Partial Match", "Finding the entity within a longer string.", "\"Acme\" → \"Acme Logistics LLC\""),
        ("Word Order", "Matches despite rearranged words.", "\"First National Bank\" → \"Bank First National\""),
        ("Noise Handling", "Filtering prefixes like 'X DO NOT USE'.", "\"X DO NOT USE - FORD\" → \"Ford Motor Company\""),
        ("Semantic Logic", "Related business concepts and synonyms.", "\"Software\" → \"Systems\""),
        ("Suffix Variation", "Handling LLC, Inc, Corp, Ltd variations.", "\"Acme Inc\" → \"Acme LLC\""),
        ("Multi-Word Overlap", "Handling companies with shared names.", "\"The Coca Cola Co\" → \"Coca-Cola Enterprises\"")
    ]
    
    for title, desc, eg in scenarios:
        header.append(f"### {title}\n{desc}\n- **Example:** `{eg}`\n\n")
    
    header.append("---\n\n")
    
    header.append("## Acronym Fidelity Score (Thorough Explanation)\n\n")
    header.append("The **Acronym Fidelity Score** is an algorithmic measure of how precisely a name expands an acronym. ")
    header.append("Unlike general semantic matching, it strictly validates the letter pattern against word starts.\n\n")
    
    header.append("### Scoring Patterns\n\n")
    header.append("| Fidelity | Relationship | Example Match Pattern |\n")
    header.append("|----------|--------------|-----------------------|\n")
    header.append("| **1.00** | Perfect Expansion | **I**nternational **B**usiness **M**achines → **IBM** |\n")
    header.append("| **0.95** | Prefix Expansion | **I**nternational **B**usiness **M**achines **C**orp → **IBM** |\n")
    header.append("| **0.90** | Subsequence | **I**nternational **B**ureau of **M**anagement → **IBM** |\n")
    header.append("| **0.70** | Word Collision | **I**B**M** Solutions → **IBM** (Internal letters matching sequence) |\n")
    header.append("| **0.65** | Partial Word | **I**ntercontinental **B**anking **M**etrics → **IBM** |\n")
    header.append("| **0.40** | Fuzzy/Broken | Some initials match, but order or significant words are skipped. |\n\n")
    
    header.append("### Detailed Pattern Logic (Example: IBM)\n")
    header.append("```\n")
    header.append("Query: \"IBM\"\n")
    header.append("Target: \"International Business Machines\"\n")
    header.append("\n")
    header.append("1. Extract Words: [International, Business, Machines]\n")
    header.append("2. Extract Initials: [I, B, M]\n")
    header.append("3. Pattern Verification:\n")
    header.append("   - Initials match \"IBM\" exactly? YES\n")
    header.append("   - Words contain internal acronym letters? NO\n")
    header.append("4. Result: 1.00 Fidelity (Highest Confidence)\n")
    header.append("```\n\n")
    
    header.append("---\n\n")
    return ''.join(header)


def draw_box(title, lines):
    """Draw a box around several lines of text"""
    width = 77
    box = []
    box.append("┌" + "─" * width + "┐")
    box.append("│ " + title.upper().ljust(width - 1) + "│")
    box.append("├" + "─" * width + "┤")
    box.append("│" + " " * width + "│")
    for line in lines:
        box.append("│  " + line.ljust(width - 3) + "│")
    box.append("│" + " " * width + "│")
    box.append("└" + "─" * width + "┘")
    return '\n'.join(box)


def generate_fidelity_diagram(query, match_name, fidelity):
    """Generate ASCII diagram for fidelity explanation"""
    lines = [
        f"Query Context:  {query}",
        f"Target Match:   {match_name}",
        "",
        "FIDELITY ANALYSIS:",
    ]
    
    # Simple letter mapping logic for the visual
    if not match_name: match_name = "Unknown"
    if not query: query = "Unknown"
    
    words = [w for w in str(match_name).split() if w.lower() not in ['the', 'of', 'and', 'for', 'in', 'at', 'by', 'to']]
    
    mapping_str = "No direct pattern mapping."
    # Check if forward or reverse
    if len(str(query)) < 8: # Likely acronym
        acro = str(query).upper()
        parts = []
        for i, char in enumerate(acro):
            if i < len(words):
                parts.append(f"{char} → {words[i]}")
        if parts:
            mapping_str = " | ".join(parts)
    else:
        # Reverse acronym
        q_str = str(query)
        acro = "".join([w[0].upper() for w in q_str.split() if w])
        mapping_str = f"Name '{q_str}' reduces to '{acro}'"
    
    lines.append(f"• Pattern: {mapping_str}")
    lines.append(f"• Score:   {fidelity:.2f}")
    
    if fidelity >= 1.0:
        lines.append("✓ PERFECT: Clean one-to-one expansion.")
    elif fidelity >= 0.9:
        lines.append("✓ STRONG: Direct expansion with extra context.")
    else:
        lines.append("! MODERATE: Partial or overlapping match.")
        
    return "```\n" + draw_box("Acronym Fidelity Analysis", lines) + "\n```"


def generate_comparison_diagram(m1, m2, rank1, rank2):
    """Generate ASCII diagram comparing two matches"""
    n1 = m1.get('company_name', 'Unknown')
    n2 = m2.get('company_name', 'Unknown')
    s1 = m1.get('likeness_percent', 0.0)
    s2 = m2.get('likeness_percent', 0.0)
    
    lines = [
        f"Match #{rank1}: {n1} ({s1:.2f}%)",
        f"Match #{rank2}: {n2} ({s2:.2f}%)",
        "",
        f"Score Difference: {s1 - s2:.2f}%",
        "",
        "RANKING RATIONALE:",
    ]
    
    # Comparison of logic
    diff = s1 - s2
    if diff < 0.01:
        lines.append("• Virtual Tie: Negligible difference in score components.")
    elif m1.get('match_type') == 'exact' and m2.get('match_type') != 'exact':
        lines.append(f"• Text Identity: #{rank1} is a perfect character match.")
    elif m1.get('acronym_fidelity', 0) > m2.get('acronym_fidelity', 0):
        lines.append(f"• Acronym Priority: #{rank1} is a cleaner expansion.")
    elif m1.get('string_score', 0) > m2.get('string_score', 0):
        lines.append(f"• Lexical Precision: #{rank1} has better character alignment.")
    else:
        lines.append(f"• Semantic Preference: #{rank1} has stronger conceptual link.")
        
    return "```\n" + draw_box(f"Ranking: Match #{rank1} vs Match #{rank2}", lines) + "\n```"


def format_company_section(query, matches, rank_idx):
    """Format a results section for a single query"""
    md = []
    md.append(f"## {rank_idx}. Query: `{query}`\n\n")
    
    if not matches:
        md.append("❌ **No matches found.**\n\n---\n\n")
        return ''.join(md)
    
    # Check for exact match
    top_match = matches[0]
    top_name = top_match.get('company_name', '')
    is_exact = query.lower().strip() == top_name.lower().strip()
    
    if is_exact:
        md.append(f"✅ **Exact Match Found:** `{top_name}` (100.0%)\n\n")
    else:
        md.append(f"⚠️ **Exact Match NOT Found.** Showing top candidates.\n\n")
    
    # Create table for Top 15
    md.append("### Top 15 Search Results\n\n")
    md.append("| Rank | Company Name | Score | String | Semantic | Acronym | Match Insight |\n")
    md.append("|------|--------------|-------|--------|----------|---------|---------------|\n")
    
    for i, match in enumerate(matches[:15], 1):
        name = match.get('company_name', 'Unknown')
        score = f"{match.get('likeness_percent', 0.0):.2f}%"
        string = f"{match.get('string_score', 0.0):.3f}"
        sem = f"{match.get('normalized_semantic_score', match.get('semantic_score', 0.0)):.3f}"
        acro = f"{match.get('acronym_fidelity', 0.0):.2f}"
        
        # Get summary rationale
        insight = RationaleService.generate_concise_rationale(query, name, match.get('explanation_details', {}), match.get('raw_score', 0.0))
        
        md.append(f"| {i} | {name} | {score} | {string} | {sem} | {acro} | {insight} |\n")
    
    md.append("\n")
    
    # Narrative for Top 3
    md.append("### Match Narratives\n\n")
    for i in range(min(3, len(matches))):
        match = matches[i]
        md.append(f"**Rank #{i+1}: {match.get('company_name')}**\n")
        # Generate full rationale
        full_rationale = str(match.get('match_rationale', "No detailed rationale available."))
        # indent for better reading
        indented = '\n'.join(['> ' + line for line in full_rationale.split('\n')])
        md.append(f"{indented}\n\n")

    # ASCII Diagrams
    if top_match.get('match_type') in ['acronym_expansion', 'acronym_reverse'] or top_match.get('acronym_fidelity', 0) > 0:
        md.append("### Fidelity Breakdown\n")
        md.append(generate_fidelity_diagram(query, top_name, top_match.get('acronym_fidelity', 0)))
        md.append("\n")
        
    if len(matches) >= 2:
        md.append("### Ranking Rationale\n")
        md.append(generate_comparison_diagram(matches[0], matches[1], 1, 2))
        md.append("\n")
            
    md.append("---\n\n")
    return ''.join(md)


def main():
    print("="*80)
    print("🚀 GENERATING ULTRA-DETAILED CONTROL SET REPORT")
    print("="*80)
    
    # Initialize SearchService
    service = SearchService()
    # In RPC mode, we check status instead of explicit load
    status = service.get_status()
    if status.get('status') != 'ready':
        print(f"❌ Error: Search service not ready: {status.get('message')}")
        return
    
    # Load control set from Root
    control_set_path = os.path.join(os.path.dirname(__file__), '..', 'companies_control_set.json')
    if not os.path.exists(control_set_path):
        print(f"❌ Error: Control set not found at {control_set_path}")
        return
        
    with open(control_set_path, 'r', encoding='utf-8') as f:
        control_data = json.load(f)
    
    queries = [item['Company Name'] for item in control_data]
    
    # Check for limit
    limit = 15 # Default limit for this run as requested
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass
            
    if limit:
        queries = queries[:limit]
        
    print(f"✅ Loaded {len(queries)} queries from control set (Limit: {limit}).")
    
    output_file = 'control_set_report_ULTRA.md'
    report_parts = [generate_report_header()]
    
    # Process
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Processing: {query}...", end='', flush=True)
        t0 = time.time()
        
        # Search Top 15
        matches = service.search(query, top_k=15)
        
        # Format
        section = format_company_section(query, matches, i)
        report_parts.append(section)
        
        print(f" OK ({time.time()-t0:.2f}s)")
        
        # Interim save at 10 companies
        if i == 10:
            print(f"\n💾 Saving interim report (10 companies) to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(''.join(report_parts))
            print("💾 Interim save complete.\n")
            
    # Final save
    print(f"\n💾 Saving final ultra report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report_parts))
    print(f"✅ SUCCESS! Generated report for {len(queries)} companies.")
    print(f"📍 Location: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
