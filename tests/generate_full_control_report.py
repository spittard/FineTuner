#!/usr/bin/env python3
"""
Generate comprehensive control set report with detailed explanations.
Writes report after first 10 companies, then continues processing all 104.
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
    header.append("# Company Matching Control Set Report (Location-Aware)\n\n")
    header.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    header.append("---\n\n")
    
    # Scenarios this project handles
    header.append("## Matching Scenarios Handled\n\n")
    header.append("This system is designed to handle the following real-world company matching challenges:\n\n")
    
    header.append("### 1. **Exact Matches**\n")
    header.append("Perfect character-for-character matching.\n")
    header.append("- `\"IBM\"` → `\"IBM\"` (100%)\n")
    header.append("- `\"Microsoft\"` → `\"Microsoft\"` (100%)\n")
    header.append("- `\"Apple Inc.\"` → `\"Apple Inc.\"` (100%)\n")
    header.append("- `\"Google\"` → `\"Google\"` (100%)\n")
    header.append("- `\"Amazon.com\"` → `\"Amazon.com\"` (100%)\n\n")
    
    header.append("### 2. **Acronym Expansions**\n")
    header.append("Matching acronyms to their full company names or vice-versa.\n")
    header.append("- `\"IBM\"` → `\"International Business Machines\"` (Strong expansion)\n")
    header.append("- `\"AWS\"` → `\"Amazon Web Services\"` (Strong expansion)\n")
    header.append("- `\"GE\"` → `\"General Electric\"` (Strong expansion)\n")
    header.append("- `\"AT&T\"` → `\"American Telephone and Telegraph\"` (Strong expansion)\n")
    header.append("- `\"FedEx\"` → `\"Federal Express\"` (Strong expansion)\n\n")
    
    header.append("### 3. **Location-Aware Matching (NEW)**\n")
    header.append("Using city/state context to resolve ambiguity between identical or similar names.\n")
    header.append("- `\"Acme\"` (Chicago) → `\"Acme Corp\"` (Chicago, IL) vs (Miami, FL)\n")
    header.append("- `\"Northwestern\"` (Evanston) → `\"Northwestern University\"` (Evanston, IL) vs `\"Northwestern Mutual\"` (Milwaukee, WI)\n")
    header.append("- `\"Pizza Hut\"` (London, KY) → `\"Pizza Hut\"` (London, KY) vs `\"Pizza Hut\"` (London, UK)\n")
    header.append("- `\"Springfield Power\"` (Springfield, IL) → Resolved to Illinois entity over Massachusetts\n")
    header.append("- `\"Regency Hotel\"` (Paris, TX) → Resolved to Texas entity over France or Nevada\n\n")
    
    header.append("### 4. **Popularity/Frequency Bias (NEW)**\n")
    header.append("Using occurrence counts to break ties, prioritizing major entities over obscure ones.\n")
    header.append("- `\"McDonalds\"` → Global chain (5,000+ records) vs `\"McDonalds Hardware\"` (1 record)\n")
    header.append("- `\"Starbucks\"` → National brand vs `\"Starbucks Coffee Roasters\"` (local shop)\n")
    header.append("- `\"Walmart\"` → Major retailer vs `\"Walmarts Antiques\"` (single entry)\n")
    header.append("- `\"Chase\"` → `\"JP Morgan Chase\"` (Bank) vs `\"Chase & Sons Trucking\"` \n")
    header.append("- `\"Ford\"` → `\"Ford Motor Company\"` vs `\"Ford's Diner\"`\n\n")

    header.append("---\n\n")
    
    # Scoring formula
    header.append("## Advanced Scoring Formula\n\n")
    header.append("```\n")
    header.append("Base Score = (String Similarity × 70%) + (Semantic Similarity × 30%)\n")
    header.append("Fidelity Boost = Acronym Fidelity × 15%\n")
    header.append("Location Boost = Location Score × 5% (Post-Inference)\n")
    header.append("Final Score = Base Score + Fidelity Boost + Location Boost + Popularity Boost\n")
    header.append("```\n\n")
    
    header.append("---\n\n")
    header.append("## Control Set Results\n\n")
    
    return ''.join(header)


def format_company_result(query, result_data, rank):
    """Format a single company result in compact format with detailed rationales"""
    
    md = []
    
    # Get matches
    original_matches = result_data.get('matches', [])
    if not original_matches:
        md.append(f"## {rank}. {query}\n\n")
        md.append("❌ **No matches found**\n\n")
        md.append("---\n\n")
        return ''.join(md)
    
    # Determine exact match status (name-only)
    has_exact_name = any(m['company_name'].lower().strip() == query.lower().strip() for m in original_matches)
    
    # FILTER: Only discard if it's truly the "same" entity as the query (if location provided)
    # Otherwise, keep at least one exact match if we want to show it, or keep all if they are different entities
    filtered_matches = []
    found_self = False
    
    query_name_lower = query.lower().strip()
    query_city_lower = (result_data.get('query_city') or "").lower().strip()
    query_state_lower = (result_data.get('query_state') or "").lower().strip()
    
    for m in original_matches:
        match_name_lower = m['company_name'].lower().strip()
        match_city_lower = (m.get('city') or "").lower().strip()
        match_state_lower = (m.get('state') or "").lower().strip()
        
        # Is this a perfect identity match (Name + Location)?
        is_identity = (match_name_lower == query_name_lower)
        if query_city_lower or query_state_lower:
            # If query has location, identity requires location match too
            is_identity = is_identity and (match_city_lower == query_city_lower) and (match_state_lower == query_state_lower)
        
        if is_identity and not found_self:
            # Filter out ONLY the FIRST perfect identity match as the "Self-Match"
            found_self = True
            continue
        
        filtered_matches.append(m)
    
    has_exact = found_self or has_exact_name
    
    # Extract query location
    query_city = result_data.get('query_city', '')
    query_state = result_data.get('query_state', '')
    location_str = ""
    if query_city.strip() or query_state.strip():
        location_parts = []
        if query_city.strip():
            location_parts.append(query_city.strip())
        if query_state.strip():
            location_parts.append(query_state.strip())
        location_str = f" ({', '.join(location_parts)})"
    
    # Header with inline metadata
    md.append(f"## {rank}. {query}{location_str}\n\n")
    
    # Inline query details
    query_meta = f"**Query:** `{query}`"
    if query_city.strip() or query_state.strip():
        loc_parts = []
        if query_city.strip():
            loc_parts.append(query_city.strip())
        if query_state.strip():
            loc_parts.append(query_state.strip())
        query_meta += f" • **Location:** {', '.join(loc_parts)}"
    else:
        query_meta += " • **Location:** None (name-only search)"
    
    if has_exact:
        query_meta += " • **Self-Match:** ✅ Found & Filtered"
    else:
        query_meta += " • **Self-Match:** ❌ Not Found"
    
    md.append(f"{query_meta}\n\n")
    
    if not filtered_matches:
        md.append("> No non-identical matches found for this query.\n\n")
        md.append("---\n\n")
        return ''.join(md)
        
    top_match = filtered_matches[0]
    company_name = top_match.get('company_name', 'Unknown')
    score = top_match.get('likeness_percent', 0.0)
    city = top_match.get('city', '')
    state = top_match.get('state', '')
    
    match_location_str = f" ({city}, {state})" if city or state else ""
    
    # Top match header
    md.append(f"**Top Match:** {company_name}{match_location_str} • **Score:** {score:.1f}%\n\n")
    
    # Detailed score breakdown from RationaleService
    md.append("<details>\n")
    md.append("<summary><b>📊 Scoring Breakdown</b></summary>\n\n")
    breakdown = RationaleService.generate_detailed_score_breakdown(top_match, query)
    md.append(f"{breakdown}\n")
    md.append("</details>\n\n")
    
    # Detailed match rationale from RationaleService
    md.append("**Match Rationale (Narrative):**  \n")
    explanation_dict = top_match.get('explanation_details', {})
    rationale = RationaleService.generate_match_rationale(query, company_name, explanation_dict, score / 100.0)
    md.append(f"{rationale}\n\n")
    
    # Top 10 matches as table with expandable details
    md.append("**Top 10 Matches:**\n\n")
    md.append("| Rank | Company | Score | Summary |\n")
    md.append("|:----:|---------|:-----:|:--------|\n")
    
    for i, match in enumerate(filtered_matches[:10], 1):
        match_name = match.get('company_name', 'Unknown')
        match_score = match.get('likeness_percent', 0.0)
        m_city = match.get('city', '')
        m_state = match.get('state', '')
        m_loc = f" ({m_city}, {m_state})" if m_city or m_state else ""
        
        # Use RationaleService for a concise summary note
        m_explanation = match.get('explanation_details', {})
        note = RationaleService.get_short_summary(query, match_name, m_explanation)
        
        # Determine confidence indicator
        if match_score >= 95:
            indicator = "🟢 EXACT"
        elif match_score >= 80:
            indicator = "🟢 HIGH"
        elif match_score >= 60:
            indicator = "🟡 MEDIUM"
        elif match_score >= 40:
            indicator = "🟠 LOW"
        else:
            indicator = "🔴 VERY LOW"
        
        # Table row
        md.append(f"| {i} | {match_name}{m_loc} | **{match_score:.1f}%** | {indicator} {match_score:.1f}% |\n")
    
    md.append("\n")
    
    # Expandable detailed rationales for each match
    for i, match in enumerate(filtered_matches[:10], 1):
        match_name = match.get('company_name', 'Unknown')
        match_score = match.get('likeness_percent', 0.0)
        m_city = match.get('city', '')
        m_state = match.get('state', '')
        m_loc = f" ({m_city}, {m_state})" if m_city or m_state else ""
        m_explanation = match.get('explanation_details', {})
        
        md.append(f"<details>\n")
        md.append(f"<summary><b>#{i} {match_name}{m_loc}</b> — {match_score:.1f}% — Click for detailed rationale</summary>\n\n")
        
        # Detailed scoring breakdown
        breakdown = RationaleService.generate_detailed_score_breakdown(match, query)
        md.append(f"**📊 Score Breakdown:**\n{breakdown}\n\n")
        
        # Full narrative rationale
        rationale = RationaleService.generate_match_rationale(query, match_name, m_explanation, match_score / 100.0)
        md.append(f"**📝 Match Rationale:**\n{rationale}\n\n")
        
        # Relative positioning (why below the one above)
        if i > 1:
            match_above = filtered_matches[i-2]
            rel_pos = RationaleService.generate_relative_positioning_explanation(match, match_above, None, i)
            md.append(f"**📍 Why below #{i-1}?**\n{rel_pos}\n\n")
        
        md.append("</details>\n\n")
    
    
    md.append("---\n\n")
    
    return ''.join(md)


# Removed local rationale/note generators - now using RationaleService


def main():
    print("="*70)
    print("🔬 GENERATING FULL CONTROL SET REPORT (LOCATION-AWARE)")
    print("="*70)
    
    # Initialize service
    print("\n📦 Initializing SearchService...")
    service = SearchService()
    # Explicitly load the location-aware dataset
    if not service.load_company_data(model_name='paraphrase-MiniLM-L3-v2', filename='companies_with_location.json'):
        print("❌ Failed to load company data")
        return
    
    print("✅ Company data loaded\n")
    
    # Load control set
    control_set_file = os.path.join(os.path.dirname(__file__), '..', 'companies_control_set.json')
    
    if not os.path.exists(control_set_file):
        print(f"❌ Control set file not found: {control_set_file}")
        return
    
    # Read control set JSON
    with open(control_set_file, 'r', encoding='utf-8') as f:
        control_data = json.load(f)
    
    # Keep the full dictionary entries to preserve City/State data
    companies = control_data
    
    print(f"📋 Loaded {len(companies)} companies from control set\n")
    
    # Generate header
    report_content = [generate_report_header()]
    
    # Process companies
    results = []
    
    for i, company_entry in enumerate(companies, 1):
        # Extract company name and optional location
        if isinstance(company_entry, dict):
            company = company_entry.get('Company Name', '')
            city = company_entry.get('City', None)
            state = company_entry.get('State', None)
        else:
            # Backward compatibility: if it's just a string
            company = company_entry
            city = None
            state = None
        
        print(f"Processing {i}/{len(companies)}: {company}")
        if city or state:
            print(f"   With location: {city}, {state}")
        
        # Search with location parameters
        matches = service.search(company, top_k=10, city=city, state=state)
        
        result_data = {
            'query': company,
            'query_city': city or '',
            'query_state': state or '',
            'matches': matches
        }
        results.append(result_data)
        
        # Format result
        company_md = format_company_result(company, result_data, i)
        report_content.append(company_md)
        
        # Write report after first 10 companies
        if i == 10:
            output_file = 'control_set_report_ULTRA.md'
            print(f"\n📄 Writing initial report (first 10 companies) to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(''.join(report_content))
            print(f"✅ Initial report written! Continuing with remaining {len(companies) - 10} companies...\n")
    
    # Write final report
    output_file = 'control_set_report_ULTRA.md'
    print(f"\n📄 Writing final report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report_content))
    
    print(f"\n✅ COMPLETE! Processed {len(companies)} companies")
    print(f"📊 Report saved: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
