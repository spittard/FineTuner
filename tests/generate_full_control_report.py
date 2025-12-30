#!/usr/bin/env python3
"""
Generate comprehensive control set report with detailed explanations.
Writes report after first 10 companies, then continues processing all 104.
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core import cache_rpc
from finetuner.core.matcher import TextPreprocessor
from finetuner.web.services.rationale_service import RationaleService

def enrich_rpc_results(query, raw_matches):
    """
    Enrich raw RPC matches with rationales and explanations locally.
    This replaces what SearchService.search() used to do.
    """
    results = []
    for i, match in enumerate(raw_matches, 1):
        # 1. Replicate CompanyMatcher.explain_match() locally
        # We need this because RPC doesn't return the 'explanation' object
        q_tokens = set(TextPreprocessor.clean_company_name(query).split())
        t_tokens = set(TextPreprocessor.clean_company_name(match['name']).split())
        overlap = q_tokens.intersection(t_tokens)
        overlap_score = len(overlap) / len(q_tokens) if q_tokens else 0.0
        
        explanation = {
            "query_tokens": list(q_tokens),
            "match_tokens": list(t_tokens),
            "overlap": list(overlap),
            "overlap_score": overlap_score,
            "string_score": match.get('string_score', 0.0),
            "semantic_score": match.get('semantic_score', 0.0),
            "normalized_semantic_score": match.get('normalized_semantic_score', 0.0),
            "acronym_fidelity": match.get('acronym_fidelity', 0.0),
            "match_type": match.get('match_type', 'hybrid'),
            "location_score": match.get('location_score', 0.0),
            "count": match.get('count', 0),
            "popularity_boost": match.get('popularity_boost', 0.0),
            "location_boost": match.get('location_boost', 0.0),
            "match_city": match.get('city', '')
        }

        # 2. Generate Rationale using RationaleService
        rationale = RationaleService.generate_match_rationale(query, match['name'], explanation, match['score'])
        
        result_entry = {
            'rank': i,
            'company_name': match['name'],
            'likeness_percent': round(match['score'] * 100, 1),
            'match_rationale': rationale,
            'raw_score': match['score'],
            'explanation_details': explanation
        }
        
        # Add top-level fields
        result_entry['string_score'] = explanation['string_score']
        result_entry['semantic_score'] = explanation['semantic_score']
        result_entry['normalized_semantic_score'] = explanation['normalized_semantic_score']
        result_entry['acronym_fidelity'] = explanation['acronym_fidelity']
        result_entry['match_type'] = explanation['match_type']
        
        # Add location/count
        if 'city' in match:
            result_entry['city'] = match.get('city', '')
        if 'state' in match:
            result_entry['state'] = match.get('state', '')
        if 'count' in match:
            result_entry['record_count'] = match.get('count', 0)
        if 'location_score' in match:
            result_entry['location_score'] = round(match.get('location_score', 0) * 100, 1)
        if 'name_score' in match:
            result_entry['name_score'] = round(match.get('name_score', 0) * 100, 1)
            
        results.append(result_entry)
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate control set report')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of companies to process')
    args = parser.parse_args()

    print("="*70)
    print("🔬 GENERATING FULL CONTROL SET REPORT (RPC CLIENT MODE)")
    if args.limit:
        print(f"⚠️ LIMIT SET: Processing only {args.limit} companies")
    print("="*70)
    
    # Initialize RPC Client
    print("\n🔗 Connecting to RPC Server...")
    try:
        if not cache_rpc.is_server_running():
            print("❌ RPC Server is not running! Please start it with: python -m finetuner.core.cache_rpc --serve")
            return
        
        client = cache_rpc.connect()
        status = client.get_status()
        print(f"✅ Connected to RPC Server (PID: {status.get('pid')})")
        
        # Check loaded caches
        loaded_caches = client.list_loaded_caches()
        
        # We need the location-aware cache
        # Ideally, we find a cache key that corresponds to 'companies_with_location.json'
        # For now, we'll try to load it by filename if we can, or check if any large cache is loaded
        
        # Try to ensure a cache is loaded
        if not loaded_caches:
            print("⚠️ No caches loaded on server. Attempting to search to trigger auto-load...")
            # This relies on server having a default or us knowing the key.
            # Let's try to list available and load the largest one
            avail = client.list_available_caches()
            if avail:
                # Pick largest
                best_cache = sorted(avail, key=lambda x: x.get('num_companies', 0), reverse=True)[0]
                cache_key = best_cache['cache_key']
                print(f"   Loading largest cache: {cache_key} ({best_cache.get('num_companies')} companies)...")
                client.load_cache(cache_key)
            else:
                print("❌ No caches found on server disk.")
                return
        
        print("✅ Cache ready on server\n")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
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
        if args.limit and i > args.limit:
            print(f"\n⚠️ Limit reached ({args.limit}). Stopping early.")
            break
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
        
        # Search via RPC (fetch more to allow for filtering)
        rpc_response = client.search(company, top_k=15, city=city, state=state)
        
        # Enrich results locally
        if 'error' in rpc_response:
             print(f"   ❌ RPC Error: {rpc_response['error']}")
             matches = []
        else:
             raw_matches = rpc_response.get('results', [])
             matches = enrich_rpc_results(company, raw_matches)
        
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
    
    for i, match in enumerate(filtered_matches[:10], 1):
        match_name = match.get('company_name', 'Unknown')
        match_score = match.get('likeness_percent', 0.0)
        m_city = match.get('city', '')
        m_state = match.get('state', '')
        m_loc = f" ({m_city}, {m_state})" if m_city or m_state else ""
        
        # Concise one-line summary for the collapsed state
        visual_indicator = RationaleService.get_visual_indicator(match_score / 100.0)
        summary_line = f"{visual_indicator} <b>#{i}</b> | <b>{match_name}</b>{m_loc} | Score: <b>{match_score:.1f}%</b>"
        
        # 1. Detailed Score Breakdown
        exp = match.get('explanation_details', {})
        string_score = exp.get('string_score', 0.0)
        semantic_norm = exp.get('normalized_semantic_score', exp.get('semantic_score', 0.0))
        location_boost = exp.get('location_boost', 0.0)
        popularity_boost = exp.get('popularity_boost', 0.0)
        contrib_string = string_score * 0.7
        contrib_semantic = semantic_norm * 0.3
        base_score = contrib_string + contrib_semantic
        
        # Content Generation
        html_parts = []
        # Full width container (removed width: 600px constraint)
        html_parts.append(f"    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>") 
        
        # Score Breakdown Table (Small, compact)
        html_parts.append(f"      <b>📊 Score Breakdown:</b><br>")
        html_parts.append(f"      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>")
        html_parts.append(f"        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>")
        html_parts.append(f"        <tr><td>String Similarity</td><td>{string_score:.4f}</td><td>70%</td><td>{contrib_string:.4f}</td></tr>")
        html_parts.append(f"        <tr><td>Semantic Similarity (Norm)</td><td>{semantic_norm:.4f}</td><td>30%</td><td>{contrib_semantic:.4f}</td></tr>")
        html_parts.append(f"        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>{base_score:.4f}</em></td></tr>")
        if location_boost > 0:
            html_parts.append(f"        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+{location_boost:.4f}</td></tr>")
        if popularity_boost > 0:
            html_parts.append(f"        <tr><td>Frequency Boost</td><td></td><td></td><td>+{popularity_boost:.4f}</td></tr>")
        html_parts.append(f"        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>{match_score/100:.4f}</strong></td></tr>")
        html_parts.append(f"      </table>")
        
        # Match Rationale
        rationale_text = RationaleService.generate_match_rationale(query, match_name, exp, match_score / 100.0)
        
        html_parts.append(f"      <b>📝 Match Rationale:</b><br>")
        html_parts.append(f"      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>")
        html_parts.append(f"{rationale_text}") 
        html_parts.append(f"      </div>")
        
        html_parts.append(f"    </div>")
        
        detail_content = "\n".join(html_parts)
        
        # Create the details block
        md.append(f"<details style='margin-bottom: 5px; padding: 5px;'>\n")
        md.append(f"  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>{summary_line}</summary>\n")
        md.append(f"{detail_content}\n")
        md.append(f"</details>\n")
    
    md.append("\n")
    md.append("---\n\n")
    
    return ''.join(md)

if __name__ == "__main__":
    main()
