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
    header.append("Perfect text matching when query exactly equals company name.\n")
    header.append("- Example: `\"IBM\"` → `\"IBM\"` (100% match)\n\n")
    
    header.append("### 2. **Acronym Expansions**\n")
    header.append("Matching acronyms to their full company names.\n")
    header.append("- Example: `\"IBM\"` → `\"International Business Machines\"` (98%+ match)\n\n")
    
    header.append("### 3. **Location-Aware Matching (NEW)**\n")
    header.append("Differentiates identical names using geographic context.\n")
    header.append("- Example: `\"Acme\"` in `\"Chicago\"` → Matches `\"Acme Corp (Chicago)\"` higher than `\"Acme Corp (Miami)\"`\n\n")
    
    header.append("### 4. **Popularity/Frequency Bias (NEW)**\n")
    header.append("Uses occurrence counts to break ties and prioritize larger entities.\n")
    header.append("- Example: Frequent national brands rank higher than obscure single-occurrence entries.\n\n")

    header.append("---\n\n")
    
    # Scoring formula
    header.append("## Advanced Scoring Formula\n\n")
    header.append("```\n")
    header.append("Base Score = (String Similarity × 70%) + (Semantic Similarity × 30%)\n")
    header.append("Fidelity Boost = Acronym Fidelity × 15%\n")
    header.append("Location Boost = Location Score × 5% (Implicit via Location Baking)\n")
    header.append("Final Score = Base Score + Fidelity Boost + Location Boost + Popularity Boost\n")
    header.append("```\n\n")
    
    header.append("---\n\n")
    header.append("## Control Set Results\n\n")
    
    return ''.join(header)


def format_company_result(query, result_data, rank):
    """Format a single company result with self-exclusion for bias analysis"""
    
    md = []
    
    # Get matches
    original_matches = result_data.get('matches', [])
    if not original_matches:
        md.append(f"### {rank}. {query}\n\n")
        md.append("❌ **No matches found**\n\n")
        md.append("---\n\n")
        return ''.join(md)
    
    # FILTER: Programmatically discard exact self-matches to focus on model confusion
    filtered_matches = [m for m in original_matches if m['company_name'].lower().strip() != query.lower().strip()]
    
    # Determine exact match status
    has_exact = any(m['company_name'].lower().strip() == query.lower().strip() for m in original_matches)
    
    md.append(f"### {rank}. {query}\n\n")
    
    if has_exact:
        md.append(f"✅ **Exact Match (Self) Found & Filtered**\n\n")
    
    if not filtered_matches:
        md.append("> No non-identical matches found for this query.\n\n")
        md.append("---\n\n")
        return ''.join(md)
        
    top_match = filtered_matches[0]
    company_name = top_match.get('company_name', 'Unknown')
    score = top_match.get('likeness_percent', 0.0)
    city = top_match.get('city', '')
    state = top_match.get('state', '')
    count = top_match.get('count', 0)
    
    location_str = f" ({city}, {state})" if city or state else ""
    
    # Highlight that this is the first non-identical result
    md.append(f"**Top Non-Self Match (Rank 2):** `{company_name}`{location_str} ({score:.1f}%)\n\n")
    
    # Extract score components
    explanation = top_match.get('explanation_details', {})
    string_score = explanation.get('string_score', top_match.get('string_score', 0.0))
    semantic_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
    acronym_fidelity = explanation.get('acronym_fidelity', top_match.get('acronym_fidelity', 0.0))
    location_score = explanation.get('location_score', top_match.get('location_score', 0.0))
    
    # Score breakdown table
    md.append("**Score Breakdown for Best Non-Self Match:**\n\n")
    md.append("| Component | Value | Weight | Contribution |\n")
    md.append("|-----------|-------|--------|-------------|\n")
    md.append(f"| String Similarity | {string_score:.4f} | 70% | {string_score * 0.70:.4f} |\n")
    md.append(f"| Semantic Similarity | {semantic_score:.4f} | 30% | {semantic_score * 0.30:.4f} |\n")
    
    base_score = (string_score * 0.70) + (semantic_score * 0.30)
    md.append(f"| **Base Score** | **{base_score:.4f}** | - | **{base_score * 100:.2f}%** |\n")
    
    if acronym_fidelity > 0.0:
        acro_boost = acronym_fidelity * 0.15
        md.append(f"| Acronym Fidelity Boost | {acronym_fidelity:.4f} | 15% max | +{acro_boost:.4f} |\n")

    if location_score > 0.0:
        loc_boost = location_score * 0.05
        md.append(f"| Location Match Boost | {location_score:.4f} | 5% max | +{loc_boost:.4f} |\n")

    if count > 1:
        # Show that popularity played a role if count > 1
        md.append(f"| Popularity Boost | {count} counts | log-scale | YES |\n")
        
    md.append(f"| **Final Score** | **{score/100:.4f}** | - | **{score:.2f}%** |\n\n")
    
    # Show top 5 non-identical matches
    md.append("**Top 5 Non-Self Matches:**\n\n")
    for i, match in enumerate(filtered_matches[:5], 2):
        match_name = match.get('company_name', 'Unknown')
        match_score = match.get('likeness_percent', 0.0)
        m_city = match.get('city', '')
        m_state = match.get('state', '')
        m_loc = f" ({m_city}, {m_state})" if m_city or m_state else ""
        md.append(f"{i}. {match_name}{m_loc} ({match_score:.1f}%)\n")
    md.append("\n")
    
    md.append("---\n\n")
    
    return ''.join(md)


def main():
    print("="*70)
    print("🔬 GENERATING FULL CONTROL SET REPORT (LOCATION-AWARE)")
    print("="*70)
    
    # Initialize service
    print("\n📦 Initializing SearchService...")
    service = SearchService()
    # Explicitly load the location-aware dataset
    if not service.load_company_data(model_name='paraphrase-MiniLM-L3-v2', filename='companies_sample_100k.json'):
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
    
    companies = [item['Company Name'] for item in control_data]
    
    print(f"📋 Loaded {len(companies)} companies from control set\n")
    
    # Generate header
    report_content = [generate_report_header()]
    
    # Process companies
    results = []
    
    for i, company in enumerate(companies, 1):
        print(f"Processing {i}/{len(companies)}: {company}")
        
        # Search
        matches = service.search(company, top_k=10)
        
        result_data = {
            'query': company,
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
