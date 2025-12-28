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
    header.append("# Company Matching Control Set Report\n\n")
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
    header.append("- Example: `\"IBM\"` → `\"International Business Machines\"` (98%+ match)\n")
    header.append("- Example: `\"ABA\"` → `\"American Bar Association\"` (98%+ match)\n\n")
    
    header.append("### 3. **Typo Tolerance**\n")
    header.append("Fuzzy matching handles common spelling errors.\n")
    header.append("- Example: `\"Microsft\"` → `\"Microsoft\"` (94%+ match)\n")
    header.append("- Example: `\"Gogle\"` → `\"Google\"` (92%+ match)\n\n")
    
    header.append("### 4. **Abbreviation Variations**\n")
    header.append("Recognizes common business abbreviations.\n")
    header.append("- Example: `\"Corp\"` ↔ `\"Corporation\"`\n")
    header.append("- Example: `\"Inc\"` ↔ `\"Incorporated\"`\n")
    header.append("- Example: `\"Intl\"` ↔ `\"International\"`\n\n")
    
    header.append("### 5. **Plural/Singular Variations**\n")
    header.append("Handles grammatical number differences.\n")
    header.append("- Example: `\"International Business Machine\"` vs `\"International Business Machines\"`\n\n")
    
    header.append("### 6. **Word Order Variations**\n")
    header.append("Matches despite different word arrangements.\n")
    header.append("- Example: `\"Bank First National\"` → `\"First National Bank\"`\n\n")
    
    header.append("### 7. **Partial Name Matches**\n")
    header.append("Finds matches when only part of the company name is provided.\n")
    header.append("- Example: `\"Acme\"` → `\"Acme Corporation\"`\n\n")
    
    header.append("### 8. **Legal Entity Suffix Variations**\n")
    header.append("Handles different legal entity designations.\n")
    header.append("- Example: `\"Acme LLC\"` vs `\"Acme Inc\"` vs `\"Acme Corporation\"`\n\n")
    
    header.append("---\n\n")
    
    # Scoring formula
    header.append("## Scoring Formula\n\n")
    header.append("```\n")
    header.append("Base Score = (String Similarity × 70%) + (Semantic Similarity × 30%)\n")
    header.append("Final Score = Base Score + Acronym Fidelity Boost (up to +15%)\n")
    header.append("```\n\n")
    
    # Fidelity explanation with examples
    header.append("## Acronym Fidelity Score Explained\n\n")
    header.append("The **Acronym Fidelity Score** measures how well a company name expands an acronym ")
    header.append("using **pure algorithmic pattern matching**. It analyzes whether the first letters of ")
    header.append("significant words match the acronym letters in order.\n\n")
    
    header.append("### How It Works\n\n")
    header.append("The algorithm extracts the first letter of each significant word (excluding common words ")
    header.append("like 'the', 'of', 'and') and compares them to the acronym:\n\n")
    
    header.append("**Example 1: Perfect Match (Fidelity = 1.00)**\n")
    header.append("```\n")
    header.append("Query: \"IBM\"\n")
    header.append("Match: \"International Business Machines\"\n")
    header.append("\n")
    header.append("Step 1: Extract first letters\n")
    header.append("  Words: [International, Business, Machines]\n")
    header.append("  First letters: [I, B, M]\n")
    header.append("  Joined: \"IBM\"\n")
    header.append("\n")
    header.append("Step 2: Compare to query\n")
    header.append("  Query: \"IBM\"\n")
    header.append("  Word starts: \"IBM\"\n")
    header.append("  Match: EXACT ✓\n")
    header.append("\n")
    header.append("Step 3: Check for overlaps\n")
    header.append("  Does any word contain multiple acronym letters? NO ✓\n")
    header.append("\n")
    header.append("Result: Fidelity = 1.00 (Perfect Expansion)\n")
    header.append("```\n\n")
    
    header.append("**Example 2: Prefix Match (Fidelity = 0.95)**\n")
    header.append("```\n")
    header.append("Query: \"IBM\"\n")
    header.append("Match: \"International Business Machines Corporation\"\n")
    header.append("\n")
    header.append("First letters: [I, B, M, C] → \"IBMC\"\n")
    header.append("Query: \"IBM\"\n")
    header.append("Pattern: \"IBM\" is a prefix of \"IBMC\" ✓\n")
    header.append("\n")
    header.append("Result: Fidelity = 0.95 (Acronym matches start, extra words after)\n")
    header.append("```\n\n")
    
    header.append("**Example 3: Subsequence Match (Fidelity = 0.90)**\n")
    header.append("```\n")
    header.append("Query: \"IBM\"\n")
    header.append("Match: \"International Bureau of Management\"\n")
    header.append("\n")
    header.append("First letters: [I, B, M] → \"IBM\" (skipping 'of')\n")
    header.append("Pattern: \"IBM\" appears as subsequence in word starts ✓\n")
    header.append("\n")
    header.append("Result: Fidelity = 0.90 (Subsequence match)\n")
    header.append("```\n\n")
    
    header.append("### Fidelity Score Reference\n\n")
    header.append("| Score | Pattern | Example |\n")
    header.append("|-------|---------|----------|\n")
    header.append("| 1.00 | Perfect: Each letter = first letter of distinct word, no overlaps | IBM → International Business Machines |\n")
    header.append("| 0.95 | Prefix: Acronym matches start, extra words after | IBM → International Business Machines Corp |\n")
    header.append("| 0.90 | Subsequence: Acronym appears in word-starts | IBM → International Bureau of Management |\n")
    header.append("| 0.70 | Collision: Words overlap with acronym letters | (penalized) |\n")
    header.append("| 0.65 | Word prefix: First word starts with acronym | IBMA → IBM... |\n")
    header.append("| 0.40 | Partial: Some letters match out of order | (scaled down) |\n\n")
    
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
    # This is critical for identifying abbreviation bias in Rank 2+ results
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
    
    # Highlight that this is the first non-identical result
    md.append(f"**Top Non-Self Match (Rank 2):** `{company_name}` ({score:.1f}%)\n\n")
    
    # Extract score components
    string_score = top_match.get('string_score', 0.0)
    semantic_score = top_match.get('normalized_semantic_score', top_match.get('semantic_score', 0.0))
    acronym_fidelity = top_match.get('acronym_fidelity', 0.0)
    match_type = top_match.get('match_type', 'hybrid')
    
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
        md.append(f"| **Final Score** | **{score/100:.4f}** | - | **{score:.2f}%** |\n\n")
    else:
        md.append(f"| **Final Score** | **{score/100:.4f}** | - | **{score:.2f}%** |\n\n")
    
    # Show top 5 non-identical matches
    md.append("**Top 5 Non-Self Matches:**\n\n")
    for i, match in enumerate(filtered_matches[:5], 2):
        match_name = match.get('company_name', 'Unknown')
        match_score = match.get('likeness_percent', 0.0)
        md.append(f"{i}. {match_name} ({match_score:.1f}%)\n")
    md.append("\n")
    
    md.append("---\n\n")
    
    return ''.join(md)


def main():
    print("="*70)
    print("🔬 GENERATING FULL CONTROL SET REPORT")
    print("="*70)
    
    # Initialize service
    print("\n📦 Initializing SearchService...")
    service = SearchService()
    if not service.load_company_data(model_name='paraphrase-MiniLM-L3-v2'):
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
            output_file = 'control_set_report_FULL.md'
            print(f"\n📄 Writing initial report (first 10 companies) to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(''.join(report_content))
            print(f"✅ Initial report written! Continuing with remaining {len(companies) - 10} companies...\n")
    
    # Write final report
    output_file = 'control_set_report_FULL.md'
    print(f"\n📄 Writing final report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report_content))
    
    print(f"\n✅ COMPLETE! Processed {len(companies)} companies")
    print(f"📊 Report saved: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
