#!/usr/bin/env python3
"""
ENHANCED FAST ITERATION Verification Script - Complete scoring details and relative positioning.

Key features:
1. Uses SearchService singleton - data loaded ONCE and reused
2. Complete score breakdown for all components
3. Relative positioning explanations (why ranked above/below adjacent matches)
4. Score differentials and component analysis

Usage:
    python tests\\verify_control_set_fast_enhanced.py              # Run all 104 companies
    python tests\\verify_control_set_fast_enhanced.py --limit 10   # Test first 10 only
    python tests\\verify_control_set_fast_enhanced.py --acronyms   # Test only acronyms
"""

import json
import os
import sys
import time
from datetime import datetime
import argparse

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.web.services.search_service import SearchService
from finetuner.web.services.rationale_service import RationaleService

# Try to import tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable


def load_control_set(filepath='companies_control_set.json'):
    """Load company names from control set JSON"""
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    company_names = [item["Company Name"] for item in data if "Company Name" in item]
    return company_names


def generate_enhanced_markdown_report(results, output_file='companies_control_set_results_enhanced.md', top_k=10):
    """Generate enhanced markdown report with complete scoring details"""
    
    md_content = []
    md_content.append("# Company Match Control Set Results (Enhanced)\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"Total Companies Tested: {len(results)}\n")
    md_content.append("---\n\n")
    
    md_content.append("> [!NOTE]\n")
    md_content.append("> This enhanced report includes complete score breakdowns and relative positioning explanations.\n")
    md_content.append("> Each match shows all scoring components and why it ranks above/below adjacent matches.\n\n")
    
    # Detailed results
    md_content.append("## Detailed Results\n\n")
    
    for i, result in enumerate(results, 1):
        query = result['query']
        top_match = result['top_match']
        all_matches = result['matches']
        
        md_content.append(f"### {i}. {query}\n\n")
        md_content.append(f"**Top Match**: {top_match['company_name']}\n\n")
        md_content.append(f"- **Final Score**: {top_match['raw_score']:.4f} ({top_match['likeness_percent']}%) \n")
        
        # Format the rationale as a blockquote
        rationale_lines = top_match.get('match_rationale', '').split('\n')
        md_content.append(f"- **Top Match Rationale**:\n")
        for line in rationale_lines:
            md_content.append(f"  > {line}\n")
        md_content.append("\n")
        
        # Add detailed score breakdown for top match
        score_breakdown = RationaleService.generate_detailed_score_breakdown(top_match, query)
        md_content.append(score_breakdown)
        md_content.append("\n")
        
        if len(all_matches) > 1:
            num_to_show = min(len(all_matches), top_k)
            md_content.append(f"**Top {num_to_show} Matches with Complete Scoring:**\n\n")
            
            # Enhanced table with all score components
            md_content.append("| Rank | Company Name | Final | String | Semantic | Acronym | Location | Match Type |\n")
            md_content.append("|------|--------------|-------|--------|----------|---------|----------|------------|\n")
            
            for j, match in enumerate(all_matches[:num_to_show], 1):
                name = match['company_name'].replace('|', '\\|')[:40]  # Truncate long names
                final_score = match.get('raw_score', match.get('score', 0.0))
                string_score = match.get('string_score', 0.0)
                sem_score = match.get('normalized_semantic_score', match.get('semantic_score', 0.0))
                acro_score = match.get('acronym_fidelity', 0.0)
                loc_score = match.get('location_score', 0.0)
                match_type = match.get('match_type', 'hybrid')[:10]
                
                md_content.append(f"| {j} | {name} | {final_score:.4f} | {string_score:.4f} | {sem_score:.4f} | {acro_score:.4f} | {loc_score:.4f} | {match_type} |\n")
            
            md_content.append("\n")
            
            # Add detailed breakdown for each match with relative positioning
            md_content.append(f"### Detailed Match Analysis (Top {min(5, num_to_show)})\n\n")
            
            for j, match in enumerate(all_matches[:min(5, num_to_show)], 1):
                md_content.append(f"#### Match #{j}: {match['company_name']}\n\n")
                
                # Score breakdown
                breakdown = RationaleService.generate_detailed_score_breakdown(match, query)
                md_content.append(breakdown)
                md_content.append("\n")
                
                # Relative positioning
                match_above = all_matches[j-2] if j > 1 else None
                match_below = all_matches[j] if j < len(all_matches) else None
                
                positioning = RationaleService.generate_relative_positioning_explanation(
                    match, match_above, match_below, j
                )
                md_content.append(positioning)
                md_content.append("\n")
                
                # Rationale
                md_content.append("### Match Rationale\n\n")
                rationale_lines = match.get('match_rationale', '').split('\n')
                for line in rationale_lines:
                    md_content.append(f"> {line}\n")
                md_content.append("\n---\n\n")
        
        md_content.append("---\n\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    print(f"📄 Enhanced report saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced fast iteration control set verification')
    parser.add_argument('--limit', type=int, help='Limit number of companies to test')
    parser.add_argument('--acronyms', action='store_true', help='Test only acronyms (ABA, PDMA, IBM, GE)')
    parser.add_argument('--top-k', type=int, default=20, help='Number of matches to retrieve (default: 20)')
    parser.add_argument('--output', default='companies_control_set_results_enhanced.md', help='Output markdown file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 ENHANCED FAST ITERATION - Control Set Verification")
    print("=" * 70)
    
    # Step 1: Initialize SearchService (singleton - reuses loaded data)
    print("\n📦 Step 1: Initializing SearchService...")
    init_start = time.time()
    
    service = SearchService()
    if not service.load_company_data():
        print("❌ Failed to load company data")
        return
    
    init_time = time.time() - init_start
    print(f"✅ Service ready in {init_time:.2f}s")
    
    # Step 2: Load control set
    print("\n📋 Step 2: Loading Control Set...")
    control_names = load_control_set()
    if not control_names:
        return
    
    # Filter if requested
    if args.acronyms:
        acronym_list = ['ABA', 'PDMA', 'IBM', 'GE']
        control_names = [name for name in control_names if name in acronym_list]
        print(f"🔍 Filtering to acronyms only: {control_names}")
    elif args.limit:
        control_names = control_names[:args.limit]
        print(f"🔍 Limited to first {args.limit} companies")
    
    print(f"✅ Testing {len(control_names)} companies")
    
    # Step 3: Run matching tests
    print(f"\n🔬 Step 3: Running {len(control_names)} Matching Tests...")
    print(f"   Top-K: {args.top_k}")
    
    results = []
    match_start = time.time()
    
    for query in tqdm(control_names, desc="   Matching", unit="query", ncols=70):
        matches = service.search(query, top_k=args.top_k)
        
        if matches:
            top_match = matches[0]
            results.append({
                'query': query,
                'top_match': top_match,
                'matches': matches
            })
        else:
            results.append({
                'query': query,
                'top_match': {'company_name': 'No matches found', 'raw_score': 0.0, 'likeness_percent': 0.0},
                'matches': []
            })
    
    match_time = time.time() - match_start
    avg_time = match_time / len(control_names) if control_names else 0
    
    print(f"✅ Completed in {match_time:.2f}s ({avg_time*1000:.1f}ms per query)")
    
    # Step 4: Generate enhanced report
    print(f"\n📊 Step 4: Generating Enhanced Report...")
    generate_enhanced_markdown_report(results, output_file=args.output, top_k=args.top_k)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print(f"   Total time: {time.time() - init_start:.2f}s")
    print(f"   Initialization: {init_time:.2f}s")
    print(f"   Matching: {match_time:.2f}s ({avg_time*1000:.1f}ms/query)")
    print(f"   Companies tested: {len(control_names)}")
    print(f"   Enhanced Report: {args.output}")
    print("=" * 70)
    
    # Quick preview of acronym results if testing acronyms
    if args.acronyms:
        print("\n🔍 Acronym Results Preview:")
        for result in results:
            query = result['query']
            top = result['top_match']
            print(f"   {query:10s} → {top['company_name'][:50]:50s} ({top['likeness_percent']:.1f}%)")


if __name__ == "__main__":
    main()
