#!/usr/bin/env python3
"""
FAST ITERATION Verification Script - Optimized for rapid testing of matching logic changes.

Key optimizations:
1. Uses SearchService singleton - data loaded ONCE and reused
2. Minimal output during processing (progress bar only)
3. Quick re-runs without reloading 176MB JSON or 4.5GB cache
4. Timing instrumentation to measure performance

Usage:
    python tests\verify_control_set_fast.py              # Run all 104 companies
    python tests\verify_control_set_fast.py --limit 10   # Test first 10 only
    python tests\verify_control_set_fast.py --acronyms   # Test only acronyms (ABA, PDMA, IBM, GE)
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


def generate_markdown_report(results, output_file='companies_control_set_results.md', top_k=10):
    """Generate markdown report from test results"""
    
    md_content = []
    md_content.append("# Company Match Control Set Results\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"Total Companies Tested: {len(results)}\n")
    md_content.append("---\n\n")
    
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
        
        if len(all_matches) > 1:
            num_to_show = min(len(all_matches), top_k)
            md_content.append(f"**Top {num_to_show} Matches:**\n\n")
            md_content.append("| Rank | Company Name | Score | Rationale |\n")
            md_content.append("|------|--------------|-------|-----------|\\n")
            for j, match in enumerate(all_matches[:num_to_show], 1):
                name = match['company_name'].replace('|', '\\|')
                score_pct = match['likeness_percent']
                # Truncate rationale for table
                rationale = match.get('match_rationale', '').split('\n')[0][:50] + "..."
                md_content.append(f"| {j} | {name} | {score_pct:.1f}% | {rationale} |\n")
            md_content.append("\n")
        
        md_content.append("---\n\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    print(f"📄 Report saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Fast iteration control set verification')
    parser.add_argument('--limit', type=int, help='Limit number of companies to test')
    parser.add_argument('--acronyms', action='store_true', help='Test only acronyms (ABA, PDMA, IBM, GE)')
    parser.add_argument('--top-k', type=int, default=20, help='Number of matches to retrieve (default: 20)')
    parser.add_argument('--output', default='companies_control_set_results.md', help='Output markdown file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 FAST ITERATION - Control Set Verification")
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
    
    # Step 4: Generate report
    print(f"\n📊 Step 4: Generating Report...")
    generate_markdown_report(results, output_file=args.output, top_k=args.top_k)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print(f"   Total time: {time.time() - init_start:.2f}s")
    print(f"   Initialization: {init_time:.2f}s")
    print(f"   Matching: {match_time:.2f}s ({avg_time*1000:.1f}ms/query)")
    print(f"   Companies tested: {len(control_names)}")
    print(f"   Report: {args.output}")
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
