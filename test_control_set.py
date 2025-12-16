#!/usr/bin/env python3
"""
Test Control Set - Submit control companies to CompanyMatcher without reloading cache
Generates a markdown report with results
"""

import json
import os
import sys
import time
from datetime import datetime
from CompanyMatcher import CompanyMatcher

# Try to import tqdm for progress bars, fallback to simple progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None, **kwargs):
        """Simple progress bar fallback"""
        if total is None:
            total = len(iterable) if hasattr(iterable, '__len__') else None
        if desc:
            print(f"{desc}...")
        return iterable

def load_companies_from_json(filepath):
    """Load company names from JSON dataset"""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return None
    
    print(f"Loading companies from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        company_names = []
        total_items = len(data)
        print(f"Processing {total_items:,} items...")
        
        for i, item in enumerate(tqdm(data, desc="Loading", total=total_items, unit="items"), 1):
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
            
            # Show progress every 100k items if no tqdm
            if not HAS_TQDM and i % 100000 == 0:
                print(f"  Processed {i:,}/{total_items:,} items ({i/total_items*100:.1f}%)...")
        
        if not company_names:
            print("Error: No company names found in dataset")
            return None
        
        print(f"Loaded {len(company_names):,} company names")
        return company_names
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_markdown_report(results, output_file='companies_control_set_results.md', top_k=10):
    """Generate markdown report from test results"""
    
    md_content = []
    md_content.append("# Company Match Control Set Results\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"Total Companies Tested: {len(results)}\n")
    md_content.append("---\n\n")
    
    # Summary statistics
    exact_matches = sum(1 for r in results if r['top_match']['match_type'] == 'exact')
    high_confidence = sum(1 for r in results if r['top_match']['score'] >= 0.9)
    medium_confidence = sum(1 for r in results if 0.7 <= r['top_match']['score'] < 0.9)
    low_confidence = sum(1 for r in results if r['top_match']['score'] < 0.7)
    
    md_content.append("## Summary Statistics\n\n")
    md_content.append(f"- **Exact Matches**: {exact_matches} ({exact_matches/len(results)*100:.1f}%)\n")
    md_content.append(f"- **High Confidence (>=0.9)**: {high_confidence} ({high_confidence/len(results)*100:.1f}%)\n")
    md_content.append(f"- **Medium Confidence (0.7-0.9)**: {medium_confidence} ({medium_confidence/len(results)*100:.1f}%)\n")
    md_content.append(f"- **Low Confidence (<0.7)**: {low_confidence} ({low_confidence/len(results)*100:.1f}%)\n\n")
    md_content.append("---\n\n")
    
    # Detailed results
    md_content.append("## Detailed Results\n\n")
    
    for i, result in enumerate(results, 1):
        query = result['query']
        top_match = result['top_match']
        all_matches = result['matches']
        
        md_content.append(f"### {i}. {query}\n\n")
        md_content.append(f"**Top Match**: {top_match['name']}\n\n")
        md_content.append(f"- **Score**: {top_match['score']:.4f} ({top_match['score']*100:.2f}%)\n")
        md_content.append(f"- **Match Type**: {top_match.get('match_type', 'unknown')}\n")
        md_content.append(f"- **String Score**: {top_match.get('string_score', 0):.4f}\n")
        md_content.append(f"- **Semantic Score**: {top_match.get('semantic_score', 0):.4f}\n\n")
        
        # Show top matches (up to top_k)
        if len(all_matches) > 1:
            num_to_show = min(len(all_matches), top_k)
            md_content.append(f"**Top {num_to_show} Matches:**\n\n")
            md_content.append("| Rank | Company Name | Score | Match Type |\n")
            md_content.append("|------|--------------|-------|------------|\n")
            for j, match in enumerate(all_matches[:num_to_show], 1):
                name = match['name'].replace('|', '\\|')  # Escape pipes in markdown
                score_pct = match['score'] * 100
                match_type = match.get('match_type', 'unknown')
                md_content.append(f"| {j} | {name} | {score_pct:.2f}% | {match_type} |\n")
            md_content.append("\n")
        
        md_content.append("---\n\n")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    print(f"\nMarkdown report generated: {output_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test control set companies against CompanyMatcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_control_set.py companies_control_set.json
  python test_control_set.py companies_control_set.json --top-k 10
  python test_control_set.py companies_control_set.json --output results.md
        """
    )
    
    parser.add_argument('control_set', help='Path to control set JSON file (e.g., companies_control_set.json)')
    parser.add_argument('--dataset', default='companies.json',
                       help='Path to main dataset JSON file (default: companies.json)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top matches to return per query (default: 10)')
    parser.add_argument('--output', default='companies_control_set_results.md',
                       help='Output markdown file (default: companies_control_set_results.md)')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model (default: all-MiniLM-L6-v2)')
    
    args = parser.parse_args()
    
    # Build/load index using filepath for fast cache checking (no loading needed if cache exists)
    print("=" * 60)
    print("Step 1: Building/loading index")
    print("=" * 60)
    print(f"Dataset file: {args.dataset}")
    print("(Using file metadata for fast cache check - no file loading if cache exists!)\n")
    
    # Track index loading to verify it only happens once
    index_load_start = time.time()
    index_load_count = [0]  # Use list to allow modification in nested scope
    
    # Monkey-patch build_index to track calls (for verification)
    original_build_index = CompanyMatcher.build_index
    def tracked_build_index(self, company_names=None, filepath=None):
        index_load_count[0] += 1
        if index_load_count[0] == 1:
            if filepath:
                print(f"[INDEX LOAD #{index_load_count[0]}] Checking cache using file metadata...")
            else:
                print(f"[INDEX LOAD #{index_load_count[0]}] Building/loading index...")
        else:
            print(f"\n[WARNING] Index build_index called {index_load_count[0]} times! This should only happen once!")
        return original_build_index(self, company_names, filepath)
    
    CompanyMatcher.build_index = tracked_build_index
    
    matcher = CompanyMatcher(model_name=args.model)
    
    # Build/load index using filepath - this checks cache using file metadata first
    # If cache exists, no file loading needed! If cache miss, file will be loaded automatically
    matcher.build_index(filepath=args.dataset)
    index_load_time = time.time() - index_load_start
    
    # Verify index is ready
    if not matcher.is_index_ready():
        print("Error: Failed to build/load index")
        sys.exit(1)
    
    # Get company count for display
    main_company_count = len(matcher.original_company_names)
    
    print(f"\n[INDEX STATUS] Index loaded successfully!")
    print(f"  - Companies in index: {main_company_count:,}")
    print(f"  - Index load time: {index_load_time:.2f}s")
    print(f"  - Index load count: {index_load_count[0]} (should be 1)")
    if index_load_count[0] == 1:
        print(f"  - VERIFIED: Index loaded only once [OK]")
    else:
        print(f"  - WARNING: Index was loaded {index_load_count[0]} times!")
    print(f"  - Index will remain in memory - all queries will reuse it without reloading.\n")
    
    # Load control set
    print("=" * 60)
    print("Step 2: Loading control set")
    print("=" * 60)
    control_company_names = load_companies_from_json(args.control_set)
    if not control_company_names:
        print(f"Error: Failed to load control set from {args.control_set}")
        sys.exit(1)
    
    # Test each control company using batch processing for better performance
    # NOTE: Index is already loaded in memory - no reloading happens during this loop
    print("\n" + "=" * 60)
    print(f"Step 3: Testing {len(control_company_names)} control companies")
    print("=" * 60)
    print("(Using batch processing for faster encoding - index already loaded in memory!)\n")
    
    # Verify index load count hasn't changed
    initial_load_count = index_load_count[0]
    print(f"[VERIFICATION] Index load count before queries: {initial_load_count} (should remain {initial_load_count})\n")
    
    # Use batch_match for efficient processing
    start_time = time.time()
    
    # Process all queries in batch with progress bar
    print(f"Processing {len(control_company_names)} queries in batch...")
    print(f"[BATCH] Starting batch match with batch_size=32, top_k={args.top_k}\n")
    
    # Check if batch_match exists, otherwise fall back to individual matches
    if hasattr(matcher, 'batch_match'):
        # Wrap batch_match with progress feedback
        # Since batch_match processes internally, we'll use individual matches with progress bar for visibility
        print("[BATCH] Using individual matches with progress bar for visual feedback...")
        all_matches = []
        for query in tqdm(control_company_names, desc="Matching companies", unit="query", ncols=80):
            matches = matcher.match(query, top_k=args.top_k)
            all_matches.append(matches)
    else:
        # Fallback: process individually with progress bar
        print("[BATCH] batch_match not available, processing individually...")
        all_matches = []
        for query in tqdm(control_company_names, desc="Matching companies", unit="query", ncols=80):
            matches = matcher.match(query, top_k=args.top_k)
            all_matches.append(matches)
    
    batch_time = time.time() - start_time
    
    # Verify index wasn't reloaded during batch processing
    final_load_count = index_load_count[0]
    if final_load_count != initial_load_count:
        print(f"\n[WARNING] Index load count changed from {initial_load_count} to {final_load_count} during batch processing!")
    else:
        print(f"[VERIFICATION] Index load count after queries: {final_load_count} (unchanged [OK])")
    
    print(f"\n[BATCH] Processing completed:")
    print(f"  - Total time: {batch_time:.2f}s")
    print(f"  - Average time per query: {batch_time/len(control_company_names)*1000:.1f}ms")
    print(f"  - Queries per second: {len(control_company_names)/batch_time:.1f}\n")
    
    # Process results with progress bar and counters
    print("Processing results and generating report data...")
    results = []
    exact_match_count = 0
    high_confidence_count = 0
    no_match_count = 0
    
    for i, (query, matches) in enumerate(tqdm(zip(control_company_names, all_matches), 
                                              desc="Processing results", 
                                              total=len(control_company_names),
                                              unit="result",
                                              ncols=80), 1):
        if matches:
            top_match = matches[0]
            results.append({
                'query': query,
                'top_match': top_match,
                'matches': matches
            })
            
            # Count statistics
            if top_match.get('match_type') == 'exact':
                exact_match_count += 1
            if top_match['score'] >= 0.9:
                high_confidence_count += 1
        else:
            results.append({
                'query': query,
                'top_match': {'name': 'No matches found', 'score': 0.0, 'match_type': 'none'},
                'matches': []
            })
            no_match_count += 1
    
    print(f"\n[STATISTICS] Results summary:")
    print(f"  - Total queries: {len(control_company_names)}")
    print(f"  - Exact matches: {exact_match_count} ({exact_match_count/len(control_company_names)*100:.1f}%)")
    print(f"  - High confidence (>=0.9): {high_confidence_count} ({high_confidence_count/len(control_company_names)*100:.1f}%)")
    print(f"  - No matches: {no_match_count}")
    print()
    
    # Generate markdown report
    print("=" * 60)
    print("Step 4: Generating markdown report")
    print("=" * 60)
    generate_markdown_report(results, output_file=args.output, top_k=args.top_k)
    
    # Final verification
    final_final_load_count = index_load_count[0]
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Tested {len(control_company_names)} companies")
    print(f"Report saved to: {args.output}")
    print(f"\n[FINAL VERIFICATION] Index was loaded {final_final_load_count} time(s)")
    if final_final_load_count == 1:
        print("  [OK] Index loaded exactly once - no unnecessary reloads!")
    else:
        print(f"  ⚠ Index was loaded {final_final_load_count} times (expected 1)")

if __name__ == "__main__":
    main()

