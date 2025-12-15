#!/usr/bin/env python3
"""
Test Control Set - Submit control companies to CompanyMatcher without reloading cache
Generates a markdown report with results
"""

import json
import os
import sys
from datetime import datetime
from CompanyMatcher import CompanyMatcher

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
        for item in data:
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
        
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
    md_content.append(f"- **High Confidence (≥0.9)**: {high_confidence} ({high_confidence/len(results)*100:.1f}%)\n")
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
    
    # Load main dataset and build index (will use cache if available)
    print("=" * 60)
    print("Step 1: Loading main dataset and building index")
    print("=" * 60)
    main_company_names = load_companies_from_json(args.dataset)
    if not main_company_names:
        print(f"Error: Failed to load main dataset from {args.dataset}")
        sys.exit(1)
    
    # Initialize CompanyMatcher - index will be loaded once and reused for all queries
    print(f"\nInitializing CompanyMatcher with {len(main_company_names):,} companies...")
    print("(Index will be loaded once from cache if available - no reloading during queries!)")
    
    matcher = CompanyMatcher(model_name=args.model)
    
    # Build/load index once - this will use cache if available, otherwise build new
    # After this call, the index stays in memory and is reused for all queries
    matcher.build_index(main_company_names)
    
    # Verify index is ready
    if not matcher.is_index_ready():
        print("Error: Failed to build/load index")
        sys.exit(1)
    
    print(f"Index loaded successfully! {len(matcher.original_company_names):,} companies ready.")
    print("Index will remain in memory - all queries will reuse it without reloading.\n")
    
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
    
    # Use batch_match for efficient processing
    import time
    start_time = time.time()
    
    # Process all queries in batch
    print(f"Processing {len(control_company_names)} queries in batch...")
    all_matches = matcher.batch_match(control_company_names, top_k=args.top_k, batch_size=32)
    
    batch_time = time.time() - start_time
    print(f"Batch processing completed in {batch_time:.2f}s ({batch_time/len(control_company_names)*1000:.1f}ms per query)\n")
    
    # Process results
    results = []
    for i, (query, matches) in enumerate(zip(control_company_names, all_matches), 1):
        print(f"[{i}/{len(control_company_names)}] Testing: {query}")
        
        if matches:
            top_match = matches[0]
            results.append({
                'query': query,
                'top_match': top_match,
                'matches': matches
            })
            print(f"  -> Top match: {top_match['name']} (score: {top_match['score']:.4f}, type: {top_match.get('match_type', 'unknown')})")
        else:
            results.append({
                'query': query,
                'top_match': {'name': 'No matches found', 'score': 0.0, 'match_type': 'none'},
                'matches': []
            })
            print(f"  -> No matches found")
        print()
    
    # Generate markdown report
    print("=" * 60)
    print("Step 4: Generating markdown report")
    print("=" * 60)
    generate_markdown_report(results, output_file=args.output, top_k=args.top_k)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Tested {len(control_company_names)} companies")
    print(f"Report saved to: {args.output}")

if __name__ == "__main__":
    main()

