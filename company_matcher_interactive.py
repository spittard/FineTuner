#!/usr/bin/env python3
"""
Interactive Company Matcher
Direct interactive mode for CompanyMatcher - keeps cache loaded for multiple queries
"""

import json
import os
import sys

# Add src to python path to access finetuner package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.core.matcher import CompanyMatcher

def load_company_names(dataset_path):
    """Load company names from JSON dataset"""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found")
        return None
    
    print(f"Loading company data from: {dataset_path}")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        company_names = []
        for item in data:
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
        
        if not company_names:
            print("Error: No company names found in dataset")
            return None
        
        print(f"[OK] Loaded {len(company_names):,} company names")
        return company_names
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def print_results(query, matches):
    """Print search results in a formatted table"""
    if not matches:
        print(f"\nNo matches found for '{query}'")
        return
    
    print(f"\nTop {len(matches)} matches:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Company Name':<50} {'Score':<8} {'Type':<10}")
    print("-" * 80)
    
    for i, match in enumerate(matches, 1):
        score_percent = match['score'] * 100
        match_type = match.get('match_type', 'unknown')
        company_name = match['name'][:48]  # Truncate if too long
        print(f"{i:<6} {company_name:<50} {score_percent:>6.1f}%  {match_type:<10}")
    
    print("-" * 80)

def print_explanation(query, top_match, matcher):
    """Print detailed explanation for top match"""
    explanation = matcher.explain_match(query, top_match['name'])
    
    print(f"\nTop Match Details: '{top_match['name']}'")
    print(f"  Match type: {explanation.get('match_type', 'unknown')}")
    
    if 'string_score' in explanation:
        print(f"  String score: {explanation.get('string_score', 0):.3f}")
        print(f"  Semantic score: {explanation.get('semantic_score', 0):.3f}")
        print(f"  Final score: {explanation.get('final_score', 0):.3f}")
    
    if explanation.get('query_tokens'):
        print(f"  Query tokens: {', '.join(explanation['query_tokens'])}")
    if explanation.get('match_tokens'):
        print(f"  Match tokens: {', '.join(explanation['match_tokens'])}")
    if explanation.get('overlap'):
        print(f"  Overlap: {', '.join(explanation['overlap'])}")
        print(f"  Overlap score: {explanation.get('overlap_score', 0):.3f}")

def interactive_mode(matcher, top_k=10):
    """Run interactive search mode"""
    print("\n" + "=" * 60)
    print("Company Matcher - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  <query>     - Search for company name")
    print("  help        - Show this help")
    print("  stats       - Show statistics")
    print("  quit/exit/q - Exit interactive mode")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nEnter company name to search: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode...")
                break
            elif query.lower() == 'help':
                print("\nCommands:")
                print("  <query>     - Search for company name")
                print("  help        - Show this help")
                print("  stats       - Show statistics")
                print("  quit/exit/q - Exit interactive mode")
                continue
            elif query.lower() == 'stats':
                print(f"\nStatistics:")
                print(f"  Total companies: {len(matcher.original_company_names):,}")
                print(f"  Model: {matcher.model_name}")
                print(f"  Index ready: {matcher.is_index_ready()}")
                continue
            
            # Perform search
            print(f"\nSearching for: '{query}'")
            matches = matcher.match(query, top_k=top_k)
            
            # Print results
            print_results(query, matches)
            
            # Show explanation for top match
            if matches:
                print_explanation(query, matches[0], matcher)
        
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError during search: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive Company Matcher - Direct CompanyMatcher interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python company_matcher_interactive.py companies.json
  python company_matcher_interactive.py companies.json --top-k 5
  python company_matcher_interactive.py companies.json --model all-MiniLM-L6-v2
        """
    )
    
    parser.add_argument('dataset', help='Path to JSON dataset file (e.g., companies.json)')
    parser.add_argument('--top-k', type=int, default=10, 
                       help='Number of top matches to return (default: 10)')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model (default: all-MiniLM-L6-v2)')
    
    args = parser.parse_args()
    
    # Load company names
    company_names = load_company_names(args.dataset)
    if not company_names:
        sys.exit(1)
    
    # Initialize CompanyMatcher
    print(f"\nBuilding company matching index with {len(company_names):,} companies...")
    print("(This will use cache if available - much faster!)")
    
    matcher = CompanyMatcher(model_name=args.model)
    matcher.build_index(company_names)
    
    print(f"[OK] Index ready! {len(company_names):,} companies loaded.")
    print("[OK] Cache loaded - all queries will be fast!")
    
    # Start interactive mode
    interactive_mode(matcher, top_k=args.top_k)

if __name__ == "__main__":
    main()
