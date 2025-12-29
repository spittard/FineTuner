#!/usr/bin/env python3
"""
Company Name Search CLI
Refactored to use the optimized src.finetuner.core.matcher.CompanyMatcher.
"""
import os
import sys
import argparse
import time

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from finetuner.core.matcher import CompanyMatcher
except ImportError as e:
    print(f"Error importing CompanyMatcher: {e}")
    print("Please ensure the 'src' directory is in your Python path.")
    sys.exit(1)

def print_results(query, results, verbose=False):
    if not results:
        print(f"\nNo matches found for '{query}'")
        return
    
    print(f"\nTop {len(results)} matches for '{query}':")
    print("-" * 80)
    print(f"{'#':<3} {'Company':<40} {'Score':<8} {'Type':<10} {'Details' if verbose else ''}")
    print("-" * 80)
    
    for i, res in enumerate(results, 1):
        score_pct = res['score'] * 100
        match_type = res.get('match_type', 'hybrid')
        
        details = ""
        if verbose:
            details = f"Str: {res.get('string_score', 0):.2f}, Sem: {res.get('semantic_score', 0):.2f}"
            if 'location_score' in res:
                details += f", Loc: {res['location_score']:.2f}"
                
        print(f"{i:<3} {res['name'][:40]:<40} {score_pct:5.1f}%   {match_type:<10} {details}")
    print("-" * 80)

def search_loop(matcher):
    print("\nInteractive Search Mode")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            query = input("\nEnter company name: ").strip()
            if query.lower() in ('quit', 'exit'):
                break
            if not query:
                continue
            
            start = time.time()
            # Pass city/state from args if available in loop context (simplified for now to global args)
            results = matcher.match_with_location(query, city=getattr(matcher, '_default_city', None), 
                                                state=getattr(matcher, '_default_state', None), top_k=10)
            elapsed = time.time() - start
            
            print_results(query, results, verbose=True)
            print(f"(Search took {elapsed:.3f}s)")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Company Name Search (Powered by FineTuner)")
    parser.add_argument("query", nargs="?", help="Company name to search for")
    parser.add_argument("--data", default="training_data.json", help="Path to training data JSON file")
    parser.add_argument("--top", type=int, default=10, help="Number of results to return")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--batch", nargs="+", help="Batch search multiple queries")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild index ignoring cache")
    parser.add_argument("--city", help="City for location-aware matching")
    parser.add_argument("--state", help="State for location-aware matching")
    
    args = parser.parse_args()
    
    print(f"Initializing CompanyMatcher...")
    try:
        matcher = CompanyMatcher()
        
        if args.no_cache:
            print("Forcing index rebuild...")
            # Ideally matcher would have force_rebuild param, but checks file timestamp. 
            # We can clear cache manually using clear_cache if strictly needed, 
            # but for now we rely on build_index logic.
            matcher.clear_cache()
            
        # Initialize index
        if not os.path.exists(args.data):
            print(f"Error: Data file '{args.data}' not found.")
            return
            
        matcher.build_index(filepath=args.data)
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Mode selection
    if args.interactive or (not args.query and not args.batch):
        search_loop(matcher)
    elif args.batch:
        print(f"Batch searching {len(args.batch)} queries...")
        for q in args.batch:
            results = matcher.match_with_location(q, city=args.city, state=args.state, top_k=args.top)
            print_results(q, results)
    elif args.query:
        # Store for interactive loop if needed
        matcher._default_city = args.city
        matcher._default_state = args.state
        results = matcher.match_with_location(args.query, city=args.city, state=args.state, top_k=args.top)
        print_results(args.query, results, verbose=True)

if __name__ == "__main__":
    main()
