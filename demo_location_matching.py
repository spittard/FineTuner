#!/usr/bin/env python3
"""
Interactive demo of location-aware company matching.
"""

from CompanyMatcher import CompanyMatcher

# Sample data with companies in multiple locations
DEMO_DATA = [
    {"Company Name": "Acme Corporation", "City": "New York", "State": "NY", "Count": 150},
    {"Company Name": "Acme Corporation", "City": "Los Angeles", "State": "CA", "Count": 85},
    {"Company Name": "Acme Corporation", "City": "Chicago", "State": "IL", "Count": 42},
    {"Company Name": "Acme Inc", "City": "Boston", "State": "MA", "Count": 28},
    {"Company Name": "First National Bank", "City": "Houston", "State": "TX", "Count": 200},
    {"Company Name": "First National Bank", "City": "Miami", "State": "FL", "Count": 175},
    {"Company Name": "First National Bank of Texas", "City": "Dallas", "State": "TX", "Count": 95},
    {"Company Name": "Global Tech Solutions", "City": "San Francisco", "State": "CA", "Count": 320},
    {"Company Name": "Global Tech", "City": "Seattle", "State": "WA", "Count": 180},
    {"Company Name": "Smith & Associates", "City": "Denver", "State": "CO", "Count": 55},
]

def print_results(results, title):
    """Pretty print search results"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Company Name':<30} {'Score':<8} {'City':<15} {'State':<6} {'Count':<8}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['name']:<30} {r['score']*100:>5.1f}%  {r.get('city', ''):<15} {r.get('state', ''):<6} {r.get('count', 0):<8}")
    print("-" * 70)


def main():
    print("\n" + "#"*70)
    print("#  LOCATION-AWARE COMPANY MATCHING DEMO")
    print("#"*70)
    
    # Initialize matcher with demo data
    print("\nLoading demo data with 10 companies across multiple locations...")
    matcher = CompanyMatcher()
    matcher.build_index_with_location(data=DEMO_DATA)
    
    print(f"\n[OK] Loaded {len(matcher.original_company_names)} companies")
    print(f"[OK] Location data: {'Available' if matcher.has_location_data else 'Not available'}")
    
    # Demo 1: Basic search (no location filter)
    print_results(
        matcher.match_with_location("Acme", top_k=5),
        "DEMO 1: Search for 'Acme' (no location filter)"
    )
    
    # Demo 2: Search with city filter
    print_results(
        matcher.match_with_location("Acme", city="New York", top_k=5),
        "DEMO 2: Search for 'Acme' in New York"
    )
    
    # Demo 3: Search with state filter
    print_results(
        matcher.match_with_location("First National Bank", state="TX", top_k=5),
        "DEMO 3: Search for 'First National Bank' in Texas"
    )
    
    # Demo 4: Search with both city and state
    print_results(
        matcher.match_with_location("Global Tech", city="San Francisco", state="CA", top_k=5),
        "DEMO 4: Search for 'Global Tech' in San Francisco, CA"
    )
    
    # Demo 5: Show record counts
    print("\n" + "="*70)
    print(" DEMO 5: Record Counts")
    print("="*70)
    for company in ["First National Bank", "Global Tech Solutions", "Smith & Associates"]:
        count = matcher.get_company_count(company)
        location = matcher.get_company_location(company)
        print(f"  {company}: {count} records in {location.get('city', 'N/A')}, {location.get('state', 'N/A')}")
    
    print("\n" + "#"*70)
    print("#  DEMO COMPLETE")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()

