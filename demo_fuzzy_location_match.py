#!/usr/bin/env python3
"""
Demo: Fuzzy company name + location matching scenario.

This demonstrates when a query company doesn't exact match,
and the system uses BOTH company name similarity AND location
to find the best match.
"""


import os
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.core.matcher import CompanyMatcher


# Scenario: Multiple similar companies across different locations
# The query will NOT exact match - system must use name + location to disambiguate
DEMO_DATA = [
    # Same company name in different locations
    {"ID": 1001, "Company Name": "First National Bank", "City": "New York", "State": "NY", "Count": 500},
    {"ID": 1002, "Company Name": "First National Bank", "City": "Houston", "State": "TX", "Count": 350},
    {"ID": 1003, "Company Name": "First National Bank", "City": "Los Angeles", "State": "CA", "Count": 275},
    {"ID": 1004, "Company Name": "First National Bank", "City": "Chicago", "State": "IL", "Count": 400},
    
    # Similar but different company names
    {"ID": 2001, "Company Name": "First Natl Bank of Texas", "City": "Dallas", "State": "TX", "Count": 200},
    {"ID": 2002, "Company Name": "First National Bank & Trust", "City": "Miami", "State": "FL", "Count": 150},
    {"ID": 2003, "Company Name": "1st National Bank", "City": "Phoenix", "State": "AZ", "Count": 100},
    
    # Another company family with location variations
    {"ID": 3001, "Company Name": "Global Tech Solutions", "City": "San Francisco", "State": "CA", "Count": 600},
    {"ID": 3002, "Company Name": "Global Tech Solutions", "City": "Seattle", "State": "WA", "Count": 450},
    {"ID": 3003, "Company Name": "Global Tech Solutions", "City": "Austin", "State": "TX", "Count": 300},
    {"ID": 3004, "Company Name": "Global Technology Solutions Inc", "City": "Boston", "State": "MA", "Count": 250},
    {"ID": 3005, "Company Name": "GlobalTech Solutions LLC", "City": "Denver", "State": "CO", "Count": 175},
    
    # Acme variations
    {"ID": 4001, "Company Name": "Acme Corporation", "City": "New York", "State": "NY", "Count": 800},
    {"ID": 4002, "Company Name": "Acme Corporation", "City": "Los Angeles", "State": "CA", "Count": 650},
    {"ID": 4003, "Company Name": "Acme Corp", "City": "Chicago", "State": "IL", "Count": 400},
    {"ID": 4004, "Company Name": "ACME Inc", "City": "Houston", "State": "TX", "Count": 300},
    {"ID": 4005, "Company Name": "Acme Industries", "City": "Detroit", "State": "MI", "Count": 225},
]

def print_results(results, max_show=5):
    print(f"{'Rank':<5} {'ID':<7} {'Company Name':<35} {'Score':<7} {'City':<15} {'State':<6} {'LocScore':<8} {'NameScore'}")
    print("-" * 105)
    for i, r in enumerate(results[:max_show], 1):
        loc_score = r.get('location_score', 0) * 100
        name_score = r.get('name_score', 0) * 100
        print(f"{i:<5} {r.get('id', 'N/A'):<7} {r['name'][:35]:<35} {r['score']*100:>5.1f}%  {r.get('city',''):<15} {r.get('state',''):<6} {loc_score:>5.1f}%   {name_score:>5.1f}%")


print("=" * 105)
print(" FUZZY COMPANY + LOCATION MATCHING DEMO")
print(" Scenario: Query does NOT exact match - must use name + location to find best match")
print("=" * 105)

# Build index
matcher = CompanyMatcher()
matcher.build_index_with_location(data=DEMO_DATA)

print(f"\nLoaded {len(DEMO_DATA)} companies across multiple locations")
print("=" * 105)

# ============================================================================
# SCENARIO 1: Typo in company name + location to disambiguate
# ============================================================================
print("\n" + "=" * 105)
print("SCENARIO 1: Typo in company name - '1st Natinal Bank' (typo) in Houston, TX")
print("Expected: Should find 'First National Bank' in Houston, TX (ID 1002)")
print("=" * 105)
results = matcher.match_with_location("1st Natinal Bank", city="Houston", state="TX", top_k=5)
print_results(results)

# ============================================================================
# SCENARIO 2: Abbreviated name + state to find correct location
# ============================================================================
print("\n" + "=" * 105)
print("SCENARIO 2: Abbreviated name - 'First Natl Bank' in Texas")
print("Expected: Texas locations should rank higher than others")
print("=" * 105)
results = matcher.match_with_location("First Natl Bank", state="TX", top_k=5)
print_results(results)

# ============================================================================
# SCENARIO 3: Similar names, need location to pick the right one
# ============================================================================
print("\n" + "=" * 105)
print("SCENARIO 3: 'Global Tech' in 'SF, California' (using city abbreviation)")
print("Expected: San Francisco, CA location should rank #1")
print("=" * 105)
results = matcher.match_with_location("Global Tech", city="SF", state="California", top_k=5)
print_results(results)

# ============================================================================
# SCENARIO 4: Multiple exact name matches - location breaks the tie
# ============================================================================
print("\n" + "=" * 105)
print("SCENARIO 4: 'First National Bank' (exact name exists 4x) in 'LA, CA'")
print("Expected: Los Angeles, CA should rank #1 despite same name scores")
print("=" * 105)
results = matcher.match_with_location("First National Bank", city="LA", state="CA", top_k=5)
print_results(results)

# ============================================================================
# SCENARIO 5: Partial name + city variation
# ============================================================================
print("\n" + "=" * 105)
print("SCENARIO 5: 'Acme' in 'Chi-Town, Illinois' (slang + full state name)")
print("Expected: Chicago, IL 'Acme Corp' should rank highest")
print("=" * 105)
results = matcher.match_with_location("Acme", city="Chi-Town", state="Illinois", top_k=5)
print_results(results)

# ============================================================================
# SCENARIO 6: Without location - ambiguous results
# ============================================================================
print("\n" + "=" * 105)
print("SCENARIO 6: 'First National Bank' WITHOUT location (for comparison)")
print("Expected: Multiple matches with same score - no way to disambiguate")
print("=" * 105)
results = matcher.match_with_location("First National Bank", top_k=5)
print_results(results)

# ============================================================================
# SCENARIO 7: Same query WITH location - disambiguation works
# ============================================================================
print("\n" + "=" * 105)
print("SCENARIO 7: 'First National Bank' WITH location 'NYC, NY'")
print("Expected: New York location should clearly rank #1")
print("=" * 105)
results = matcher.match_with_location("First National Bank", city="NYC", state="NY", top_k=5)
print_results(results)

print("\n" + "=" * 105)
print(" DEMO COMPLETE")
print("=" * 105)
print("""
KEY TAKEAWAYS:
1. When company name doesn't exact match, fuzzy name matching kicks in
2. Location score helps disambiguate between similar companies
3. City/state variations are normalized (NYC->New York, CA->California, etc.)
4. For non-exact matches: Final Score = (Name Score * 0.8) + (Location Score * 0.2)
5. For exact matches: Location adds small boost (5%) to break ties between same-name companies
""")

