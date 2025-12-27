#!/usr/bin/env python3
"""Test enhanced location fuzzy matching."""

import json
from CompanyMatcher import CompanyMatcher

# Create sample data with various locations
DEMO_DATA = [
    {"ID": 1, "Company Name": "Acme Corp", "City": "New York", "State": "NY", "Count": 100},
    {"ID": 2, "Company Name": "Acme Inc", "City": "New York City", "State": "New York", "Count": 50},
    {"ID": 3, "Company Name": "Acme LLC", "City": "Los Angeles", "State": "CA", "Count": 75},
    {"ID": 4, "Company Name": "Acme Ltd", "City": "LA", "State": "California", "Count": 25},
    {"ID": 5, "Company Name": "Acme Co", "City": "San Francisco", "State": "CA", "Count": 60},
    {"ID": 6, "Company Name": "Acme Group", "City": "SF", "State": "California", "Count": 40},
    {"ID": 7, "Company Name": "Acme Holdings", "City": "Chicago", "State": "IL", "Count": 80},
    {"ID": 8, "Company Name": "Acme Partners", "City": "Chi-Town", "State": "Illinois", "Count": 30},
    {"ID": 9, "Company Name": "Acme Services", "City": "Philadelphia", "State": "PA", "Count": 55},
    {"ID": 10, "Company Name": "Acme Solutions", "City": "Philly", "State": "Pennsylvania", "Count": 45},
    {"ID": 11, "Company Name": "Acme Tech", "City": "Saint Louis", "State": "MO", "Count": 35},
    {"ID": 12, "Company Name": "Acme Digital", "City": "St. Louis", "State": "Missouri", "Count": 65},
    {"ID": 13, "Company Name": "Acme Systems", "City": "Washington", "State": "DC", "Count": 90},
    {"ID": 14, "Company Name": "Acme Data", "City": "Washington DC", "State": "District of Columbia", "Count": 20},
]

print("="*70)
print("ENHANCED LOCATION FUZZY MATCHING TEST")
print("="*70)

# Build index
matcher = CompanyMatcher()
matcher.build_index_with_location(data=DEMO_DATA)

def print_results(results, title):
    print(f"\n{title}")
    print("-" * 70)
    print(f"{'#':<3} {'Company':<20} {'Score':<8} {'City':<20} {'State':<8} {'LocScore'}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        loc_score = r.get('location_score', 0) * 100
        print(f"{i:<3} {r['name']:<20} {r['score']*100:>5.1f}%  {r.get('city',''):<20} {r.get('state',''):<8} {loc_score:>5.1f}%")

# Test 1: NYC variations
print("\n" + "="*70)
print("TEST 1: Search for 'Acme' in 'NYC' (should match 'New York', 'New York City')")
print("="*70)
results = matcher.match_with_location("Acme", city="NYC", top_k=5)
print_results(results, "Results:")

# Test 2: LA variations  
print("\n" + "="*70)
print("TEST 2: Search for 'Acme' in 'LA, California' (should match 'Los Angeles', 'CA')")
print("="*70)
results = matcher.match_with_location("Acme", city="LA", state="California", top_k=5)
print_results(results, "Results:")

# Test 3: San Fran variations
print("\n" + "="*70)
print("TEST 3: Search for 'Acme' in 'San Fran, CA' (should match 'San Francisco', 'SF')")
print("="*70)
results = matcher.match_with_location("Acme", city="San Fran", state="CA", top_k=5)
print_results(results, "Results:")

# Test 4: Philly variations
print("\n" + "="*70)
print("TEST 4: Search for 'Acme' in 'Philly' (should match 'Philadelphia')")
print("="*70)
results = matcher.match_with_location("Acme", city="Philly", top_k=5)
print_results(results, "Results:")

# Test 5: State abbreviation matching
print("\n" + "="*70)
print("TEST 5: Search for 'Acme' with state='Illinois' (should match 'IL')")
print("="*70)
results = matcher.match_with_location("Acme", state="Illinois", top_k=5)
print_results(results, "Results:")

# Test 6: St. Louis variations
print("\n" + "="*70)
print("TEST 6: Search for 'Acme' in 'St Louis, MO' (should match 'Saint Louis', 'St. Louis')")
print("="*70)
results = matcher.match_with_location("Acme", city="St Louis", state="MO", top_k=5)
print_results(results, "Results:")

# Test 7: Washington DC variations
print("\n" + "="*70)
print("TEST 7: Search for 'Acme' in 'Washington D.C.' (should match 'Washington', 'Washington DC')")
print("="*70)
results = matcher.match_with_location("Acme", city="Washington D.C.", state="DC", top_k=5)
print_results(results, "Results:")

print("\n" + "="*70)
print("TESTS COMPLETE")
print("="*70)

