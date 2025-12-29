#!/usr/bin/env python3
"""
Verification test for Location Baking and Frequency Boost.
"""
import os
import sys
import math

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from finetuner.core.matcher import CompanyMatcher

def test_location_integration():
    print("\n" + "="*70)
    print("VERIFICATION: LOCATION BAKING & FREQUENCY BOOST")
    print("="*70)
    
    # 1. Setup Test Data
    # Ambiguous names in different locations + different frequencies
    test_data = [
        {"Company Name": "Smith Inc", "City": "Philadelphia", "State": "PA", "Count": 10},
        {"Company Name": "Smith Inc", "City": "Los Angeles", "State": "CA", "Count": 10},
        {"Company Name": "Global Tech", "City": "Seattle", "State": "WA", "Count": 1000},
        {"Company Name": "Global Tech", "City": "Miami", "State": "FL", "Count": 5},
    ]
    
    matcher = CompanyMatcher(model_name='paraphrase-MiniLM-L3-v2') 
    print(f"Building index with {len(test_data)} companies...")
    matcher.build_index_with_location(data=test_data)
    
    # --- TEST 1: LOCATION BAKING DISAMBIGUATION ---
    print("\n--- Test 1: Location Baking (\"Smith PA\") ---")
    # Even without explicit city/state parameters, "PA" in the query should 
    # find the Pennsylvania entry better because it's baked into the vector.
    results = matcher.match("Smith PA", top_k=2)
    
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['name']} in {res['city']}, {res['state']} | Score: {res['score']:.4f}")
    
    assert results[0]['state'] == "PA", "PA entry should rank higher for 'Smith PA' query"
    print("[OK] Location baking successfully disambiguated search.")
    
    # --- TEST 2: FREQUENCY BOOST ---
    print("\n--- Test 2: Frequency Boost (\"Global Tech\") ---")
    # Both names are identical, locations are different. Without explicit location,
    # the one with 1000 occurrences should win over the one with 5.
    results = matcher.match("Global Tech", top_k=2)
    
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['name']} in {res['city']}, {res['state']} (Count: {res['count']}) | Score: {res['score']:.4f}")
        
    assert results[0]['count'] == 1000, "Higher count entry should rank higher for ambiguous identical names"
    print("[OK] Frequency boost successfully prioritized popular entry.")

    # --- TEST 3: EXPLICIT LOCATION OVERRIDE ---
    print("\n--- Test 3: Explicit Location Override (\"Smith\" + CA) ---")
    results = matcher.match_with_location("Smith", city="Los Angeles", state="CA", top_k=2)
    
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['name']} in {res['city']}, {res['state']} | Score: {res['score']:.4f}")
        
    assert results[0]['state'] == "CA", "Explicit CA filter should prioritize CA entry"
    print("[OK] Explicit location parameters work as expected.")

    print("\n" + "="*70)
    print("VERIFICATION SUCCESSFUL")
    print("="*70)

if __name__ == "__main__":
    try:
        test_location_integration()
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
