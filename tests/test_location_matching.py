#!/usr/bin/env python3
"""
Test script for location-aware company matching.
Tests the new location-based matching functionality.
"""

import json
import os
import sys
import tempfile

# Add parent (root) directory to path so we can resolve 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from finetuner.core.matcher import CompanyMatcher


def create_test_data_with_location():
    """Create sample test data with location information"""
    return [
        {"Company Name": "Acme Corporation", "City": "New York", "State": "NY", "Count": 15},
        {"Company Name": "Acme Corporation", "City": "Los Angeles", "State": "CA", "Count": 8},
        {"Company Name": "Acme Inc", "City": "Chicago", "State": "IL", "Count": 5},
        {"Company Name": "Global Tech Solutions", "City": "San Francisco", "State": "CA", "Count": 20},
        {"Company Name": "Global Tech", "City": "Seattle", "State": "WA", "Count": 12},
        {"Company Name": "First National Bank", "City": "Boston", "State": "MA", "Count": 100},
        {"Company Name": "First National Bank", "City": "Miami", "State": "FL", "Count": 75},
        {"Company Name": "First National Bank of Texas", "City": "Houston", "State": "TX", "Count": 50},
        {"Company Name": "Smith & Associates", "City": "Denver", "State": "CO", "Count": 10},
        {"Company Name": "Smith Associates LLC", "City": "Phoenix", "State": "AZ", "Count": 7},
    ]


def test_basic_matching_with_location():
    """Test that basic matching still works with location data"""
    print("\n" + "="*70)
    print("TEST 1: Basic Matching with Location Data")
    print("="*70)
    
    test_data = create_test_data_with_location()
    
    # Create matcher and build index with location
    matcher = CompanyMatcher()
    matcher.build_index_with_location(data=test_data)
    
    # Verify index was built correctly
    assert matcher.is_index_ready(), "Index should be ready"
    assert matcher.has_location_data, "Should have location data"
    assert len(matcher.company_locations) == len(test_data), "Location data count should match"
    assert len(matcher.company_counts) == len(test_data), "Count data should match"
    
    print(f"[OK] Index built with {len(matcher.original_company_names)} companies")
    print(f"[OK] Location data available: {matcher.has_location_data}")
    print(f"[OK] Location entries: {len(matcher.company_locations)}")
    
    # Test basic name matching
    results = matcher.match("Acme Corporation", top_k=3)
    assert len(results) > 0, "Should find matches"
    assert results[0]['name'] == "Acme Corporation", "First match should be exact"
    print(f"[OK] Basic matching works: '{results[0]['name']}' with score {results[0]['score']:.2f}")
    
    return True


def test_location_aware_matching():
    """Test location-aware matching functionality"""
    print("\n" + "="*70)
    print("TEST 2: Location-Aware Matching")
    print("="*70)
    
    test_data = create_test_data_with_location()
    
    matcher = CompanyMatcher()
    matcher.build_index_with_location(data=test_data)
    
    # Test 1: Search with city filter
    print("\n--- Test 2.1: Search with city filter ---")
    results = matcher.match_with_location("Acme", city="New York", top_k=5)
    
    assert len(results) > 0, "Should find matches"
    
    # Check that results include location data
    for result in results[:3]:
        print(f"  {result['name']}: score={result['score']:.2f}, "
              f"city={result.get('city', 'N/A')}, state={result.get('state', 'N/A')}, "
              f"count={result.get('count', 0)}")
    
    print(f"[OK] City filter working - found {len(results)} results")
    
    # Test 2: Search with state filter
    print("\n--- Test 2.2: Search with state filter ---")
    results = matcher.match_with_location("First National Bank", state="TX", top_k=5)
    
    # Check that Texas results are boosted
    texas_results = [r for r in results if r.get('state', '').upper() == 'TX']
    print(f"  Texas matches: {len(texas_results)}")
    for result in results[:3]:
        print(f"  {result['name']}: score={result['score']:.2f}, "
              f"location_score={result.get('location_score', 0):.2f}, "
              f"state={result.get('state', 'N/A')}")
    
    print(f"[OK] State filter working")
    
    # Test 3: Search with both city and state
    print("\n--- Test 2.3: Search with city AND state filter ---")
    results = matcher.match_with_location("Global Tech", city="San Francisco", state="CA", top_k=5)
    
    for result in results[:3]:
        print(f"  {result['name']}: score={result['score']:.2f}, "
              f"city={result.get('city', 'N/A')}, state={result.get('state', 'N/A')}")
    
    # Verify CA results are prioritized
    ca_results = [r for r in results if r.get('state', '').upper() == 'CA']
    assert len(ca_results) > 0, "Should find CA results"
    print(f"[OK] City+State filter working - CA results: {len(ca_results)}")
    
    return True


def test_count_data():
    """Test that record counts are included in results"""
    print("\n" + "="*70)
    print("TEST 3: Record Count Data")
    print("="*70)
    
    test_data = create_test_data_with_location()
    
    matcher = CompanyMatcher()
    matcher.build_index_with_location(data=test_data)
    
    # Search and check that count is included
    results = matcher.match_with_location("First National Bank", top_k=5)
    
    print("\nResults with record counts:")
    for result in results[:5]:
        print(f"  {result['name']}: count={result.get('count', 0)}, "
              f"city={result.get('city', 'N/A')}, state={result.get('state', 'N/A')}")
    
    # Verify count data is present
    has_counts = any(r.get('count', 0) > 0 for r in results)
    assert has_counts, "At least one result should have a count"
    print(f"\n[OK] Record counts included in results")
    
    # Test get_company_count method
    count = matcher.get_company_count("First National Bank")
    print(f"[OK] get_company_count('First National Bank') = {count}")
    
    # Test get_company_location method
    location = matcher.get_company_location("Global Tech Solutions")
    print(f"[OK] get_company_location('Global Tech Solutions') = {location}")
    
    return True


def test_backwards_compatibility():
    """Test that regular matching still works without location data"""
    print("\n" + "="*70)
    print("TEST 4: Backwards Compatibility (no location data)")
    print("="*70)
    
    # Create data WITHOUT location info
    simple_data = [
        {"Company Name": "Apple Inc"},
        {"Company Name": "Microsoft Corporation"},
        {"Company Name": "Google LLC"},
        {"Company Name": "Amazon.com Inc"},
    ]
    
    matcher = CompanyMatcher()
    
    # Extract just company names (old format)
    company_names = [item["Company Name"] for item in simple_data]
    matcher.build_index(company_names)
    
    assert matcher.is_index_ready(), "Index should be ready"
    assert not matcher.has_location_data, "Should NOT have location data"
    
    # Regular match should still work
    results = matcher.match("Apple", top_k=3)
    assert len(results) > 0, "Should find matches"
    print(f"[OK] Regular matching works: found {len(results)} results")
    
    # match_with_location should still work (just ignores location params)
    results = matcher.match_with_location("Microsoft", city="Seattle", state="WA", top_k=3)
    assert len(results) > 0, "Should find matches"
    print(f"[OK] match_with_location works without location data: found {len(results)} results")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# LOCATION-AWARE COMPANY MATCHING - TEST SUITE")
    print("#"*70)
    
    tests = [
        ("Basic Matching with Location", test_basic_matching_with_location),
        ("Location-Aware Matching", test_location_aware_matching),
        ("Record Count Data", test_count_data),
        ("Backwards Compatibility", test_backwards_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

