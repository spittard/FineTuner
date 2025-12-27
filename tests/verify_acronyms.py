import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from finetuner.core.matcher import CompanyMatcher

def test_acronym(matcher, acronym, expected_partial):
    print(f"\nScanning for Acronym: '{acronym}'")
    results = matcher.match(acronym, top_k=5)
    
    found = False
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']} (Score: {r['score']:.4f}, Type: {r['match_type']})")
        if expected_partial.lower() in r['name'].lower():
            found = True
            if r['match_type'] == 'acronym_expansion' or r['score'] >= 0.9:
                print(f"  [SUCCESS] Found '{expected_partial}' via Acronym Expansion!")
            else:
                print(f"  [WARNING] Found '{expected_partial}' but maybe not via Acronym (Type: {r['match_type']})")
                
    if not found:
        print(f"  [FAILURE] Did not find '{expected_partial}' in top 5")

def main():
    print("Initializing Matcher...")
    matcher = CompanyMatcher()
    
    # Force load of the known valid cache key to avoid rebuilds
    # (The key mismatch issue identified during update process)
    forced_key = "e6988a56da041497449aae41fb61b00f"
    print(f"Forcing load of cache key: {forced_key}")
    
    if matcher.load_from_cache(forced_key):
        print(f"Success! Loaded {len(matcher.original_company_names):,} companies.")
    else:
        print("Failed to load cache! Attempting build (this might be slow if cache missing)...")
        matcher.build_index("companies.json")

    # Test 1: ABA -> American Bar Association (User Example)
    test_acronym(matcher, "ABA", "American Bar Association")
    
    # Test 2: IBM -> International Business Machines
    # Note: "International Business Machines" might be stored as "IBM" or full name.
    # We will search for generic known ones.
    test_acronym(matcher, "IBM", "International Business Machines")
    
    # Test 3: GE -> General Electric
    test_acronym(matcher, "GE", "General Electric")
    
    # Test 4: PDMA -> Product Development and Management Association (From our control set!)
    test_acronym(matcher, "PDMA", "Product Development and Management Association")

if __name__ == "__main__":
    main()
