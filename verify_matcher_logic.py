
import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from finetuner.core.matcher import CompanyMatcher

def verify():
    print("Initializing CompanyMatcher...")
    matcher = CompanyMatcher()
    
    # Try to load the specific cache used in the report
    cache_key = 'e1894e93a84bbc84a9ec980508a5fec4_loc'
    print(f"Loading cache: {cache_key}")
    if not matcher.load_from_cache(cache_key):
        print("Failed to load cache. Using raw file.")
        # Fallback to loading from file if cache missing (shouldn't happen if RPC has it)
        matcher.build_index_with_location(filepath='companies_with_location.json')
    
    print(f"Max Company Count: {matcher.max_company_count}")
    
    query = "Next Level Events"
    print(f"\nSearching for: {query}")
    results = matcher.match(query, top_k=5)
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1}: {res['name']}")
        print(f"  Score: {res['score']:.6f}")
        print(f"  Raw Semantic: {res.get('semantic_score', 0):.4f}")
        print(f"  String: {res.get('string_score', 0):.4f}")
        print(f"  Count: {res.get('count', 0)}")
        print(f"  Exposed Pop Boost: {res.get('popularity_boost', 0):.6f}")
        print(f"  Exposed Loc Boost: {res.get('location_boost', 0):.6f}")
        
        record_count = res.get('count', 0)
        if record_count > 0 and matcher.max_company_count > 0:
            import math
            freq_score = math.log1p(record_count) / math.log1p(matcher.max_company_count)
            # Approximate name score as 1.0 for this check since these are top matches
            calc_boost = freq_score * 0.05 * 1.0
            print(f"  Calc Freq Score: {freq_score:.4f}")
            print(f"  Expected Boost (~): {calc_boost:.6f}")
        else:
            print("  Max company count or record count is 0, no boost calculation.")

if __name__ == "__main__":
    verify()
