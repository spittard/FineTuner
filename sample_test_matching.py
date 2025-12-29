import json
import os
import sys
import time

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.core.matcher import CompanyMatcher

def main():
    source_file = "companies_with_location.json"
    sample_size = 10000
    
    print(f"Loading first {sample_size:,} records from {source_file}...")
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found!")
        return
        
    with open(source_file, 'r', encoding='utf-8') as f:
        # We need to read carefully as it's a large file
        data = []
        count = 0
        # Simple JSON loader for the sample
        raw_data = json.load(f)
        data = raw_data[:sample_size]
    
    print(f"Loaded {len(data):,} records for sampling.")
    
    # Initialize Matcher
    print("\nInitializing CompanyMatcher...")
    matcher = CompanyMatcher(model_name='paraphrase-MiniLM-L3-v2')
    
    # Build index with location data
    print("Building location-aware index...")
    start_time = time.time()
    matcher.build_index_with_location(data=data)
    print(f"Index built in {time.time() - start_time:.2f} seconds.")
    
    # Test 1: Ambiguous name (e.g., "Kehilat Ariel")
    # Let's find some names in our sample to test
    print("\n" + "="*70)
    print("TEST MATCHING")
    print("="*70)
    
    # Find a name with a count > 1 if possible
    freq_data = sorted(data, key=lambda x: x.get('Count', 0), reverse=True)
    if freq_data:
        top_name = freq_data[0]['Company Name']
        top_city = freq_data[0]['City']
        top_state = freq_data[0]['State']
        
        print(f"\n1. Searching for: '{top_name}' (No filter)")
        matches = matcher.match_with_location(top_name, top_k=3)
        for i, m in enumerate(matches, 1):
            print(f"   {i}. {m['name']} ({m['city']}, {m['state']}) - Score: {m['score']*100:.2f}%, Count: {m.get('count', 0)}")
            
        if top_city or top_state:
            print(f"\n2. Searching for: '{top_name}' in {top_city}, {top_state}")
            matches = matcher.match_with_location(top_name, city=top_city, state=top_state, top_k=3)
            for i, m in enumerate(matches, 1):
                print(f"   {i}. {m['name']} ({m['city']}, {m['state']}) - Score: {m['score']*100:.2f}%, Count: {m.get('count', 0)}")

    # Test 3: Near match with location help
    print(f"\n3. Searching for partial name with city: 'National Philip' in 'San Jose, CA'")
    matches = matcher.match_with_location("National Philip", city="San Jose", state="CA", top_k=3)
    for i, m in enumerate(matches, 1):
        print(f"   {i}. {m['name']} ({m['city']}, {m['state']}) - Score: {m['score']*100:.2f}%, Count: {m.get('count', 0)}")

    print("\nSample testing complete!")

if __name__ == "__main__":
    main()
