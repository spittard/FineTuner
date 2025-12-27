import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from finetuner.core.matcher import CompanyMatcher

def main():
    print("Initializing Matcher...")
    matcher = CompanyMatcher()
    
    filename = "companies.json"
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found in {os.getcwd()}")
        return

    # Debug Cache Key Generation
    stat = os.stat(filename)
    print(f"File: {os.path.abspath(filename)}")
    print(f"Size: {stat.st_size}")
    print(f"Mtime: {stat.st_mtime}")
    print(f"Model: {matcher.model_name}")
    
    key = matcher.get_cache_key_from_file(filename)
    print(f"Calculated Key: {key}")
    
    expected_key = "e6988a56da041497449aae41fb61b00f"
    if key != expected_key:
        print(f"WARNING: Key mismatch! Expected {expected_key}")
        print("Forcing use of expected key...")
        key = expected_key
        
    print(f"Attempting to load from cache key: {key}")
    if matcher.load_from_cache(key):
        print(f"Success! Loaded {len(matcher.original_company_names):,} companies.")
    else:
        print("Failed to load from cache.")
        return

    # Check for existing acronyms
    if hasattr(matcher, 'acronym_index') and matcher.acronym_index:
        print(f"Acronym index already exists with {len(matcher.acronym_index)} entries.")
        # choice = input("Rebuild anyway? (y/n): ")
        # if choice.lower() != 'y':
        #     return
            
    # Manually create the index
    print("\nGenerating Acronym Index manually...")
    start = time.time()
    matcher._create_acronym_index()
    print(f"Generated in {time.time() - start:.1f}s")
    
    # Save back to cache
    print("\nSaving updated cache...")
    matcher.save_to_cache(
        key, 
        None, None, 
        matcher.company_names, 
        matcher.original_company_names,
        locations=matcher.company_locations,
        counts=matcher.company_counts,
        ids=matcher.company_ids
    )
    print("Done!")

if __name__ == "__main__":
    main()
