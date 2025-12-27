import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from finetuner.core.matcher import CompanyMatcher

def main():
    # Source: The key we know exists and has full data (including new acronyms)
    src_key = "e6988a56da041497449aae41fb61b00f"
    # Dest: The key the current environment calculates (from logs)
    dst_key = "e3e4df77d6e01ad2f37f4c59be6124ae"

    print(f"Goal: Duplicate cache {src_key} -> {dst_key}")
    
    matcher = CompanyMatcher()
    print(f"Loading source cache: {src_key}...")
    # Force load from the known good key
    if not matcher.load_from_cache(src_key):
        print("Error: Could not load source cache. Aborting.")
        return

    print(f"Loaded {len(matcher.original_company_names):,} companies.")
    print(f"Saving to destination cache key: {dst_key}...")
    
    start = time.time()
    # Save to the new key. This will write embeddings, index, metadata, names, etc.
    matcher.save_to_cache(
        dst_key, 
        None, None, 
        matcher.company_names, 
        matcher.original_company_names,
        locations=matcher.company_locations,
        counts=matcher.company_counts,
        ids=matcher.company_ids
    )
    print(f"Success! Cache duplicated in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
