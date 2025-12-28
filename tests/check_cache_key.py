#!/usr/bin/env python3
"""Quick diagnostic to check cache key generation"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core.matcher import CompanyMatcher

# Initialize matcher with the ULTRA-fast model
matcher = CompanyMatcher(model_name='paraphrase-MiniLM-L3-v2')

# Get cache key from file
filename = 'companies.json'
if os.path.exists(filename):
    file_cache_key_reg = matcher.get_cache_key_from_file(filename)
    file_cache_key_loc = file_cache_key_reg + "_loc" if file_cache_key_reg else None
    
    print(f"File: {filename}")
    print(f"File cache key (regular): {file_cache_key_reg}")
    print(f"File cache key (location): {file_cache_key_loc}")
    print()
    
    # List existing cache files
    print("Existing cache files:")
    cache_dir = 'company_matcher_cache'
    if os.path.exists(cache_dir):
        import glob
        metadata_files = glob.glob(os.path.join(cache_dir, '*_metadata.pkl'))
        for mf in metadata_files:
            cache_key = os.path.basename(mf).replace('_metadata.pkl', '')
            print(f"  - {cache_key}")
            
            # Check if this matches our file cache key
            if cache_key == file_cache_key_reg:
                print(f"    ✓ MATCHES regular file cache key!")
            if cache_key == file_cache_key_loc:
                print(f"    ✓ MATCHES location file cache key!")
else:
    print(f"Error: {filename} not found")
