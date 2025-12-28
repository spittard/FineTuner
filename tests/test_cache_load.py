#!/usr/bin/env python3
"""Force load from existing cache"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core.matcher import CompanyMatcher

# Initialize matcher
matcher = CompanyMatcher(model_name='paraphrase-MiniLM-L3-v2')

# Try to load from the known good cache
cache_key = 'e3e4df77d6e01ad2f37f4c59be6124ae'
print(f"Attempting to load from cache: {cache_key}")

if matcher.load_from_cache(cache_key):
    print(f"✅ SUCCESS! Loaded {len(matcher.original_company_names):,} companies from cache")
    print(f"   Has location data: {matcher.has_location_data}")
    print(f"   Model: {matcher.model_name}")
else:
    print(f"❌ FAILED to load from cache")
