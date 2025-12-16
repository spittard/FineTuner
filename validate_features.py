#!/usr/bin/env python3
"""Validate all recent features are present"""

import inspect

print("=" * 60)
print("FEATURE VALIDATION REPORT")
print("=" * 60)

# Check CompanyMatcher features
from CompanyMatcher import CompanyMatcher

m = CompanyMatcher()

checks = []

# 1. Filepath-based cache checking
checks.append(("Filepath-based cache key", hasattr(m, 'get_cache_key_from_file')))
if hasattr(m, 'get_cache_key_from_file'):
    checks.append(("get_cache_key_from_file works", callable(getattr(m, 'get_cache_key_from_file'))))

# 2. build_index accepts filepath
sig = inspect.signature(m.build_index)
checks.append(("build_index accepts filepath", 'filepath' in sig.parameters))
checks.append(("build_index accepts company_names", 'company_names' in sig.parameters))

# 3. Hyphen normalization
with open('CompanyMatcher.py', 'r', encoding='utf-8') as f:
    cm_content = f.read()
checks.append(("Hyphen normalization", ".replace('-', ' ')" in cm_content))

# 4. Coverage-based boosting
checks.append(("Coverage-based boosting", "coverage_ratio" in cm_content))
checks.append(("Coverage boost logic", "coverage_boost" in cm_content))

# 5. Test script features
with open('test_control_set.py', 'r', encoding='utf-8') as f:
    test_content = f.read()
checks.append(("Progress bars (tqdm)", "tqdm" in test_content))
checks.append(("Index load tracking", "index_load_count" in test_content))
checks.append(("Filepath usage", "build_index(filepath=" in test_content))
checks.append(("Statistics counters", "exact_match_count" in test_content))

# Print results
all_passed = True
for name, result in checks:
    status = "PASS" if result else "FAIL"
    print(f"{status:4} {name}: {result}")
    if not result:
        all_passed = False

print("=" * 60)
if all_passed:
    print("ALL FEATURES VALIDATED - Everything is present!")
else:
    print("WARNING: Some features may be missing!")
print("=" * 60)
