#!/usr/bin/env python3
"""Build index with 1000 records and test it."""

import json
from CompanyMatcher import CompanyMatcher

# Load first 1000 companies from existing data
print('Loading companies.json...')
with open('companies.json', 'r', encoding='utf-8') as f:
    data = json.load(f)[:1000]

print(f'Loaded {len(data)} companies')

# Extract names
names = [item['Company Name'] for item in data if 'Company Name' in item]
print(f'Extracted {len(names)} company names')

# Build index
print('')
print('Building index...')
matcher = CompanyMatcher()
matcher.build_index(names)

print('')
print('=' * 50)
print('INDEX READY')
print('=' * 50)
print(f'Companies indexed: {len(matcher.original_company_names)}')

# Test searches
print('')
print('--- Test Search: "Acme" ---')
results = matcher.match('Acme', top_k=5)
for i, r in enumerate(results, 1):
    print(f'  {i}. {r["name"]}: {r["score"]*100:.1f}%')

print('')
print('--- Test Search: "Travel" ---')
results = matcher.match('Travel', top_k=5)
for i, r in enumerate(results, 1):
    print(f'  {i}. {r["name"]}: {r["score"]*100:.1f}%')

print('')
print('--- Test Search: "National Bank" ---')
results = matcher.match('National Bank', top_k=5)
for i, r in enumerate(results, 1):
    print(f'  {i}. {r["name"]}: {r["score"]*100:.1f}%')

print('')
print('--- Test Search: "School" ---')
results = matcher.match('School', top_k=5)
for i, r in enumerate(results, 1):
    print(f'  {i}. {r["name"]}: {r["score"]*100:.1f}%')

print('')
print('Done!')

