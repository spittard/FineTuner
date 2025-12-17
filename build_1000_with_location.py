#!/usr/bin/env python3
"""Build index with 1000 records including ID, City, State, Count."""

import json
from CompanyMatcher import CompanyMatcher

# Create sample data with IDs and location info
# Simulating what would come from SQL Server
print('Creating sample data with ID, City, State, Count...')

# Load base company names from existing data
with open('companies.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)[:1000]

# Add fake IDs and location data for demo
# In production, this would come from your SQL Server query
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
          'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
          'Seattle', 'Denver', 'Boston', 'Detroit', 'Miami']
states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA',
          'TX', 'FL', 'TX', 'OH', 'NC', 'WA', 'CO', 'MA', 'MI', 'FL']

import random
random.seed(42)  # For reproducibility

data_with_location = []
for i, item in enumerate(raw_data):
    if 'Company Name' in item:
        city_idx = i % len(cities)
        data_with_location.append({
            'ID': 10000 + i,  # Simulated database ID
            'Company Name': item['Company Name'],
            'City': cities[city_idx],
            'State': states[city_idx],
            'Count': random.randint(1, 500)  # Simulated record count
        })

print(f'Created {len(data_with_location)} records with location data')

# Save to file for inspection
with open('companies_1000_with_location.json', 'w', encoding='utf-8') as f:
    json.dump(data_with_location[:10], f, indent=2)  # Save first 10 as sample
print('Saved sample to companies_1000_with_location.json')

# Build index with location data
print('')
print('Building index with location data...')
matcher = CompanyMatcher()
matcher.build_index_with_location(data=data_with_location)

print('')
print('=' * 70)
print('INDEX READY WITH FULL DATA')
print('=' * 70)
print(f'Companies indexed: {len(matcher.original_company_names)}')
print(f'Location data: {len(matcher.company_locations)} entries')
print(f'Company IDs: {len(matcher.company_ids)} entries')
print(f'Record counts: {len(matcher.company_counts)} entries')

# Test searches with location
print('')
print('=' * 70)
print('TEST 1: Search "Travel" (no location filter)')
print('=' * 70)
results = matcher.match_with_location('Travel', top_k=5)
print(f'{"Rank":<5} {"ID":<8} {"Company Name":<35} {"Score":<8} {"City":<15} {"State":<6} {"Count"}')
print('-' * 90)
for i, r in enumerate(results, 1):
    print(f'{i:<5} {str(r.get("id", "N/A")):<8} {r["name"][:35]:<35} {r["score"]*100:>5.1f}%  {r.get("city", ""):<15} {r.get("state", ""):<6} {r.get("count", 0)}')

print('')
print('=' * 70)
print('TEST 2: Search "Travel" in Texas (state filter)')
print('=' * 70)
results = matcher.match_with_location('Travel', state='TX', top_k=5)
print(f'{"Rank":<5} {"ID":<8} {"Company Name":<35} {"Score":<8} {"City":<15} {"State":<6} {"Count"}')
print('-' * 90)
for i, r in enumerate(results, 1):
    print(f'{i:<5} {str(r.get("id", "N/A")):<8} {r["name"][:35]:<35} {r["score"]*100:>5.1f}%  {r.get("city", ""):<15} {r.get("state", ""):<6} {r.get("count", 0)}')

print('')
print('=' * 70)
print('TEST 3: Search "School" in Chicago, IL')
print('=' * 70)
results = matcher.match_with_location('School', city='Chicago', state='IL', top_k=5)
print(f'{"Rank":<5} {"ID":<8} {"Company Name":<35} {"Score":<8} {"City":<15} {"State":<6} {"Count"}')
print('-' * 90)
for i, r in enumerate(results, 1):
    print(f'{i:<5} {str(r.get("id", "N/A")):<8} {r["name"][:35]:<35} {r["score"]*100:>5.1f}%  {r.get("city", ""):<15} {r.get("state", ""):<6} {r.get("count", 0)}')

print('')
print('=' * 70)
print('TEST 4: Exact match lookup')
print('=' * 70)
# Get a specific company name from the data
test_name = data_with_location[50]['Company Name']
print(f'Looking up: "{test_name}"')
results = matcher.match_with_location(test_name, top_k=3)
for i, r in enumerate(results, 1):
    print(f'  {i}. ID={r.get("id", "N/A")}, Name="{r["name"]}", Score={r["score"]*100:.1f}%, City={r.get("city", "")}, State={r.get("state", "")}')

print('')
print('Done!')

