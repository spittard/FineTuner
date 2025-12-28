#!/usr/bin/env python3
"""Quick test for IBM matching"""
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.web.services.search_service import SearchService

print("="*60)
print("QUICK TEST: IBM")
print("="*60)

# Initialize and load
s = SearchService()
print("\nLoading company data...")
start = time.time()
s.load_company_data()
load_time = time.time() - start
print(f"✅ Loaded in {load_time:.1f}s")

# Search for IBM
print("\n" + "="*60)
print("Searching for: IBM")
print("="*60)
search_start = time.time()
matches = s.search('IBM', top_k=10)
search_time = time.time() - search_start

print(f"\n✅ Search completed in {search_time:.2f}s")
print(f"\nTop 10 matches for 'IBM':\n")
print(f"{'Rank':<6} {'Company Name':<50} {'Score':<8}")
print("-" * 70)

for i, m in enumerate(matches[:10], 1):
    name = m['company_name'][:48]
    score = m['likeness_percent']
    print(f"{i:<6} {name:<50} {score:>6.1f}%")

print("\n" + "="*60)
print("DETAILED RATIONALE FOR TOP MATCH")
print("="*60)
if matches:
    top = matches[0]
    print(f"\nCompany: {top['company_name']}")
    print(f"Score: {top['likeness_percent']:.1f}%")
    print(f"\nRationale:")
    print(top['match_rationale'])
