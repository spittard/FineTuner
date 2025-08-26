#!/usr/bin/env python3
"""
Test script to verify CLI and webapp consistency
This ensures both systems return identical results for the same queries
"""

import json
import subprocess
import requests
import sys
from typing import List, Dict, Any

def run_cli_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Run CLI search and return results"""
    try:
        # Run CLI command
        cmd = [
            'python', 'FineTuner.py', 
            '--mode', 'match',
            '--dataset', 'training_data.json',
            '--query', query,
            '--top-k', str(top_k)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"CLI Error: {result.stderr}")
            return []
        
        # Parse CLI output to extract results
        lines = result.stdout.split('\n')
        results = []
        
        # Find the results section
        in_results = False
        for line in lines:
            line = line.strip()
            if 'Top' in line and 'matches for' in line:
                in_results = True
                continue
            elif line.startswith('-' * 60):
                if in_results:
                    break
                continue
            
            if in_results and line and '.' in line and '%' in line:
                # Parse line like " 1. 11th Armored Division National Conventio  60.0%"
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        rank = int(parts[0].rstrip('.'))
                        score_str = parts[-1].rstrip('%')
                        score = float(score_str) / 100.0
                        
                        # Company name is everything between rank and score
                        company_name = ' '.join(parts[1:-1])
                        
                        results.append({
                            'rank': rank,
                            'company_name': company_name,
                            'score': score,
                            'likeness_percent': float(score_str)
                        })
                except (ValueError, IndexError):
                    continue
        
        return results
        
    except Exception as e:
        print(f"CLI execution error: {e}")
        return []

def run_webapp_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Run webapp search and return results"""
    try:
        # Make HTTP request to webapp
        response = requests.post(
            'http://localhost:5000/search',
            data={'query': query, 'top_k': top_k},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"Webapp Error: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        if not data.get('success'):
            print(f"Webapp Error: {data.get('error', 'Unknown error')}")
            return []
        
        # Extract and format results to match CLI format
        results = []
        for result in data['results']:
            results.append({
                'rank': result['rank'],
                'company_name': result['company_name'],
                'score': result['raw_score'],
                'likeness_percent': result['likeness_percent']
            })
        
        return results
        
    except Exception as e:
        print(f"Webapp request error: {e}")
        return []

def compare_results(cli_results: List[Dict], webapp_results: List[Dict], query: str) -> bool:
    """Compare CLI and webapp results for consistency"""
    print(f"\n🔍 Comparing results for query: '{query}'")
    print("=" * 60)
    
    if not cli_results:
        print("❌ CLI returned no results")
        return False
    
    if not webapp_results:
        print("❌ Webapp returned no results")
        return False
    
    print(f"CLI Results ({len(cli_results)}):")
    for result in cli_results:
        print(f"  {result['rank']:2d}. {result['company_name']:<40} {result['likeness_percent']:5.1f}%")
    
    print(f"\nWebapp Results ({len(webapp_results)}):")
    for result in webapp_results:
        print(f"  {result['rank']:2d}. {result['company_name']:<40} {result['likeness_percent']:5.1f}%")
    
    # Check if results are identical
    if len(cli_results) != len(webapp_results):
        print(f"\n❌ Result count mismatch: CLI={len(cli_results)}, Webapp={len(webapp_results)}")
        return False
    
    # Compare each result
    all_match = True
    for i, (cli_result, webapp_result) in enumerate(zip(cli_results, webapp_results)):
        if (cli_result['company_name'] != webapp_result['company_name'] or
            abs(cli_result['score'] - webapp_result['score']) > 0.001):
            
            print(f"\n❌ Result {i+1} mismatch:")
            print(f"  CLI:    {cli_result['company_name']} ({cli_result['score']:.3f})")
            print(f"  Webapp: {webapp_result['company_name']} ({webapp_result['score']:.3f})")
            all_match = False
    
    if all_match:
        print(f"\n✅ All results match perfectly!")
        return True
    else:
        print(f"\n❌ Results do not match!")
        return False

def test_consistency():
    """Run consistency tests"""
    print("🚀 Starting CLI vs Webapp Consistency Tests")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "eleventh armor",
        "tech solutions", 
        "microsoft",
        "apple inc",
        "national bank"
    ]
    
    all_passed = True
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        
        # Run both searches
        cli_results = run_cli_search(query, top_k=5)
        webapp_results = run_webapp_search(query, top_k=5)
        
        # Compare results
        if not compare_results(cli_results, webapp_results, query):
            all_passed = False
        
        print("-" * 40)
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! CLI and Webapp are perfectly synchronized.")
    else:
        print("❌ SOME TESTS FAILED! CLI and Webapp are not synchronized.")
        print("   Check the differences above and ensure both systems use identical logic.")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = test_consistency()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
