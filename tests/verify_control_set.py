
#!/usr/bin/env python3
"""
Verify Control Set - Submit control companies to CompanyMatcher using the new package structure.
Generates a markdown report for comparison.
"""

import json
import os
import sys
import time
from datetime import datetime

# Add src to python path to access finetuner package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core.matcher import CompanyMatcher

# Try to import tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if total is None:
            total = len(iterable) if hasattr(iterable, '__len__') else None
        if desc:
            print(f"{desc}...")
        return iterable

def load_companies_from_json(filepath):
    """Load company names from JSON dataset"""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return None
    
    print(f"Loading companies from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        company_names = []
        for item in data:
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
        
        if not company_names:
            print("Error: No company names found in dataset")
            return None
        
        print(f"Loaded {len(company_names):,} company names")
        return company_names
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def generate_markdown_report(results, output_file='companies_control_set_results.md', top_k=10):
    """Generate markdown report from test results"""
    
    md_content = []
    md_content.append("# Company Match Control Set Results\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"Total Companies Tested: {len(results)}\n")
    md_content.append("---\n\n")
    
    # Summary statistics
    exact_matches = sum(1 for r in results if r['top_match']['match_type'] == 'exact')
    high_confidence = sum(1 for r in results if r['top_match']['score'] >= 0.9)
    medium_confidence = sum(1 for r in results if 0.7 <= r['top_match']['score'] < 0.9)
    low_confidence = sum(1 for r in results if r['top_match']['score'] < 0.7)
    
    md_content.append("## Summary Statistics\n\n")
    md_content.append(f"- **Exact Matches**: {exact_matches} ({exact_matches/len(results)*100:.1f}%)\n")
    md_content.append(f"- **High Confidence (>=0.9)**: {high_confidence} ({high_confidence/len(results)*100:.1f}%)\n")
    md_content.append(f"- **Medium Confidence (0.7-0.9)**: {medium_confidence} ({medium_confidence/len(results)*100:.1f}%)\n")
    md_content.append(f"- **Low Confidence (<0.7)**: {low_confidence} ({low_confidence/len(results)*100:.1f}%)\n\n")
    md_content.append("---\n\n")
    
    # Detailed results
    md_content.append("## Detailed Results\n\n")
    
    for i, result in enumerate(results, 1):
        query = result['query']
        top_match = result['top_match']
        all_matches = result['matches']
        
        md_content.append(f"### {i}. {query}\n\n")
        md_content.append(f"**Top Match**: {top_match['name']}\n\n")
        md_content.append(f"- **Score**: {top_match['score']:.4f} ({top_match['score']*100:.2f}%)\n")
        md_content.append(f"- **Match Type**: {top_match.get('match_type', 'unknown')}\n")
        md_content.append(f"- **String Score**: {top_match.get('string_score', 0):.4f}\n")
        md_content.append(f"- **Semantic Score**: {top_match.get('semantic_score', 0):.4f}\n\n")
        
        if len(all_matches) > 1:
            num_to_show = min(len(all_matches), top_k)
            md_content.append(f"**Top {num_to_show} Matches:**\n\n")
            md_content.append("| Rank | Company Name | Score | String | Semantic | Type |\n")
            md_content.append("|------|--------------|-------|--------|----------|------|\n")
            for j, match in enumerate(all_matches[:num_to_show], 1):
                name = match['name'].replace('|', '\\|')
                score_pct = match['score'] * 100
                string_score = match.get('string_score', 0)
                semantic_score = match.get('semantic_score', 0)
                match_type = match.get('match_type', 'unknown')
                md_content.append(f"| {j} | {name} | {score_pct:.2f}% | {string_score:.4f} | {semantic_score:.4f} | {match_type} |\n")
            md_content.append("\n")
        
        md_content.append("---\n\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    print(f"\nMarkdown report generated: {output_file}")

def main():
    control_set_file = 'companies_control_set.json'
    dataset_file = 'companies.json'
    output_file = 'companies_control_set_results.md'
    top_k = 20  # Use 20 to match previous report
    
    print("=" * 60)
    print("Step 1: Initializing Matcher")
    print("=" * 60)
    
    matcher = CompanyMatcher(model_name='all-MiniLM-L6-v2')
    matcher.build_index(filepath=dataset_file)
    
    print("=" * 60)
    print("Step 2: Loading Control Set")
    print("=" * 60)
    
    control_names = load_companies_from_json(control_set_file)
    if not control_names:
        return
        
    print("=" * 60)
    print(f"Step 3: Running {len(control_names)} Tests")
    print("=" * 60)
    
    results = []
    
    for query in tqdm(control_names, desc="Running matches", unit="query"):
        matches = matcher.match(query, top_k=top_k)
        
        if matches:
            top_match = matches[0]
            results.append({
                'query': query,
                'top_match': top_match,
                'matches': matches
            })
        else:
            results.append({
                'query': query,
                'top_match': {'name': 'No matches found', 'score': 0.0, 'match_type': 'none'},
                'matches': []
            })
            
    print("=" * 60)
    print("Step 4: Generating Report")
    print("=" * 60)
    
    generate_markdown_report(results, output_file=output_file, top_k=top_k)
    print("Done!")

if __name__ == "__main__":
    main()
