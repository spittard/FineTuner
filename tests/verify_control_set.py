
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


from finetuner.web.services.search_service import SearchService

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
    
    # Detailed results
    md_content.append("## Detailed Results\n\n")
    
    for i, result in enumerate(results, 1):
        query = result['query']
        top_match = result['top_match']
        all_matches = result['matches']
        
        md_content.append(f"### {i}. {query}\n\n")
        md_content.append(f"**Top Match**: {top_match['company_name']}\n\n")
        md_content.append(f"- **Final Score**: {top_match['raw_score']:.4f} ({top_match['likeness_percent']}%) \n")
        
        # Format the rationale as a blockquote
        rationale_lines = top_match.get('match_rationale', '').split('\n')
        md_content.append(f"- **Top Match Rationale**:\n")
        for line in rationale_lines:
            md_content.append(f"  > {line}\n")
        md_content.append("\n")
        
        if len(all_matches) > 1:
            num_to_show = min(len(all_matches), top_k)
            md_content.append(f"**Top {num_to_show} Matches:**\n\n")
            md_content.append("| Rank | Company Name | Score | Rationale |\n")
            md_content.append("|------|--------------|-------|-----------|\n")
            for j, match in enumerate(all_matches[:num_to_show], 1):
                name = match['company_name'].replace('|', '\\|')
                score_pct = match['likeness_percent']
                # Truncate rationale for table
                rationale = match.get('match_rationale', '').split('\n')[0][:50] + "..."
                md_content.append(f"| {j} | {name} | {score_pct:.1f}% | {rationale} |\n")
            md_content.append("\n")
        
        md_content.append("---\n\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    print(f"\nMarkdown report generated: {output_file}")

def main():
    control_set_file = 'companies_control_set.json'
    output_file = 'companies_control_set_results.md'
    top_k = 20
    
    print("=" * 60)
    print("Step 1: Initializing Search Service")
    print("=" * 60)
    
    service = SearchService()
    if not service.load_company_data():
        print("Failed to load company data via SearchService.")
        return

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
        # Use SearchService.search()
        matches = service.search(query, top_k=top_k)
        
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
                'top_match': {'company_name': 'No matches found', 'raw_score': 0.0, 'likeness_percent': 0.0},
                'matches': []
            })
            
    print("=" * 60)
    print("Step 4: Generating Report")
    print("=" * 60)
    
    generate_markdown_report(results, output_file=output_file, top_k=top_k)
    print("Done!")

if __name__ == "__main__":
    main()
