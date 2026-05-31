
#!/usr/bin/env python3
"""
Verify Control Set - Submit control companies to CompanyMatcher using the new package structure.
Generates a markdown report for comparison.

When a control entry includes City/State, search uses match_with_location (via RPC).
Optional fields control_match_state and control_match_city assert the top match's geography
(case study: duplicate legal name, multiple offices).
"""

import json
import os
import sys
import time
from datetime import datetime

# Add src to python path to access finetuner package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT, 'src'))


from finetuner.web.services.search_service import SearchService
from finetuner.utils.text_preprocessor import TextPreprocessor

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


def load_control_set(filepath):
    """Load full control set list (dicts with Company Name and optional City/State/assertions)."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return None
    print(f"Loading control set from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Error: Control set must be a JSON array")
        return None
    items = [x for x in data if isinstance(x, dict) and "Company Name" in x]
    print(f"Loaded {len(items):,} control entries")
    return items


def _norm_loc(s):
    if not s:
        return ""
    return str(s).strip().lower()


def check_location_assertion(item, top_match):
    """Return None if OK, else error string."""
    exp_state = item.get("control_match_state")
    exp_city = item.get("control_match_city")
    if not exp_state and not exp_city:
        return None
    got_state = TextPreprocessor.normalize_state(top_match.get("state") or "")
    got_city = _norm_loc(top_match.get("city") or "")
    if exp_state:
        want = TextPreprocessor.normalize_state(exp_state)
        if got_state != want:
            return f"expected state {exp_state!r} (norm {want!r}), top match has {top_match.get('state')!r} (norm {got_state!r})"
    if exp_city:
        want_c = _norm_loc(exp_city)
        if want_c and want_c not in got_city and got_city not in want_c:
            # allow prefix / normalized match
            if want_c != got_city:
                return f"expected city containing {want_c!r}, got {got_city!r}"
    return None


def generate_markdown_report(results, output_file='companies_control_set_results.md', top_k=10):
    """Generate markdown report from test results"""
    
    md_content = []
    md_content.append("# Company Match Control Set Results\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"Total Companies Tested: {len(results)}\n")
    failed = [r for r in results if r.get('assertion_error')]
    if failed:
        md_content.append(f"**Location assertions failed:** {len(failed)}\n")
    md_content.append("---\n\n")
    
    md_content.append("## Detailed Results\n\n")
    
    for i, result in enumerate(results, 1):
        query = result['query']
        loc_note = result.get('query_location') or ""
        top_match = result['top_match']
        all_matches = result['matches']
        err = result.get('assertion_error')
        
        title = f"{i}. {query}"
        if loc_note:
            title += f" — {loc_note}"
        md_content.append(f"### {title}\n\n")
        if err:
            md_content.append(f"**ASSERTION FAILED:** {err}\n\n")
        md_content.append(f"**Top Match**: {top_match['company_name']}\n\n")
        if top_match.get('city') or top_match.get('state'):
            md_content.append(f"- **Match location**: {top_match.get('city', '')}, {top_match.get('state', '')}\n")
        md_content.append(f"- **Final Score**: {top_match['raw_score']:.4f} ({top_match['likeness_percent']}%) \n")
        
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
                rationale = match.get('match_rationale', '').split('\n')[0][:50] + "..."
                md_content.append(f"| {j} | {name} | {score_pct:.1f}% | {rationale} |\n")
            md_content.append("\n")
        
        md_content.append("---\n\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    print(f"\nMarkdown report generated: {output_file}")


def main():
    control_set_file = os.path.join(ROOT, 'companies_control_set.json')
    output_file = os.path.join(ROOT, 'companies_control_set_results.md')
    top_k = 20
    
    print("=" * 60)
    print("Step 1: Initializing Search Service")
    print("=" * 60)
    
    service = SearchService()
    if not service.load_company_data():
        return 1

    print("=" * 60)
    print("Step 2: Loading Control Set")
    print("=" * 60)
    
    control_items = load_control_set(control_set_file)
    if not control_items:
        return 1
        
    print("=" * 60)
    print(f"Step 3: Running {len(control_items)} Tests")
    print("=" * 60)
    
    results = []
    assertion_failures = []
    
    for item in tqdm(control_items, desc="Running matches", unit="query"):
        query = item["Company Name"]
        city = item.get("City") or None
        state = item.get("State") or None
        if city is not None and str(city).strip() == "":
            city = None
        if state is not None and str(state).strip() == "":
            state = None
        loc_label = f"{city or ''}, {state or ''}".strip(", ") if (city or state) else ""
        
        try:
            matches = service.search(query, top_k=top_k, city=city, state=state)
        except Exception as e:
            results.append({
                'query': query,
                'query_location': loc_label,
                'top_match': {'company_name': f'Error: {e}', 'raw_score': 0.0, 'likeness_percent': 0.0, 'match_rationale': str(e)},
                'matches': [],
                'assertion_error': f'search error: {e}',
            })
            assertion_failures.append(query)
            continue
        
        if matches:
            top_match = matches[0]
            err = check_location_assertion(item, top_match)
            if err:
                assertion_failures.append(f"{query} ({loc_label}): {err}")
            results.append({
                'query': query,
                'query_location': loc_label,
                'top_match': top_match,
                'matches': matches,
                'assertion_error': err,
            })
        else:
            results.append({
                'query': query,
                'query_location': loc_label,
                'top_match': {'company_name': 'No matches found', 'raw_score': 0.0, 'likeness_percent': 0.0},
                'matches': [],
                'assertion_error': 'no matches' if (item.get('control_match_state') or item.get('control_match_city')) else None,
            })
            
    print("=" * 60)
    print("Step 4: Generating Report")
    print("=" * 60)
    
    generate_markdown_report(results, output_file=output_file, top_k=top_k)
    
    if assertion_failures:
        print("\n--- ASSERTION FAILURES ---")
        for f in assertion_failures:
            print(f"  - {f}")
        print(f"\nTotal assertion failures: {len(assertion_failures)}")
        return 1
    
    print("Done! All location assertions passed (where specified).")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
