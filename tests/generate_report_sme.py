#!/usr/bin/env python3
"""
Generate comprehensive SME-ready control set report via RPC.

Uses FULL RationaleService rationales with complete explanations:
- What This Means
- Action Required  
- Why This Happens
- Complete Score Breakdown
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core.cache_rpc import connect, is_server_running
from finetuner.web.services.rationale_service import RationaleService


def generate_report_header():
    """Generate the report header."""
    return f"""# Control Set Report - SME Review
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Matching Methodology

| Component | Weight | Description |
|-----------|--------|-------------|
| String Similarity | 70% | Jaro-Winkler lexical comparison |
| Semantic Similarity | 30% | Neural embedding (MiniLM) |
| Acronym Fidelity | +15% | Bonus for acronym expansion |
| Location Boost | +5% | Bonus for location match |

---

"""


def format_company_result(query, result_data, query_num):
    """Format with COMPREHENSIVE rationales using RationaleService."""
    output = []
    results = result_data.get('results', [])
    
    output.append(f"## {query_num}. Query: `{query}`")
    output.append("")
    
    if not results:
        output.append("*No matches found*")
        output.append("")
        return "\n".join(output)
    
    # Table header
    output.append("| # | Score | Company | Type | Rationale |")
    output.append("|:-:|:-----:|---------|------|-----------|")
    
    for i, match in enumerate(results[:10], 1):
        name = match.get('name', 'Unknown')
        score = match.get('score', 0)
        string_score = match.get('string_score', 0)
        semantic_score = match.get('semantic_score', 0)
        acronym_fidelity = match.get('acronym_fidelity', 0)
        
        # Icon
        icon = "🟢" if score >= 0.80 else "🟡" if score >= 0.60 else "🟠" if score >= 0.40 else "🔴"
        
        # Match type
        if string_score >= 0.9:
            match_type = "Exact"
        elif acronym_fidelity > 0.5:
            match_type = "Acronym"
        elif semantic_score > string_score:
            match_type = "Semantic"
        else:
            match_type = "Lexical"
        
        # Get COMPREHENSIVE rationale
        full_rationale = RationaleService.generate_match_rationale(
            query=query,
            company_name=name,
            explanation=match,
            score=score
        )
        
        # Get detailed score breakdown
        score_breakdown = RationaleService.generate_detailed_score_breakdown(match, query)
        
        # Combine and format for HTML
        combined = f"{full_rationale}\n\n---\n\n{score_breakdown}"
        rationale_html = combined.replace('\n', '<br>').replace('|', '&#124;')
        
        output.append(f"| {i} | {icon} **{score:.1%}** | `{name}` | {match_type} | <details><summary>📊 View</summary><br>{rationale_html}</details> |")
    
    output.append("")
    output.append("---")
    output.append("")
    
    return "\n".join(output)


def main():
    if not is_server_running():
        print("ERROR: Cache server not running!")
        print('Start: python -c "import sys; sys.path.insert(0,\'src\'); from finetuner.core.cache_rpc import run_server; run_server()"')
        sys.exit(1)
    
    print("Connecting to RPC cache server...")
    server = connect()
    
    loaded = server.list_loaded_caches()
    if not loaded:
        print("Loading cache...")
        server.load_cache('e1894e93a84bbc84a9ec980508a5fec4_loc')
        loaded = ['e1894e93a84bbc84a9ec980508a5fec4_loc']
    
    print(f"Using cache: {loaded[0]}")
    
    control_set_path = os.path.join(os.path.dirname(__file__), '..', 'companies_control_set.json')
    with open(control_set_path, 'r', encoding='utf-8') as f:
        control_set = json.load(f)
    
    print(f"Processing {len(control_set)} queries with COMPREHENSIVE rationales...")
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'control_set_report_SME.md')
    report_content = generate_report_header()
    
    start_time = time.time()
    
    for i, entry in enumerate(control_set):
        query = entry.get('Company Name', entry.get('query', ''))
        if not query:
            continue
        
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(control_set) - i - 1) if i > 0 else 0
        print(f"[{i+1}/{len(control_set)}] ({elapsed:.0f}s, ETA: {eta:.0f}s) {query}")
        
        result = server.search(query, top_k=10)
        report_content += format_company_result(query, result, i+1)
        
        if (i + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"   [saved] Checkpoint written")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[DONE] Report: {output_path}")
    print(f"   Time: {total_time:.1f}s ({len(control_set)} queries)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
