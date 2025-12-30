#!/usr/bin/env python3
"""
Generate control set report using RPC cache server.

This script connects to the running cache server for fast searches
instead of loading the cache directly.

Usage:
    1. Start the cache server: python -c "import sys; sys.path.insert(0,'src'); from finetuner.core.cache_rpc import run_server; run_server()"
    2. Load the cache (in another terminal): python test_rpc_search.py  (or via curl)
    3. Run this script: python tests/generate_report_via_rpc.py
"""

import json
import os
import sys
import time
from datetime import datetime

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core.cache_rpc import connect, is_server_running
from finetuner.web.services.rationale_service import RationaleService


def generate_report_header():
    """Generate the report header."""
    return f"""# Control Set Report (via RPC Cache Server)
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report was generated using the RPC cache server for efficient matching.

---

"""


def format_company_result(query, result_data, rank):
    """Format a single company result."""
    output = []
    
    results = result_data.get('results', [])
    
    output.append(f"## {rank}. Query: `{query}`\n")
    output.append(f"| Rank | Company | Score | Match Type |")
    output.append(f"|------|---------|-------|------------|")
    
    for i, match in enumerate(results[:10], 1):
        name = match.get('name', 'Unknown')
        score = match.get('score', 0)
        match_type = match.get('match_type', 'unknown')
        output.append(f"| {i} | {name} | {score:.3f} | {match_type} |")
    
    output.append("")
    return "\n".join(output)


def main():
    # Check if server is running
    if not is_server_running():
        print("ERROR: Cache server is not running!")
        print("Start it with:")
        print("  python -c \"import sys; sys.path.insert(0,'src'); from finetuner.core.cache_rpc import run_server; run_server()\"")
        sys.exit(1)
    
    # Connect to server
    print("Connecting to RPC cache server...")
    server = connect()
    
    # Check if cache is loaded
    loaded = server.list_loaded_caches()
    if not loaded:
        print("No caches loaded. Loading main cache...")
        if not server.load_cache('e1894e93a84bbc84a9ec980508a5fec4_loc'):
            print("Failed to load cache!")
            sys.exit(1)
    
    print(f"Using cache: {loaded[0] if loaded else 'e1894e93a84bbc84a9ec980508a5fec4_loc'}")
    
    # Load control set
    control_set_path = os.path.join(os.path.dirname(__file__), '..', 'companies_control_set.json')
    if not os.path.exists(control_set_path):
        print(f"Control set not found: {control_set_path}")
        sys.exit(1)
    
    with open(control_set_path, 'r', encoding='utf-8') as f:
        control_set = json.load(f)
    
    print(f"Processing {len(control_set)} queries...")
    
    # Generate report
    output_path = os.path.join(os.path.dirname(__file__), '..', 'control_set_report_ULTRA.md')
    
    report_content = generate_report_header()
    
    start_time = time.time()
    
    for i, entry in enumerate(control_set):
        query = entry.get('Company Name', entry.get('query', ''))
        if not query:
            continue
            
        print(f"[{i+1}/{len(control_set)}] Searching: {query}")
        
        # Search via RPC
        result = server.search(query, top_k=10)
        
        report_content += format_company_result(query, result, i+1)
        
        # Write intermediate results every 10 queries
        if (i + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            elapsed = time.time() - start_time
            print(f"   Saved intermediate report ({i+1} queries in {elapsed:.1f}s)")
    
    # Write final report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Report generated: {output_path}")
    print(f"Total time: {total_time:.1f}s ({len(control_set)} queries)")
    print(f"Average time per query: {total_time/len(control_set):.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
