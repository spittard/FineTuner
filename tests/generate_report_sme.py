#!/usr/bin/env python3
"""
Generate comprehensive SME-ready control set report via RPC.

This creates a detailed report with:
- Short summary for quick scanning
- Expandable detailed rationales (click to reveal)
- Complete scoring breakdown
- Relative ranking explanations
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
    """Generate the report header with methodology explanation."""
    return f"""# Control Set Report - SME Review
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Matching Methodology Overview

This report uses a **hybrid semantic + lexical matching** approach:

| Component | Weight | Description |
|-----------|--------|-------------|
| **String Similarity** | 70% | Lexical comparison using Jaro-Winkler distance on normalized company names |
| **Semantic Similarity** | 30% | Neural embedding comparison using SentenceTransformers (paraphrase-MiniLM-L3-v2) |
| **Acronym Fidelity** | +15% max | Bonus for matching acronym expansions (e.g., IBM → International Business Machines) |
| **Location Boost** | +5% max | Bonus when query location matches company location |

### Score Interpretation

| Score Range | Confidence | Recommendation |
|-------------|------------|----------------|
| ≥ 95% | **Exact/Near-Exact** | Auto-accept match |
| 80-94% | **High** | Likely correct, quick verification |
| 60-79% | **Medium** | Requires human review |
| 40-59% | **Low** | Multiple candidates, careful selection needed |
| < 40% | **Very Low** | May need manual search |

---

"""


def format_short_summary(query, match, rank):
    """Generate a concise one-line summary."""
    name = match.get('name', 'Unknown')
    score = match.get('score', 0)
    match_type = match.get('match_type', 'hybrid')
    string_score = match.get('string_score', 0)
    semantic_score = match.get('semantic_score', 0)
    
    # Determine match quality
    if score >= 0.95:
        quality = "🟢 EXACT"
    elif score >= 0.80:
        quality = "🟢 HIGH"
    elif score >= 0.60:
        quality = "🟡 MEDIUM"
    elif score >= 0.40:
        quality = "🟠 LOW"
    else:
        quality = "🔴 WEAK"
    
    # Create key insight
    if string_score >= 0.9:
        insight = "Near-exact text match"
    elif match.get('acronym_fidelity', 0) > 0.5:
        insight = f"Acronym expansion ({match.get('acronym_fidelity', 0):.0%} fidelity)"
    elif semantic_score > string_score:
        insight = "Semantic/meaning-based match"
    else:
        insight = "Lexical word matching"
    
    return f"{quality} **{score:.1%}** | {insight}"


def format_detailed_rationale(query, match, rank, matches_above=None, matches_below=None):
    """Generate comprehensive detailed rationale for SME review."""
    name = match.get('name', 'Unknown')
    score = match.get('score', 0)
    string_score = match.get('string_score', 0)
    semantic_score = match.get('semantic_score', 0)
    acronym_fidelity = match.get('acronym_fidelity', 0)
    match_type = match.get('match_type', 'hybrid')
    
    detail = []
    
    # Score breakdown table
    detail.append("**Score Breakdown:**")
    detail.append("")
    detail.append("| Component | Value | Weight | Contribution |")
    detail.append("|-----------|-------|--------|--------------|")
    
    string_contrib = string_score * 0.7
    detail.append(f"| String Similarity | {string_score:.4f} | 70% | {string_contrib:.4f} |")
    
    sem_contrib = semantic_score * 0.3
    detail.append(f"| Semantic Similarity | {semantic_score:.4f} | 30% | {sem_contrib:.4f} |")
    
    base_score = string_contrib + sem_contrib
    detail.append(f"| **Base Score** | - | - | **{base_score:.4f}** |")
    
    if acronym_fidelity > 0:
        acr_boost = acronym_fidelity * 0.15
        detail.append(f"| Acronym Fidelity | {acronym_fidelity:.4f} | +15% max | +{acr_boost:.4f} |")
    
    detail.append(f"| **Final Score** | - | - | **{score:.4f}** ({score:.1%}) |")
    detail.append("")
    
    # Why this matched
    detail.append("**Why This Matched:**")
    detail.append("")
    
    # Analyze text overlap
    query_words = set(query.lower().split())
    name_words = set(name.lower().split())
    overlap = query_words & name_words
    
    if overlap:
        detail.append(f"- **Word Overlap:** {', '.join(sorted(overlap))}")
    
    # Explain string similarity
    if string_score >= 0.9:
        detail.append(f"- **Lexical Match:** Nearly identical text (Jaro-Winkler: {string_score:.3f})")
    elif string_score >= 0.7:
        detail.append(f"- **Lexical Match:** Strong word alignment (Jaro-Winkler: {string_score:.3f})")
    elif string_score >= 0.5:
        detail.append(f"- **Lexical Match:** Moderate word alignment (Jaro-Winkler: {string_score:.3f})")
    else:
        detail.append(f"- **Lexical Match:** Weak word alignment (Jaro-Winkler: {string_score:.3f}) - relies on semantic similarity")
    
    # Explain semantic similarity
    if semantic_score >= 0.8:
        detail.append(f"- **Semantic Match:** Very strong meaning-based connection (cosine: {semantic_score:.3f})")
    elif semantic_score >= 0.6:
        detail.append(f"- **Semantic Match:** Good meaning-based connection (cosine: {semantic_score:.3f})")
    elif semantic_score >= 0.4:
        detail.append(f"- **Semantic Match:** Moderate meaning-based connection (cosine: {semantic_score:.3f})")
    else:
        detail.append(f"- **Semantic Match:** Weak meaning-based connection (cosine: {semantic_score:.3f})")
    
    # Acronym analysis
    if acronym_fidelity > 0:
        if acronym_fidelity >= 0.9:
            detail.append(f"- **Acronym:** Query appears to be acronym of this company ({acronym_fidelity:.0%} fidelity)")
        elif acronym_fidelity >= 0.5:
            detail.append(f"- **Acronym:** Possible acronym relationship ({acronym_fidelity:.0%} fidelity)")
    
    detail.append("")
    
    # Ranking justification
    detail.append("**Ranking Justification:**")
    detail.append("")
    
    if rank == 1:
        detail.append(f"- Ranked **#1** because it has the highest combined score ({score:.4f})")
        if matches_below:
            gap = score - matches_below[0].get('score', 0)
            detail.append(f"- Score gap to #2: {gap:.4f} ({gap:.1%})")
    else:
        if matches_above:
            gap_above = matches_above[-1].get('score', 0) - score
            detail.append(f"- Ranked **#{rank}** - score is {gap_above:.4f} lower than #{rank-1}")
        if matches_below:
            gap_below = score - matches_below[0].get('score', 0)
            detail.append(f"- Score is {gap_below:.4f} higher than #{rank+1}")
    
    return "\n".join(detail)


def format_company_result(query, result_data, rank):
    """Format a single company result with expandable details."""
    output = []
    
    results = result_data.get('results', [])
    
    output.append(f"## {rank}. Query: `{query}`")
    output.append("")
    
    if not results:
        output.append("*No matches found*")
        output.append("")
        return "\n".join(output)
    
    # Summary table (always visible)
    output.append("| Rank | Company | Score | Summary |")
    output.append("|:----:|---------|:-----:|---------|")
    
    for i, match in enumerate(results[:10], 1):
        name = match.get('name', 'Unknown')
        score = match.get('score', 0)
        summary = format_short_summary(query, match, i)
        output.append(f"| {i} | {name} | **{score:.1%}** | {summary} |")
    
    output.append("")
    
    # Detailed rationales (expandable - using HTML details tag)
    output.append("<details>")
    output.append("<summary><strong>📊 Click for Detailed Match Rationales</strong></summary>")
    output.append("")
    
    for i, match in enumerate(results[:5], 1):  # Top 5 detailed
        name = match.get('name', 'Unknown')
        score = match.get('score', 0)
        
        matches_above = results[:i-1] if i > 1 else None
        matches_below = results[i:i+1] if i < len(results) else None
        
        output.append(f"### Rank #{i}: {name}")
        output.append("")
        
        detail = format_detailed_rationale(query, match, i, matches_above, matches_below)
        output.append(detail)
        output.append("")
        output.append("---")
        output.append("")
    
    output.append("</details>")
    output.append("")
    
    return "\n".join(output)


def main():
    # Check if server is running
    if not is_server_running():
        print("ERROR: Cache server is not running!")
        print("Start it with:")
        print('  python -c "import sys; sys.path.insert(0,\'src\'); from finetuner.core.cache_rpc import run_server; run_server()"')
        sys.exit(1)
    
    # Connect to server
    print("Connecting to RPC cache server...")
    server = connect()
    
    # Check if cache is loaded
    loaded = server.list_loaded_caches()
    if not loaded:
        print("No caches loaded. Loading main cache...")
        cache_key = 'e1894e93a84bbc84a9ec980508a5fec4_loc'
        print(f"Loading {cache_key}...")
        if not server.load_cache(cache_key):
            print("Failed to load cache!")
            sys.exit(1)
        loaded = [cache_key]
    
    print(f"Using cache: {loaded[0]}")
    
    # Load control set
    control_set_path = os.path.join(os.path.dirname(__file__), '..', 'companies_control_set.json')
    if not os.path.exists(control_set_path):
        print(f"Control set not found: {control_set_path}")
        sys.exit(1)
    
    with open(control_set_path, 'r', encoding='utf-8') as f:
        control_set = json.load(f)
    
    print(f"Processing {len(control_set)} queries with detailed rationales...")
    
    # Generate report
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
        
        # Search via RPC
        result = server.search(query, top_k=10)
        
        report_content += format_company_result(query, result, i+1)
        
        # Write intermediate results every 10 queries
        if (i + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"   ✓ Saved checkpoint")
    
    # Write final report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Report generated: {output_path}")
    print(f"   Total time: {total_time:.1f}s ({len(control_set)} queries)")
    print(f"   Average: {total_time/len(control_set):.2f}s per query")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
