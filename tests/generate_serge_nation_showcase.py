#!/usr/bin/env python3
"""
Generate Serge Nation Showcase Report.

Curated examples demonstrating ALL matching nuances:
1. Perfect exact matches
2. Acronym expansions (IBM, ABA, PDMA)
3. Semantic matches (similar meaning, different words)
4. Word overlap matches (partial name matches)
5. Location-boosted matches
6. Popularity-boosted matches
7. Edge cases and false positives
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core.cache_rpc import connect, is_server_running
from finetuner.web.services.rationale_service import RationaleService


# Curated queries prioritized by LIKELIHOOD OF OCCURRENCE
SHOWCASE_QUERIES = [
    # === TIER 1: HIGH FREQUENCY (~60% of cases) ===
    # Exact and near-exact matches are the bread and butter.
    {
        "query": "Next Level Events",
        "category": "Exact Match",
        "likelihood": "Very High (~60%)",
        "why": "The system's primary job: finding the exact name quickly and incorrectly handling duplicates."
    },
    {
        "query": "DermaQuest Inc",
        "category": "Corporate Suffix",
        "likelihood": "Very High (~60%)",
        "why": "Users often omit 'Inc', 'LLC', etc. The system handles this trivially."
    },

    # === TIER 2: COMMON VARIATIONS (~25% of cases) ===
    # Partial names, typos, and word overlaps.
    {
        "query": "Hartford Hospital School of Nursing",
        "category": "Partial/Word Overlap",
        "likelihood": "High (~25%)",
        "why": "Long names where the user query matches a significant chunk (but not all) of the official name.",
    },
    {
        "query": "Mitsubishi Motor Sales of America, Incorporated",
        "category": "Complex Corporate Name",
        "likelihood": "High (~25%)",
        "why": "Complex names with multiple variations in the database."
    },
    {
        "query": "Southern Vermont Deerfield Valley Chamber of commerce",
        "category": "Location/Regional",
        "likelihood": "Medium (~15%)",
        "why": "Names where the location (Vermont) is part of the entity name itself.",
    },

    # === TIER 3: ACRONYMS (~10% of cases) ===
    # High value, but less frequent. The "magic" layer.
    {
        "query": "IBM",
        "category": "Acronym Expansion",
        "likelihood": "Medium (~10%)",
        "why": "Classic 3-letter acronym. Shows the system knowing 'IBM' = 'International Business Machines'."
    },
    {
        "query": "PDMA",
        "category": "Acronym Ambiguity",
        "likelihood": "Medium (~10%)",
        "why": "Acronyms that might expand to multiple different organizations (Product Dev vs. others)."
    },
    {
        "query": "NFC Forum",
        "category": "Mixed Acronym",
        "likelihood": "Low (~5%)",
        "why": "Mixture of acronyms and regular words."
    },

    # === TIER 4: SEMANTIC & EDGE CASES (<5% of cases) ===
    # Rare but "noisy" if noticed.
    {
        "query": "Chicago South Swim Club",
        "category": "Semantic/Topic",
        "likelihood": "Low (<5%)",
        "why": "No word overlap, but conceptually similar to other swim clubs (Semantic Search)."
    },
    {
        "query": "Nicolas/Sanchez Wedding",
        "category": "Noise/Events",
        "likelihood": "Rare (<1%)",
        "why": "Personal events that pollute the company database. Shows how scoring deprioritizes them."
    },
    {
        "query": "X DO NOT USE - FRANCIS PARKER SCHOOL",
        "category": "Dirty Data",
        "likelihood": "Rare (<1%)",
        "why": "Database artifacts (DO NOT USE) that can confuse search engines."
    },
]


def generate_report_header():
    """Generate the Serge Nation report header."""
    return f"""# Serge Nation Scenario Showcase
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This report demonstrates the **nuances and edge cases** of our company matching system.
Each scenario shows a different challenge the algorithm must handle, with full rationale
explanations to help Serge Nation understand the matching logic.

---

## Scoring Formula

| Component | Weight | Description |
|-----------|--------|-------------|
| String Similarity | 70% | Jaro-Winkler lexical comparison |
| Semantic Similarity | 30% | Neural embedding (MiniLM) |
| Acronym Fidelity | +15% max | Bonus for acronym → expansion |
| Location Boost | +5% max | Bonus when location matches |
| Popularity Boost | +5% max | Bonus for frequently appearing names |

---

"""


def format_company_result(query, result_data, query_num, category, why, likelihood):
    """Format with COMPREHENSIVE rationales for Serge Nation."""
    output = []
    results = result_data.get('results', [])
    
    # Header containing the vital context for the SME
    output.append(f"## {query_num}. {query}")
    output.append(f"**Scenario Category:** {category}  |  **Likelihood:** `{likelihood}`")
    output.append("")
    output.append(f"> **Why matches happen here:** {why}")
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
        
        # Match type logic
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
        sys.exit(1)
    
    print("Connecting to RPC cache server...")
    server = connect()
    
    loaded = server.list_loaded_caches()
    if not loaded:
        print("Loading cache...")
        server.load_cache('e1894e93a84bbc84a9ec980508a5fec4_loc')
        loaded = ['e1894e93a84bbc84a9ec980508a5fec4_loc']
    
    print(f"Using cache: {loaded[0]}")
    print(f"Processing {len(SHOWCASE_QUERIES)} curated scenarios for Serge Nation...")
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'serge_nation_showcase.md')
    report_content = generate_report_header()
    
    start_time = time.time()
    
    for i, entry in enumerate(SHOWCASE_QUERIES):
        query = entry['query']
        category = entry['category']
        why = entry['why']
        likelihood = entry['likelihood']
        
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(SHOWCASE_QUERIES) - i - 1) if i > 0 else 0
        print(f"[{i+1}/{len(SHOWCASE_QUERIES)}] ({elapsed:.0f}s, ETA: {eta:.0f}s) [{likelihood}] {query}")
        
        result = server.search(query, top_k=10)
        report_content += format_company_result(query, result, i+1, category, why, likelihood)
    
    # Add summary section
    report_content += """
## Key Takeaways for Serge Nation

### Acronym Matching
- The algorithm uses **pure pattern matching** (first letters of words)
- It does NOT know if an organization is "well-known"
- Multiple valid expansions get similar fidelity scores

### Semantic vs Lexical
- **Semantic** matches find conceptually similar names
- **Lexical** matches find text-similar names
- The 70/30 weighting balances both approaches

### Edge Cases
- Wedding/event names match other similar events
- Data quality issues (markers like "DO NOT USE") are exposed
- Non-English text may have lower semantic scores

### What This Means for Data Entry
- High scores (≥80%) = likely correct, quick verification
- Medium scores (60-79%) = requires human judgment
- Low scores (<60%) = may need manual search
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[DONE] Serge Nation Showcase: {output_path}")
    print(f"   Scenarios: {len(SHOWCASE_QUERIES)}")
    print(f"   Time: {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
