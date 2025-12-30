#!/usr/bin/env python3
"""
Quick single-query test with COMPREHENSIVE rationales.
Uses RationaleService.generate_match_rationale for FULL detailed output.
"""
import sys
sys.path.insert(0, 'src')

from finetuner.core.cache_rpc import connect
from finetuner.web.services.rationale_service import RationaleService

server = connect()

# Single query test
query = "PDMA Association"
result = server.search(query, top_k=10)

# Generate output in TABLE format with COMPREHENSIVE rationale
output = []
output.append(f"## 1. Query: `{query}`")
output.append("")

# Table header
output.append("| # | Score | Company | Type | Rationale |")
output.append("|:-:|:-----:|---------|------|-----------|")

for i, match in enumerate(result['results'][:10], 1):
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
    
    # Get COMPREHENSIVE rationale from RationaleService
    full_rationale = RationaleService.generate_match_rationale(
        query=query,
        company_name=name,
        explanation=match,
        score=score
    )
    
    # Also get detailed score breakdown
    score_breakdown = RationaleService.generate_detailed_score_breakdown(match, query)
    
    # Combine into expandable section
    combined_rationale = f"{full_rationale}\n\n---\n\n{score_breakdown}"
    
    # Escape for HTML and format for collapsible
    rationale_html = combined_rationale.replace('\n', '<br>').replace('|', '\\|')
    
    # Table row with expand link
    output.append(f"| {i} | {icon} **{score:.1%}** | `{name}` | {match_type} | <details><summary>📊 View</summary><br>{rationale_html}</details> |")

output.append("")

# Write test output
with open('test_single_query_output.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("Written to test_single_query_output.md")
print("\n" + "="*60)
print("PREVIEW OF FIRST MATCH RATIONALE:")
print("="*60)

# Print first match comprehensive rationale
first_match = result['results'][0]
print(RationaleService.generate_match_rationale(
    query=query,
    company_name=first_match['name'],
    explanation=first_match,
    score=first_match['score']
))
