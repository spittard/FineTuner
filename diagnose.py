
import os
import sys
import argparse
import json
import time

# Add src to python path to allow imports from finetuner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.web.services.search_service import SearchService
from finetuner.web.services.rationale_service import RationaleService

def format_diagnostic_result(query, city, state, matches):
    """Format matching results into a detailed Markdown report."""
    md = []
    md.append(f"# Diagnostic Report: '{query}'\n")
    
    loc_str = f"Location: {city}, {state}" if (city or state) else "Location: None (Name-only search)"
    md.append(f"**{loc_str}**\n\n")
    
    if not matches:
        md.append("❌ **No matches found below the threshold.**\n")
        return "\n".join(md)
    
    top_match = matches[0]
    name = top_match.get('company_name', 'Unknown')
    score = top_match.get('likeness_percent', 0.0)
    m_city = top_match.get('city', '')
    m_state = top_match.get('state', '')
    m_loc = f" ({m_city}, {m_state})" if (m_city or m_state) else ""
    
    md.append(f"## Top Match: {name}{m_loc} • Score: {score:.1f}%\n\n")
    
    # 1. Detailed Score Breakdown
    md.append("### 📊 Scoring Breakdown\n")
    breakdown = RationaleService.generate_detailed_score_breakdown(top_match, query)
    md.append(f"{breakdown}\n")
    
    # 2. Narrative Rationale
    md.append("### 📝 Match Rationale\n")
    explanation_dict = top_match.get('explanation_details', {})
    rationale = RationaleService.generate_match_rationale(query, name, explanation_dict, score / 100.0)
    md.append(f"{rationale}\n")
    
    # 3. Top 5 Results Overview
    md.append("### 🔝 Top 5 Candidates\n")
    for i, match in enumerate(matches[:5], 1):
        m_name = match.get('company_name', 'Unknown')
        m_score = match.get('likeness_percent', 0.0)
        mc = match.get('city', '')
        ms = match.get('state', '')
        ml = f" ({mc}, {ms})" if (mc or ms) else ""
        
        m_explanation = match.get('explanation_details', {})
        note = RationaleService.get_short_summary(query, m_name, m_explanation)
        
        md.append(f"{i}. **{m_name}{ml}** - {m_score:.1f}%  \n")
        md.append(f"   *Insight: {note}*\n")
        
        if i > 1:
            prev_match = matches[i-2]
            rel_pos = RationaleService.generate_relative_positioning_explanation(match, prev_match, None, i)
            # Indent the relative positioning
            indented_rel = "\n".join([f"   > {line}" for line in rel_pos.split('\n')])
            md.append(f"{indented_rel}\n")
            
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="Diagnose company matching for a single query.")
    parser.add_argument("query", help="The company name to search for.")
    parser.add_argument("--city", help="Optional city for location-aware matching.")
    parser.add_argument("--state", help="Optional state for location-aware matching.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of matches to retrieve.")
    parser.add_argument("--model", default="paraphrase-MiniLM-L3-v2", help="Model name to use.")
    parser.add_argument("--data", default="companies_with_location.json", help="Data file to search.")
    
    args = parser.parse_args()
    
    print(f"🔍 Initializing diagnostic for: '{args.query}'...", file=sys.stderr)
    service = SearchService()
    
    # Load data
    if not service.load_company_data(model_name=args.model, filename=args.data):
        print(f"❌ Error: Could not load data file {args.data}", file=sys.stderr)
        sys.exit(1)
        
    # Perform Search
    start_time = time.time()
    matches = service.search(
        args.query, 
        top_k=args.top_k, 
        city=args.city, 
        state=args.state
    )
    search_duration = time.time() - start_time
    
    # Generate Report
    report = format_diagnostic_result(args.query, args.city, args.state, matches)
    
    print("-" * 40, file=sys.stderr)
    print(f"✅ Search completed in {search_duration:.2f}s", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    print("\n" + report)

if __name__ == "__main__":
    main()
