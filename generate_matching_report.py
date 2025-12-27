
import sys
import os
import json
import time

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.core.matcher import CompanyMatcher

# Selected 10 companies from companies_control_set.json
SELECTED_QUERIES = [
    "PDMA Association",
    "DermaQuest Inc",
    "Site Foundation Golf Tournament",
    "THE SOCA GROUP ORGANIZATION",
    "Shiroyama Junior High School",
    "Amedysis, Incorporated",
    "Mitsubishi M501G",
    "Linklaters CIS",
    "Volkswagen Group China",
    "The Jones Assembly"
]

def generate_report():
    print("Initializing Matcher...")
    matcher = CompanyMatcher()
    
    # Use companies.json as it is the standard for control set verification
    data_file = 'companies.json'
    # data_file = 'training_data.json' # Temporarily use smaller dataset for verification
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found, falling back to training_data.json")
        data_file = 'training_data.json'
        
    print(f"Building index from {data_file}...")
    matcher.build_index(filepath=data_file)
    
    report_lines = []
    report_lines.append("# Control Set Matching Process Report\n")
    report_lines.append(f"**Generated**: {time.asctime()}\n")
    report_lines.append(f"**Database**: `{data_file}`\n")
    report_lines.append("---\n")
    
    for query in SELECTED_QUERIES:
        print(f"Processing: {query}")
        report_lines.append(f"## Query: `{query}`\n")
        
        # Run match (Top 10 as requested)
        matches = matcher.match(query, top_k=10)
        
        if not matches:
            report_lines.append("No matches found.\n\n")
            continue

        report_lines.append("| Rank | Candidate | Score | Type | Process Description |")
        report_lines.append("|---|---|---|---|---|")
        
        for i, match in enumerate(matches, 1):
            name = match['name']
            score = match['score']
            match_type = match.get('match_type', 'hybrid')
            
            # Get explanation
            explanation = matcher.explain_match(query, name)
            
            # Construct process description
            desc = []
            
            # Basic token overlap
            q_tok = explanation.get('query_tokens', [])
            m_tok = explanation.get('match_tokens', [])
            overlap = explanation.get('overlap', [])
            
            desc.append(f"**Tokens Matched**: {len(overlap)}/{len(q_tok)} ({', '.join(str(x) for x in overlap)})<br>")
            
            # Scores
            str_sc = explanation.get('string_score', 0.0)
            sem_sc = explanation.get('semantic_score', 0.0)
            sem_sc_norm = explanation.get('normalized_semantic_score', 0.0)
            
            desc.append(f"**Components**: String={str_sc:.2f}, Semantic={sem_sc_norm:.2f} (Raw={sem_sc:.2f})<br>")
            
            # Logic flow
            if match_type == 'exact':
                desc.append("**Logic**: Exact string match found (O(1) lookup). Forced to top.")
            else:
                desc.append(f"**Logic**: Hybrid score = ({str_sc:.2f} * 0.7) + ({sem_sc_norm:.2f} * 0.3) = {score:.4f}.")
                if str_sc > sem_sc_norm:
                     desc.append("Lexical similarity dominated.")
                else:
                     desc.append("Semantic similarity dominated.")
            
            # Location (if applicable)
            if 'location_score' in match and match.get('location_score', 0) > 0:
                 desc.append(f"<br>**Location**: Boost applied (Score={match['location_score']:.2f}).")
                 
            process_text = " ".join(desc)
            
            # Escape pipes for markdown table
            safe_name = name.replace('|', '\\|')
            
            report_lines.append(f"| {i} | {safe_name} | {score:.4f} | {match_type} | {process_text} |")
        
        report_lines.append("\n")

    output_path = r'C:\Users\scott\.gemini\antigravity\brain\1debfe08-7eca-4589-9bad-93e3eb94889a\matching_process_report.md'
    # Also save locally for easy access
    local_path = 'matching_process_report.md'
    
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Report generated: {local_path}")
    
    # Copy to artifact
    try:
        import shutil
        shutil.copy(local_path, output_path)
    except Exception as e:
        print(f"Failed to copy to artifact: {e}")

if __name__ == '__main__':
    generate_report()
