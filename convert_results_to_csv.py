#!/usr/bin/env python3
"""
Convert control set results markdown to CSV spreadsheet
"""

import re
import csv

def parse_markdown_to_csv(md_file, csv_file):
    """Parse the markdown results file and create a CSV spreadsheet"""
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by case sections (### N. Company Name)
    case_pattern = r'### (\d+)\. (.+?)\n\n\*\*Top Match\*\*: (.+?)\n\n- \*\*Score\*\*: ([\d.]+) \(([\d.]+)%\)\n- \*\*Match Type\*\*: (\w+)\n- \*\*String Score\*\*: ([\d.]+)\n- \*\*Semantic Score\*\*: ([\d.]+)'
    
    cases = re.findall(case_pattern, content)
    
    # Also extract the match tables
    table_pattern = r'### (\d+)\. .+?\n.*?\|.*?\n\|[-\|]+\n((?:\|[^\n]+\n)+)'
    tables = re.findall(table_pattern, content, re.DOTALL)
    
    # Create a dict of case_num -> table rows
    case_tables = {}
    for case_num, table_content in tables:
        rows = []
        for line in table_content.strip().split('\n'):
            if line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 4:
                    rows.append(parts)
        case_tables[case_num] = rows
    
    # Prepare CSV data
    # Header row
    headers = [
        'Case #', 'Query', 'Top Match', 'Score', 'Score %', 'Match Type', 
        'String Score', 'Semantic Score'
    ]
    
    # Add columns for top 20 matches
    for i in range(1, 21):
        headers.extend([f'Match {i} Name', f'Match {i} Score', f'Match {i} Type'])
    
    rows = []
    for case in cases:
        case_num, query, top_match, score, score_pct, match_type, string_score, semantic_score = case
        
        row = [
            case_num, query, top_match, score, score_pct, match_type,
            string_score, semantic_score
        ]
        
        # Add match details
        table_rows = case_tables.get(case_num, [])
        for i in range(20):
            if i < len(table_rows):
                rank, name, match_score, mtype = table_rows[i]
                row.extend([name, match_score, mtype])
            else:
                row.extend(['', '', ''])
        
        rows.append(row)
    
    # Write CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Created {csv_file} with {len(rows)} cases")
    return len(rows)

def create_summary_csv(md_file, csv_file):
    """Create a simpler summary CSV with just query and top 5 matches"""
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse case info
    case_pattern = r'### (\d+)\. (.+?)\n\n\*\*Top Match\*\*: (.+?)\n\n- \*\*Score\*\*: ([\d.]+) \(([\d.]+)%\)\n- \*\*Match Type\*\*: (\w+)\n- \*\*String Score\*\*: ([\d.]+)\n- \*\*Semantic Score\*\*: ([\d.]+)'
    cases = re.findall(case_pattern, content)
    
    # Parse tables
    table_pattern = r'### (\d+)\. .+?\n.*?\|.*?\n\|[-\|]+\n((?:\|[^\n]+\n)+)'
    tables = re.findall(table_pattern, content, re.DOTALL)
    
    case_tables = {}
    for case_num, table_content in tables:
        rows = []
        for line in table_content.strip().split('\n'):
            if line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 4:
                    rows.append(parts)
        case_tables[case_num] = rows
    
    # Create summary format
    headers = [
        'Case', 'Query', 
        'Top Match Score', 'Match Type', 'String Score', 'Semantic Score',
        'Match 1', 'Score 1',
        'Match 2', 'Score 2', 
        'Match 3', 'Score 3',
        'Match 4', 'Score 4',
        'Match 5', 'Score 5'
    ]
    
    rows = []
    for case in cases:
        case_num, query, top_match, score, score_pct, match_type, string_score, semantic_score = case
        
        row = [case_num, query, score_pct + '%', match_type, string_score, semantic_score]
        
        table_rows = case_tables.get(case_num, [])
        for i in range(5):
            if i < len(table_rows):
                rank, name, match_score, mtype = table_rows[i]
                row.extend([name, match_score])
            else:
                row.extend(['', ''])
        
        rows.append(row)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Created {csv_file} with {len(rows)} cases (summary format)")
    return len(rows)

if __name__ == '__main__':
    md_file = 'companies_control_set_results.md'
    
    # Full CSV with all 20 matches
    create_summary_csv(md_file, 'control_set_results_summary.csv')
    
    # Detailed CSV with all 20 matches  
    parse_markdown_to_csv(md_file, 'control_set_results_full.csv')
    
    print("\nDone! Created:")
    print("  - control_set_results_summary.csv (top 5 matches)")
    print("  - control_set_results_full.csv (all 20 matches)")

