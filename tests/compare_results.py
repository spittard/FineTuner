
import re
import sys

def parse_markdown_results(filepath):
    """
    Parses the markdown results file to extract rankings for each case.
    Returns a dict: {case_query: {rank: {name, score}}}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    results = {}
    
    # Split by "### " to get each case
    cases = content.split('### ')[1:]
    
    for case in cases:
        lines = case.split('\n')
        # Line 0 usually has "1. Query Name"
        header_match = re.match(r'^\d+\.\s+(.+)$', lines[0].strip())
        if not header_match:
            continue
            
        query = header_match.group(1).strip()
        
        # Find the markdown table
        table_start = -1
        for i, line in enumerate(lines):
            if '| Rank | Company Name |' in line:
                table_start = i + 2 # Skip header and separator
                break
        
        if table_start == -1:
            continue
            
        case_ranks = {}
        for line in lines[table_start:]:
            if not line.strip().startswith('|'):
                break
                
            parts = [p.strip() for p in line.split('|')]
            # | 1 | Name | Score | ... |
            # parts[0] is empty, parts[1] is rank, parts[2] is name, parts[3] is score
            if len(parts) >= 4:
                try:
                    rank = int(parts[1])
                    name = parts[2]
                    score_str = parts[3].replace('%', '')
                    score = float(score_str)
                    case_ranks[rank] = {'name': name, 'score': score}
                except ValueError:
                    continue
        
        results[query] = case_ranks
        
    return results

def compare_results(old_file, new_file):
    print(f"Comparing:")
    print(f"  OLD: {old_file}")
    print(f"  NEW: {new_file}")
    print("-" * 60)
    
    old_data = parse_markdown_results(old_file)
    new_data = parse_markdown_results(new_file)
    
    discrepancies = 0
    total_checks = 0
    
    queries = sorted(list(set(old_data.keys()) | set(new_data.keys())))
    
    for query in queries:
        if query not in old_data:
            print(f"[NEW QUERY] '{query}' found in new results but not old.")
            continue
        if query not in new_data:
            print(f"[MISSING QUERY] '{query}' missing from new results.")
            continue
            
        old_ranks = old_data[query]
        new_ranks = new_data[query]
        
        # Check ranks 2-20 (or whatever is available)
        max_rank = max(max(old_ranks.keys(), default=0), max(new_ranks.keys(), default=0))
        
        for rank in range(2, max_rank + 1):
            if rank not in old_ranks and rank not in new_ranks:
                continue
                
            total_checks += 1
            
            old_item = old_ranks.get(rank, {'name': 'N/A', 'score': 0})
            new_item = new_ranks.get(rank, {'name': 'N/A', 'score': 0})
            
            # Compare Name and Score (allow small float diff)
            name_match = old_item['name'] == new_item['name']
            score_diff = abs(old_item['score'] - new_item['score'])
            score_match = score_diff < 0.05 # 0.05% tolerance
            
            if not name_match or not score_match:
                discrepancies += 1
                print(f"DIFF: '{query}' at Rank {rank}")
                if not name_match:
                    print(f"  Name: '{old_item['name']}' -> '{new_item['name']}'")
                if not score_match:
                    print(f"  Score: {old_item['score']}% -> {new_item['score']}% (Diff: {score_diff:.4f})")

    print("-" * 60)
    if discrepancies == 0:
        print(f"SUCCESS: No discrepancies found in secondary matches (Ranks 2+) across {len(queries)} cases.")
    else:
        print(f"FOUND {discrepancies} discrepancies in secondary matches.")

if __name__ == "__main__":
    compare_results('companies_control_set_results_OLD.md', 'companies_control_set_results.md')
