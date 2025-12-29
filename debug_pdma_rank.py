
import os
import sys
import json

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.web.services.search_service import SearchService

def debug_pdma_ranking():
    service = SearchService()
    if not service.load_company_data(model_name='paraphrase-MiniLM-L3-v2', filename='companies_with_location.json'):
        print("❌ Failed to load company data")
        return
    
    query = "PDMA Association"
    target_name = "Association Headquarters-PDMA"
    
    print(f"\n--- Debugging Ranking for query: '{query}' ---")
    print(f"Targeting: '{target_name}'")
    
    # Get more matches to find our target
    matches = service.search(query, top_k=50)
    
    found = False
    print(f"\nTop 50 matches:")
    for i, m in enumerate(matches, 1):
        name = m['company_name']
        score = m['likeness_percent']
        if i <= 10:
            print(f"{i}. {name} (Score: {score:.2f}%)")
        
        if name.lower().strip() == target_name.lower().strip():
            print(f"\n🎯 FOUND TARGET at Rank {i}!")
            print(f"Name: {name}")
            print(f"Score: {score:.2f}%")
            print(f"Details: {m.get('explanation_details', {})}")
            found = True
            if i > 10:
                print(f"Ranked below #10 because score {score:.2f}% < {matches[9]['likeness_percent']:.2f}%")
    
    if not found:
        print(f"\n❌ Target '{target_name}' not found in top 50 matches.")

if __name__ == "__main__":
    debug_pdma_ranking()
