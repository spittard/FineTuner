
import os
import sys
import json

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.web.services.search_service import SearchService

def debug_search():
    service = SearchService()
    # Explicitly load the location-aware dataset
    if not service.load_company_data(model_name='paraphrase-MiniLM-L3-v2', filename='companies_with_location.json'):
        print("❌ Failed to load company data")
        return
    
    query = "Next Level Events"
    print(f"\n--- Debugging Search for: '{query}' ---")
    
    # We want to see what CompanyMatcher returns BEFORE the report script filters it
    matches = service.search(query, top_k=10)
    
    print(f"\nFound {len(matches)} matches in SearchService:")
    for i, m in enumerate(matches, 1):
        loc = f" ({m.get('city', 'N/A')}, {m.get('state', 'N/A')})"
        print(f"{i}. {m['company_name']}{loc} [ID: {m.get('id', 'N/A')}] (Score: {m['likeness_percent']}%)")

if __name__ == "__main__":
    debug_search()
