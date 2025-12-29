import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.web.services.search_service import SearchService
import json

# Quick test script to generate sample output with new format
service = SearchService()
print("Loading 1M sample...")
service.load_company_data(model_name='paraphrase-MiniLM-L3-v2', filename='companies_sample_1m.json')

# Test queries
test_queries = [
    {"Company Name": "Hartford Hospital School of Nursing", "City": "Hartford", "State": "CT"},
    {"Company Name": "Chicago South Swim Club", "City": "Chicago", "State": "IL"},
    {"Company Name": "IBM"},
]

# Import the formatting function
from tests.generate_full_control_report import format_company_result

print("\n" + "="*70)
print("SAMPLE OUTPUT - NEW COMPACT FORMAT")
print("="*70 + "\n")

for i, query_entry in enumerate(test_queries, 1):
    company = query_entry.get('Company Name', '')
    city = query_entry.get('City', None)
    state = query_entry.get('State', None)
    
    print(f"Processing {i}/3: {company}")
    matches = service.search(company, top_k=10, city=city, state=state)
    
    result_data = {
        'query': company,
        'query_city': city or '',
        'query_state': state or '',
        'matches': matches
    }
    
    # Generate formatted output
    output = format_company_result(company, result_data, i)
    print(output)

print("="*70)
print("SAMPLE GENERATION COMPLETE")
print("="*70)
