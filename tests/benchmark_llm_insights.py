from finetuner.web.services.rationale_service import RationaleService
import json

def test_semantic_enrichment():
    test_cases = [
        ("IBM", "International Business Machines"),
        ("Kehilat Ariel Synagogue", "Kehilat Ariel Synagogue (San Diego, CA)"),
        ("Pizza Hut", "Pizza Hut (London)"),
        ("Apple", "Fruit Market Inc")
    ]
    
    print(f"{'Query':<30} | {'Company':<40} | {'AI Insight'}")
    print("-" * 110)
    
    for query, company in test_cases:
        # Mock explanation for _append_location_frequency_analysis
        explanation = {
            'normalized_semantic_score': 0.85,
            'location_boost': 0.1,
            'match_city': 'New York',
            'count': 5
        }
        
        # We target the specific part of the rationale that contains the AI Insight
        rationale = RationaleService._append_location_frequency_analysis("", explanation, "New York", query, company)
        
        # Extract AI Insight
        import re
        insight_match = re.search(r"<b>AI Insight:</b> (.*?)<br>", rationale)
        insight = insight_match.group(1) if insight_match else "N/A"
        
        print(f"{query:<30} | {company:<40} | {insight}")

if __name__ == "__main__":
    test_semantic_enrichment()
