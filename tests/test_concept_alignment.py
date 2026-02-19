from finetuner.core.matcher import CompanyMatcher
import os

def test_alignment():
    print("Testing Concept Alignment Scoring...")
    matcher = CompanyMatcher()
    
    # We'll mock the index if needed, but let's assume we can add a few companies
    matcher.original_company_names = ["Apple Inc.", "Apple Orchard Farm", "IBM", "International Business Machines"]
    matcher.company_names = matcher.original_company_names # Preprocessed
    
    # Formal index build
    matcher.build_index(["Apple Inc.", "Apple Orchard Farm", "IBM", "International Business Machines"])
    matcher.has_location_data = False
    
    # Query: Apple (looking for tech)
    query = "Apple"
    print(f"\nQUERY: '{query}'")
    results = matcher.match_with_location(query, top_k=5)
    
    for i, res in enumerate(results):
        print(f"{i+1}. {res['name']}")
        print(f"   Final Score: {res['score']:.4f}")
        print(f"   String Score: {res['string_score']:.4f}")
        print(f"   Concept Align: {res['concept_alignment']:.4f}")
        print(f"   Match Type: {res['match_type']}")
        print("")

if __name__ == "__main__":
    test_alignment()
