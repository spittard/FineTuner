from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load the "Brain" (The Ultra Model)
# We use the same model as the main application to ensure consistency
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# 2. Define the "Probe Concepts" (The questions we want to ask the vector)
# These are the yardsticks we measure your company against.
concepts = {
    "Geography": ["Pennsylvania", "Florida", "New York", "Texas", "London", "Canada"],
    "Industry":  ["Medical", "Technology", "Construction", "Legal", "Automotive", "Food"],
    "Structure": ["Corporate", "Non-Profit", "Government", "Small Business"]
}

def scan_company(company_name, location_context=""):
    print(f"\n🔬 SCANNING: '{company_name}' {f'(Context: {location_context})' if location_context else ''}")
    print("-" * 50)
    
    # Bake the input just like your app does
    full_text = f"{company_name} {location_context}".strip()
    company_vector = model.encode(full_text, convert_to_tensor=True)

    # Check against every concept
    for category, anchors in concepts.items():
        print(f"[{category.upper()}]")
        
        # Encode all anchors in this category at once
        anchor_vectors = model.encode(anchors, convert_to_tensor=True)
        
        # Calculate similarity (How close is the company to these words?)
        scores = util.cos_sim(company_vector, anchor_vectors)[0]
        
        # Print results
        results = []
        for i, anchor in enumerate(anchors):
            score_percent = scores[i].item() * 100
            results.append((anchor, score_percent))
        
        # Sort by strongest match
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Display top matches
        for anchor, score in results:
            bar_len = int(score / 5)
            bar = "█" * bar_len
            # Highlight strong matches
            if score > 35: 
                print(f"  ✅ {anchor:<12} {score:.1f}%  {bar}")
            else:
                print(f"     {anchor:<12} {score:.1f}%  {bar}")
        print("")

if __name__ == "__main__":
    print("🧠 Loading Semantic Scanner (Concept Probing)...")
    
    # --- RUN THE TESTS ---

    # Test 1: The "PA" Problem (Does it contain 'Pennsylvania'?)
    scan_company("Smith PA")

    # Test 2: The Location Baking (Does 'London' change the meaning?)
    scan_company("Western University", "London Ontario")

    # Test 3: Industry Check (Does it know what 'General Motors' does?)
    scan_company("General Motors")
