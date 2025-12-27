#!/usr/bin/env python3
"""
Batch testing script for the fine-tuned company names model
"""

import os
import json
from FineTuner import FineTuner

def batch_test_model():
    # Check if model directory exists
    model_dir = "company_names_model"
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory '{model_dir}' not found!")
        return
    
    print("🚀 Loading fine-tuned model...")
    
    # Initialize FineTuner with the fine-tuned model
    fine_tuner = FineTuner(
        model_name=model_dir,
        max_seq_length=2048,
        device="cpu"
    )
    
    # Load the fine-tuned model
    fine_tuner.load_model()
    
    print("✅ Model loaded successfully!")
    
    # Test prompts
    test_prompts = [
        "Company Name: ",
        "Company Name: Microsoft",
        "Company Name: American",
        "Company Name: *Bristol",
        "Company Name: #6766V2",
        "What company name starts with 'A'?",
        "Generate a company name: ",
        "Company: ",
        "Business: ",
        "Organization: "
    ]
    
    print(f"\n🧪 Testing {len(test_prompts)} prompts...")
    print("=" * 60)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}/{len(test_prompts)}")
        print(f"📤 Input: {prompt}")
        
        try:
            # Generate response
            response = fine_tuner.predict(prompt, max_new_tokens=25)
            
            print(f"🤖 Generated: {response}")
            
            # Store result
            results.append({
                "prompt": prompt,
                "response": response,
                "status": "success"
            })
            
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"❌ {error_msg}")
            
            results.append({
                "prompt": prompt,
                "response": error_msg,
                "status": "error"
            })
        
        print("-" * 40)
    
    # Save results
    output_file = "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved to: {output_file}")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(successful/len(results)*100):.1f}%")

if __name__ == "__main__":
    batch_test_model()
