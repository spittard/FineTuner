#!/usr/bin/env python3
"""
Interactive testing script for the fine-tuned company names model
"""

import os
import sys
from FineTuner import FineTuner

def test_model():
    # Check if model directory exists
    model_dir = "company_names_model"
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory '{model_dir}' not found!")
        print("Please run the training first:")
        print("python FineTuner.py --mode train --dataset training_data.json --epochs 3 --batch-size 2 --lr 5e-5 --output-dir company_names_model")
        return
    
    print("🚀 Loading fine-tuned model...")
    
    # Initialize FineTuner with the fine-tuned model
    fine_tuner = FineTuner(
        model_name=model_dir,  # Use local fine-tuned model
        max_seq_length=2048,
        device="cpu"
    )
    
    # Load the fine-tuned model
    fine_tuner.load_model()
    
    print("✅ Model loaded successfully!")
    print("\n🧪 Interactive Testing Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            prompt = input("\n📝 Enter your prompt (or 'quit' to exit): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                print("⚠️  Please enter a prompt")
                continue
            
            # Get number of tokens to generate
            try:
                max_tokens = input("🔢 Max tokens to generate (default 20): ").strip()
                max_tokens = int(max_tokens) if max_tokens else 20
            except ValueError:
                max_tokens = 20
            
            print(f"\n🔄 Generating response...")
            print(f"📤 Input: {prompt}")
            print(f"📊 Max tokens: {max_tokens}")
            print("-" * 40)
            
            # Generate response
            response = fine_tuner.predict(prompt, max_new_tokens=max_tokens)
            
            print(f"🤖 Generated: {response}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'quit' to exit")

if __name__ == "__main__":
    test_model()
