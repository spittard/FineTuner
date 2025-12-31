import ollama
import sys

def verify_ollama():
    try:
        # Check connection
        models_resp = ollama.list()
        print(f"Ollama Response: {models_resp}")
        
        # Access based on observed structure
        models = models_resp.models if hasattr(models_resp, 'models') else models_resp.get('models', [])
        
        model_names = [m.model if hasattr(m, 'model') else m.get('name', '') for m in models]
        print(f"Available models: {model_names}")
        
        target_model = 'llama3.2:3b'
        if any(target_model in name for name in model_names):
            print(f"SUCCESS: {target_model} is available.")
            
            # Simple test chat
            print("Running test prompt...")
            response = ollama.chat(model=target_model, messages=[
                {'role': 'user', 'content': 'Say "Ollama is ready" if you can read this.'}
            ])
            print(f"Response: {response['message']['content'].strip()}")
        else:
            print(f"FAILURE: {target_model} not found in {model_names}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    verify_ollama()
