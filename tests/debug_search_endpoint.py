import requests
import json

def debug_search():
    url = "http://127.0.0.1:5000/search"
    payload = {"query": "aba", "top_k": 10}
    
    print(f"Sending POST request to {url} with payload {payload}...")
    try:
        response = requests.post(url, data=payload)
        
        print(f"Status Code: {response.status_code}")
        
        try:
            data = response.json()
            # Save to file for inspection
            with open('debug_response.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print("Saved response to debug_response.json")
            
            if not data.get('success'):
                print("\n[!] Response 'success' field is falsy or missing!")
                
            results = data.get('results', [])
            print(f"\nReceived {len(results)} results.")
            
        except json.JSONDecodeError:
            print("\n[!] Failed to decode JSON. Raw text:")
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_search()
