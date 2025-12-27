
import requests
import json

url = 'http://localhost:5000/search'
data = {
    'query': 'School',
    'top_k': 10
}

try:
    print(f"Sending POST request to {url} with data: {data}")
    response = requests.post(url, data=data)
    
    print(f"Response Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except:
        print("Response Text:")
        print(response.text)
        
except Exception as e:
    print(f"Request failed: {e}")
