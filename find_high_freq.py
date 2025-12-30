import json

data = json.load(open('companies_with_location.json', encoding='utf-8'))
high = sorted([c for c in data if c['Count'] > 10], key=lambda x: -x['Count'])[:20]
for c in high:
    print(f"{c['Count']}: {c['Company Name']} ({c['City']}, {c['State']})")
