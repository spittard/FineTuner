import json
import os

def add_acronyms():
    control_file = 'companies_control_set.json'
    
    # New acronym and location test cases
    new_cases = [
        {"Company Name": "ABA"},  # Expect: American Bar Association
        {"Company Name": "PDMA"}, # Expect: Product Development and Management Association
        {"Company Name": "IBM"},  # Expect: International Business Machines
        {"Company Name": "GE"},   # Expect: General Electric
        {"Company Name": "American Bar Association", "City": "Chicago", "State": "IL"},
        {"Company Name": "IBM", "City": "Armonk", "State": "NY"}
    ]
    
    if os.path.exists(control_file):
        with open(control_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check specific names
        existing_names = {item.get("Company Name", "") for item in data}
        
        added = 0
        for case in new_cases:
            if case["Company Name"] not in existing_names:
                data.append(case)
                added += 1
                print(f"Added: {case['Company Name']}")
            else:
                print(f"Skipped (Exists): {case['Company Name']}")
                
        if added > 0:
            with open(control_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"Successfully added {added} acronyms to {control_file}")
        else:
            print("No new acronyms added.")
            
    else:
        print(f"Error: {control_file} not found.")

if __name__ == "__main__":
    add_acronyms()
