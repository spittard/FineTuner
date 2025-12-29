import json
import re
import os
import sys
from collections import Counter
from typing import List, Dict, Tuple

# Optional: Using scikit-learn for the lightweight model if installed
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class EntityClassifier:
    def __init__(self):
        # 1. Event Keywords (High Precision)
        self.event_keywords = {
            'wedding', 'bar mitzvah', 'bat mitzvah', 'tournament', 'meeting', 
            'conference', 'summit', 'party', 'breakfast', 'luncheon', 'dinner', 
            'gala', 'expo', 'workshop', 'seminar', 'convention', 'symposium',
            'festival', 'celebration', 'nuptials', 'reception', 'outing'
        }
        
        # 2. Business Indicators (Safeguards)
        self.business_indicators = {
            'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'association', 
            'group', 'foundation', 'co', 'company', 'systems', 'technology', 
            'technologies', 'industries', 'associates', 'solutions', 'services',
            'llp', 'pvt', 'pllc'
        }

        # 3. Structural Patterns (Regex)
        self.wedding_pattern = re.compile(r'.+[/&].+\sWedding', re.IGNORECASE)
        self.vs_pattern = re.compile(r'.+\svs\.?\s.+', re.IGNORECASE)
        self.year_pattern = re.compile(r'\b(20\d{2}|19\d{2})\b')

        self.model = None
        self.vectorizer = None

    def classify_by_rules(self, name: str) -> Tuple[str, str]:
        """Classify using fast rule-based logic."""
        name_lower = name.lower()
        tokens = set(re.findall(r'\w+', name_lower))

        # Check for Business Indicators first (High Priority Safeguard)
        if any(indicator in tokens for indicator in self.business_indicators):
            return "ORGANIZATION", "business_indicator"

        # Check for Wedding Patterns
        if self.wedding_pattern.search(name):
            return "EVENT", "wedding_pattern"

        # Check for vs. Pattern (often sports or legal, but usually not a company name)
        if self.vs_pattern.search(name):
            return "EVENT", "vs_pattern"

        # Check for Event Keywords
        found_keywords = tokens.intersection(self.event_keywords)
        if found_keywords:
            return "EVENT", f"keyword_{list(found_keywords)[0]}"

        return "UNKNOWN", "no_rules_matched"

    def train_lightweight_model(self):
        """Trains a simple Logistic Regression on basic features if unknown."""
        if not HAS_SKLEARN:
            print("Scikit-learn not available, skipping model training.")
            return

        # Synthetic/Heuristic training data for demonstration
        # In a real scenario, this would be a small curated set.
        train_texts = [
            "Apple Inc", "Google Corp", "Microsoft Solutions", 
            "The Smith & Jones Wedding", "Annual Charity Gala 2024",
            "Regional Sales Meeting", "World Cup Tournament",
            "Goldman Sachs Group", "United Nations Foundation"
        ]
        train_labels = [0, 0, 0, 1, 1, 1, 1, 0, 0] # 0 = ORG, 1 = EVENT

        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        X = self.vectorizer.fit_transform(train_texts)
        self.model = LogisticRegression()
        self.model.fit(X, train_labels)

    def infer_type(self, name: str) -> str:
        """Infers the specific type of event."""
        name_lower = name.lower()
        if 'wedding' in name_lower or 'nuptials' in name_lower:
            return "Social/Wedding"
        if any(k in name_lower for k in ['meeting', 'conference', 'summit', 'breakfast', 'luncheon', 'dinner', 'seminar']):
            return "Business/Meeting"
        if any(k in name_lower for k in ['tournament', 'cup', 'match', 'vs']):
            return "Sports/Tournament"
        if any(k in name_lower for k in ['festival', 'gala', 'party', 'celebration']):
            return "Social/General"
        return "Miscellaneous Event"

    def process_dataset(self, file_path: str, limit: int = 100000):
        print(f"Loading dataset: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return

        results = []
        event_counts = Counter()
        
        print(f"Processing up to {limit} records...")
        for i, record in enumerate(data[:limit]):
            name = record.get("Company Name", "")
            if not name: continue

            entity_type, reason = self.classify_by_rules(name)
            
            # If rules are uncertain, we could use the model here if trained
            # For now, we focus on the rule-based plus inference for report
            
            if entity_type == "EVENT":
                event_type = self.infer_type(name)
                results.append({
                    "name": name,
                    "reason": reason,
                    "event_type": event_type,
                    "location": f"{record.get('City', '')}, {record.get('State', '')}"
                })
                event_counts[event_type] += 1

            if (i + 1) % 10000 == 0:
                print(f"Processed {i+1} records...")

        return results, event_counts

def generate_report(results, event_counts, output_file):
    print(f"Generating report: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Entity Classification Report: Organizations vs Events\n\n")
        f.write("## Summary of Event Types\n\n")
        
        for etype, count in event_counts.items():
            # Get examples for this type
            examples = [r['name'] for r in results if r['event_type'] == etype][:3]
            f.write(f"- **{etype}**: {count} found. *Examples: {', '.join(examples)}*\n")
        
        f.write("\n---\n\n")
        f.write("## Detailed List of Flagged Events (Top 100)\n\n")
        f.write("| Entity Name | Inferred Event Type | Classification Reason | Location |\n")
        f.write("|-------------|---------------------|-----------------------|----------|\n")
        
        for res in results[:100]:
            f.write(f"| {res['name']} | {res['event_type']} | {res['reason']} | {res['location']} |\n")

if __name__ == "__main__":
    classifier = EntityClassifier()
    # classifier.train_lightweight_model() # Optional for now
    
    input_json = 'e:/projects/FineTuner/FineTuner/companies_with_location.json'
    output_md = 'e:/projects/FineTuner/FineTuner/entity_classification_report.md'
    
    flagged_events, counts = classifier.process_dataset(input_json, limit=100000)
    generate_report(flagged_events, counts, output_md)
    print("Done.")
