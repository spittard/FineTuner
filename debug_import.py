
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from finetuner.web.services.rationale_service import RationaleService

print("Loaded RationaleService from:", sys.modules['finetuner.web.services.rationale_service'].__file__)

query = "Kehilat Ariel Synagogue"
company = "Kehilat Ariel Synagogue (San Diego, CA)"
context = RationaleService.analyze_business_context(query, company)
print(f"Context for '{query}' vs '{company}':")
print(context)

# Check specifically for non-profit keywords
keywords = ['association', 'foundation', 'club', 'society', 'coalition', 'initiative', 'center', 'charity']
print("\nChecking non-profit keywords in query:")
for k in keywords:
    if k in query.lower():
        print(f"MATCH: {k}")

print("\nChecking non-profit keywords in company:")
for k in keywords:
    if k in company.lower():
        print(f"MATCH: {k}")
