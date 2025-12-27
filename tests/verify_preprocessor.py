import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    print("Importing TextPreprocessor...")
    from finetuner.utils.text_preprocessor import TextPreprocessor
    print("SUCCESS: Imported.")
    
    print("Testing clean_company_name...")
    cleaned = TextPreprocessor.clean_company_name("Apple Inc.")
    print(f"cleaned: '{cleaned}'")
    assert cleaned == "apple"
    
    print("Testing get_term_weight...")
    w = TextPreprocessor.get_term_weight("Center")
    print(f"weight of 'Center': {w}")
    assert w == 0.3
    
    print("ALL TESTS PASSED")
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
