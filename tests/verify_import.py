import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    print("Attempting to import finetuner.core.matcher...")
    from finetuner.core.matcher import CompanyMatcher
    print("SUCCESS: CompanyMatcher imported.")
    
    print("Attempting to import finetuner.utils.text_preprocessor...")
    from finetuner.utils.text_preprocessor import TextPreprocessor
    print("SUCCESS: TextPreprocessor imported.")
    
    # Verify TextPreprocessor is used
    print("Verifying TextPreprocessor usage...")
    weight = TextPreprocessor.get_term_weight('Center')
    print(f"TextPreprocessor.get_term_weight('Center') = {weight}")
    
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
