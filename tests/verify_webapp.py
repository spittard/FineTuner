
import sys
import os
import io

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    print("Importing app...")
    from finetuner.web.app import app
    print("   [OK] App imported")
    
    print("Importing RationaleService...")
    from finetuner.web.services.rationale_service import RationaleService
    print("   [OK] RationaleService imported")
    
    print("Testing RationaleService...")
    # Test rationale generation with dummy data
    query = "test company"
    match_name = "Test Company Inc"
    explanation = {'query_tokens': [], 'match_tokens': [], 'overlap': [], 'overlap_score': 0.8}
    score = 0.95
    
    rationale = RationaleService.generate_match_rationale(query, match_name, explanation, score)
    print(f"   [OK] Rationale generated: {rationale[:50]}...")
    
    print("\nVerification SUCCESS!")
    sys.exit(0)
except Exception as e:
    print(f"\nVerification FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
