
import sys
import os
import io

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock flask stuff to avoid import errors if not in valid context
import flask
from unittest.mock import MagicMock

try:
    print("Importing SearchService...")
    from finetuner.web.services.search_service import SearchService
    print("   [OK] SearchService imported")
    
    service = SearchService()
    print("   [OK] SearchService instantiated (Singleton check: " + str(service is SearchService()) + ")")
    
    # Check status
    status = service.get_status()
    print(f"   [OK] Initial Status: {status['status']}")
    
    print("\nImporting app...")
    from finetuner.web.app import app
    print("   [OK] App imported")
    
    print("\nVerification SUCCESS!")
    sys.exit(0)
except Exception as e:
    print(f"\nVerification FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
