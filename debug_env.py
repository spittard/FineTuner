
import sys
import os
print("Python is running!")
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
    print("Added src to path")
    import finetuner
    print(f"Imported finetuner from {finetuner.__file__}")
    from finetuner.core.matcher import CompanyMatcher
    print("Imported CompanyMatcher")
except Exception as e:
    print(f"Error: {e}")
