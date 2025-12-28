
import sys
import os

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.utils.text_preprocessor import TextPreprocessor

def debug_fidelity(acronym, text):
    fidelity = TextPreprocessor.calculate_acronym_fidelity(acronym, text)
    print(f"Acronym: '{acronym}'")
    print(f"Text: '{text}'")
    print(f"Fidelity: {fidelity}")
    print("-" * 20)
    return fidelity

print("Debugging Acronym Fidelity Scores:")
debug_fidelity("IBM", "International Business Machines")
debug_fidelity("IBM", "IBMA/ THE BATTERY MAN")
debug_fidelity("IBM", "International Business Machines Corp")
debug_fidelity("IBM", "Imaging Business Machines")
