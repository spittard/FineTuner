import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finetuner.utils.text_preprocessor import TextPreprocessor

def test_ibm_ranking_fast():
    # We test the scoring logic on strings using the new fidelity method
    names = [
        "International Business Machines",
        "IBMA/ THE BATTERY MAN", 
        "Imaging Business Machines", 
        "International Business Models"
    ]
    acronym = "IBM"
    
    print(f"Testing ranking for acronym: {acronym}")
    results = []
    for name in names:
        fidelity = TextPreprocessor.calculate_acronym_fidelity(acronym, name)
        
        # Simulate final score calculation from matcher.py:
        # final_score_ac = 0.85 + (fidelity * 0.10) + (sem_score_ac * 0.05)
        
        # Scenario A: Real expansion has good semantic link (0.8+)
        # Scenario B: IBMA has high semantic link (0.9) because "IBM" is a literal prefix
        
        sem_score = 0.8
        if "BATTERY" in name: sem_score = 0.95 # IBMA usually gets high semantic boost
        if "International Business Machines" == name: sem_score = 0.9 # Very high
            
        score = 0.85 + (fidelity * 0.10) + (sem_score * 0.05)
        results.append((name, fidelity, sem_score, score))
    
    results.sort(key=lambda x: x[3], reverse=True)
    
    print("-" * 60)
    for name, fidelity, sem, score in results:
        print(f"Score: {score:.4f} | Fidelity: {fidelity:.2f} | Sem: {sem:.2f} | Name: {name}")
    print("-" * 60)

    if results[0][0] == "International Business Machines":
        print("SUCCESS: International Business Machines is now #1!")
    else:
        print(f"FAILURE: {results[0][0]} is #1")

if __name__ == "__main__":
    test_ibm_ranking_fast()
