# Company Matching Control Set Report (Location-Aware)

**Generated:** 2025-12-30 06:38:26

---

## Matching Scenarios Handled

This system is designed to handle the following real-world company matching challenges:

### 1. **Exact Matches**
Perfect character-for-character matching.
- `"IBM"` → `"IBM"` (100%)
- `"Microsoft"` → `"Microsoft"` (100%)
- `"Apple Inc."` → `"Apple Inc."` (100%)
- `"Google"` → `"Google"` (100%)
- `"Amazon.com"` → `"Amazon.com"` (100%)

### 2. **Acronym Expansions**
Matching acronyms to their full company names or vice-versa.
- `"IBM"` → `"International Business Machines"` (Strong expansion)
- `"AWS"` → `"Amazon Web Services"` (Strong expansion)
- `"GE"` → `"General Electric"` (Strong expansion)
- `"AT&T"` → `"American Telephone and Telegraph"` (Strong expansion)
- `"FedEx"` → `"Federal Express"` (Strong expansion)

### 3. **Location-Aware Matching (NEW)**
Using city/state context to resolve ambiguity between identical or similar names.
- `"Acme"` (Chicago) → `"Acme Corp"` (Chicago, IL) vs (Miami, FL)
- `"Northwestern"` (Evanston) → `"Northwestern University"` (Evanston, IL) vs `"Northwestern Mutual"` (Milwaukee, WI)
- `"Pizza Hut"` (London, KY) → `"Pizza Hut"` (London, KY) vs `"Pizza Hut"` (London, UK)
- `"Springfield Power"` (Springfield, IL) → Resolved to Illinois entity over Massachusetts
- `"Regency Hotel"` (Paris, TX) → Resolved to Texas entity over France or Nevada

### 4. **Popularity/Frequency Bias (NEW)**
Using occurrence counts to break ties, prioritizing major entities over obscure ones.
- `"McDonalds"` → Global chain (5,000+ records) vs `"McDonalds Hardware"` (1 record)
- `"Starbucks"` → National brand vs `"Starbucks Coffee Roasters"` (local shop)
- `"Walmart"` → Major retailer vs `"Walmarts Antiques"` (single entry)
- `"Chase"` → `"JP Morgan Chase"` (Bank) vs `"Chase & Sons Trucking"` 
- `"Ford"` → `"Ford Motor Company"` vs `"Ford's Diner"`

---

## Advanced Scoring Formula

```
Base Score = (String Similarity × 70%) + (Semantic Similarity × 30%)
Fidelity Boost = Acronym Fidelity × 15%
Location Boost = Location Score × 5% (Post-Inference)
Final Score = Base Score + Fidelity Boost + Location Boost + Popularity Boost
```

---

## Control Set Results

## 1. PDMA Association

**Query:** `PDMA Association` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Association Headquarters-PDMA (Mount Laurel, NJ) • **Score:** 95.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity (Normalized) | 0.4491 | 30% | 0.1347 |
| Semantic Similarity (Raw) | 2.8192 | - | - |
| **Base Score** | **0.7393** | - | - |
| **FINAL SCORE** | **0.9558** | - | **95.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9558
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (FAIR):** Some meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• association<br><br><b>Your Search Also Includes:</b><br>• pdma<br><br><b>Company Name Also Includes:</b><br>• headquarters-pdma<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.4491 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.58%)</span>. Frequency boost applied (1 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#1</b> | <b>Association Headquarters-PDMA</b> (Mount Laurel, NJ) | Score: <b>95.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4491</td><td>30%</td><td>0.1347</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9560</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• association<br><br><b>Your Search Also Includes:</b><br>• pdma<br><br><b>Company Name Also Includes:</b><br>• headquarters-pdma<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.4491 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.58%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#2</b> | <b>PDMA Alliance</b> (York, SC) | Score: <b>90.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5445</td><td>30%</td><td>0.1633</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9090</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.5445 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#3</b> | <b>PDMA ALLIANCE</b> | Score: <b>90.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8098</td><td>30%</td><td>0.2430</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9050</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.8098 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.55%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#4</b> | <b>PDMA ALLIANCE</b> (, FL) | Score: <b>90.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6902</td><td>30%</td><td>0.2071</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9050</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.6902 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.55%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#5</b> | <b>PDMA ALLIANCE</b> (CHARLOTTE, NC) | Score: <b>90.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5560</td><td>30%</td><td>0.1668</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9050</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.5560 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.55%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#6</b> | <b>PDMA Alliance Inc.</b> (Charlotte, NC) | Score: <b>90.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4832</td><td>30%</td><td>0.1450</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9050</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance, inc.<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.4832 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.55%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#7</b> | <b>PDMA Alliance</b> (Valhalla, NY) | Score: <b>90.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4751</td><td>30%</td><td>0.1425</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9050</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.4751 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.55%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#8</b> | <b>PDMA Alliance</b> (Valballa, NY) | Score: <b>90.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4579</td><td>30%</td><td>0.1374</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9050</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.4579 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.55%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#9</b> | <b>PDMA</b> | Score: <b>80.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7200</td><td>70%</td><td>0.5040</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8090</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7200 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.49%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#10</b> | <b>PDMA inc</b> | Score: <b>78.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7200</td><td>70%</td><td>0.5040</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9129</td><td>30%</td><td>0.2739</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7830</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• inc<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7200 (Weight: 70%)<br>• Semantic Similarity: 0.9129 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.47%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>

---

## 2. Nicolas/Sanchez Wedding

**Query:** `Nicolas/Sanchez Wedding` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Sanchez/Justin Wedding • **Score:** 80.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9159 | 30% | 0.2748 |
| Semantic Similarity (Raw) | 4.6285 | - | - |
| **Base Score** | **0.7954** | - | - |
| **FINAL SCORE** | **0.8002** | - | **80.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8002
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/justin<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9159 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.48%)</span>. Frequency boost applied (1 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#1</b> | <b>Sanchez/Justin Wedding</b> | Score: <b>80.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9159</td><td>30%</td><td>0.2748</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/justin<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9159 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.48%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#2</b> | <b>Sanchez Wedding</b> (Marietta, GA) | Score: <b>79.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7875</td><td>70%</td><td>0.5512</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7885</td><td>30%</td><td>0.2365</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7930</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7875 (Weight: 70%)<br>• Semantic Similarity: 0.7885 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.48%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#3</b> | <b>Sanchez/Ramirez Wedding</b> (Miami, FL) | Score: <b>78.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8753</td><td>30%</td><td>0.2626</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7880</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/ramirez<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8753 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.48%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#4</b> | <b>Garcia Sanchez Wedding</b> (Miami, FL) | Score: <b>77.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7780</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• garcia, sanchez<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.47%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#5</b> | <b>Gibson Sanchez Wedding</b> | Score: <b>77.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9749</td><td>30%</td><td>0.2925</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7700</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• gibson, sanchez<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 0.9749 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.47%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#6</b> | <b>Sanchez/Puerto Wedding</b> (Miami Beach, Fl) | Score: <b>76.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8113</td><td>30%</td><td>0.2434</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7690</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/puerto<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8113 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.46%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#7</b> | <b>Sanchez/Cohen Wedding</b> (Hollywood, FL) | Score: <b>76.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7882</td><td>30%</td><td>0.2365</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7620</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/cohen<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7882 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.46%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#8</b> | <b>Sanchez/Naranjo Wedding</b> (Miami, FL) | Score: <b>75.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7785</td><td>30%</td><td>0.2336</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7590</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/naranjo<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7785 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.46%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#9</b> | <b>Sanchez/Fuentes Wedding</b> (Dallas, TX) | Score: <b>75.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7780</td><td>30%</td><td>0.2334</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7590</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/fuentes<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7780 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.46%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'><b>#10</b> | <b>Sanchez/Ramos Wedding</b> (Anaheim, CA) | Score: <b>75.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7614</td><td>30%</td><td>0.2284</td></tr>
        <tr><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7540</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/ramos<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7614 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.46%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>

---

