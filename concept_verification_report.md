# Company Matching Control Set Report (Location-Aware)

**Generated:** 2025-12-31 16:53:46

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
Base Score = (String Sim × 50%) + (Semantic Sim × 25%) + (Concept Align × 25%)
Fidelity Boost = Acronym Fidelity × 15% (Dynamic)
Location Boost = Location Score × 5% (Post-Inference)
Final Score = Base Score + Fidelity Boost + Location Boost + Popularity Boost
```

---

## Control Set Results

## 1. NIH (Bethesda, MD)

**Query:** `NIH` • **Location:** Bethesda, MD • **Self-Match:** ✅ Found & Filtered

**Top Match:** NIH (Bedthesa, MD) • **Score:** 104.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 50% | 0.5000 |
| Semantic Similarity (Normalized) | 0.4259 | 25% | 0.1065 |
| Concept Alignment (Scanner) | 0.0000 | 25% | 0.0000 |
| Semantic Similarity (Raw) | 2.8778 | - | - |
| **Base Score (Name)** | **100.0000** | - | - |
| **FINAL SCORE** | **1.0462** | - | **104.6%** |

### Score Calculation Formula

```
Base Score = (String Sim × 0.50) + (Semantic Sim × 0.25) + (Concept Alignment × 0.25)
Location Boost = Location Score × 0.05 = 92.5000 × 0.05 = 4.6250

Final Score = Base Score + Location Boost = 1.0462
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (FAIR):** Some meaning-based connection
- **Location Match (EXCELLENT):** 92.50 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.43<br>• <b>Concept Match:</b> 🟡 0.65<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+4.63%)</span>. The record's location (Bedthesa) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Industry: ✅ Medical 31.2%<br>• Insight: The model detects a strong 'Medical' influence in the company's semantic vector.<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>NIH</b> (Bedthesa, MD) | Score: <b>104.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4259</td><td>25%</td><td>0.1065</td></tr>
        <tr><td>Concept Alignment</td><td>0.6488</td><td>25%</td><td>0.1622</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.1813</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0463</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0460</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.43<br>• <b>Concept Match:</b> 🟡 0.65<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+4.63%)</span>. The record's location (Bedthesa) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Industry: ✅ Medical 31.2%<br>• Insight: The model detects a strong 'Medical' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>NIH</b> (Rockville, MD) | Score: <b>103.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4200</td><td>25%</td><td>0.1050</td></tr>
        <tr><td>Concept Alignment</td><td>0.8458</td><td>25%</td><td>0.2114</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.1335</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0135</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0340</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.42<br>• <b>Concept Match:</b> 🟡 0.85<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Rockville) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.35%)</span>. This is a high-frequency record (46 occurrences), suggesting it is a well-known entity.<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Pennsylvania 32.8%<br>• Insight: The model detects a strong 'Pennsylvania' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>NIH</b> (Bethesda, ) | Score: <b>103.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5855</td><td>25%</td><td>0.1464</td></tr>
        <tr><td>Concept Alignment</td><td>0.8706</td><td>25%</td><td>0.2177</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0860</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0300</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0300</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.59<br>• <b>Concept Match:</b> 🟢 0.87<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Bethesda) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>NIH</b> (Upper Marlboro, MD) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4276</td><td>25%</td><td>0.1069</td></tr>
        <tr><td>Concept Alignment</td><td>0.7754</td><td>25%</td><td>0.1938</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.1493</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.43<br>• <b>Concept Match:</b> 🟡 0.78<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Upper Marlboro) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br><br><b>Concept Analysis:</b><br>• Industry: ✅ Medical 26.2%<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>NIH</b> (Gaithersburg, MD) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>25%</td><td>0.2500</td></tr>
        <tr><td>Concept Alignment</td><td>0.0000</td><td>25%</td><td>0.0000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Gaithersburg) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>NIH</b> (Silver Springs, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.3997</td><td>25%</td><td>0.0999</td></tr>
        <tr><td>Concept Alignment</td><td>0.8540</td><td>25%</td><td>0.2135</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.1366</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.40<br>• <b>Concept Match:</b> 🟢 0.85<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Silver Springs) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>NIH</b> (Silver Spring, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4303</td><td>25%</td><td>0.1076</td></tr>
        <tr><td>Concept Alignment</td><td>0.8468</td><td>25%</td><td>0.2117</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.1307</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.43<br>• <b>Concept Match:</b> 🟡 0.85<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Silver Spring) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>NIH</b> (Laurel, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4240</td><td>25%</td><td>0.1060</td></tr>
        <tr><td>Concept Alignment</td><td>0.8466</td><td>25%</td><td>0.2117</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.1323</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.42<br>• <b>Concept Match:</b> 🟡 0.85<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Laurel) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>NIH</b> (National Institute Of Health, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4527</td><td>25%</td><td>0.1132</td></tr>
        <tr><td>Concept Alignment</td><td>0.7178</td><td>25%</td><td>0.1795</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.1574</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🔴 0.45<br>• <b>Concept Match:</b> 🟡 0.72<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (National Institute Of Health) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Pennsylvania 25.2%<br>• Industry: ✅ Medical 41.2%<br>• Insight: The model detects a strong 'Medical' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>NIH</b> (Balitmore, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>25%</td><td>0.2500</td></tr>
        <tr><td>Concept Alignment</td><td>0.0000</td><td>25%</td><td>0.0000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Balitmore) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 2. Ohio University (Athens, OH)

**Query:** `Ohio University` • **Location:** Athens, OH • **Self-Match:** ✅ Found & Filtered

**Top Match:** Ohio University (Athens, GA) • **Score:** 103.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 50% | 0.5000 |
| Semantic Similarity (Normalized) | 0.5914 | 25% | 0.1479 |
| Concept Alignment (Scanner) | 0.0000 | 25% | 0.0000 |
| Semantic Similarity (Raw) | 4.2003 | - | - |
| **Base Score (Name)** | **100.0000** | - | - |
| **FINAL SCORE** | **1.0300** | - | **103.0%** |

### Score Calculation Formula

```
Base Score = (String Sim × 0.50) + (Semantic Sim × 0.25) + (Concept Alignment × 0.25)
Location Boost = Location Score × 0.05 = 60.0000 × 0.05 = 3.0000

Final Score = Base Score + Location Boost = 1.0300
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection
- **Location Match (EXCELLENT):** 60.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.59<br>• <b>Concept Match:</b> 🟢 0.93<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Athens) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 48.8%, ✅ Illinois 28.1%<br>• Industry: ✅ Education 31.8%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Ohio University</b> (Athens, GA) | Score: <b>103.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5914</td><td>25%</td><td>0.1479</td></tr>
        <tr><td>Concept Alignment</td><td>0.9280</td><td>25%</td><td>0.2320</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0701</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0300</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0300</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.59<br>• <b>Concept Match:</b> 🟢 0.93<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Athens) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 48.8%, ✅ Illinois 28.1%<br>• Industry: ✅ Education 31.8%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Ohio University</b> (Columbus, OH) | Score: <b>103.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7832</td><td>25%</td><td>0.1958</td></tr>
        <tr><td>Concept Alignment</td><td>0.9870</td><td>25%</td><td>0.2467</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0075</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0095</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0300</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.78<br>• <b>Concept Match:</b> 🟢 0.99<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Columbus) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.95%)</span>. This is a high-frequency record (14 occurrences), suggesting it is a well-known entity.<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 72.1%, ✅ Illinois 32.4%, ✅ Miami 30.8%, ✅ Chicago 30.5%, ✅ New York 28.0%, ✅ California 27.0%, ✅ Pennsylvania 25.6%<br>• Industry: ✅ Education 33.0%<br>• Nature: ✅ Professional 26.9%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Ohio University</b> (Shade, OH) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7140</td><td>25%</td><td>0.1785</td></tr>
        <tr><td>Concept Alignment</td><td>0.9719</td><td>25%</td><td>0.2430</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0285</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.71<br>• <b>Concept Match:</b> 🟢 0.97<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Shade) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 64.5%, ✅ Miami 35.6%, ✅ Chicago 33.3%, ✅ Illinois 31.6%, ✅ California 31.4%<br>• Nature: ✅ Local 29.1%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Ohio University</b> (Dublin, OH) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6965</td><td>25%</td><td>0.1741</td></tr>
        <tr><td>Concept Alignment</td><td>0.9502</td><td>25%</td><td>0.2376</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0383</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.70<br>• <b>Concept Match:</b> 🟢 0.95<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Dublin) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 63.9%, ✅ London 36.7%, ✅ Chicago 31.4%, ✅ Illinois 29.7%, ✅ Pennsylvania 28.7%, ✅ California 26.2%, ✅ New York 25.6%<br>• Industry: ✅ Education 40.9%<br>• Nature: ✅ Local 28.3%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Ohio University</b> (, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9156</td><td>25%</td><td>0.2289</td></tr>
        <tr><td>Concept Alignment</td><td>0.9926</td><td>25%</td><td>0.2481</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟢 0.92<br>• <b>Concept Match:</b> 🟢 0.99<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location () is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 81.7%, ✅ Illinois 43.0%, ✅ Chicago 37.9%, ✅ California 35.1%, ✅ Miami 29.3%, ✅ New York 26.6%<br>• Industry: ✅ Education 42.9%<br>• Nature: ✅ Professional 33.9%, ✅ Local 25.1%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Ohio University</b> (Cincinnati, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8274</td><td>25%</td><td>0.2069</td></tr>
        <tr><td>Concept Alignment</td><td>0.9865</td><td>25%</td><td>0.2466</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟢 0.83<br>• <b>Concept Match:</b> 🟢 0.99<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Cincinnati) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 76.7%, ✅ Chicago 38.1%, ✅ Illinois 34.3%, ✅ California 29.3%, ✅ Miami 27.8%<br>• Industry: ✅ Education 30.4%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Ohio University</b> (West Chester, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7068</td><td>25%</td><td>0.1767</td></tr>
        <tr><td>Concept Alignment</td><td>0.9816</td><td>25%</td><td>0.2454</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0279</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.71<br>• <b>Concept Match:</b> 🟢 0.98<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (West Chester) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 69.0%, ✅ Illinois 40.2%, ✅ Chicago 34.1%, ✅ Pennsylvania 31.5%, ✅ California 25.1%<br>• Industry: ✅ Education 36.8%<br>• Nature: ✅ Local 29.0%, ✅ Professional 26.9%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Ohio University</b> (Dayton, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7437</td><td>25%</td><td>0.1859</td></tr>
        <tr><td>Concept Alignment</td><td>0.9788</td><td>25%</td><td>0.2447</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0194</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.74<br>• <b>Concept Match:</b> 🟢 0.98<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Dayton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 69.7%, ✅ Illinois 34.2%, ✅ Chicago 32.6%<br>• Industry: ✅ Education 30.5%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Ohio University</b> (Ironton, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6633</td><td>25%</td><td>0.1658</td></tr>
        <tr><td>Concept Alignment</td><td>0.9766</td><td>25%</td><td>0.2441</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0400</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.66<br>• <b>Concept Match:</b> 🟢 0.98<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Ironton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 64.0%, ✅ Illinois 34.9%<br>• Industry: ✅ Education 26.0%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Ohio University</b> (Coal Grove, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>50%</td><td>0.5000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6749</td><td>25%</td><td>0.1687</td></tr>
        <tr><td>Concept Alignment</td><td>0.9733</td><td>25%</td><td>0.2433</td></tr>
        <tr><td>Lexical Alignment Boost</td><td></td><td>Floor</td><td>+0.0380</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Name Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5-20%</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry• <b>Strength:</b> 🟡 0.67<br>• <b>Concept Match:</b> 🟢 0.97<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Coal Grove) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br><br><b>Concept Analysis:</b><br>• Geography: ✅ Ohio 67.6%, ✅ Illinois 41.7%, ✅ Chicago 35.0%, ✅ New York 29.9%, ✅ California 29.9%, ✅ Texas 26.7%, ✅ Pennsylvania 26.3%<br>• Industry: ✅ Education 32.6%<br>• Insight: The model detects a strong 'Ohio' influence in the company's semantic vector.<br>
      </div>
    </div>
</details>

---

