# Company Matching Control Set Report (Location-Aware)

**Generated:** 2025-12-30 08:38:52

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

## 1. NIH (Bethesda, MD)

**Query:** `NIH` • **Location:** Bethesda, MD • **Self-Match:** ✅ Found & Filtered

**Top Match:** NIH (Bedthesa, MD) • **Score:** 104.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.4259 | 30% | 0.1278 |
| Semantic Similarity (Raw) | 2.8778 | - | - |
| **Base Score** | **0.8278** | - | - |
| Location Context Boost | 92.5000 | 5% max | +4.6250 |
| **FINAL SCORE** | **1.0487** | - | **104.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 92.5000 × 0.05 = 4.6250

Final Score = Base Score + Location Boost = 1.0487
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (FAIR):** Some meaning-based connection
- **Location Match (EXCELLENT):** 92.50 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+4.63%)</span>. The record's location (Bedthesa) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>NIH</b> (Bedthesa, MD) | Score: <b>104.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4259</td><td>30%</td><td>0.1278</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8278</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0463</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0490</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+4.63%)</span>. The record's location (Bedthesa) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>NIH</b> (Rockville, MD) | Score: <b>103.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4200</td><td>30%</td><td>0.1260</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8260</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0135</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0340</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.42<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Rockville) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.35%)</span>. This is a high-frequency record (46 occurrences), suggesting it is a well-known entity.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>NIH</b> (Bethesda, ) | Score: <b>103.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5855</td><td>30%</td><td>0.1757</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8757</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0300</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0320</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Bethesda) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>NIH</b> (Upper Marlboro, MD) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4276</td><td>30%</td><td>0.1283</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8283</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Upper Marlboro) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>NIH</b> (Gaithersburg, MD) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Gaithersburg) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>NIH</b> (National Institute Of Health, MD) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4527</td><td>30%</td><td>0.1358</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8358</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.45<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (National Institute Of Health) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>NIH</b> (Silver Spring, MD) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4303</td><td>30%</td><td>0.1291</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8291</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Silver Spring) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>NIH</b> (Laurel, MD) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4240</td><td>30%</td><td>0.1272</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8272</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.42<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Laurel) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>NIH</b> (Silver Springs, MD) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.3997</td><td>30%</td><td>0.1199</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8199</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.40<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Silver Springs) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>NIH</b> (Balitmore, MD) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Balitmore) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>

---

## 2. Ohio University (Athens, OH)

**Query:** `Ohio University` • **Location:** Athens, OH • **Self-Match:** ✅ Found & Filtered

**Top Match:** Ohio University (Athens, GA) • **Score:** 103.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.5914 | 30% | 0.1774 |
| Semantic Similarity (Raw) | 4.2003 | - | - |
| **Base Score** | **0.8774** | - | - |
| Location Context Boost | 60.0000 | 5% max | +3.0000 |
| **FINAL SCORE** | **1.0324** | - | **103.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 60.0000 × 0.05 = 3.0000

Final Score = Base Score + Location Boost = 1.0324
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection
- **Location Match (EXCELLENT):** 60.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Athens) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Ohio University</b> (Athens, GA) | Score: <b>103.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5914</td><td>30%</td><td>0.1774</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8774</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0300</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0320</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Athens) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Ohio University</b> (Columbus, OH) | Score: <b>103.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7832</td><td>30%</td><td>0.2350</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9350</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0095</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0300</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Columbus) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.95%)</span>. This is a high-frequency record (14 occurrences), suggesting it is a well-known entity.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Ohio University</b> (Shade, OH) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7140</td><td>30%</td><td>0.2142</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9142</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Shade) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Ohio University</b> (Dublin, OH) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6965</td><td>30%</td><td>0.2090</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9090</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Dublin) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Ohio University</b> (, OH) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9156</td><td>30%</td><td>0.2747</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9747</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.92<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location () is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Ohio University</b> (Cincinnati, OH) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8274</td><td>30%</td><td>0.2482</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9482</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.83<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Cincinnati) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Ohio University</b> (Dayton, OH) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7437</td><td>30%</td><td>0.2231</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9231</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.74<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Dayton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>OHIO UNIVERSITY</b> (Lancaster, OH) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7434</td><td>30%</td><td>0.2230</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9230</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.74<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Lancaster) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Ohio University</b> (West Chester, OH) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7068</td><td>30%</td><td>0.2120</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9120</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (West Chester) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Ohio University</b> (Coal Grove, OH) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6749</td><td>30%</td><td>0.2025</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9025</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Coal Grove) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>

---

## 3. Western University (London, ON)

**Query:** `Western University` • **Location:** London, ON • **Self-Match:** ✅ Found & Filtered

**Top Match:** Western University (London, ) • **Score:** 103.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8136 | 30% | 0.2441 |
| Semantic Similarity (Raw) | 5.3052 | - | - |
| **Base Score** | **0.9441** | - | - |
| Location Context Boost | 60.0000 | 5% max | +3.0000 |
| **FINAL SCORE** | **1.0349** | - | **103.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 60.0000 × 0.05 = 3.0000

Final Score = Base Score + Location Boost = 1.0349
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 60.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (London) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.49%)</span>. Frequency boost applied (3 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Western University</b> (London, ) | Score: <b>103.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8136</td><td>30%</td><td>0.2441</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9441</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0300</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0049</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0350</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (London) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.49%)</span>. Frequency boost applied (3 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Western University</b> (Toronto, ON) | Score: <b>102.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6299</td><td>30%</td><td>0.1890</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8890</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0073</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0270</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Toronto) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.73%)</span>. Frequency boost applied (7 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Western University</b> (Ottawa, ON) | Score: <b>102.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5979</td><td>30%</td><td>0.1794</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8794</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.60<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Ottawa) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Western University</b> (, ON) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8364</td><td>30%</td><td>0.2509</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9509</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.84<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location () is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Western University</b> (Chatham, ON) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6382</td><td>30%</td><td>0.1914</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8914</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Chatham) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Western University</b> (Illderton, ON) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5650</td><td>30%</td><td>0.1695</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8695</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.57<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Illderton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Western University</b> (Ilderton, ON) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Ilderton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Western University</b> (Kincardine, ON) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Kincardine) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Western University</b> (Pomona, CA) | Score: <b>100.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0077</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0080</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.77%)</span>. Frequency boost applied (8 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Western University</b> (Halifax, NS) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6115</td><td>30%</td><td>0.1835</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8835</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>

---

## 4. Kruger Products (Bentonville, AR)

**Query:** `Kruger Products` • **Location:** Bentonville, AR • **Self-Match:** ✅ Found & Filtered

**Top Match:** Kruger Products (Fort Smith, AR) • **Score:** 102.2%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.5810 | 30% | 0.1743 |
| Semantic Similarity (Raw) | 4.1602 | - | - |
| **Base Score** | **0.8743** | - | - |
| Location Context Boost | 40.0000 | 5% max | +2.0000 |
| **FINAL SCORE** | **1.0224** | - | **102.2%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0224
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Fort Smith) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Kruger Products</b> (Fort Smith, AR) | Score: <b>102.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5810</td><td>30%</td><td>0.1743</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8743</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Fort Smith) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Kruger Products</b> (Mississauga, ON) | Score: <b>100.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6290</td><td>30%</td><td>0.1887</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8887</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0063</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0060</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.63%)</span>. Frequency boost applied (5 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Kruger Products</b> (Toronto, ON) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7254</td><td>30%</td><td>0.2176</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9176</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.73<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Kruger Products</b> | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Kruger Products</b> (Vancouver, BC) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6789</td><td>30%</td><td>0.2037</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9037</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Kruger Products</b> (Delta, BC) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6632</td><td>30%</td><td>0.1989</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8989</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Kruger Products</b> (New Westminster, ) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6381</td><td>30%</td><td>0.1914</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8914</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Kruger Products</b> (New Westminster, BC) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5329</td><td>30%</td><td>0.1599</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8599</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.53<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Kruger Products</b> (Minnetonka, MN) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4611</td><td>30%</td><td>0.1383</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8383</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.46<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Kruger Products USA Inc</b> (Bentonville, AR) | Score: <b>97.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8182</td><td>70%</td><td>0.5727</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4610</td><td>30%</td><td>0.1383</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7110</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0116</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9720</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'Kruger Products' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.46<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br>• <b>Context:</b> Match indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Bentonville) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.16%)</span>. Frequency boost applied (3 occurrences).<br>
      </div>
    </div>
</details>

---

## 5. Vision America

**Query:** `Vision America` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Vision America (Houston, TX) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.5802 | 30% | 0.1741 |
| Semantic Similarity (Raw) | 3.9116 | - | - |
| **Base Score** | **0.8741** | - | - |
| **FINAL SCORE** | **1.0039** | - | **100.4%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0039
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Vision America</b> (Houston, TX) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5802</td><td>30%</td><td>0.1741</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8741</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Vision America</b> (Lufkin, TX) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5615</td><td>30%</td><td>0.1685</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8685</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.56<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Vision America</b> | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Vision America</b> (Washington, DC) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7631</td><td>30%</td><td>0.2289</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9289</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.76<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Vision America</b> (Birmingham, AL) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6949</td><td>30%</td><td>0.2085</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9085</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>VISION AMERICA</b> (Keller, TX) | Score: <b>100.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5994</td><td>30%</td><td>0.1798</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8798</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0024</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.60<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.24%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Vision Council of America</b> (Alexandria, VA) | Score: <b>96.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4974</td><td>30%</td><td>0.1492</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7538</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0162</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9660</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Vision America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, vision<br><br><b>Company Name Also Includes:</b><br>• council, of<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.4974 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.50<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.62%)</span>. Frequency boost applied (6 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Vision action america</b> (Houston, TX) | Score: <b>95.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5388</td><td>30%</td><td>0.1616</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7662</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0092</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9590</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Vision America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, vision<br><br><b>Company Name Also Includes:</b><br>• action<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.5388 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.54<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.92%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Vision America Action</b> | Score: <b>95.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8182</td><td>70%</td><td>0.5727</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8690</td><td>30%</td><td>0.2607</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8334</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0058</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9560</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'Vision America' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.87<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.58%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Vision Council of America</b> | Score: <b>95.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6730</td><td>30%</td><td>0.2019</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8064</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0058</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9560</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Vision America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, vision<br><br><b>Company Name Also Includes:</b><br>• council, of<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.6730 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.58%)</span>. Frequency boost applied (1 occurrences).<br>
      </div>
    </div>
</details>

---

