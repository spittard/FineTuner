# Company Matching Control Set Report (Location-Aware)

**Generated:** 2025-12-30 10:54:34

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

**Top Match:** NIH (Bedthesa, MD) • **Score:** 104.6%

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
| **FINAL SCORE** | **1.0462** | - | **104.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 92.5000 × 0.05 = 4.6250

Final Score = Base Score + Location Boost = 1.0462
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (FAIR):** Some meaning-based connection
- **Location Match (EXCELLENT):** 92.50 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+4.63%)</span>. The record's location (Bedthesa) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>NIH</b> (Bedthesa, MD) | Score: <b>104.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4259</td><td>30%</td><td>0.1278</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8278</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0463</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0460</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+4.63%)</span>. The record's location (Bedthesa) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.42<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Rockville) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.35%)</span>. This is a high-frequency record (46 occurrences), suggesting it is a well-known entity.<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>NIH</b> (Bethesda, ) | Score: <b>103.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5855</td><td>30%</td><td>0.1757</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8757</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0300</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0300</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Bethesda) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Upper Marlboro) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
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
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>NIH</b> (National Institute Of Health, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4527</td><td>30%</td><td>0.1358</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8358</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.45<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (National Institute Of Health) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>NIH</b> (Silver Spring, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4303</td><td>30%</td><td>0.1291</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8291</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Silver Spring) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>NIH</b> (Laurel, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4240</td><td>30%</td><td>0.1272</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8272</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.42<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Laurel) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>NIH</b> (Silver Springs, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.3997</td><td>30%</td><td>0.1199</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8199</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.40<br>• <b>AI Insight:</b> The National Institutes of Health (NIH) is the same entity, a branch of the United States federal government.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Silver Springs) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>NIH</b> (Balitmore, MD) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Balitmore) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.5914 | 30% | 0.1774 |
| Semantic Similarity (Raw) | 4.2003 | - | - |
| **Base Score** | **0.8774** | - | - |
| Location Context Boost | 60.0000 | 5% max | +3.0000 |
| **FINAL SCORE** | **1.0300** | - | **103.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 60.0000 × 0.05 = 3.0000

Final Score = Base Score + Location Boost = 1.0300
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection
- **Location Match (EXCELLENT):** 60.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Athens) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Ohio University</b> (Athens, GA) | Score: <b>103.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5914</td><td>30%</td><td>0.1774</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8774</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0300</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0300</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (Athens) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Columbus) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.95%)</span>. This is a high-frequency record (14 occurrences), suggesting it is a well-known entity.<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Shade) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Dublin) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Ohio University</b> (, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9156</td><td>30%</td><td>0.2747</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9747</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.92<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location () is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Ohio University</b> (Cincinnati, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8274</td><td>30%</td><td>0.2482</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9482</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.83<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Cincinnati) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Ohio University</b> (Dayton, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7437</td><td>30%</td><td>0.2231</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9231</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.74<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Dayton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>OHIO UNIVERSITY</b> (Lancaster, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7434</td><td>30%</td><td>0.2230</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9230</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.74<br>• <b>AI Insight:</b> Ohio University is an educational institution and a university within the Ohio University System, making it a branch of the same entity as OHIO UNIVERSITY.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Lancaster) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Ohio University</b> (West Chester, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7068</td><td>30%</td><td>0.2120</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9120</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (West Chester) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Ohio University</b> (Coal Grove, OH) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6749</td><td>30%</td><td>0.2025</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9025</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> Ohio University is the same entity as Ohio University, a public research university located in Athens, Ohio.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Coal Grove) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (London) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.49%)</span>. Frequency boost applied (3 occurrences).<br>

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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+3.00%)</span>. The record's location (London) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.49%)</span>. Frequency boost applied (3 occurrences).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Toronto) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.73%)</span>. Frequency boost applied (7 occurrences).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.60<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Ottawa) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Western University</b> (, ON) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8364</td><td>30%</td><td>0.2509</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9509</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.84<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location () is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Western University</b> (Chatham, ON) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6382</td><td>30%</td><td>0.1914</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8914</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Chatham) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Western University</b> (Illderton, ON) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5650</td><td>30%</td><td>0.1695</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8695</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.57<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Illderton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Western University</b> (Ilderton, ON) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Ilderton) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Western University</b> (Kincardine, ON) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Kincardine) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> Western University is an educational institution and a research-intensive university, providing higher education services in various fields including business, with its main campus located in London, Ontario, Canada.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>

---

## 4. Kruger Products (Bentonville, AR)

**Query:** `Kruger Products` • **Location:** Bentonville, AR • **Self-Match:** ✅ Found & Filtered

**Top Match:** Kruger Products (Fort Smith, AR) • **Score:** 102.0%

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
| **FINAL SCORE** | **1.0200** | - | **102.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0200
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (GOOD):** Moderate meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Fort Smith) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Kruger Products</b> (Fort Smith, AR) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5810</td><td>30%</td><td>0.1743</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8743</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Fort Smith) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.63%)</span>. Frequency boost applied (5 occurrences).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.73<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Kruger Products</b> | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Kruger Products</b> (Vancouver, BC) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6789</td><td>30%</td><td>0.2037</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9037</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Kruger Products</b> (Delta, BC) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6632</td><td>30%</td><td>0.1989</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8989</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Kruger Products</b> (New Westminster, ) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6381</td><td>30%</td><td>0.1914</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8914</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Kruger Products</b> (New Westminster, BC) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5329</td><td>30%</td><td>0.1599</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8599</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.53<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Kruger Products</b> (Minnetonka, MN) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4611</td><td>30%</td><td>0.1383</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8383</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.46<br>• <b>AI Insight:</b> Kruger Products is a company that manufactures and distributes meat products, particularly beef, in South Africa, and is part of the larger global meat processing industry.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'Kruger Products' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.46<br>• <b>AI Insight:</b> Kruger Products USA Inc is the same entity as Kruger Products, a leading manufacturer of baby food and infant formula in North America.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br>• <b>Context:</b> Match indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Bentonville) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.16%)</span>. Frequency boost applied (3 occurrences).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>AI Insight:</b> Vision America is Vision America, a non-profit organization that provides financial assistance to low-income families in the United States through various programs and services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>

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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>AI Insight:</b> Vision America is Vision America, a non-profit organization that provides financial assistance to low-income families in the United States through various programs and services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.56<br>• <b>AI Insight:</b> Vision America is Vision America, a non-profit organization that provides financial assistance to low-income families in the United States through various programs and services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Vision America</b> | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Vision America</b> (Washington, DC) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7631</td><td>30%</td><td>0.2289</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9289</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.76<br>• <b>AI Insight:</b> Vision America is Vision America, a non-profit organization that provides financial assistance to low-income families in the United States through various programs and services.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Vision America</b> (Birmingham, AL) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6949</td><td>30%</td><td>0.2085</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9085</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> Vision America is Vision America, a non-profit organization that provides financial assistance to low-income families in the United States through various programs and services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>VISION AMERICA</b> (Keller, TX) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5994</td><td>30%</td><td>0.1798</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8798</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.60<br>• <b>AI Insight:</b> Vision America is a non-profit organization that provides financial assistance to low-income families in the United States, and it is also the same entity as Vision America, a company that offers travel packages and vacation clubs for seniors.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
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
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Vision America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, vision<br><br><b>Company Name Also Includes:</b><br>• council, of<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.4974 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.50<br>• <b>AI Insight:</b> The Vision Council of America is the same entity as Vision America, a non-profit organization that provides training and resources for the vision care industry.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.62%)</span>. Frequency boost applied (6 occurrences).<br>
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
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Vision America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, vision<br><br><b>Company Name Also Includes:</b><br>• action<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.5388 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.54<br>• <b>AI Insight:</b> Vision America is Vision Action America, a non-profit organization that operates within the healthcare and wellness industry, providing vision services to underserved communities.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.92%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Vision America Action</b> | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8182</td><td>70%</td><td>0.5727</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8690</td><td>30%</td><td>0.2607</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8334</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'Vision America' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.87<br>• <b>AI Insight:</b> Vision America is Vision America Action, a non-profit organization focused on promoting social justice and human rights in the United States.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Vision Council of America</b> | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6730</td><td>30%</td><td>0.2019</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8064</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Vision America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, vision<br><br><b>Company Name Also Includes:</b><br>• council, of<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.6730 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The Vision Council of America is the same entity as Vision America, a non-profit organization that provides training and resources for the vision care industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 6. PDMA Association

**Query:** `PDMA Association` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Association Headquarters-PDMA (Mount Laurel, NJ) • **Score:** 95.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity (Normalized) | 0.4491 | 30% | 0.1347 |
| Semantic Similarity (Raw) | 2.8192 | - | - |
| **Base Score** | **0.7393** | - | - |
| **FINAL SCORE** | **0.9500** | - | **95.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9500
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (FAIR):** Some meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• association<br><br><b>Your Search Also Includes:</b><br>• pdma<br><br><b>Company Name Also Includes:</b><br>• headquarters-pdma<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.4491 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.45<br>• <b>AI Insight:</b> The PDMA Association is the same entity as the Association Headquarters-PDMA, indicating that they are the same organization with different names for their headquarters.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Association Headquarters-PDMA</b> (Mount Laurel, NJ) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4491</td><td>30%</td><td>0.1347</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7393</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• association<br><br><b>Your Search Also Includes:</b><br>• pdma<br><br><b>Company Name Also Includes:</b><br>• headquarters-pdma<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.4491 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.45<br>• <b>AI Insight:</b> The PDMA Association is the same entity as the Association Headquarters-PDMA, indicating that they are the same organization with different names for their headquarters.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>PDMA Alliance</b> (York, SC) | Score: <b>90.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5445</td><td>30%</td><td>0.1633</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7583</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0087</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9090</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.5445 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.54<br>• <b>AI Insight:</b> The PDMA Association and the PDMA Alliance are the same entity, a non-profit professional association, with a shared industry context in the business process management (BPM) and project management (PMM) communities.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>PDMA ALLIANCE</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8098</td><td>30%</td><td>0.2430</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8380</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.8098 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> The PDMA Association is the same entity as the PDMA ALLIANCE, which indicates that they are the same organization with two different names.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>PDMA ALLIANCE</b> (, FL) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6902</td><td>30%</td><td>0.2071</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8021</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.6902 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> The PDMA Association is the same entity as the PDMA ALLIANCE, which indicates that they are the same organization with two different names.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>PDMA ALLIANCE</b> (CHARLOTTE, NC) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5560</td><td>30%</td><td>0.1668</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7618</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.5560 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.56<br>• <b>AI Insight:</b> The PDMA Association is the same entity as the PDMA ALLIANCE, which indicates that they are the same organization with two different names.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>PDMA Alliance Inc.</b> (Charlotte, NC) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4832</td><td>30%</td><td>0.1450</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7400</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance, inc.<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.4832 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.48<br>• <b>AI Insight:</b> The PDMA Association is the same entity as the PDMA Alliance Inc., which is a branch of the larger organization, indicating a clear hierarchical relationship between the two entities in the shared industry context of Project Management.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br>• <b>Context:</b> Different contexts: non_profit, partnership vs corporate<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>PDMA Alliance</b> (Valhalla, NY) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4751</td><td>30%</td><td>0.1425</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7375</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.4751 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.48<br>• <b>AI Insight:</b> The PDMA Association and the PDMA Alliance are the same entity, a non-profit professional association, with a shared industry context in the business process management (BPM) and project management (PMM) communities.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br>• <b>Context:</b> Query indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>PDMA Alliance</b> (Valballa, NY) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4579</td><td>30%</td><td>0.1374</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7324</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.4579 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.46<br>• <b>AI Insight:</b> The PDMA Association and the PDMA Alliance are the same entity, a non-profit professional association, with a shared industry context in the business process management (BPM) and project management (PMM) communities.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br>• <b>Context:</b> Query indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>PDMA</b> | Score: <b>80.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7200</td><td>70%</td><td>0.5040</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8040</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7200 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> The PDMA Association is the same entity as the PDMA company, which is a professional association for product development management professionals.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>PDMA inc</b> | Score: <b>77.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7200</td><td>70%</td><td>0.5040</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9129</td><td>30%</td><td>0.2739</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7779</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7780</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• pdma<br><br><b>Your Search Also Includes:</b><br>• association<br><br><b>Company Name Also Includes:</b><br>• inc<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7200 (Weight: 70%)<br>• Semantic Similarity: 0.9129 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.91<br>• <b>AI Insight:</b> PDMA Association is the same entity as PDMA inc, a company that operates in the same industry of Product Development Management (PDM).<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Different contexts: non_profit, partnership vs corporate<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 7. Nicolas/Sanchez Wedding

**Query:** `Nicolas/Sanchez Wedding` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Sanchez/Justin Wedding • **Score:** 79.5%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9159 | 30% | 0.2748 |
| Semantic Similarity (Raw) | 4.6285 | - | - |
| **Base Score** | **0.7954** | - | - |
| **FINAL SCORE** | **0.7954** | - | **79.5%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7954
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/justin<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9159 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.92<br>• <b>AI Insight:</b> Nicolas and Sanchez are individuals, making them Unrelated.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#1</b> | <b>Sanchez/Justin Wedding</b> | Score: <b>79.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9159</td><td>30%</td><td>0.2748</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7954</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7950</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/justin<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9159 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.92<br>• <b>AI Insight:</b> Nicolas and Sanchez are individuals, making them Unrelated.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>Sanchez Wedding</b> (Marietta, GA) | Score: <b>78.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7875</td><td>70%</td><td>0.5512</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7885</td><td>30%</td><td>0.2365</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7878</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7880</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7875 (Weight: 70%)<br>• Semantic Similarity: 0.7885 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> Nicolas and Sanchez Wedding are related as they are both wedding-related businesses, with Nicolas being the parent company of Sanchez Wedding.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Sanchez/Ramirez Wedding</b> (Miami, FL) | Score: <b>78.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8753</td><td>30%</td><td>0.2626</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7832</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7830</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/ramirez<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8753 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.88<br>• <b>AI Insight:</b> Nicolas/Sanchez Wedding and Sanchez/Ramirez Wedding are related as they are both wedding planning companies, with Nicolas/Sanchez focusing on high-end weddings and Sanchez/Ramirez specializing in more affordable, yet still luxurious, events.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>Garcia Sanchez Wedding</b> (Miami, FL) | Score: <b>77.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7733</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7730</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• garcia, sanchez<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> The Garcia Sanchez Wedding company is the same entity as Nicolas/Sanchez, indicating that they are related through a common individual or business name.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Gibson Sanchez Wedding</b> | Score: <b>76.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9749</td><td>30%</td><td>0.2925</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7658</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7660</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• gibson, sanchez<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 0.9749 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.97<br>• <b>AI Insight:</b> Nicolas/Sanchez Wedding is related to Gibson Sanchez Wedding as it appears to be an entity within the same industry context, specifically wedding planning services.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Sanchez/Puerto Wedding</b> (Miami Beach, Fl) | Score: <b>76.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8113</td><td>30%</td><td>0.2434</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7640</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7640</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/puerto<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8113 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> Nicolas/Sanchez Wedding is an unrelated entity as it pertains to wedding services, whereas Sanchez/Puerto Wedding is a company specializing in wedding planning and coordination.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Sanchez/Cohen Wedding</b> (Hollywood, FL) | Score: <b>75.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7882</td><td>30%</td><td>0.2365</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7571</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7570</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/cohen<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7882 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> Nicolas/Sanchez Wedding is an unrelated entity as it pertains to wedding services, whereas Sanchez/Cohen Wedding is the company providing such services.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Sanchez/Naranjo Wedding</b> (Miami, FL) | Score: <b>75.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7785</td><td>30%</td><td>0.2336</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7542</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7540</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/naranjo<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7785 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>AI Insight:</b> Nicolas and Sanchez are the same entity, a branch of the company Sanchez/Naranjo Wedding.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>Sanchez/Fuentes Wedding</b> (Dallas, TX) | Score: <b>75.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7780</td><td>30%</td><td>0.2334</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7540</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7540</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/fuentes<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7780 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>AI Insight:</b> Nicolas and Sanchez are the same entity, a branch of the company Sanchez/Fuentes Wedding.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Sanchez/Ramos Wedding</b> (Anaheim, CA) | Score: <b>74.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7614</td><td>30%</td><td>0.2284</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7490</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7490</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• wedding<br><br><b>Your Search Also Includes:</b><br>• nicolas/sanchez<br><br><b>Company Name Also Includes:</b><br>• sanchez/ramos<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7614 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.76<br>• <b>AI Insight:</b> Nicolas/Sanchez and Sanchez/Ramos are related as they are both wedding planning companies, with Nicolas being a branch of the larger company, Sanchez/Ramos.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 8. Kehilat Ariel Synagogue (Los Angeles, CA)

**Query:** `Kehilat Ariel Synagogue` • **Location:** Los Angeles, CA • **Self-Match:** ✅ Found & Filtered

**Top Match:** Kehilat Ariel Synagogue (San Diego, CA) • **Score:** 102.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.9348 | 30% | 0.2805 |
| Semantic Similarity (Raw) | 4.4039 | - | - |
| **Base Score** | **0.9805** | - | - |
| Location Context Boost | 40.0000 | 5% max | +2.0000 |
| **FINAL SCORE** | **1.0200** | - | **102.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0200
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.93<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (San Diego) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Kehilat Ariel Synagogue</b> (San Diego, CA) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9348</td><td>30%</td><td>0.2805</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9805</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.93<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (San Diego) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Kehilat Ariel Messianic Synagogue</b> (San Diego, CA) | Score: <b>84.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8225</td><td>30%</td><td>0.2468</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8513</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0800</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8400</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Kehilat Ariel Synagogue' was found in this company name.<br><br><b>Matching Words:</b><br>• ariel, kehilat, synagogue<br><br><b>Company Name Also Includes:</b><br>• messianic<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.8225 (Weight: 30%)<br>• Location Bonus: +0.4000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.82<br>• <b>AI Insight:</b> Kehilat Ariel Synagogue and Kehilat Ariel Messianic Synagogue are the same entity, a branch of the Kehilat Ariel Messianic Congregation.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+8.00%)</span>. The record's location (San Diego) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Kehilat Ariel</b> (San Diego, CA) | Score: <b>80.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8182</td><td>70%</td><td>0.5727</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5886</td><td>30%</td><td>0.1766</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7493</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0800</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ariel, kehilat<br><br><b>Your Search Also Includes:</b><br>• synagogue<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8182 (Weight: 70%)<br>• Semantic Similarity: 0.5886 (Weight: 30%)<br>• Location Bonus: +0.4000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> Kehilat Ariel Synagogue is a religious entity, specifically a synagogue affiliated with the Kehilat Ariel branch of the Jewish community.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates religious context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+8.00%)</span>. The record's location (San Diego) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Kehilat Ariel Passover</b> (San Diego, CA) | Score: <b>80.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8833</td><td>70%</td><td>0.6183</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5558</td><td>30%</td><td>0.1667</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7851</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0800</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ariel, kehilat<br><br><b>Your Search Also Includes:</b><br>• synagogue<br><br><b>Company Name Also Includes:</b><br>• passover<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8833 (Weight: 70%)<br>• Semantic Similarity: 0.5558 (Weight: 30%)<br>• Location Bonus: +0.4000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.56<br>• <b>AI Insight:</b> Kehilat Ariel Synagogue and Kehilat Ariel Passover are the same entity, a branch of the same organization.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates religious context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+8.00%)</span>. The record's location (San Diego) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>KAS</b> | Score: <b>69.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6900</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ACRONYM MATCH<br><br><b>What This Means:</b><br>The system identified a direct link between an exact acronym and its full company name.<br><br><b>Match Type:</b><br>• Acronym Reverse<br><br><b>Score Breakdown:</b><br>• Expansion Quality: 0.70<br>• Lexical match: 1.00<br>• Semantic link: 1.0000<br><br><b>Action Required:</b><br>• Verify if the acronym 'Kehilat Ariel Synagogue' correctly represents 'KAS'<br>• Expansion quality is moderate<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> Kehilat Ariel Synagogue is a religious entity, specifically a branch of Kehilat Ariel, which is an independent Jewish congregation in Israel.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Query indicates religious context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Ohel Moshe Synagogue</b> (Los Angeles, CA) | Score: <b>62.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4750</td><td>70%</td><td>0.3325</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6645</td><td>30%</td><td>0.1994</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5319</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6250</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• synagogue<br><br><b>Your Search Also Includes:</b><br>• ariel, kehilat<br><br><b>Company Name Also Includes:</b><br>• moshe, ohel<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4750 (Weight: 70%)<br>• Semantic Similarity: 0.6645 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> Kehilat Ariel Synagogue and Ohel Moshe Synagogue are the same entity, specifically a branch of the Ohel Moshe Synagogue in Israel.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Los Angeles) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Sephardic Temple-Synagogue</b> (Los Angeles, CA) | Score: <b>61.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4750</td><td>70%</td><td>0.3325</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6269</td><td>30%</td><td>0.1881</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5206</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6160</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
LINGUISTIC MATCH<br><br><b>What This Means:</b><br>The names look different but are linguistically related.<br><br><b>Key Relationships Found:</b><br>• 'synagogue' ↔ 'temple-synagogue' (abbreviation/expansion)<br><br><b>Details:</b><br>• 'synagogue' is abbreviation of 'temple-synagogue'<br><br><b>Real-World Scenario:</b><br>• Someone used the short form 'synagogue' instead of 'temple-synagogue'<br><br><b>Action Required:</b><br>• Verify if this variation makes sense<br>• Likely the same company<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>AI Insight:</b> Kehilat Ariel Synagogue and Sephardic Temple-Synagogue are related as the same entity, specifically a synagogue, within the broader context of Sephardic Jewish communities.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Los Angeles) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Synagogue 3000</b> (Los Angeles, CA) | Score: <b>61.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4318</td><td>70%</td><td>0.3023</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7146</td><td>30%</td><td>0.2144</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5167</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6130</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• synagogue<br><br><b>Your Search Also Includes:</b><br>• ariel, kehilat<br><br><b>Company Name Also Includes:</b><br>• 3000<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4318 (Weight: 70%)<br>• Semantic Similarity: 0.7146 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> Synagogue 3000 is the same entity as Kehilat Ariel Synagogue, which is a branch of Synagogue 3000.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Los Angeles) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>Temple Sinai Synagogue</b> (Oakland, CA) | Score: <b>52.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4750</td><td>70%</td><td>0.3325</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7069</td><td>30%</td><td>0.2121</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5446</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0800</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0077</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5230</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• synagogue<br><br><b>Your Search Also Includes:</b><br>• ariel, kehilat<br><br><b>Company Name Also Includes:</b><br>• sinai, temple<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4750 (Weight: 70%)<br>• Semantic Similarity: 0.7069 (Weight: 30%)<br>• Location Bonus: +0.4000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> Kehilat Ariel Synagogue is a branch of Temple Sinai Synagogue, sharing the same industry context as a synagogue in the United States.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+8.00%)</span>. The record's location (Oakland) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.77%)</span>. Frequency boost applied (4 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Synagogue Temple Aliyah</b> (Woodland Hills, CA) | Score: <b>50.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4750</td><td>70%</td><td>0.3325</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6519</td><td>30%</td><td>0.1956</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5281</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0800</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• synagogue<br><br><b>Your Search Also Includes:</b><br>• ariel, kehilat<br><br><b>Company Name Also Includes:</b><br>• aliyah, temple<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4750 (Weight: 70%)<br>• Semantic Similarity: 0.6519 (Weight: 30%)<br>• Location Bonus: +0.4000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The Synagogue Temple Aliyah is the same entity as Kehilat Ariel Synagogue, indicating that they are the same branch or location of the same company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have religious indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+8.00%)</span>. The record's location (Woodland Hills) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 9. Next Level Events

**Query:** `Next Level Events` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Next Level Events (New York, NY) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6612 | 30% | 0.1983 |
| Semantic Similarity (Raw) | 3.9835 | - | - |
| **Base Score** | **0.8983** | - | - |
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> Next Level Events is the same entity as Next Level Events, a company that specializes in event planning and management services for various industries, including corporate events, conferences, and festivals.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Next Level Events</b> (New York, NY) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6612</td><td>30%</td><td>0.1983</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8983</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> Next Level Events is the same entity as Next Level Events, a company that specializes in event planning and management services for various industries, including corporate events, conferences, and festivals.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Next Level Events</b> (Lehi, UT) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5924</td><td>30%</td><td>0.1777</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8777</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> Next Level Events is the same entity as Next Level Events, a company that specializes in event planning and management services for various industries, including corporate events, conferences, and festivals.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Next Level Events</b> | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Next Level Events</b> (Atlanta, GA) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6291</td><td>30%</td><td>0.1887</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8887</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>AI Insight:</b> Next Level Events is the same entity as Next Level Events, a company that specializes in event planning and management services for various industries, including corporate events, conferences, and festivals.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>NEXT LEVEL EVENTS</b> (Dallas, TX) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6171</td><td>30%</td><td>0.1851</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8851</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.62<br>• <b>AI Insight:</b> Next Level Events is the same entity as NEXT LEVEL EVENTS, a company that specializes in event planning and management services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Next Level Events</b> (Elizabeth, NJ) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6089</td><td>30%</td><td>0.1827</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8827</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> Next Level Events is the same entity as Next Level Events, a company that specializes in event planning and management services for various industries, including corporate events, conferences, and festivals.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>NEXT LEVEL EVENTS</b> (LOS ANGELES, CA) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5804</td><td>30%</td><td>0.1741</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8741</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>AI Insight:</b> Next Level Events is the same entity as NEXT LEVEL EVENTS, a company that specializes in event planning and management services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Next Level Events</b> (Woodbridge, VA) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Next Level Events</b> (Salt Lake City, UT) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Next Level Events</b> (Scottsdale, AZ) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 10. Site Foundation Golf Tournament

**Query:** `Site Foundation Golf Tournament` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Fore County Golf Tournament • **Score:** 76.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8191 | 30% | 0.2457 |
| Semantic Similarity (Raw) | 4.6740 | - | - |
| **Base Score** | **0.7664** | - | - |
| **FINAL SCORE** | **0.7664** | - | **76.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7664
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• golf, tournament<br><br><b>Your Search Also Includes:</b><br>• foundation, site<br><br><b>Company Name Also Includes:</b><br>• county, fore<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8191 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.82<br>• <b>AI Insight:</b> The Fore County Golf Tournament is the same entity as Site Foundation, a non-profit organization that provides golf tournaments and fundraising events for children with life-threatening illnesses.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#1</b> | <b>Fore County Golf Tournament</b> | Score: <b>76.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8191</td><td>30%</td><td>0.2457</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7664</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7660</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• golf, tournament<br><br><b>Your Search Also Includes:</b><br>• foundation, site<br><br><b>Company Name Also Includes:</b><br>• county, fore<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8191 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.82<br>• <b>AI Insight:</b> The Fore County Golf Tournament is the same entity as Site Foundation, a non-profit organization that provides golf tournaments and fundraising events for children with life-threatening illnesses.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>House Victory Golf Tournament</b> | Score: <b>76.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8036</td><td>30%</td><td>0.2411</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7617</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7620</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• golf, tournament<br><br><b>Your Search Also Includes:</b><br>• foundation, site<br><br><b>Company Name Also Includes:</b><br>• house, victory<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8036 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.80<br>• <b>AI Insight:</b> The House Victory Golf Tournament is the same entity as Site Foundation, a non-profit organization that hosts golf tournaments to support various charitable causes.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Women In Golf Foundation</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7983</td><td>30%</td><td>0.2395</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7601</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• foundation, golf<br><br><b>Your Search Also Includes:</b><br>• site, tournament<br><br><b>Company Name Also Includes:</b><br>• in, women<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7983 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.80<br>• <b>AI Insight:</b> The Women In Golf Foundation is the same entity as Site Foundation, a non-profit organization that provides golf-related educational and charitable initiatives for women in golf.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>ANNIKA Foundation - Golf Tournament</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7875</td><td>70%</td><td>0.5512</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6945</td><td>30%</td><td>0.2083</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7596</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• foundation, golf, tournament<br><br><b>Your Search Also Includes:</b><br>• site<br><br><b>Company Name Also Includes:</b><br>• -, annika<br><br><b>Match Strength:</b><br>• 60% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7875 (Weight: 70%)<br>• Semantic Similarity: 0.6945 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> The ANNIKA Foundation - Golf Tournament is an entity that shares the same industry context as Site Foundation, specifically in the golf course management and maintenance sector.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Golf Tournament</b> | Score: <b>75.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6562</td><td>70%</td><td>0.4594</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7594</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7590</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• golf, tournament<br><br><b>Your Search Also Includes:</b><br>• foundation, site<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6562 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> The Golf Tournament company is the same entity as the Site Foundation, which is a branch of the Golf Tournament organization.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Bunker To Bunker Golf Tournament</b> | Score: <b>75.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7885</td><td>30%</td><td>0.2365</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7572</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7570</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• golf, tournament<br><br><b>Your Search Also Includes:</b><br>• foundation, site<br><br><b>Company Name Also Includes:</b><br>• bunker, to<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7885 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> The Bunker To Bunker Golf Tournament is the same entity as the Site Foundation Golf Tournament, as they share the same industry context of organizing golf tournaments.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>National Youth Golf Foundation</b> | Score: <b>75.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7825</td><td>30%</td><td>0.2347</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7554</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7550</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• foundation, golf<br><br><b>Your Search Also Includes:</b><br>• site, tournament<br><br><b>Company Name Also Includes:</b><br>• national, youth<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7825 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>AI Insight:</b> The National Youth Golf Foundation is the same entity as Site Foundation, a non-profit organization that provides golf course construction and renovation services to youth golf programs.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>World Golf Foundation</b> | Score: <b>75.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9335</td><td>30%</td><td>0.2800</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7533</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7530</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• foundation, golf<br><br><b>Your Search Also Includes:</b><br>• site, tournament<br><br><b>Company Name Also Includes:</b><br>• world<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 0.9335 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.93<br>• <b>AI Insight:</b> The World Golf Foundation is the same entity as Site Foundation, a non-profit organization that supports golf course construction and renovation initiatives in developing countries.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>National Golf Foundation</b> | Score: <b>74.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9103</td><td>30%</td><td>0.2731</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7464</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7460</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• foundation, golf<br><br><b>Your Search Also Includes:</b><br>• site, tournament<br><br><b>Company Name Also Includes:</b><br>• national<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 0.9103 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.91<br>• <b>AI Insight:</b> The National Golf Foundation is the same entity as Site Foundation, a non-profit organization that provides funding and support for golf courses in the United States.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Golf Tournament-Central Florida</b> | Score: <b>74.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7499</td><td>30%</td><td>0.2250</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7456</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7460</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• golf<br><br><b>Your Search Also Includes:</b><br>• foundation, site, tournament<br><br><b>Company Name Also Includes:</b><br>• florida, tournament-central<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7499 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.75<br>• <b>AI Insight:</b> Golf Tournament-Central Florida is the same entity as Golf Tournament-Central Florida Foundation, a branch of Golf Tournament-Central Florida, which is the shared industry context within the golf tournament-corporation.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 11. Interim WG Meeting - BIER

**Query:** `Interim WG Meeting - BIER` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Interim Healthcare TEAMM Meeting • **Score:** 68.8%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7083 | 70% | 0.4958 |
| Semantic Similarity (Normalized) | 0.6420 | 30% | 0.1926 |
| Semantic Similarity (Raw) | 2.9424 | - | - |
| **Base Score** | **0.6884** | - | - |
| **FINAL SCORE** | **0.6884** | - | **68.8%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.6884
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• interim, meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, wg<br><br><b>Company Name Also Includes:</b><br>• healthcare, teamm<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7083 (Weight: 70%)<br>• Semantic Similarity: 0.6420 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> The Interim WG Meeting - BIER is an Interim Healthcare TEAMM Meeting, indicating that they are the same entity with a shared industry context of healthcare.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#1</b> | <b>Interim Healthcare TEAMM Meeting</b> | Score: <b>68.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7083</td><td>70%</td><td>0.4958</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6420</td><td>30%</td><td>0.1926</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6884</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6880</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• interim, meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, wg<br><br><b>Company Name Also Includes:</b><br>• healthcare, teamm<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7083 (Weight: 70%)<br>• Semantic Similarity: 0.6420 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> The Interim WG Meeting - BIER is an Interim Healthcare TEAMM Meeting, indicating that they are the same entity with a shared industry context of healthcare.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>AACP 2012 Interim Meeting</b> | Score: <b>68.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7083</td><td>70%</td><td>0.4958</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6374</td><td>30%</td><td>0.1912</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6871</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6870</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• interim, meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, wg<br><br><b>Company Name Also Includes:</b><br>• 2012, aacp<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7083 (Weight: 70%)<br>• Semantic Similarity: 0.6374 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> The AACP 2012 Interim WG Meeting - BIER is an entity that shares the same industry context as the search query, specifically in the field of Academic and Professional Computing (AACP).<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Legislative Interim Meeting</b> (Weston, WV) | Score: <b>65.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6439</td><td>70%</td><td>0.4508</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6808</td><td>30%</td><td>0.2042</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6550</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6550</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• interim, meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, wg<br><br><b>Company Name Also Includes:</b><br>• legislative<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6439 (Weight: 70%)<br>• Semantic Similarity: 0.6808 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> The Legislative Interim Meeting is the same entity as the Interim WG Meeting - BIER, sharing the same industry context of legislative proceedings and interim meetings in government or regulatory settings.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>BI Meeting</b> | Score: <b>56.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3750</td><td>70%</td><td>0.2625</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5625</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5620</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, interim, wg<br><br><b>Company Name Also Includes:</b><br>• bi<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3750 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> The Interim WG Meeting - BIER is an entity within the BI Meeting, sharing the same industry context as the company.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Bi Annual Meeting</b> | Score: <b>55.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4091</td><td>70%</td><td>0.2864</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9004</td><td>30%</td><td>0.2701</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5565</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5560</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, interim, wg<br><br><b>Company Name Also Includes:</b><br>• annual, bi<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4091 (Weight: 70%)<br>• Semantic Similarity: 0.9004 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.90<br>• <b>AI Insight:</b> The Bi Annual Meeting is the same entity as the Interim WG Meeting - BIER, as they refer to the same organization with different names for their internal meetings.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>HRC Advisory Board meeting</b> | Score: <b>53.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4500</td><td>70%</td><td>0.3150</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7343</td><td>30%</td><td>0.2203</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5353</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5350</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, interim, wg<br><br><b>Company Name Also Includes:</b><br>• advisory, board, hrc<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4500 (Weight: 70%)<br>• Semantic Similarity: 0.7343 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.73<br>• <b>AI Insight:</b> The HRC Advisory Board meeting is the same entity as the Interim WG Meeting - BIER, sharing the same industry context of Human Resource Management and related topics.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Match indicates consulting context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Biz Library January Meeting</b> | Score: <b>52.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4500</td><td>70%</td><td>0.3150</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7125</td><td>30%</td><td>0.2138</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5288</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5290</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, interim, wg<br><br><b>Company Name Also Includes:</b><br>• biz, january, library<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4500 (Weight: 70%)<br>• Semantic Similarity: 0.7125 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> The Biz Library January Meeting is an Interim Working Group (WG) meeting, which implies it is related to the company's internal working group structure and meetings.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Bim Object Meeting</b> | Score: <b>52.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4091</td><td>70%</td><td>0.2864</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7930</td><td>30%</td><td>0.2379</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5243</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5240</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, interim, wg<br><br><b>Company Name Also Includes:</b><br>• bim, object<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4091 (Weight: 70%)<br>• Semantic Similarity: 0.7930 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> The Bim Object Meeting is an entity of the Bim Object Meeting company, which shares the same industry context as Interim WG Meeting - BIER.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>American Biz Meeting</b> | Score: <b>52.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4091</td><td>70%</td><td>0.2864</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7840</td><td>30%</td><td>0.2352</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5216</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, interim, wg<br><br><b>Company Name Also Includes:</b><br>• american, biz<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4091 (Weight: 70%)<br>• Semantic Similarity: 0.7840 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>AI Insight:</b> The American Biz Meeting is the same entity as the Interim WG Meeting - BIER, sharing the same industry context of business and management.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Executive Advisory Board Meeting</b> | Score: <b>51.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4500</td><td>70%</td><td>0.3150</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6713</td><td>30%</td><td>0.2014</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5164</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5160</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• meeting<br><br><b>Your Search Also Includes:</b><br>• -, bier, interim, wg<br><br><b>Company Name Also Includes:</b><br>• advisory, board, executive<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4500 (Weight: 70%)<br>• Semantic Similarity: 0.6713 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The Interim WG Meeting - BIER is an Executive Advisory Board Meeting, which shares the same industry context as the company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates consulting context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 12. DermaQuest Inc

**Query:** `DermaQuest Inc` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Dermaquest Skin Care • **Score:** 95.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7500 | 70% | 0.5250 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 4.3077 | - | - |
| **Base Score** | **0.8250** | - | - |
| **FINAL SCORE** | **0.9500** | - | **95.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9500
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• dermaquest<br><br><b>Your Search Also Includes:</b><br>• inc<br><br><b>Company Name Also Includes:</b><br>• care, skin<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> DermaQuest Inc and Dermaquest Skin Care are the same entity, a branch of the company, within the skin care industry context.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Different contexts: corporate vs healthcare<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Dermaquest Skin Care</b> | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8250</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• dermaquest<br><br><b>Your Search Also Includes:</b><br>• inc<br><br><b>Company Name Also Includes:</b><br>• care, skin<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> DermaQuest Inc and Dermaquest Skin Care are the same entity, a branch of the company, within the skin care industry context.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Different contexts: corporate vs healthcare<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Dermaquest, Incorporated</b> (Hayward, CA) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7240</td><td>30%</td><td>0.2172</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9172</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
LINGUISTIC MATCH<br><br><b>What This Means:</b><br>The names look different but are linguistically related.<br><br><b>Key Relationships Found:</b><br>• 'dermaquest' ↔ 'dermaquest,' (abbreviation/expansion)<br>• 'inc' ↔ 'incorporated' (abbreviation/expansion)<br><br><b>Details:</b><br>• 'dermaquest' is abbreviation of 'dermaquest,'<br>• 'inc' is abbreviation of 'incorporated'<br><br><b>Real-World Scenario:</b><br>• Someone used the short form 'dermaquest' instead of 'dermaquest,'<br>• Someone used the short form 'inc' instead of 'incorporated'<br><br><b>Action Required:</b><br>• Verify if this variation makes sense<br>• Likely the same company<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.72<br>• <b>AI Insight:</b> DermaQuest Inc and Dermaquest Incorporated are the same entity, a US-based biotechnology company that specializes in developing dermal repair products for skin health issues.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have corporate indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Dermaquest Skin Therapy</b> (Hayward, CA) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6384</td><td>30%</td><td>0.1915</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7165</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• dermaquest<br><br><b>Your Search Also Includes:</b><br>• inc<br><br><b>Company Name Also Includes:</b><br>• skin, therapy<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 0.6384 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> DermaQuest Inc and Dermaquest Skin Therapy are the same entity, a US-based company that specializes in skin therapy products and solutions.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#4</b> | <b>Dermapen</b> | Score: <b>49.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3000</td><td>70%</td><td>0.2100</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9470</td><td>30%</td><td>0.2841</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4941</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0048</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4990</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.50)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3000 (Weight: 70%)<br>• Semantic Similarity: 0.9470 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#5</b> | <b>DERMA E</b> | Score: <b>49.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.2888</td><td>70%</td><td>0.2021</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9707</td><td>30%</td><td>0.2912</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4934</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4930</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
LINGUISTIC MATCH<br><br><b>What This Means:</b><br>The names look different but are linguistically related.<br><br><b>Key Relationships Found:</b><br>• 'dermaquest' ↔ 'derma' (abbreviation/expansion)<br>• 'dermaquest' ↔ 'e' (abbreviation/expansion)<br><br><b>Details:</b><br>• 'derma' is abbreviation of 'dermaquest'<br>• 'e' is abbreviation of 'dermaquest'<br><br><b>Real-World Scenario:</b><br>• Your system has the short form 'derma' but someone wrote 'dermaquest'<br>• Your system has the short form 'e' but someone wrote 'dermaquest'<br><br><b>Action Required:</b><br>• Verify if this variation makes sense<br>• Likely the same company<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.97<br>• <b>AI Insight:</b> DermaQuest Inc and DERMA E are the same entity, a US-based company that specializes in dermal science research and development.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#6</b> | <b>Mapquest</b> | Score: <b>48.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3500</td><td>70%</td><td>0.2450</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7941</td><td>30%</td><td>0.2382</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4832</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4830</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.48)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3500 (Weight: 70%)<br>• Semantic Similarity: 0.7941 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#7</b> | <b>Perquest</b> | Score: <b>47.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3500</td><td>70%</td><td>0.2450</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7750</td><td>30%</td><td>0.2325</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4775</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4780</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.48)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3500 (Weight: 70%)<br>• Semantic Similarity: 0.7750 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#8</b> | <b>Interquest</b> | Score: <b>47.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3150</td><td>70%</td><td>0.2205</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8312</td><td>30%</td><td>0.2494</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4699</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4700</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.47)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3150 (Weight: 70%)<br>• Semantic Similarity: 0.8312 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#9</b> | <b>RamQuest</b> | Score: <b>45.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3500</td><td>70%</td><td>0.2450</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7097</td><td>30%</td><td>0.2129</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4579</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4580</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.46)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3500 (Weight: 70%)<br>• Semantic Similarity: 0.7097 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#10</b> | <b>ENTREQUEST</b> | Score: <b>45.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3150</td><td>70%</td><td>0.2205</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7674</td><td>30%</td><td>0.2302</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4507</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0043</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4550</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.46)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3150 (Weight: 70%)<br>• Semantic Similarity: 0.7674 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>

---

## 13. Ellwood Group Inc (Chicago, IL)

**Query:** `Ellwood Group Inc` • **Location:** Chicago, IL • **Self-Match:** ✅ Found & Filtered

**Top Match:** Ellwood Group Inc (Ellwood City, PA) • **Score:** 100.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 5.2426 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0000** | - | **100.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0000
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have corporate indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Ellwood Group Inc</b> (Ellwood City, PA) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have corporate indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Ellwood Associates</b> (Chicago, IL) | Score: <b>96.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.9000</td><td>70%</td><td>0.6300</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7235</td><td>30%</td><td>0.2171</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8471</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0092</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9690</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• associates<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.9000 (Weight: 70%)<br>• Semantic Similarity: 0.7235 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.72<br>• <b>AI Insight:</b> Ellwood Group Inc and Ellwood Associates are the same entity, a subsidiary of Ellwood Capital Corporation.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Different contexts: corporate vs partnership<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Chicago) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.92%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Ellwood TX Forge Houston</b> (Houston, TX) | Score: <b>76.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6923</td><td>70%</td><td>0.4846</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6505</td><td>30%</td><td>0.1951</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6798</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0092</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7690</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• forge, houston, tx<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6923 (Weight: 70%)<br>• Semantic Similarity: 0.6505 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> Ellwood Group Inc is Ellwood TX Forge Houston, a company that operates in the same industry of steel fabrication and manufacturing.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.92%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>Ellwood Community Church</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8259</td><td>30%</td><td>0.2478</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7728</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• church, community<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 0.8259 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.83<br>• <b>AI Insight:</b> Ellwood Group Inc and Ellwood Community Church are unrelated entities, as Ellwood Group is a construction company specializing in building materials and services, while Ellwood Community Church is a non-profit organization focused on spiritual growth and community service.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Different contexts: corporate vs religious<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Ellwood Rose Machine</b> (Houston, TX) | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7028</td><td>30%</td><td>0.2108</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7358</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• machine, rose<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 0.7028 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> Ellwood Group Inc and Ellwood Rose Machine are the same entity, a subsidiary of Ellwood Industries, Inc., which is a leading manufacturer of rose machinery and related products in the agricultural industry.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Ellwood TX Forge</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6956</td><td>30%</td><td>0.2087</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7337</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• forge, tx<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 0.6956 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> Ellwood Group Inc is Ellwood TX Forge.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Ellwood TX Forge Houston</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6923</td><td>70%</td><td>0.4846</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6921</td><td>30%</td><td>0.2076</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6922</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• forge, houston, tx<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6923 (Weight: 70%)<br>• Semantic Similarity: 0.6921 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> Ellwood Group Inc is Ellwood TX Forge Houston, a company that operates in the same industry of steel fabrication and manufacturing.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Ellwood Closed Die Group</b> (Houston, TX) | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6883</td><td>30%</td><td>0.2065</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7315</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood, group<br><br><b>Your Search Also Includes:</b><br>• inc<br><br><b>Company Name Also Includes:</b><br>• closed, die<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 0.6883 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> Ellwood Group Inc and Ellwood Closed Die Group are the same entity, a subsidiary of Ellwood Industries, Inc., which is a leading manufacturer of closed die forging equipment and services in the metal forming industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>Ellwood Specialty Steel</b> (Ellwood City, PA) | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6816</td><td>30%</td><td>0.2045</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7295</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• specialty, steel<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 0.6816 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> Ellwood Specialty Steel is Ellwood Group Inc's wholly-owned subsidiary, providing access to its expertise in the steel industry within the company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Ellwood City Area School District (inc)</b> (Ellwood City, PA) | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6000</td><td>70%</td><td>0.4200</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6656</td><td>30%</td><td>0.1997</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6197</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• ellwood<br><br><b>Your Search Also Includes:</b><br>• group, inc<br><br><b>Company Name Also Includes:</b><br>• (inc), area, city, district, school<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6000 (Weight: 70%)<br>• Semantic Similarity: 0.6656 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> Ellwood Group Inc and Ellwood City Area School District (inc) are the same entity, a school district, with Ellwood being a branch or location of the main organization.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have corporate indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 14. American Miniature Horse Registry

**Query:** `American Miniature Horse Registry` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** American Miniature Horse Association (Alvarado, TX) • **Score:** 90.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 0.5919 | 30% | 0.1776 |
| Semantic Similarity (Raw) | 3.7158 | - | - |
| **Base Score** | **0.7682** | - | - |
| **FINAL SCORE** | **0.9000** | - | **90.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9000
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse, miniature<br><br><b>Your Search Also Includes:</b><br>• registry<br><br><b>Company Name Also Includes:</b><br>• association<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8438 (Weight: 70%)<br>• Semantic Similarity: 0.5919 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> The American Miniature Horse Registry is the same entity as the American Miniature Horse Association, which is a branch of the larger organization.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>American Miniature Horse Association</b> (Alvarado, TX) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8438</td><td>70%</td><td>0.5906</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5919</td><td>30%</td><td>0.1776</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7682</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse, miniature<br><br><b>Your Search Also Includes:</b><br>• registry<br><br><b>Company Name Also Includes:</b><br>• association<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8438 (Weight: 70%)<br>• Semantic Similarity: 0.5919 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> The American Miniature Horse Registry is the same entity as the American Miniature Horse Association, which is a branch of the larger organization.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>American Miniature Horse Association Headquarters</b> | Score: <b>76.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7670</td><td>70%</td><td>0.5369</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7608</td><td>30%</td><td>0.2282</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7652</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7650</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse, miniature<br><br><b>Your Search Also Includes:</b><br>• registry<br><br><b>Company Name Also Includes:</b><br>• association, headquarters<br><br><b>Match Strength:</b><br>• 60% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7670 (Weight: 70%)<br>• Semantic Similarity: 0.7608 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.76<br>• <b>AI Insight:</b> The American Miniature Horse Registry is the same entity as the American Miniature Horse Association Headquarters, sharing the industry context of miniature horse breeding and management.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Match indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>American Saddle Horse Association</b> | Score: <b>69.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5717</td><td>30%</td><td>0.1715</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6921</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6920</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse<br><br><b>Your Search Also Includes:</b><br>• miniature, registry<br><br><b>Company Name Also Includes:</b><br>• association, saddle<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5717 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.57<br>• <b>AI Insight:</b> The American Miniature Horse Registry is a related entity to the American Saddle Horse Association, as both organizations are involved in the care and management of miniature horses within their respective industries.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>American Horse Defense Fund</b> | Score: <b>68.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5554</td><td>30%</td><td>0.1666</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6872</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6870</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse<br><br><b>Your Search Also Includes:</b><br>• miniature, registry<br><br><b>Company Name Also Includes:</b><br>• defense, fund<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5554 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.56<br>• <b>AI Insight:</b> The American Miniature Horse Registry is a non-profit organization that serves as a branch of the American Miniature Horse Association, which is an industry-specific entity focused on promoting and preserving miniature horse breeding and ownership.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Miniature Horse & Pony Show</b> (Farr West, UT) | Score: <b>68.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5432</td><td>30%</td><td>0.1629</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6836</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6840</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• horse, miniature<br><br><b>Your Search Also Includes:</b><br>• american, registry<br><br><b>Company Name Also Includes:</b><br>• &, pony, show<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5432 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.54<br>• <b>AI Insight:</b> The American Miniature Horse Registry is an entity that shares the same industry context as the Miniature Horse & Pony Show, which is related to equestrian activities and competitions.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>American Youth & Horse Council</b> | Score: <b>68.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5377</td><td>30%</td><td>0.1613</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6819</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6820</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse<br><br><b>Your Search Also Includes:</b><br>• miniature, registry<br><br><b>Company Name Also Includes:</b><br>• &, council, youth<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5377 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.54<br>• <b>AI Insight:</b> The American Youth & Horse Council is the same entity as the American Miniature Horse Registry, serving as a branch of the organization that oversees and promotes miniature horses in the United States.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>American Youth Horse Council</b> (Lexington, KY) | Score: <b>67.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5064</td><td>30%</td><td>0.1519</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6726</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0065</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6790</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse<br><br><b>Your Search Also Includes:</b><br>• miniature, registry<br><br><b>Company Name Also Includes:</b><br>• council, youth<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5064 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.51<br>• <b>AI Insight:</b> The American Youth Horse Council is the same entity as the American Miniature Horse Registry, serving as a branch of the organization that oversees and promotes miniature horses in the United States.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.65%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>American Youth Horse Council</b> (Storrs, CT) | Score: <b>67.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5262</td><td>30%</td><td>0.1578</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6785</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6780</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse<br><br><b>Your Search Also Includes:</b><br>• miniature, registry<br><br><b>Company Name Also Includes:</b><br>• council, youth<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5262 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.53<br>• <b>AI Insight:</b> The American Youth Horse Council is the same entity as the American Miniature Horse Registry, serving as a branch of the organization that oversees and promotes miniature horses in the United States.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>American Miniature Hores Association</b> | Score: <b>67.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5243</td><td>30%</td><td>0.1573</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6779</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6780</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, miniature<br><br><b>Your Search Also Includes:</b><br>• horse, registry<br><br><b>Company Name Also Includes:</b><br>• association, hores<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5243 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.52<br>• <b>AI Insight:</b> The American Miniature Horse Registry is the same entity as the American Miniature Horses Association, which is a shared industry context within the equine industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>American Horse Show Association</b> (Lexington, KY) | Score: <b>67.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5054</td><td>30%</td><td>0.1516</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6723</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6720</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, horse<br><br><b>Your Search Also Includes:</b><br>• miniature, registry<br><br><b>Company Name Also Includes:</b><br>• association, show<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.5054 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.51<br>• <b>AI Insight:</b> The American Miniature Horse Registry is a related entity to the American Horse Show Association, as both organizations are involved in the care and management of miniature horses within their respective industries.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates non_profit, partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 15. YADA ENTERPRISES, INC

**Query:** `YADA ENTERPRISES, INC` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Yada Yada (Kirkland, WA) • **Score:** 95.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.9000 | 70% | 0.6300 |
| Semantic Similarity (Normalized) | 0.7863 | 30% | 0.2359 |
| Semantic Similarity (Raw) | 3.3808 | - | - |
| **Base Score** | **0.8659** | - | - |
| **FINAL SCORE** | **0.9500** | - | **95.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9500
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• yada<br><br><b>Your Search Also Includes:</b><br>• enterprises,, inc<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.9000 (Weight: 70%)<br>• Semantic Similarity: 0.7863 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> Yada Yada Enterprises, Inc is the same entity as Yada Yada.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Yada Yada</b> (Kirkland, WA) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.9000</td><td>70%</td><td>0.6300</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7863</td><td>30%</td><td>0.2359</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8659</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• yada<br><br><b>Your Search Also Includes:</b><br>• enterprises,, inc<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.9000 (Weight: 70%)<br>• Semantic Similarity: 0.7863 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> Yada Yada Enterprises, Inc is the same entity as Yada Yada.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>Yasuda Corporation Limited</b> | Score: <b>53.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3600</td><td>70%</td><td>0.2520</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9295</td><td>30%</td><td>0.2788</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5308</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5310</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.53)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3600 (Weight: 70%)<br>• Semantic Similarity: 0.9295 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Fair (50-59%) - Some semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Yama</b> | Score: <b>52.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3375</td><td>70%</td><td>0.2362</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9544</td><td>30%</td><td>0.2863</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5226</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5230</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.52)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3375 (Weight: 70%)<br>• Semantic Similarity: 0.9544 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Fair (50-59%) - Some semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>Yama Group</b> | Score: <b>51.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3375</td><td>70%</td><td>0.2362</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9225</td><td>30%</td><td>0.2768</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5130</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5130</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.51)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3375 (Weight: 70%)<br>• Semantic Similarity: 0.9225 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Fair (50-59%) - Some semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Ya</b> | Score: <b>51.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3000</td><td>70%</td><td>0.2100</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5100</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5100</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
LINGUISTIC MATCH<br><br><b>What This Means:</b><br>The names look different but are linguistically related.<br><br><b>Key Relationships Found:</b><br>• 'yada' ↔ 'ya' (abbreviation/expansion)<br><br><b>Details:</b><br>• 'ya' is abbreviation of 'yada'<br><br><b>Real-World Scenario:</b><br>• Your system has the short form 'ya' but someone wrote 'yada'<br><br><b>Action Required:</b><br>• Verify if this variation makes sense<br>• Likely the same company<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> Ya Enterprises, Inc and Ya are the same entity, a branch of the company YADA ENTERPRISES, INC.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Yara</b> | Score: <b>50.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3375</td><td>70%</td><td>0.2362</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9008</td><td>30%</td><td>0.2702</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5065</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5060</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.51)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3375 (Weight: 70%)<br>• Semantic Similarity: 0.9008 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Fair (50-59%) - Some semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>YATA</b> | Score: <b>50.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3375</td><td>70%</td><td>0.2362</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8827</td><td>30%</td><td>0.2648</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5011</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5010</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.50)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3375 (Weight: 70%)<br>• Semantic Similarity: 0.8827 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Fair (50-59%) - Some semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Yama Enterprises</b> (Smyrna, GA) | Score: <b>50.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3375</td><td>70%</td><td>0.2362</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8804</td><td>30%</td><td>0.2641</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5004</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
LINGUISTIC MATCH<br><br><b>What This Means:</b><br>The names look different but are linguistically related.<br><br><b>Key Relationships Found:</b><br>• 'enterprises,' ↔ 'enterprises' (abbreviation/expansion)<br><br><b>Details:</b><br>• 'enterprises' is abbreviation of 'enterprises,'<br><br><b>Real-World Scenario:</b><br>• Your system has the short form 'enterprises' but someone wrote 'enterprises,'<br><br><b>Action Required:</b><br>• Verify if this variation makes sense<br>• Likely the same company<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.88<br>• <b>AI Insight:</b> Yama Enterprises and Yada Enterprises, Inc are the same entity, a branch of Yama Enterprises.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates corporate context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#9</b> | <b>Yamas</b> | Score: <b>49.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3375</td><td>70%</td><td>0.2362</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8744</td><td>30%</td><td>0.2623</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4986</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4990</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.50)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3375 (Weight: 70%)<br>• Semantic Similarity: 0.8744 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#10</b> | <b>YALLA</b> | Score: <b>46.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.3000</td><td>70%</td><td>0.2100</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8504</td><td>30%</td><td>0.2551</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4651</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4650</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.47)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.3000 (Weight: 70%)<br>• Semantic Similarity: 0.8504 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Poor (40-49%) - Weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>

---

## 16. Seafood Nutrition Partnership

**Query:** `Seafood Nutrition Partnership` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Seafood Nutrition Partnership • **Score:** 100.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.7515 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0000** | - | **100.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0000
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have partnership indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Seafood Nutrition Partnership</b> | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have partnership indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Seafood Nutrition Partnership</b> (Durham, CT) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7011</td><td>30%</td><td>0.2103</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9103</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> Seafood Nutrition Partnership is the same entity as Seafood Nutrition, a company that specializes in providing nutrition information and resources for seafood consumers.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have partnership indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Seafood Nutrition Partnership</b> (Bellevue, WA) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6643</td><td>30%</td><td>0.1993</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8993</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> Seafood Nutrition Partnership is the same entity as Seafood Nutrition, a company that specializes in providing nutrition information and resources for seafood consumers.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have partnership indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Sustainable Seafood Partnership</b> (Bellingham, WA) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5755</td><td>30%</td><td>0.1726</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7394</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• partnership, seafood<br><br><b>Your Search Also Includes:</b><br>• nutrition<br><br><b>Company Name Also Includes:</b><br>• sustainable<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.5755 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>AI Insight:</b> The Sustainable Seafood Partnership is the same entity as the company, Sustainable Seafood Partnership, which is a partnership focused on promoting sustainable seafood practices and providing nutrition information to consumers.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have partnership indicators<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>SEAFOOD NUTRITION</b> (ARLINGTON, VA) | Score: <b>71.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6360</td><td>30%</td><td>0.1908</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7158</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7160</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• nutrition, seafood<br><br><b>Your Search Also Includes:</b><br>• partnership<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 0.6360 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> SEAFOOD NUTRITION is the same entity as Seafood Nutrition Partnership, which indicates that they are likely related in some capacity within the seafood industry context.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Seafood Choices Alliance</b> | Score: <b>59.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7513</td><td>30%</td><td>0.2254</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5948</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5950</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• seafood<br><br><b>Your Search Also Includes:</b><br>• nutrition, partnership<br><br><b>Company Name Also Includes:</b><br>• alliance, choices<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.7513 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.75<br>• <b>AI Insight:</b> The Seafood Choices Alliance is the same entity as the Seafood Nutrition Partnership, indicating that they are likely related through a partnership or collaboration in the seafood industry context.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Pet Nutrition Alliance</b> | Score: <b>56.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6344</td><td>30%</td><td>0.1903</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5598</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• nutrition<br><br><b>Your Search Also Includes:</b><br>• partnership, seafood<br><br><b>Company Name Also Includes:</b><br>• alliance, pet<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.6344 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>AI Insight:</b> The Seafood Nutrition Partnership is a partnership between Pet Nutrition Alliance, which is an organization focused on providing nutrition solutions for pets, and the seafood industry, indicating that their shared context is in the provision of nutritional advice related to seafood consumption for pet owners.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>East Coast Seafood</b> | Score: <b>55.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6182</td><td>30%</td><td>0.1855</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5549</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5550</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• seafood<br><br><b>Your Search Also Includes:</b><br>• nutrition, partnership<br><br><b>Company Name Also Includes:</b><br>• coast, east<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.6182 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.62<br>• <b>AI Insight:</b> East Coast Seafood and East Coast Seafood Farms are the same entity, a branch of the company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>International Boston Seafood</b> | Score: <b>55.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6149</td><td>30%</td><td>0.1845</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5539</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5540</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• seafood<br><br><b>Your Search Also Includes:</b><br>• nutrition, partnership<br><br><b>Company Name Also Includes:</b><br>• boston, international<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.6149 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> International Boston Seafood and International Boston Seafood Nutrition Partnership are the same entity, a branch of the company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Different contexts: partnership vs international<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>North Sea Seafood</b> | Score: <b>55.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6134</td><td>30%</td><td>0.1840</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5535</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5530</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• seafood<br><br><b>Your Search Also Includes:</b><br>• nutrition, partnership<br><br><b>Company Name Also Includes:</b><br>• north, sea<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.6134 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> North Sea Seafood is the same entity as North Sea Seafood Nutrition Partnership, indicating that they are the same company with different names used in different contexts.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates partnership context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 17. AVIAKOMPANIYA SIBIR, PAO

**Query:** `AVIAKOMPANIYA SIBIR, PAO` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** AVIAKOMPANIYA MIZHNARODNI AVIA • **Score:** 59.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.5278 | 70% | 0.3694 |
| Semantic Similarity (Normalized) | 0.7340 | 30% | 0.2202 |
| Semantic Similarity (Raw) | 3.2724 | - | - |
| **Base Score** | **0.5897** | - | - |
| **FINAL SCORE** | **0.5897** | - | **59.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.5897
```

### Component Analysis

- **String Similarity (FAIR):** Some lexical similarity - partial word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• aviakompaniya<br><br><b>Your Search Also Includes:</b><br>• pao, sibir,<br><br><b>Company Name Also Includes:</b><br>• avia, mizhnarodni<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.7340 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.73<br>• <b>AI Insight:</b> AVIAKOMPANIYA SIBIR, PAO and AVIAKOMPANIYA MIZHNARODNI AVIA are the same entity.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#1</b> | <b>AVIAKOMPANIYA MIZHNARODNI AVIA</b> | Score: <b>59.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7340</td><td>30%</td><td>0.2202</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5897</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5900</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• aviakompaniya<br><br><b>Your Search Also Includes:</b><br>• pao, sibir,<br><br><b>Company Name Also Includes:</b><br>• avia, mizhnarodni<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.7340 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.73<br>• <b>AI Insight:</b> AVIAKOMPANIYA SIBIR, PAO and AVIAKOMPANIYA MIZHNARODNI AVIA are the same entity.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>AVIAKOMPANIYA MIZHNARODNI AVIA</b> (KYIV, ) | Score: <b>58.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6913</td><td>30%</td><td>0.2074</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5768</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0056</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5820</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• aviakompaniya<br><br><b>Your Search Also Includes:</b><br>• pao, sibir,<br><br><b>Company Name Also Includes:</b><br>• avia, mizhnarodni<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.6913 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> AVIAKOMPANIYA SIBIR, PAO and AVIAKOMPANIYA MIZHNARODNI AVIA are the same entity.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.56%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>AVIAKOMPANIYA AEROSVIT, PRYVAT</b> (SELO GORA, ) | Score: <b>53.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5278</td><td>70%</td><td>0.3694</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5625</td><td>30%</td><td>0.1688</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5382</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5380</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• aviakompaniya<br><br><b>Your Search Also Includes:</b><br>• pao, sibir,<br><br><b>Company Name Also Includes:</b><br>• aerosvit,, pryvat<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5278 (Weight: 70%)<br>• Semantic Similarity: 0.5625 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.56<br>• <b>AI Insight:</b> AVIAKOMPANIYA SIBIR, PAO and AVIAKOMPANIYA AEROSVIT, PRYVAT are the same entity, a Russian aviation company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#4</b> | <b>AVIAKOMPANIYA AEROSVIT, PRYVATNE AT</b> (SELO GORA, ) | Score: <b>48.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4798</td><td>70%</td><td>0.3359</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5095</td><td>30%</td><td>0.1529</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.4887</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.4890</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• aviakompaniya<br><br><b>Your Search Also Includes:</b><br>• pao, sibir,<br><br><b>Company Name Also Includes:</b><br>• aerosvit,, at, pryvatne<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4798 (Weight: 70%)<br>• Semantic Similarity: 0.5095 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.51<br>• <b>AI Insight:</b> AVIAKOMPANIYA SIBIR, PAO and AVIAKOMPANIYA AEROSVIT, PRYVATNE AT are the same entity.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#5</b> | <b>Shibir  Desai</b> | Score: <b>34.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.1636</td><td>70%</td><td>0.1145</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7685</td><td>30%</td><td>0.2305</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.3451</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.3450</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.34)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.1636 (Weight: 70%)<br>• Semantic Similarity: 0.7685 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Very Poor (30-39%) - Very weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#6</b> | <b>AVIPAM Sao Paulo</b> | Score: <b>33.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.2538</td><td>70%</td><td>0.1777</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5269</td><td>30%</td><td>0.1581</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.3358</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.3360</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.34)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.2538 (Weight: 70%)<br>• Semantic Similarity: 0.5269 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Very Poor (30-39%) - Very weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#7</b> | <b>Avyaya Integrated</b> | Score: <b>33.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.1841</td><td>70%</td><td>0.1289</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6823</td><td>30%</td><td>0.2047</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.3336</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.3340</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.33)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.1841 (Weight: 70%)<br>• Semantic Similarity: 0.6823 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Very Poor (30-39%) - Very weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#8</b> | <b>Salaha Kabir</b> | Score: <b>33.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.1636</td><td>70%</td><td>0.1145</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7287</td><td>30%</td><td>0.2186</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.3332</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.3330</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.33)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.1636 (Weight: 70%)<br>• Semantic Similarity: 0.7287 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Very Poor (30-39%) - Very weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#9</b> | <b>AMANDA MAHABIR</b> | Score: <b>33.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.1990</td><td>70%</td><td>0.1393</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6406</td><td>30%</td><td>0.1922</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.3315</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.3310</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SEMANTIC MATCH (Score: 0.33)<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.1990 (Weight: 70%)<br>• Semantic Similarity: 0.6406 (Weight: 30%)<br>• Popularity Boost: Log-weighted frequency<br><br><b>Confidence Level:</b><br>• Very Poor (30-39%) - Very weak semantic relationship<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🔴 <b>#10</b> | <b>Avia Sis</b> (Kiev, ) | Score: <b>33.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.1848</td><td>70%</td><td>0.1293</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6620</td><td>30%</td><td>0.1986</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.3279</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0032</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.3310</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
LINGUISTIC MATCH<br><br><b>What This Means:</b><br>The names look different but are linguistically related.<br><br><b>Key Relationships Found:</b><br>• 'aviakompaniya' ↔ 'avia' (abbreviation/expansion)<br><br><b>Details:</b><br>• 'avia' is abbreviation of 'aviakompaniya'<br><br><b>Real-World Scenario:</b><br>• Your system has the short form 'avia' but someone wrote 'aviakompaniya'<br><br><b>Action Required:</b><br>• Verify if this variation makes sense<br>• Likely the same company<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> Avia Sis is the same entity as Avia Sibir, PAO, which is an aviation company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.32%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>

---

## 18. Hartford Hospital School of Nursing (Hartford, CT)

**Query:** `Hartford Hospital School of Nursing` • **Location:** Hartford, CT • **Self-Match:** ✅ Found & Filtered

**Top Match:** Hartford Hospital School of Nursing (Wethersfield, CT) • **Score:** 102.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.7613 | 30% | 0.2284 |
| Semantic Similarity (Raw) | 4.4633 | - | - |
| **Base Score** | **0.9284** | - | - |
| Location Context Boost | 40.0000 | 5% max | +2.0000 |
| **FINAL SCORE** | **1.0200** | - | **102.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 40.0000 × 0.05 = 2.0000

Final Score = Base Score + Location Boost = 1.0200
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 40.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.76<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing is the same entity as Hartford Hospital, which is a hospital and healthcare provider in Connecticut, USA, sharing an industry context with the nursing education sector.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Wethersfield) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Hartford Hospital School of Nursing</b> (Wethersfield, CT) | Score: <b>102.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7613</td><td>30%</td><td>0.2284</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9284</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.76<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing is the same entity as Hartford Hospital, which is a hospital and healthcare provider in Connecticut, USA, sharing an industry context with the nursing education sector.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+2.00%)</span>. The record's location (Wethersfield) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Hartford Public High School</b> (Hartford, Ct) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7692</td><td>30%</td><td>0.2308</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8258</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• high, public<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.7692 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.77<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing is an educational institution within Hartford Public High School, which serves as the primary provider of nursing education in the city of Hartford.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Hartford Magnet Middle School</b> (Hartford, CT) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6747</td><td>30%</td><td>0.2024</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7974</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• magnet, middle<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.6747 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing and Hartford Magnet Middle School are the same entity, a school within the same educational institution chain.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>East Hartford Middle School</b> (East Hartford, CT) | Score: <b>91.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6898</td><td>30%</td><td>0.2070</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8020</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.1862</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0087</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9150</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• east, middle<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.6898 (Weight: 30%)<br>• Location Bonus: +0.9308<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> The Hartford Hospital School of Nursing and East Hartford Middle School are unrelated entities, as they represent different levels of education within the healthcare industry context.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+18.62%)</span>. The record's location (East Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Hartford Public High School</b> (West Hartford, CT) | Score: <b>90.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7360</td><td>30%</td><td>0.2208</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8158</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.1862</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9060</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• high, public<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.7360 (Weight: 30%)<br>• Location Bonus: +0.9308<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.74<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing is an educational institution within Hartford Public High School, which serves as the primary provider of nursing education in the city of Hartford.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+18.62%)</span>. The record's location (West Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>West Hartford Public School</b> (West Hartford, CT) | Score: <b>90.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7268</td><td>30%</td><td>0.2180</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8130</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.1862</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9060</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• public, west<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.7268 (Weight: 30%)<br>• Location Bonus: +0.9308<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.73<br>• <b>AI Insight:</b> The Hartford Hospital School of Nursing and West Hartford Public School are related as they both serve students in the education sector, with the school providing nursing programs at Hartford Hospital.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+18.62%)</span>. The record's location (West Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Hartford Hospital</b> (Hartford, CT) | Score: <b>86.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8250</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, hospital<br><br><b>Your Search Also Includes:</b><br>• nursing, of, school<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7500 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing is the same entity as Hartford Hospital, which is a healthcare organization providing medical services and facilities in Connecticut.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Hartford Union High School</b> (Hartford, WI) | Score: <b>84.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6911</td><td>30%</td><td>0.2073</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8023</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.1200</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8400</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• high, union<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.6911 (Weight: 30%)<br>• Location Bonus: +0.6000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing and Hartford Union High School are the same entity, a school within the same educational institution chain.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+12.00%)</span>. The record's location (Hartford) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Hartford School District</b> (Hartford, CT) | Score: <b>83.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7727</td><td>70%</td><td>0.5409</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8504</td><td>30%</td><td>0.2551</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7960</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8370</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• district<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7727 (Weight: 70%)<br>• Semantic Similarity: 0.8504 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.85<br>• <b>AI Insight:</b> The Hartford Hospital School of Nursing and the Hartford School District are the same entity, a branch or subsidiary of each other.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Hartford Elementary School</b> (Hartford, CT) | Score: <b>83.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7727</td><td>70%</td><td>0.5409</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8261</td><td>30%</td><td>0.2478</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7887</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8310</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• hartford, school<br><br><b>Your Search Also Includes:</b><br>• hospital, nursing, of<br><br><b>Company Name Also Includes:</b><br>• elementary<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7727 (Weight: 70%)<br>• Semantic Similarity: 0.8261 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.83<br>• <b>AI Insight:</b> Hartford Hospital School of Nursing and Hartford Elementary School are the same entity, a school within the same educational institution chain.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Query indicates healthcare context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Hartford) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 19. Internal J&J Meeting and Breakfast

**Query:** `Internal J&J Meeting and Breakfast` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Internal J&J Meeting and Breakfast (Somerville, NJ) • **Score:** 100.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.8280 | 30% | 0.2484 |
| Semantic Similarity (Raw) | 3.7422 | - | - |
| **Base Score** | **0.9484** | - | - |
| **FINAL SCORE** | **1.0000** | - | **100.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0000
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.83<br>• <b>AI Insight:</b> The Internal J&J Meeting and Breakfast is an entity that shares the same industry context as the company, which is pharmaceuticals.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Internal J&J Meeting and Breakfast</b> (Somerville, NJ) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8280</td><td>30%</td><td>0.2484</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9484</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.83<br>• <b>AI Insight:</b> The Internal J&J Meeting and Breakfast is an entity that shares the same industry context as the company, which is pharmaceuticals.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>ASCO Internal Pre Meeting</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5930</td><td>30%</td><td>0.1779</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7729</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• internal, meeting<br><br><b>Your Search Also Includes:</b><br>• and, breakfast, j&j<br><br><b>Company Name Also Includes:</b><br>• asco, pre<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.5930 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> The company ASCO Internal Pre Meeting is the same entity as the search query, which refers to an internal meeting and breakfast for professionals in the pharmaceutical industry related to the American Society of Clinical Oncology (ASCO).<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>IT Management Internal Meeting</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8500</td><td>70%</td><td>0.5950</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5585</td><td>30%</td><td>0.1675</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7625</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• internal, meeting<br><br><b>Your Search Also Includes:</b><br>• and, breakfast, j&j<br><br><b>Company Name Also Includes:</b><br>• it, management<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8500 (Weight: 70%)<br>• Semantic Similarity: 0.5585 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.56<br>• <b>AI Insight:</b> The IT Management Internal Meeting is the same entity as the internal J&J meeting and breakfast, as it appears to be an in-house event within the company's management structure.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>Atea Internal Meeting</b> | Score: <b>74.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7727</td><td>70%</td><td>0.5409</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6709</td><td>30%</td><td>0.2013</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7422</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7420</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• internal, meeting<br><br><b>Your Search Also Includes:</b><br>• and, breakfast, j&j<br><br><b>Company Name Also Includes:</b><br>• atea<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7727 (Weight: 70%)<br>• Semantic Similarity: 0.6709 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The Atea Internal Meeting is related to the same industry context as J&J, specifically in the pharmaceuticals and medical devices sector.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Nov. Internal Meeting</b> | Score: <b>70.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7727</td><td>70%</td><td>0.5409</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5493</td><td>30%</td><td>0.1648</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7057</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7060</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• internal, meeting<br><br><b>Your Search Also Includes:</b><br>• and, breakfast, j&j<br><br><b>Company Name Also Includes:</b><br>• nov.<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7727 (Weight: 70%)<br>• Semantic Similarity: 0.5493 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.55<br>• <b>AI Insight:</b> The search query "Internal J&J Meeting and Breakfast" is related to the company Nov. Internal Meeting, as it appears to be an internal meeting of some sort, likely in the context of a company's operations or organizational structure.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Internal Meeting</b> | Score: <b>68.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6667</td><td>70%</td><td>0.4667</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7183</td><td>30%</td><td>0.2155</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6822</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6820</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• internal, meeting<br><br><b>Your Search Also Includes:</b><br>• and, breakfast, j&j<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6667 (Weight: 70%)<br>• Semantic Similarity: 0.7183 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.72<br>• <b>AI Insight:</b> The search query "Internal J&J Meeting and Breakfast" is related to the company Internal Meeting, as it appears to be a meeting scheduled within an internal organization of Johnson & Johnson (J&J), likely in the context of a business or operational discussion.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Greg Tolliver Breakfast Meeting</b> | Score: <b>65.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6375</td><td>70%</td><td>0.4462</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6781</td><td>30%</td><td>0.2034</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6497</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• breakfast, meeting<br><br><b>Your Search Also Includes:</b><br>• and, internal, j&j<br><br><b>Company Name Also Includes:</b><br>• greg, tolliver<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6375 (Weight: 70%)<br>• Semantic Similarity: 0.6781 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> The Greg Tolliver Breakfast Meeting is an internal meeting and breakfast event, which suggests it is related to the company's operations or employee engagement in the industry of food service or hospitality.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>GMCVB Breakfast & Meeting</b> | Score: <b>64.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5795</td><td>70%</td><td>0.4057</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8116</td><td>30%</td><td>0.2435</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6492</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6490</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• breakfast, meeting<br><br><b>Your Search Also Includes:</b><br>• and, internal, j&j<br><br><b>Company Name Also Includes:</b><br>• &, gmcvb<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5795 (Weight: 70%)<br>• Semantic Similarity: 0.8116 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> The GMCVB Breakfast & Meeting is the same entity as the Internal J&J Meeting and Breakfast, indicating that they are related to the same company.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>DXC Technology Breakfast Meeting</b> | Score: <b>64.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6375</td><td>70%</td><td>0.4462</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6678</td><td>30%</td><td>0.2003</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6466</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6470</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• breakfast, meeting<br><br><b>Your Search Also Includes:</b><br>• and, internal, j&j<br><br><b>Company Name Also Includes:</b><br>• dxc, technology<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6375 (Weight: 70%)<br>• Semantic Similarity: 0.6678 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The search query "Internal J&J Meeting and Breakfast" is related to the company DXC Technology, as it mentions a breakfast meeting, which suggests a business or professional context that may involve DXC's services or expertise in the healthcare industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates technology context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Optos Breakfast Meeting</b> | Score: <b>64.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.5795</td><td>70%</td><td>0.4057</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7947</td><td>30%</td><td>0.2384</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6441</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6440</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• breakfast, meeting<br><br><b>Your Search Also Includes:</b><br>• and, internal, j&j<br><br><b>Company Name Also Includes:</b><br>• optos<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.5795 (Weight: 70%)<br>• Semantic Similarity: 0.7947 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> The Optos Breakfast Meeting is an internal meeting and breakfast event for J&J, a pharmaceutical company, which shares the same industry context as the Optos organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 20. Spina Bifida Coalition of Cincinnati (Cincinnati, OH)

**Query:** `Spina Bifida Coalition of Cincinnati` • **Location:** Cincinnati, OH • **Self-Match:** ✅ Found & Filtered

**Top Match:** Spina Bifida Association of Cincinnati, Inc. (Cincinnati, OH) • **Score:** 92.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 0.7990 | 30% | 0.2397 |
| Semantic Similarity (Raw) | 4.0340 | - | - |
| **Base Score** | **0.8303** | - | - |
| Location Context Boost | 100.0000 | 5% max | +5.0000 |
| **FINAL SCORE** | **0.9200** | - | **92.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)
Location Boost = Location Score × 0.05 = 100.0000 × 0.05 = 5.0000

Final Score = Base Score + Location Boost = 0.9200
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection
- **Location Match (EXCELLENT):** 100.00 - Strong geographic match

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• bifida, of, spina<br><br><b>Your Search Also Includes:</b><br>• cincinnati, coalition<br><br><b>Company Name Also Includes:</b><br>• association, cincinnati,, inc.<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8438 (Weight: 70%)<br>• Semantic Similarity: 0.7990 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.80<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati and the Spina Bifida Association of Cincinnati, Inc. are the same entity, a non-profit organization, as they share the same name but operate under different names in different regions.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Spina Bifida Association of Cincinnati, Inc.</b> (Cincinnati, OH) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8438</td><td>70%</td><td>0.5906</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7990</td><td>30%</td><td>0.2397</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8303</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• bifida, of, spina<br><br><b>Your Search Also Includes:</b><br>• cincinnati, coalition<br><br><b>Company Name Also Includes:</b><br>• association, cincinnati,, inc.<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8438 (Weight: 70%)<br>• Semantic Similarity: 0.7990 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.80<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati and the Spina Bifida Association of Cincinnati, Inc. are the same entity, a non-profit organization, as they share the same name but operate under different names in different regions.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>SBCC</b> | Score: <b>75.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ACRONYM MATCH<br><br><b>What This Means:</b><br>The system identified a direct link between an exact acronym and its full company name.<br><br><b>Match Type:</b><br>• Acronym Reverse<br><br><b>Score Breakdown:</b><br>• Expansion Quality: 1.00<br>• Lexical match: 1.00<br>• Semantic link: 1.0000<br><br><b>Action Required:</b><br>• This is a HIGH CONFIDENCE acronym expansion<br>• Highly likely to be correct<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Query indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Greater Cincinnati Good Food Coalition</b> (Cincinnati, OH) | Score: <b>72.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6024</td><td>30%</td><td>0.1807</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6540</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7230</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• cincinnati, coalition<br><br><b>Your Search Also Includes:</b><br>• bifida, of, spina<br><br><b>Company Name Also Includes:</b><br>• food, good, greater<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 0.6024 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.60<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati and the Greater Cincinnati Good Food Coalition are related as both organizations are focused on improving the health and well-being of individuals with spina bifida, a congenital condition that affects the development of the spine.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>Alliance Cincinnati</b> (Cincinnati, OH) | Score: <b>63.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4062</td><td>70%</td><td>0.2844</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8481</td><td>30%</td><td>0.2544</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5388</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6310</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• cincinnati<br><br><b>Your Search Also Includes:</b><br>• bifida, coalition, of, spina<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4062 (Weight: 70%)<br>• Semantic Similarity: 0.8481 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.85<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati is the same entity as Alliance Cincinnati, serving as their parent organization and providing support to both entities within the broader context of healthcare and disability advocacy.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Different contexts: non_profit vs corporate<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Urban Appalachian Community Coalition</b> (Cincinnati, OH) | Score: <b>62.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4875</td><td>70%</td><td>0.3412</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6406</td><td>30%</td><td>0.1922</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5334</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6270</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• coalition<br><br><b>Your Search Also Includes:</b><br>• bifida, cincinnati, of, spina<br><br><b>Company Name Also Includes:</b><br>• appalachian, community, urban<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4875 (Weight: 70%)<br>• Semantic Similarity: 0.6406 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati is a non-profit organization, while the Urban Appalachian Community Coalition is an interagency coalition that serves as a regional partnership for social services and community development in Appalachia.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Health Alliance of Greater Cincinnati</b> (Cincinnati, OH) | Score: <b>62.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4875</td><td>70%</td><td>0.3412</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6220</td><td>30%</td><td>0.1866</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5278</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• cincinnati, of<br><br><b>Your Search Also Includes:</b><br>• bifida, coalition, spina<br><br><b>Company Name Also Includes:</b><br>• alliance, greater, health<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4875 (Weight: 70%)<br>• Semantic Similarity: 0.6220 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.62<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati is a non-profit organization, while Health Alliance of Greater Cincinnati is a health system and medical provider, indicating that they are unrelated in the industry context.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Different contexts: non_profit vs healthcare<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>AAA Allied Group Cincinnati</b> (Cincinnati, OH) | Score: <b>62.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4432</td><td>70%</td><td>0.3102</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7226</td><td>30%</td><td>0.2168</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5270</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• cincinnati<br><br><b>Your Search Also Includes:</b><br>• bifida, coalition, of, spina<br><br><b>Company Name Also Includes:</b><br>• aaa, allied, group<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4432 (Weight: 70%)<br>• Semantic Similarity: 0.7226 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.72<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati is a non-profit organization, while AAA Allied Group Cincinnati appears to be a private company or organization that may provide services related to the coalition's mission, but it is not directly affiliated with the coalition.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Different contexts: non_profit vs corporate<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Cincinnati Reds Baseball Club</b> (Cincinnati, OH) | Score: <b>62.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4875</td><td>70%</td><td>0.3412</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6185</td><td>30%</td><td>0.1855</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5268</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6210</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• cincinnati<br><br><b>Your Search Also Includes:</b><br>• bifida, coalition, of, spina<br><br><b>Company Name Also Includes:</b><br>• baseball, club, reds<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4875 (Weight: 70%)<br>• Semantic Similarity: 0.6185 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.62<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati is not the same entity as the Cincinnati Reds Baseball Club, but rather a non-profit organization focused on supporting individuals with spina bifida and their families, providing resources and advocacy in the context of healthcare and disability.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>Cincinnati North IMA Chapter</b> (Cincinnati, OH) | Score: <b>61.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4875</td><td>70%</td><td>0.3412</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6058</td><td>30%</td><td>0.1817</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5230</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6180</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• cincinnati<br><br><b>Your Search Also Includes:</b><br>• bifida, coalition, of, spina<br><br><b>Company Name Also Includes:</b><br>• chapter, ima, north<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4875 (Weight: 70%)<br>• Semantic Similarity: 0.6058 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> The Cincinnati North IMA Chapter of Spina Bifida Coalition of Cincinnati is a branch of the national organization, sharing an industry context as a professional association for individuals and families affected by spina bifida.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Different contexts: non_profit vs corporate<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>AAA Hartford Cincinnati</b> (Cincinnati, OH) | Score: <b>61.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4432</td><td>70%</td><td>0.3102</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6977</td><td>30%</td><td>0.2093</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5195</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6160</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• cincinnati<br><br><b>Your Search Also Includes:</b><br>• bifida, coalition, of, spina<br><br><b>Company Name Also Includes:</b><br>• aaa, hartford<br><br><b>Match Strength:</b><br>• 20% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4432 (Weight: 70%)<br>• Semantic Similarity: 0.6977 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> The Spina Bifida Coalition of Cincinnati is a non-profit organization, while AAA Hartford Cincinnati is an insurance company, indicating that they are unrelated in the industry context.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Different contexts: non_profit vs corporate<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Cincinnati) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 21. THE SOCA GROUP ORGANIZATION

**Query:** `THE SOCA GROUP ORGANIZATION` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Team SOCA • **Score:** 78.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.8942 | 30% | 0.2682 |
| Semantic Similarity (Raw) | 5.3999 | - | - |
| **Base Score** | **0.7889** | - | - |
| **FINAL SCORE** | **0.7889** | - | **78.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7889
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• soca<br><br><b>Your Search Also Includes:</b><br>• group, organization, the<br><br><b>Company Name Also Includes:</b><br>• team<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8942 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> The Team SOCA organization is the same entity as The Soc Group Organization, indicating that they are likely related in some capacity within the same industry context.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#1</b> | <b>Team SOCA</b> | Score: <b>78.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8942</td><td>30%</td><td>0.2682</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7889</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7890</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• soca<br><br><b>Your Search Also Includes:</b><br>• group, organization, the<br><br><b>Company Name Also Includes:</b><br>• team<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8942 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> The Team SOCA organization is the same entity as The Soc Group Organization, indicating that they are likely related in some capacity within the same industry context.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>Soca Society</b> | Score: <b>78.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8639</td><td>30%</td><td>0.2592</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7798</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7800</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• soca<br><br><b>Your Search Also Includes:</b><br>• group, organization, the<br><br><b>Company Name Also Includes:</b><br>• society<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8639 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.86<br>• <b>AI Insight:</b> The Soca Society is the same entity as The Soca Group Organization, indicating that they are related through a common name and context.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Match indicates non_profit context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Organization Management Group</b> | Score: <b>74.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7326</td><td>30%</td><td>0.2198</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7404</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7400</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• group, organization<br><br><b>Your Search Also Includes:</b><br>• soca, the<br><br><b>Company Name Also Includes:</b><br>• management<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7326 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.73<br>• <b>AI Insight:</b> The SOCA Group Organization and the Organization Management Group are related as both are part of the same industry, specifically in the management consulting sector.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>Four Organization</b> | Score: <b>73.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7056</td><td>30%</td><td>0.2117</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7323</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7320</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• organization<br><br><b>Your Search Also Includes:</b><br>• group, soca, the<br><br><b>Company Name Also Includes:</b><br>• four<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7056 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> The SOCA Group Organization and Four Organization are related as both are part of the same industry, specifically in the field of food processing and manufacturing.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>System Organization</b> | Score: <b>73.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6982</td><td>30%</td><td>0.2095</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7301</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7300</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• organization<br><br><b>Your Search Also Includes:</b><br>• group, soca, the<br><br><b>Company Name Also Includes:</b><br>• system<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6982 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> The SOCA Group Organization and System Organization appear to be the same entity, as both refer to the same company with varying names in different regions or languages.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Organization Management</b> | Score: <b>72.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6616</td><td>30%</td><td>0.1985</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7191</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0069</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7260</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• organization<br><br><b>Your Search Also Includes:</b><br>• group, soca, the<br><br><b>Company Name Also Includes:</b><br>• management<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6616 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> The SOCA Group Organization and Organization Management are the same entity, as they refer to the same company with varying names in different regions or languages.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.69%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Organization Meeting</b> | Score: <b>72.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6824</td><td>30%</td><td>0.2047</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7253</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7250</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• organization<br><br><b>Your Search Also Includes:</b><br>• group, soca, the<br><br><b>Company Name Also Includes:</b><br>• meeting<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6824 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> The SOCA Group Organization and the Organization Meeting are related as they both operate in the financial services sector, specifically within the context of banking and finance.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>International organization</b> | Score: <b>72.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6720</td><td>30%</td><td>0.2016</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7222</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• organization<br><br><b>Your Search Also Includes:</b><br>• group, soca, the<br><br><b>Company Name Also Includes:</b><br>• international<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6720 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The SOCA Group Organization is an international organization that shares its industry context with the United Kingdom, specifically in the field of security and counter-terrorism.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates international context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>social organization</b> | Score: <b>72.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6703</td><td>30%</td><td>0.2011</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7217</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• organization<br><br><b>Your Search Also Includes:</b><br>• group, soca, the<br><br><b>Company Name Also Includes:</b><br>• social<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6703 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The SOCA Group Organization is a social organization that operates in the non-profit sector, providing services and resources to various community groups and organizations across different regions.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Government Organization</b> | Score: <b>72.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6697</td><td>30%</td><td>0.2009</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7215</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7220</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• organization<br><br><b>Your Search Also Includes:</b><br>• group, soca, the<br><br><b>Company Name Also Includes:</b><br>• government<br><br><b>Match Strength:</b><br>• 25% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6697 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The SOCA Group Organization is a government organization that falls under the same industry context as the United States, specifically in the field of defense and security.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 22. Shiroyama Junior High School

**Query:** `Shiroyama Junior High School` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Brooks Junior High School • **Score:** 90.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8250 | 70% | 0.5775 |
| Semantic Similarity (Normalized) | 0.8103 | 30% | 0.2431 |
| Semantic Similarity (Raw) | 3.7100 | - | - |
| **Base Score** | **0.8206** | - | - |
| **FINAL SCORE** | **0.9000** | - | **90.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9000
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• brooks<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.8103 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> Shiroyama Junior High School and Brooks Junior High School are the same entity, a school branch.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Brooks Junior High School</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8103</td><td>30%</td><td>0.2431</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8206</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• brooks<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.8103 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> Shiroyama Junior High School and Brooks Junior High School are the same entity, a school branch.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Kenmore Junior High School</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7639</td><td>30%</td><td>0.2292</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8067</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• kenmore<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.7639 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.76<br>• <b>AI Insight:</b> Shiroyama Junior High School and Kenmore Junior High School are the same entity, a school branch.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Sargent Junior High School</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7491</td><td>30%</td><td>0.2247</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8022</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• sargent<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.7491 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.75<br>• <b>AI Insight:</b> Shiroyama Junior High School and Sargent Junior High School are the same entity, a school.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Greenspun Junior High School</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7066</td><td>30%</td><td>0.2120</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7895</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• greenspun<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.7066 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> Shiroyama Junior High School and Greenspun Junior High School are the same entity, a school branch.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Junior High School #275</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7037</td><td>30%</td><td>0.2111</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7886</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• #275<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.7037 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> Shiroyama Junior High School is the same entity as Junior High School #275, a school branch within Shiroyama Junior High School's larger organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Nimitz Junior High School</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6846</td><td>30%</td><td>0.2054</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7829</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• nimitz<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.6846 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> Shiroyama Junior High School and Nimitz Junior High School are the same entity, a school, with Shiroyama being one of its branches or locations.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>CARROLL JUNIOR HIGH SCHOOL</b> (Southlake, TX) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6792</td><td>30%</td><td>0.2038</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7813</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• carroll<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.6792 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> Shiroyama Junior High School and Carroll Junior High School are the same entity, a school branch.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Junior High School 45</b> (New York, NY) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6791</td><td>30%</td><td>0.2037</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7812</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• 45<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.6791 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> Shiroyama Junior High School is the same entity as Junior High School 45, a school branch within the same industry context of education services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Hardin Junior High School</b> (Hardin, TX) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6769</td><td>30%</td><td>0.2031</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7806</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• hardin<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.6769 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> Shiroyama Junior High School and Hardin Junior High School are the same entity, a school branch.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Frontier Junior High School</b> (Graham, WA) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6748</td><td>30%</td><td>0.2024</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7799</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• high, junior, school<br><br><b>Your Search Also Includes:</b><br>• shiroyama<br><br><b>Company Name Also Includes:</b><br>• frontier<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.6748 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> Shiroyama Junior High School and Frontier Junior High School are the same entity, a school branch.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 23. National Home Health (Washington, DC)

**Query:** `National Home Health` • **Location:** Washington, DC • **Self-Match:** ✅ Found & Filtered

**Top Match:** National Home Health (Herndon, VA) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.5904 | 30% | 0.1771 |
| Semantic Similarity (Raw) | 3.7280 | - | - |
| **Base Score** | **0.8771** | - | - |
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> National Home Health is the same entity as National Home Health Care, a healthcare service provider that offers home-based care services to patients.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>National Home Health</b> (Herndon, VA) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5904</td><td>30%</td><td>0.1771</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8771</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> National Home Health is the same entity as National Home Health Care, a healthcare service provider that offers home-based care services to patients.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>National Home Health</b> | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>National Home Health Care</b> (Washington, DC) | Score: <b>96.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8225</td><td>70%</td><td>0.5758</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7439</td><td>30%</td><td>0.2232</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7989</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0092</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9690</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'National Home Health' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.74<br>• <b>AI Insight:</b> National Home Health Care and National Home Health are the same entity, a branch of the larger company, Home Healthcare Services.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Washington) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.92%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Legacy Home Health Care</b> (Washington, DC) | Score: <b>92.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8030</td><td>70%</td><td>0.5621</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6663</td><td>30%</td><td>0.1999</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7620</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0087</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9290</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• health, home<br><br><b>Your Search Also Includes:</b><br>• national<br><br><b>Company Name Also Includes:</b><br>• care, legacy<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8030 (Weight: 70%)<br>• Semantic Similarity: 0.6663 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> Legacy Home Health Care is the same entity as National Home Health, serving as their parent company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Washington) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>National Association of Home Care</b> (Washington, DC) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8030</td><td>70%</td><td>0.5621</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5686</td><td>30%</td><td>0.1706</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7327</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• home, national<br><br><b>Your Search Also Includes:</b><br>• health<br><br><b>Company Name Also Includes:</b><br>• association, care, of<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8030 (Weight: 70%)<br>• Semantic Similarity: 0.5686 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.57<br>• <b>AI Insight:</b> National Home Health and National Association of Home Care are related as National Home Health is a branch or affiliate of the National Association of Home Care, indicating they operate within the same industry context of home care services.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Washington) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Human Touch Home Health</b> (Washington, DC) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8030</td><td>70%</td><td>0.5621</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5533</td><td>30%</td><td>0.1660</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7281</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• health, home<br><br><b>Your Search Also Includes:</b><br>• national<br><br><b>Company Name Also Includes:</b><br>• human, touch<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8030 (Weight: 70%)<br>• Semantic Similarity: 0.5533 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.55<br>• <b>AI Insight:</b> National Home Health and Human Touch Home Health are the same entity, a branch of National Healthcare Corporation.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Washington) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Community Home Health</b> (Arlington, VA) | Score: <b>80.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8833</td><td>70%</td><td>0.6183</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6849</td><td>30%</td><td>0.2055</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8238</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.0884</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8080</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• health, home<br><br><b>Your Search Also Includes:</b><br>• national<br><br><b>Company Name Also Includes:</b><br>• community<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8833 (Weight: 70%)<br>• Semantic Similarity: 0.6849 (Weight: 30%)<br>• Location Bonus: +0.4421<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> National Home Health and Community Home Health are the same entity, a branch of National Health Care Corporation.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+8.84%)</span>. The record's location (Arlington) is in the same region/state, providing a partial boost.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>National Home Health Care</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8225</td><td>70%</td><td>0.5758</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8923</td><td>30%</td><td>0.2677</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8435</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'National Home Health' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> National Home Health Care and National Home Health are the same entity, a branch of the larger company, Home Healthcare Services.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>National Home Health Holiday Party</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7500</td><td>70%</td><td>0.5250</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7499</td><td>30%</td><td>0.2250</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7500</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'National Home Health' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.75<br>• <b>AI Insight:</b> The National Home Health and National Home Health Holiday Party are the same entity, a company, as they both refer to the same organization that provides home health care services.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>National Association of Home Health Care Providers</b> | Score: <b>76.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7308</td><td>70%</td><td>0.5115</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6530</td><td>30%</td><td>0.1959</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7074</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7600</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'National Home Health' was found in this company name.<br><br><b>Matching Words:</b><br>• health, home, national<br><br><b>Company Name Also Includes:</b><br>• association, care, of, providers<br><br><b>Match Strength:</b><br>• 43% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7308 (Weight: 70%)<br>• Semantic Similarity: 0.6530 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The National Association of Home Health Care Providers is the same entity as National Home Health, which is a search query that targets this company in the industry context.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have healthcare indicators - strong alignment<br>• <b>Context:</b> Both in healthcare industry - strong industry alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 24. American News Women's Club

**Query:** `American News Women's Club` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Danish-American Women's Club (San Jose, CA) • **Score:** 90.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8438 | 70% | 0.5906 |
| Semantic Similarity (Normalized) | 0.5520 | 30% | 0.1656 |
| Semantic Similarity (Raw) | 2.9119 | - | - |
| **Base Score** | **0.7562** | - | - |
| **FINAL SCORE** | **0.9000** | - | **90.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9000
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, women's<br><br><b>Your Search Also Includes:</b><br>• american, news<br><br><b>Company Name Also Includes:</b><br>• danish-american<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8438 (Weight: 70%)<br>• Semantic Similarity: 0.5520 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.55<br>• <b>AI Insight:</b> The American News Women's Club and Danish-American Women's Club are related as they are both organizations focused on promoting women's education, health, and community development in the United States, with the latter being a branch of the former.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Danish-American Women's Club</b> (San Jose, CA) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8438</td><td>70%</td><td>0.5906</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5520</td><td>30%</td><td>0.1656</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7562</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, women's<br><br><b>Your Search Also Includes:</b><br>• american, news<br><br><b>Company Name Also Includes:</b><br>• danish-american<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8438 (Weight: 70%)<br>• Semantic Similarity: 0.5520 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.55<br>• <b>AI Insight:</b> The American News Women's Club and Danish-American Women's Club are related as they are both organizations focused on promoting women's education, health, and community development in the United States, with the latter being a branch of the former.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>American Slavic Women's Club</b> (Maple Valley, WA) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8438</td><td>70%</td><td>0.5906</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5476</td><td>30%</td><td>0.1643</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7549</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, club, women's<br><br><b>Your Search Also Includes:</b><br>• news<br><br><b>Company Name Also Includes:</b><br>• slavic<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8438 (Weight: 70%)<br>• Semantic Similarity: 0.5476 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.55<br>• <b>AI Insight:</b> The American News Women's Club and the American Slavic Women's Club are related entities, as they share the same industry context of promoting women's news and awareness in their respective regions.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>American Women's Club</b> | Score: <b>79.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7670</td><td>70%</td><td>0.5369</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8713</td><td>30%</td><td>0.2614</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7983</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7980</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, club, women's<br><br><b>Your Search Also Includes:</b><br>• news<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7670 (Weight: 70%)<br>• Semantic Similarity: 0.8713 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.87<br>• <b>AI Insight:</b> The American Women's Club is the same entity as the American News Women's Club, indicating that they are likely related branches or chapters of the same organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>American Women Club</b> | Score: <b>77.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7733</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7730</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, club<br><br><b>Your Search Also Includes:</b><br>• news, women's<br><br><b>Company Name Also Includes:</b><br>• women<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 1.0000 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>AI Insight:</b> The American Women's Club is the same entity as the American News Women's Club, indicating that they are related branches of the same organization within the industry context of women's news and journalism.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Indo American Press Club</b> | Score: <b>72.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6681</td><td>30%</td><td>0.2004</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7210</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7210</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• american, club<br><br><b>Your Search Also Includes:</b><br>• news, women's<br><br><b>Company Name Also Includes:</b><br>• indo, press<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6681 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.67<br>• <b>AI Insight:</b> The American News Women's Club and Indo American Press Club are related as they both represent women in the media industry, with the former focusing on news and current events for American audiences and the latter covering news and issues relevant to the Indian-American community.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Thousand Oaks Women's Club</b> | Score: <b>71.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6585</td><td>30%</td><td>0.1976</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7182</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7180</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, women's<br><br><b>Your Search Also Includes:</b><br>• american, news<br><br><b>Company Name Also Includes:</b><br>• oaks, thousand<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6585 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> The American News Women's Club is the same entity as the Thousand Oaks Women's Club, sharing the same industry context of women's news and media.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Women's Club Board Meeting</b> | Score: <b>71.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6499</td><td>30%</td><td>0.1950</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7156</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7160</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, women's<br><br><b>Your Search Also Includes:</b><br>• american, news<br><br><b>Company Name Also Includes:</b><br>• board, meeting<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6499 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The American News Women's Club is a related entity to the Women's Club Board Meeting, as it falls within the same industry context of women's organizations and community service.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>San Jose Women's Club</b> | Score: <b>71.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6448</td><td>30%</td><td>0.1934</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7141</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7140</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, women's<br><br><b>Your Search Also Includes:</b><br>• american, news<br><br><b>Company Name Also Includes:</b><br>• jose, san<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6448 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> The American News Women's Club and San Jose Women's Club are related as both are women's clubs, with the former being a national organization focused on promoting women's news and journalism, and the latter being a local chapter of the same organization.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>Los Prados Women's Club</b> | Score: <b>71.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6431</td><td>30%</td><td>0.1929</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7136</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7140</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, women's<br><br><b>Your Search Also Includes:</b><br>• american, news<br><br><b>Company Name Also Includes:</b><br>• los, prados<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6431 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> The American News Women's Club and Los Prados Women's Club are related as both are women's clubs, with the former being a national organization focused on promoting women's news and journalism and the latter being a local chapter of the same organization.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>Houston Women's Book Club</b> | Score: <b>71.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6353</td><td>30%</td><td>0.1906</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7112</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7110</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, women's<br><br><b>Your Search Also Includes:</b><br>• american, news<br><br><b>Company Name Also Includes:</b><br>• book, houston<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6353 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.64<br>• <b>AI Insight:</b> The American News Women's Club and Houston Women's Book Club are related entities as they both operate in the same industry context of promoting women's interests, with the former focusing on news and current events for women and the latter on literature and book-related activities.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 25. Denise Roberge

**Query:** `Denise Roberge` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Denise Abril • **Score:** 81.1%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.9680 | 30% | 0.2904 |
| Semantic Similarity (Raw) | 4.0800 | - | - |
| **Base Score** | **0.8110** | - | - |
| **FINAL SCORE** | **0.8110** | - | **81.1%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.8110
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• abril<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9680 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.97<br>• <b>AI Insight:</b> Denise Roberge is Denise Abril.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Denise Abril</b> | Score: <b>81.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9680</td><td>30%</td><td>0.2904</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8110</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8110</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• abril<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9680 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.97<br>• <b>AI Insight:</b> Denise Roberge is Denise Abril.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Denise Beard</b> | Score: <b>81.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9661</td><td>30%</td><td>0.2898</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8104</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8100</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• beard<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9661 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.97<br>• <b>AI Insight:</b> Denise Roberge is Denise Beard.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Denise White</b> | Score: <b>80.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9625</td><td>30%</td><td>0.2887</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8094</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8090</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• white<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9625 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.96<br>• <b>AI Insight:</b> Denise Roberge is Denise White.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Charmaine Denise</b> | Score: <b>80.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9493</td><td>30%</td><td>0.2848</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8054</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8050</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• charmaine<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9493 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.95<br>• <b>AI Insight:</b> Denise Roberge is Charmaine Denise.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Denise Wallack</b> | Score: <b>80.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9399</td><td>30%</td><td>0.2820</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8026</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8030</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• wallack<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9399 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.94<br>• <b>AI Insight:</b> Denise Wallack is Denise Roberge.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>CeCi Denise</b> | Score: <b>80.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9373</td><td>30%</td><td>0.2812</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8018</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8020</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• ceci<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9373 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.94<br>• <b>AI Insight:</b> Denise Roberge is CeCi Denise.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Denise Ivy</b> | Score: <b>80.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9356</td><td>30%</td><td>0.2807</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8013</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.8010</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• ivy<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.9356 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.94<br>• <b>AI Insight:</b> Denise Roberge is Denise Ivy.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Denise Martin</b> | Score: <b>79.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8964</td><td>30%</td><td>0.2689</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7895</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7900</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• martin<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8964 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.90<br>• <b>AI Insight:</b> Denise Roberge is Denise Martin.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>sylvia denise</b> | Score: <b>78.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8875</td><td>30%</td><td>0.2662</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7869</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7870</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• sylvia<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8875 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> Denise Roberge is the same entity as Sylvia Denise, and they share the same industry context of cybersecurity.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>DENISE MILES</b> | Score: <b>78.5%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8798</td><td>30%</td><td>0.2639</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7846</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7850</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• denise<br><br><b>Your Search Also Includes:</b><br>• roberge<br><br><b>Company Name Also Includes:</b><br>• miles<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.8798 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.88<br>• <b>AI Insight:</b> Denise Roberge is Denise Miles, an American marketing executive and former Chief Marketing Officer at LinkedIn.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 26. Synergy Soccer Club

**Query:** `Synergy Soccer Club` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Nordic Soccer Club (Colchester, VT) • **Score:** 90.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8097 | 70% | 0.5668 |
| Semantic Similarity (Normalized) | 0.6126 | 30% | 0.1838 |
| Semantic Similarity (Raw) | 3.0237 | - | - |
| **Base Score** | **0.7506** | - | - |
| **FINAL SCORE** | **0.9087** | - | **90.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9087
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• nordic<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.6126 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> Nordic Soccer Club is the same entity as Synergy Soccer Club, indicating they are likely related through a shared business or organizational context within the sports industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Nordic Soccer Club</b> (Colchester, VT) | Score: <b>90.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6126</td><td>30%</td><td>0.1838</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7506</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0087</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9090</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• nordic<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.6126 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> Nordic Soccer Club is the same entity as Synergy Soccer Club, indicating they are likely related through a shared business or organizational context within the sports industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Alliance Soccer Club</b> (Reynoldsburg, OH) | Score: <b>90.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5931</td><td>30%</td><td>0.1779</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7447</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0087</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9090</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.5931 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> Synergy Soccer Club and Alliance Soccer Club are the same entity, a branch of the larger company, with no indication that they are unrelated in terms of industry context.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Club Ohio Soccer</b> (Dublin, OH) | Score: <b>90.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5791</td><td>30%</td><td>0.1737</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7405</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0087</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9090</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• ohio<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.5791 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.58<br>• <b>AI Insight:</b> Club Ohio Soccer and Synergy Soccer Club are related as they are both soccer clubs, with the latter being a subsidiary or branch of the former.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.87%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Synergy Volleyball Club</b> (Toronto, ON) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8526</td><td>30%</td><td>0.2558</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8226</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, synergy<br><br><b>Your Search Also Includes:</b><br>• soccer<br><br><b>Company Name Also Includes:</b><br>• volleyball<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.8526 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.85<br>• <b>AI Insight:</b> Synergy Soccer Club and Synergy Volleyball Club are the same entity, a branch of the Synergy organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Club Soccer Event</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8147</td><td>30%</td><td>0.2444</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8112</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• event<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.8147 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> Synergy Soccer Club and Club Soccer Event are related as they both operate in the sports event management industry, specifically focusing on organizing soccer events such as tournaments and leagues.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Sting Soccer Club</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7887</td><td>30%</td><td>0.2366</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8034</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• sting<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.7887 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> Synergy Soccer Club and Sting Soccer Club are related as they are the same entity, specifically referring to the same organization, with Synergy being an abbreviation for Sting Entertainment, which is a subsidiary of Sting Soccer Club.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>Magic Soccer Club</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7884</td><td>30%</td><td>0.2365</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8033</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• magic<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.7884 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> The Magic Soccer Club and Synergy Soccer Club are the same entity, a branch of the company, as they both operate under the umbrella of the same organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>International Soccer Club</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7795</td><td>30%</td><td>0.2339</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8007</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• international<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.7795 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>AI Insight:</b> The International Soccer Club is the same entity as Synergy Soccer Club, which suggests a potential synergy between the two companies in the sports industry context.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>Classic Soccer Club</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7775</td><td>30%</td><td>0.2332</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• classic<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.7775 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.78<br>• <b>AI Insight:</b> The search query "Synergy Soccer Club" and the company "Classic Soccer Club" are the same entity, as they refer to the same organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>United Soccer Club</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7726</td><td>30%</td><td>0.2318</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7986</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• club, soccer<br><br><b>Your Search Also Includes:</b><br>• synergy<br><br><b>Company Name Also Includes:</b><br>• united<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.7726 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.77<br>• <b>AI Insight:</b> United Soccer Club is the parent company of Synergy Soccer Club, which is their youth soccer academy and training facility.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 27. NFC Forum

**Query:** `NFC Forum` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** NFC Forum (Wakefield, MA) • **Score:** 100.4%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 0.6788 | 30% | 0.2036 |
| Semantic Similarity (Raw) | 4.8315 | - | - |
| **Base Score** | **0.9036** | - | - |
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
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> The NFC Forum is the same entity as itself, a non-profit organization that represents the interests of the Near Field Communication industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>NFC Forum</b> (Wakefield, MA) | Score: <b>100.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6788</td><td>30%</td><td>0.2036</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9036</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0039</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0040</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.68<br>• <b>AI Insight:</b> The NFC Forum is the same entity as itself, a non-profit organization that represents the interests of the Near Field Communication industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.39%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>NFC Forum</b> | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>NFC Forum</b> (Minneapolis, MN) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7228</td><td>30%</td><td>0.2168</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.9168</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.72<br>• <b>AI Insight:</b> The NFC Forum is the same entity as itself, a non-profit organization that represents the interests of the Near Field Communication industry.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>NFC Forum</b> (Woodville, WI) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6279</td><td>30%</td><td>0.1884</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8884</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>AI Insight:</b> The NFC Forum is the same entity as itself, a non-profit organization that represents the interests of the Near Field Communication industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>NFC Forum</b> (Escondido, CA) | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6099</td><td>30%</td><td>0.1830</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8830</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.61<br>• <b>AI Insight:</b> The NFC Forum is the same entity as itself, a non-profit organization that represents the interests of the Near Field Communication industry.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>NFC Forum         .</b> (Wakfield, MA) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6543</td><td>30%</td><td>0.1963</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8963</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'NFC Forum' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The NFC Forum is an industry association that represents and promotes the development of Near Field Communication (NFC) technology, providing a shared platform for collaboration and innovation in the context of mobile payments, identification, and other applications within the telecommunications and consumer electronics industries.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>NFC Forum Members</b> (Wakefield, MA) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8182</td><td>70%</td><td>0.5727</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6536</td><td>30%</td><td>0.1961</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7688</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'NFC Forum' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The NFC Forum is a membership-based organization that serves as a global community for the development and promotion of Near Field Communication (NFC) technology, sharing its industry context with various stakeholders in the mobile payments, retail, and logistics sectors.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>NFC Forum         .</b> (Wakefield, MA) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6322</td><td>30%</td><td>0.1896</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8896</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'NFC Forum' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.63<br>• <b>AI Insight:</b> The NFC Forum is an industry association that represents and promotes the development of Near Field Communication (NFC) technology, providing a shared platform for collaboration and innovation in the context of mobile payments, identification, and other applications within the telecommunications and consumer electronics industries.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>NFC Consulting</b> | Score: <b>73.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7055</td><td>30%</td><td>0.2117</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7323</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7320</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• nfc<br><br><b>Your Search Also Includes:</b><br>• forum<br><br><b>Company Name Also Includes:</b><br>• consulting<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.7055 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> NFC Consulting is the same entity as NFC Forum, a branch of NFC Consulting that specializes in providing industry-specific expertise and solutions for the Near Field Communication (NFC) technology.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Match indicates consulting context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>NFC</b> | Score: <b>70.8%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6300</td><td>70%</td><td>0.4410</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8903</td><td>30%</td><td>0.2671</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7081</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7080</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• nfc<br><br><b>Your Search Also Includes:</b><br>• forum<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6300 (Weight: 70%)<br>• Semantic Similarity: 0.8903 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> The NFC Forum is the same entity as the company NFC, indicating that they are related in the same industry context of wireless communication and networking standards.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 28. A Better Choice Limousine & Concierge

**Query:** `A Better Choice Limousine & Concierge` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** First Choice Limousine Services (Dorchester, MA) • **Score:** 71.6%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.7438 | 70% | 0.5206 |
| Semantic Similarity (Normalized) | 0.6505 | 30% | 0.1952 |
| Semantic Similarity (Raw) | 3.7201 | - | - |
| **Base Score** | **0.7158** | - | - |
| **FINAL SCORE** | **0.7158** | - | **71.6%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.7158
```

### Component Analysis

- **String Similarity (GOOD):** Moderate lexical match - significant word overlap
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• choice, limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, concierge<br><br><b>Company Name Also Includes:</b><br>• first, services<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6505 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as First Choice Limousine Services, a branch of the larger company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates consulting context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#1</b> | <b>First Choice Limousine Services</b> (Dorchester, MA) | Score: <b>71.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7438</td><td>70%</td><td>0.5206</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6505</td><td>30%</td><td>0.1952</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7158</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7160</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• choice, limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, concierge<br><br><b>Company Name Also Includes:</b><br>• first, services<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7438 (Weight: 70%)<br>• Semantic Similarity: 0.6505 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as First Choice Limousine Services, a branch of the larger company.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Match indicates consulting context<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#2</b> | <b>Better Choice Travel</b> | Score: <b>64.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5852</td><td>30%</td><td>0.1756</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6488</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6490</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• better, choice<br><br><b>Your Search Also Includes:</b><br>• &, a, concierge, limousine<br><br><b>Company Name Also Includes:</b><br>• travel<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 0.5852 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.59<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as Better Choice Travel, a travel agency providing transportation and concierge services to clients.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#3</b> | <b>Better Choice Travel</b> (Cleveland, OH) | Score: <b>63.2%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6761</td><td>70%</td><td>0.4733</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5084</td><td>30%</td><td>0.1525</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6258</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0060</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.6320</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• better, choice<br><br><b>Your Search Also Includes:</b><br>• &, a, concierge, limousine<br><br><b>Company Name Also Includes:</b><br>• travel<br><br><b>Match Strength:</b><br>• 33% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6761 (Weight: 70%)<br>• Semantic Similarity: 0.5084 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.51<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as Better Choice Travel, a travel agency providing transportation and concierge services to clients.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.60%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#4</b> | <b>Executive Limousine</b> | Score: <b>56.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4062</td><td>70%</td><td>0.2844</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.9300</td><td>30%</td><td>0.2790</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5634</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5630</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, choice, concierge<br><br><b>Company Name Also Includes:</b><br>• executive<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4062 (Weight: 70%)<br>• Semantic Similarity: 0.9300 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.93<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge and Executive Limousine are related entities in the same industry context, providing luxury transportation services to individuals and groups.<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#5</b> | <b>Chicago Limousine Transportation</b> | Score: <b>55.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4432</td><td>70%</td><td>0.3102</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8138</td><td>30%</td><td>0.2441</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5544</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5540</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, choice, concierge<br><br><b>Company Name Also Includes:</b><br>• chicago, transportation<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4432 (Weight: 70%)<br>• Semantic Similarity: 0.8138 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.81<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as Chicago Limousine Transportation, a branch of the company.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Journey Limousine</b> | Score: <b>55.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4062</td><td>70%</td><td>0.2844</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8943</td><td>30%</td><td>0.2683</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5527</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5530</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, choice, concierge<br><br><b>Company Name Also Includes:</b><br>• journey<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4062 (Weight: 70%)<br>• Semantic Similarity: 0.8943 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as Journey Limousine, a branch of the company.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Metropolitan Limousine</b> | Score: <b>55.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4062</td><td>70%</td><td>0.2844</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8879</td><td>30%</td><td>0.2664</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5508</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5510</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, choice, concierge<br><br><b>Company Name Also Includes:</b><br>• metropolitan<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4062 (Weight: 70%)<br>• Semantic Similarity: 0.8879 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> Metropolitan Limousine is the same entity as A Better Choice Limousine & Concierge, indicating that they are the same company with different names in various locations or branches.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>Greater Atlanta Limousine</b> | Score: <b>55.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4432</td><td>70%</td><td>0.3102</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8011</td><td>30%</td><td>0.2403</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5506</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5510</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, choice, concierge<br><br><b>Company Name Also Includes:</b><br>• atlanta, greater<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4432 (Weight: 70%)<br>• Semantic Similarity: 0.8011 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.80<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as Greater Atlanta Limousine, serving as a branch or subsidiary of the parent company.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>Alliance Limousine</b> | Score: <b>55.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4062</td><td>70%</td><td>0.2844</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8871</td><td>30%</td><td>0.2661</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5505</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, choice, concierge<br><br><b>Company Name Also Includes:</b><br>• alliance<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4062 (Weight: 70%)<br>• Semantic Similarity: 0.8871 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.89<br>• <b>AI Insight:</b> A Better Choice Limousine & Concierge is the same entity as Alliance Limousine, serving as their parent company and umbrella organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>My Limousine Service</b> | Score: <b>55.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.4432</td><td>70%</td><td>0.3102</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7988</td><td>30%</td><td>0.2396</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.5499</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.5500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>1 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• limousine<br><br><b>Your Search Also Includes:</b><br>• &, a, better, choice, concierge<br><br><b>Company Name Also Includes:</b><br>• my, service<br><br><b>Match Strength:</b><br>• 17% word overlap<br>• This is a WEAK match - may be coincidental<br>• Action: Verify carefully before using<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.4432 (Weight: 70%)<br>• Semantic Similarity: 0.7988 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.80<br>• <b>AI Insight:</b> My Limousine Service and A Better Choice Limousine & Concierge are related entities in the same industry context of providing luxury transportation services, specifically limousines and concierge-style travel experiences.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 29. Danish Sisterhood of America

**Query:** `Danish Sisterhood of America` • **Location:** None (name-only search) • **Self-Match:** ✅ Found & Filtered

**Top Match:** Danish Sisterhood and Brotherhood of America (Burbank, Ca) • **Score:** 95.9%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 0.8636 | 70% | 0.6045 |
| Semantic Similarity (Normalized) | 0.6466 | 30% | 0.1940 |
| Semantic Similarity (Raw) | 3.6582 | - | - |
| **Base Score** | **0.7985** | - | - |
| **FINAL SCORE** | **0.9592** | - | **95.9%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 0.9592
```

### Component Analysis

- **String Similarity (VERY GOOD):** Strong lexical match - most words align well
- **Semantic Similarity (GOOD):** Moderate meaning-based connection

</details>

**Match Rationale (Narrative):**  
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Danish Sisterhood of America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, danish, of, sisterhood<br><br><b>Company Name Also Includes:</b><br>• and, brotherhood<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.6466 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The Danish Sisterhood of America is a non-profit organization that shares an industry context with the Danish Sisterhood and Brotherhood of America, which is primarily involved in promoting and preserving traditional Scandinavian crafts and cultural heritage.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.92%)</span>. Frequency boost applied (2 occurrences).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Danish Sisterhood and Brotherhood of America</b> (Burbank, Ca) | Score: <b>95.9%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8636</td><td>70%</td><td>0.6045</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6466</td><td>30%</td><td>0.1940</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7985</em></td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0092</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9590</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search 'Danish Sisterhood of America' was found in this company name.<br><br><b>Matching Words:</b><br>• america, danish, of, sisterhood<br><br><b>Company Name Also Includes:</b><br>• and, brotherhood<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8636 (Weight: 70%)<br>• Semantic Similarity: 0.6466 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The Danish Sisterhood of America is a non-profit organization that shares an industry context with the Danish Sisterhood and Brotherhood of America, which is primarily involved in promoting and preserving traditional Scandinavian crafts and cultural heritage.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+0.92%)</span>. Frequency boost applied (2 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>The Danish Sisterhood of America</b> (Arvada, CO) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6630</td><td>30%</td><td>0.1989</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8989</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SUBSTRING MATCH<br><br><b>What This Means:</b><br>This company name contains 'Danish Sisterhood of America' somewhere within it.<br><br><b>Action Required:</b><br>• This is likely the same company<br>• Check if the surrounding words make sense<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered a partial company name<br>• Your system has the complete name<br>• Common when people remember only part of a company name<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> The Danish Sisterhood of America is a non-profit organization that shares an industry context with the same entity, The Danish Sisterhood of America, which is a charity and humanitarian organization focused on supporting women's rights and empowerment in Denmark.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>DANISH BROTHERHOOD AND DANISH SISTERHOOD OF AMERICA</b> (Palatine, IL) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8182</td><td>70%</td><td>0.5727</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6455</td><td>30%</td><td>0.1936</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7664</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SUBSTRING MATCH<br><br><b>What This Means:</b><br>This company name contains 'Danish Sisterhood of America' somewhere within it.<br><br><b>Action Required:</b><br>• This is likely the same company<br>• Check if the surrounding words make sense<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered a partial company name<br>• Your system has the complete name<br>• Common when people remember only part of a company name<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The Danish Sisterhood of America is a non-profit organization that is a branch of the larger Danish Brotherhood and Sisterhood of America, which is an international fraternal organization with a shared industry context in the insurance and financial services sectors.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Danish Brotherhood & Danish Sisterhood of America</b> (Hamden, CT) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8182</td><td>70%</td><td>0.5727</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5719</td><td>30%</td><td>0.1716</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7443</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
SUBSTRING MATCH<br><br><b>What This Means:</b><br>This company name contains 'Danish Sisterhood of America' somewhere within it.<br><br><b>Action Required:</b><br>• This is likely the same company<br>• Check if the surrounding words make sense<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered a partial company name<br>• Your system has the complete name<br>• Common when people remember only part of a company name<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.57<br>• <b>AI Insight:</b> The Danish Brotherhood & Danish Sisterhood of America is a branch of the same entity, the Danish Sisterhood of America.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Danish Sisterhood of America National Board Mtg</b> (Los Angeles, CA) | Score: <b>95.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6923</td><td>70%</td><td>0.4846</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5320</td><td>30%</td><td>0.1596</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6442</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9500</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with 'Danish Sisterhood of America' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.53<br>• <b>AI Insight:</b> The Danish Sisterhood of America is not the same entity as the National Board Mtg, but rather a branch or subgroup within it.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#6</b> | <b>Danish Sisterhood of the Americas</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8250</td><td>70%</td><td>0.5775</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.8532</td><td>30%</td><td>0.2560</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.8335</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• danish, of, sisterhood<br><br><b>Your Search Also Includes:</b><br>• america<br><br><b>Company Name Also Includes:</b><br>• americas, the<br><br><b>Match Strength:</b><br>• 60% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8250 (Weight: 70%)<br>• Semantic Similarity: 0.8532 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 0.85<br>• <b>AI Insight:</b> The Danish Sisterhood of America is a non-profit organization that shares an industry context with the Danish Sisterhood of the Americas, which is primarily involved in promoting and supporting the interests of Danish women living abroad.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#7</b> | <b>The Dansih Sisterhood of America</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7079</td><td>30%</td><td>0.2124</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7792</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• america, of, sisterhood<br><br><b>Your Search Also Includes:</b><br>• danish<br><br><b>Company Name Also Includes:</b><br>• dansih, the<br><br><b>Match Strength:</b><br>• 60% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.7079 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> The Danish Sisterhood of America is not the same entity as The Danish Sisterhood of America, but rather a branch or affiliate organization within the larger non-profit organization with a similar name.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#8</b> | <b>Danish Sisterhood of Amercia</b> (Mundelain, IL) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5316</td><td>30%</td><td>0.1595</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7263</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• danish, of, sisterhood<br><br><b>Your Search Also Includes:</b><br>• america<br><br><b>Company Name Also Includes:</b><br>• amercia<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.5316 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.53<br>• <b>AI Insight:</b> The Danish Sisterhood of America is a non-profit organization that shares an industry context with the Danish Sisterhood of Denmark, suggesting a shared connection in the field of cultural exchange and community development.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#9</b> | <b>The Dansih Sisterhood of America</b> (Arvada, CO) | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.5166</td><td>30%</td><td>0.1550</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7218</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• america, of, sisterhood<br><br><b>Your Search Also Includes:</b><br>• danish<br><br><b>Company Name Also Includes:</b><br>• dansih, the<br><br><b>Match Strength:</b><br>• 60% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.5166 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.52<br>• <b>AI Insight:</b> The Danish Sisterhood of America is not the same entity as The Danish Sisterhood of America, but rather a branch or affiliate organization within the larger non-profit organization with a similar name.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#10</b> | <b>Museum of Danish America</b> | Score: <b>90.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.4331</td><td>30%</td><td>0.1299</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.6967</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>3 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• america, danish, of<br><br><b>Your Search Also Includes:</b><br>• sisterhood<br><br><b>Company Name Also Includes:</b><br>• museum<br><br><b>Match Strength:</b><br>• 75% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.4331 (Weight: 30%)<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🔴 0.43<br>• <b>AI Insight:</b> The Danish Sisterhood of America is a non-profit organization, and the Museum of Danish America is a museum that showcases the history and culture of the Danish-American community, indicating they share an industry context in preserving and promoting cultural heritage.<br>• <b>Meaning:</b> The model detects a weak or incidental connection.<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

## 30. Brooklyn Comics Club (Brooklyn, NY)

**Query:** `Brooklyn Comics Club` • **Location:** Brooklyn, NY • **Self-Match:** ✅ Found & Filtered

**Top Match:** Brooklyn Comics Club • **Score:** 100.0%

<details>
<summary><b>📊 Scoring Breakdown</b></summary>

## Complete Score Breakdown

| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| String Similarity | 1.0000 | 70% | 0.7000 |
| Semantic Similarity (Normalized) | 1.0000 | 30% | 0.3000 |
| Semantic Similarity (Raw) | 6.6312 | - | - |
| **Base Score** | **1.0000** | - | - |
| **FINAL SCORE** | **1.0000** | - | **100.0%** |

### Score Calculation Formula

```
Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)

Final Score = Base Score = 1.0000
```

### Component Analysis

- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely
- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection

</details>

**Match Rationale (Narrative):**  
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>

<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#1</b> | <b>Brooklyn Comics Club</b> | Score: <b>100.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>1.0000</td><td>70%</td><td>0.7000</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>1.0000</td><td>30%</td><td>0.3000</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>1.0000</em></td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>1.0000</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟢 1.00<br>• <b>Meaning:</b> The model detects a very strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> N/A (No location context used).<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#2</b> | <b>Cathedral Club of Brooklyn</b> (Brooklyn, NY) | Score: <b>93.4%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6648</td><td>30%</td><td>0.1994</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7662</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr><td>Frequency Boost</td><td></td><td></td><td>+0.0142</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9340</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• cathedral, of<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.6648 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> The Cathedral Club of Brooklyn and Brooklyn Comics Club are related entities, as they share the same industry context of comic book-related businesses.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+1.42%)</span>. Frequency boost applied (5 occurrences).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#3</b> | <b>Brooklyn Barbell Club</b> (Brooklyn, NY) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7223</td><td>30%</td><td>0.2167</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7835</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• barbell<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.7223 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.72<br>• <b>AI Insight:</b> The Brooklyn Comics Club and Brooklyn Barbell Club appear to be the same entity, as they share the same name and context of being a community or organization focused on comics in Brooklyn.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#4</b> | <b>Brooklyn Wallyball Club</b> (Brooklyn, NY) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6463</td><td>30%</td><td>0.1939</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7607</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• wallyball<br><br><b>Match Strength:</b><br>• 67% word overlap<br>• This is a STRONG match - likely the same company<br>• Action: Use this match with high confidence<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.6463 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.65<br>• <b>AI Insight:</b> The Brooklyn Comics Club and Brooklyn Wallyball Club are the same entity, a branch of the same company, Brooklyn Wallyball Club.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟢 <b>#5</b> | <b>Rotary Club of Brooklyn</b> (Brooklyn, NY) | Score: <b>92.0%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.8097</td><td>70%</td><td>0.5668</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6225</td><td>30%</td><td>0.1868</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7536</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.9200</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• of, rotary<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.8097 (Weight: 70%)<br>• Semantic Similarity: 0.6225 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.62<br>• <b>AI Insight:</b> The Brooklyn Comics Club is the same entity as the Rotary Club of Brooklyn, serving as a branch of the Rotary International organization within the context of comics and related industries.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#6</b> | <b>Brooklyn Book Club Meetup</b> (Brooklyn, NY) | Score: <b>78.3%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7361</td><td>70%</td><td>0.5153</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7134</td><td>30%</td><td>0.2140</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7293</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7830</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• book, meetup<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7361 (Weight: 70%)<br>• Semantic Similarity: 0.7134 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.71<br>• <b>AI Insight:</b> The Brooklyn Comics Club and Brooklyn Book Club Meetup are related entities, as they both operate within the same industry context of book clubs and comic book enthusiasts.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#7</b> | <b>Brooklyn College Diversity Club</b> (Brooklyn, NY) | Score: <b>78.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7361</td><td>70%</td><td>0.5153</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7047</td><td>30%</td><td>0.2114</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7267</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7810</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• college, diversity<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7361 (Weight: 70%)<br>• Semantic Similarity: 0.7047 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.70<br>• <b>AI Insight:</b> The Brooklyn Comics Club is a separate entity from Brooklyn College Diversity Club, as they are distinct organizations with different names and purposes within the same industry context of promoting diversity and inclusivity in comics and related communities.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#8</b> | <b>South Brooklyn Running Club</b> (Brooklyn, NY) | Score: <b>77.7%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7361</td><td>70%</td><td>0.5153</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6866</td><td>30%</td><td>0.2060</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7213</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7770</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• running, south<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7361 (Weight: 70%)<br>• Semantic Similarity: 0.6866 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.69<br>• <b>AI Insight:</b> The Brooklyn Comics Club and South Brooklyn Running Club are related as they both operate in the same industry, specifically within the realm of recreational sports and fitness.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#9</b> | <b>Brooklyn Bridge Rotary Club</b> (Brooklyn, NY) | Score: <b>77.1%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.7361</td><td>70%</td><td>0.5153</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.6627</td><td>30%</td><td>0.1988</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7141</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7710</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• bridge, rotary<br><br><b>Match Strength:</b><br>• 50% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.7361 (Weight: 70%)<br>• Semantic Similarity: 0.6627 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.66<br>• <b>AI Insight:</b> The Brooklyn Comics Club is a related entity to the Brooklyn Bridge Rotary Club, as both are part of the same industry context of community service and volunteer work.<br>• <b>Meaning:</b> The model detects a moderate meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>
<details style='margin-bottom: 5px; padding: 5px;'>
  <summary style='cursor: pointer; font-size: 1.1em; font-family: sans-serif; padding: 5px;'>🟡 <b>#10</b> | <b>North Brooklyn Comic Book Club</b> (Brooklyn, NY) | Score: <b>76.6%</b></summary>
    <div style='margin-top: 10px; margin-bottom: 20px; border-left: 3px solid #eee; padding-left: 15px;'>
      <b>📊 Score Breakdown:</b><br>
      <table style='width: 100%; max-width: 600px; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px;'>
        <tr style='text-align: left; border-bottom: 1px solid #ccc;'><th>Component</th><th>Raw</th><th>Weight</th><th>Contrib</th></tr>
        <tr><td>String Similarity</td><td>0.6748</td><td>70%</td><td>0.4723</td></tr>
        <tr><td>Semantic Similarity (Norm)</td><td>0.7855</td><td>30%</td><td>0.2356</td></tr>
        <tr style='border-top: 1px solid #eee;'><td><em>Base Score</em></td><td></td><td></td><td><em>0.7080</em></td></tr>
        <tr><td>Location Boost</td><td></td><td>5% max</td><td>+0.2000</td></tr>
        <tr style='border-top: 1px solid #ccc;'><td><strong>Final Score</strong></td><td></td><td></td><td><strong>0.7660</strong></td></tr>
      </table>
      <b>📝 Match Rationale:</b><br>
      <div style='padding: 10px; border: 1px solid #eee; border-radius: 4px;'>
WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>2 word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• brooklyn, club<br><br><b>Your Search Also Includes:</b><br>• comics<br><br><b>Company Name Also Includes:</b><br>• book, comic, north<br><br><b>Match Strength:</b><br>• 40% word overlap<br>• This is a MODERATE match - worth investigating<br>• Action: Check if this makes business sense<br><br><b>Score Breakdown:</b><br>• Lexical Similarity: 0.6748 (Weight: 70%)<br>• Semantic Similarity: 0.7855 (Weight: 30%)<br>• Location Bonus: +1.0000<br><br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written<br><br><b>Semantic Analysis:</b><br>• <b>Strength:</b> 🟡 0.79<br>• <b>AI Insight:</b> The North Brooklyn Comic Book Club is the same entity as Brooklyn Comics Club, indicating that they are the same business or organization.<br>• <b>Meaning:</b> The model detects a strong meaning-based connection.<br>• <b>Context:</b> Both have non_profit indicators - strong alignment<br><br><b>Location & Frequency Analysis:</b><br>• <b>Location:</b> <span style='color: green;'>Positive Impact (+20.00%)</span>. The record's location (Brooklyn) matches the city and state requirements.<br>• <b>Frequency:</b> Neutral (Single occurrence).<br>
      </div>
    </div>
</details>

---

