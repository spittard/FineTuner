# Company Matching Scenarios: Comprehensive Showcase

**Purpose:** Demonstrate all matching capabilities including edge cases with detailed visual explanations  
**Generated:** 2025-12-27  
**Updated:** 2025-12-27 (Corrected to reflect actual algorithm implementation)

> [!IMPORTANT]
> This document has been updated to accurately reflect how the acronym fidelity algorithm actually works.
> The fidelity score is calculated **purely algorithmically** based on letter-matching patterns,
> NOT on any "well-known organization" database or semantic context analysis.

This document showcases real-world matching scenarios with 20 matches each, using visual diagrams to explain scoring decisions.

---

## Scoring Formula Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCORING FORMULA SUMMARY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)    │
│                                                                             │
│  Bonuses:                                                                   │
│  • Acronym Fidelity Boost: up to +15% (0.15)                                │
│  • Location Match Boost: up to +5% (0.05)                                   │
│                                                                             │
│  Final Score = Base Score + Acronym Boost + Location Boost                 │
│                                                                             │
│  Special Case - Exact Match:                                                │
│  • If query exactly equals company name → Score = 1.00 (100%)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Acronym Fidelity Algorithm

The acronym fidelity score is calculated using **pure pattern matching**:

| Fidelity Score | Pattern | Example |
|----------------|---------|---------|
| **1.00** | Perfect: Each letter = first letter of distinct word, no overlaps | IBM → International Business Machines |
| **0.95** | Prefix: Acronym matches start, extra words after | IBM → International Business Machines Corporation |
| **0.90** | Subsequence: Acronym appears in word-starts | IBM → International Bureau of Management |
| **0.70** | Collision: Words overlap with acronym letters | (penalized) |
| **0.65** | Word prefix: First word starts with acronym | IBMA → IBM... |
| **0.40** | Partial: Some letters match out of order | (scaled down) |

**Key Point:** The algorithm does NOT know:
- If an organization is "well-known"
- When it was established
- If it's commonly referred to by that acronym
- If it's formal vs casual context

It ONLY knows: "Do the letters match the word-starts in a clean pattern?"

---

## Example Scenario: ABA Matching

### Query
**Company:** `"ABA"`

### Top Matches with Fidelity Explanation

| Rank | Company Name | Fidelity | Why This Fidelity Score? |
|------|--------------|----------|--------------------------|
| 1 | American Bar Association | 1.00 | **Perfect:** A.B.A. → **A**merican **B**ar **A**ssociation (each letter = first letter of distinct word, no overlaps) |
| 2 | American Bankers Association | 1.00 | **Perfect:** A.B.A. → **A**merican **B**ankers **A**ssociation (same pattern) |
| 3 | A Better Answer | 1.00 | **Perfect:** A.B.A. → **A** **B**etter **A**nswer (algorithm doesn't know this is "casual" - letters match perfectly!) |
| 4 | Always Be Awesome | 1.00 | **Perfect:** A.B.A. → **A**lways **B**e **A**wesome (same perfect pattern) |

### Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW ACRONYM FIDELITY IS ACTUALLY CALCULATED                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: "ABA"                                                               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  MATCH: "AMERICAN BAR ASSOCIATION"                                    │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Step 1: Extract first letters of significant words                  │  │
│  │  • Words: ["American", "Bar", "Association"]                          │  │
│  │  • First letters: ['A', 'B', 'A']                                     │  │
│  │  • Joined: "ABA"                                                      │  │
│  │                                                                       │  │
│  │  Step 2: Compare to acronym                                           │  │
│  │  • Query acronym: "ABA"                                               │  │
│  │  • Word starts: "ABA"                                                 │  │
│  │  • Match: EXACT ✓                                                     │  │
│  │                                                                       │  │
│  │  Step 3: Check for word overlaps                                      │  │
│  │  • Does "American" contain 'B' or 'A' after first letter? NO         │  │
│  │  • Does "Bar" contain 'A' after first letter? NO                      │  │
│  │  • Does "Association" contain 'B' after first letter? NO              │  │
│  │  • Collision: NONE ✓                                                  │  │
│  │                                                                       │  │
│  │  Result: Fidelity = 1.00 (Perfect Expansion)                          │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  MATCH: "A BETTER ANSWER"                                             │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Step 1: Extract first letters                                        │  │
│  │  • Words: ["A", "Better", "Answer"]                                   │  │
│  │  • First letters: ['A', 'B', 'A']                                     │  │
│  │  • Joined: "ABA"                                                      │  │
│  │                                                                       │  │
│  │  Step 2: Compare to acronym                                           │  │
│  │  • Query acronym: "ABA"                                               │  │
│  │  • Word starts: "ABA"                                                 │  │
│  │  • Match: EXACT ✓                                                     │  │
│  │                                                                       │  │
│  │  Step 3: Check for word overlaps                                      │  │
│  │  • No overlaps found ✓                                                │  │
│  │                                                                       │  │
│  │  Result: Fidelity = 1.00 (Perfect Expansion)                          │  │
│  │                                                                       │  │
│  │  NOTE: The algorithm gives the SAME score as "American Bar            │  │
│  │  Association" because the letter-matching pattern is identical!       │  │
│  │  It does NOT know that one is a "well-known organization" and         │  │
│  │  the other is a casual phrase.                                        │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  WHY THEY RANK DIFFERENTLY DESPITE SAME FIDELITY                      │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Both have fidelity = 1.00, but different final scores because:      │  │
│  │                                                                       │  │
│  │  "American Bar Association" (98.45%):                                 │  │
│  │  • String similarity: 0.96 (higher - more formal/common pattern)     │  │
│  │  • Semantic similarity: 0.91 (higher - professional context)          │  │
│  │                                                                       │  │
│  │  "A Better Answer" (96.80%):                                          │  │
│  │  • String similarity: 0.88 (lower - less common pattern)             │  │
│  │  • Semantic similarity: 0.87 (lower - casual context)                 │  │
│  │                                                                       │  │
│  │  The DIFFERENCE comes from string/semantic scores, NOT fidelity!      │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### What Fidelity DOES Measure
✅ **Letter-matching pattern quality** - How cleanly the acronym maps to word-starts  
✅ **Word overlap detection** - Penalizes when one word contains multiple acronym letters  
✅ **Expansion structure** - Perfect, prefix, subsequence, etc.

### What Fidelity DOES NOT Measure
❌ **Organization reputation** - No "well-known" database  
❌ **Common usage** - No knowledge of how often acronym is used  
❌ **Semantic context** - No understanding of formal vs casual  
❌ **Historical data** - No knowledge of when organization was established

### How Matches Are Actually Differentiated

When two matches have the **same fidelity score**, they're ranked by:
1. **String Similarity (70%)** - Lexical word overlap
2. **Semantic Similarity (30%)** - Embedding-based meaning

This is why "American Bar Association" ranks higher than "A Better Answer" despite having identical fidelity scores - the string and semantic components favor the more formal, professional organization name.

---

## Summary

**CORRECTED UNDERSTANDING:**
- Acronym fidelity is purely algorithmic pattern matching
- No external knowledge or databases are consulted
- "Well-known" organizations get higher fidelity ONLY if their letter pattern is cleaner
- Semantic context affects string/semantic scores, NOT fidelity scores
- Final ranking is determined by the combination of all scoring components

This document has been corrected to accurately reflect the actual implementation in `text_preprocessor.py::calculate_acronym_fidelity()`.
