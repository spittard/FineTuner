---
description: Always use comprehensive detailed rationales in reports
---

# CRITICAL: Comprehensive Rationales Required

When generating ANY report or match output for SME review, ALWAYS use the **full comprehensive rationale** from `RationaleService`. 

## Required Report Format

### Table Structure
```markdown
| # | Score | Company | Type | Rationale |
|:-:|:-----:|---------|------|-----------|
| 1 | 🟢 **100.2%** | `Company Name` | Exact | <details><summary>📊 View</summary>...</details> |
```

### Score Icons
- 🟢 ≥ 80% (High confidence)
- 🟡 60-79% (Medium confidence)
- 🟠 40-59% (Low confidence)
- 🔴 < 40% (Very low confidence)

### Match Type Column
- **Exact** - String score ≥ 0.9
- **Acronym** - Acronym fidelity > 0.5
- **Semantic** - Semantic score > string score
- **Lexical** - Otherwise

## Required Rationale Sections (via RationaleService)

Every expandable rationale MUST include:

1. **Match Type Header:** PERFECT MATCH / WORD OVERLAP MATCH / ACRONYM MATCH / etc.
2. **What This Means:** Plain language explanation
3. **Action Required:** Clear next steps for the data clerk
4. **Why This Happens:** Context for the match behavior
5. **Complete Score Breakdown:** Full component table with weights

## Implementation

```python
from finetuner.web.services.rationale_service import RationaleService

# Get COMPREHENSIVE rationale - NEVER abbreviated
full_rationale = RationaleService.generate_match_rationale(
    query=query,
    company_name=match['name'],
    explanation=match,  # Pass full match data dict
    score=match['score']
)

# Also get detailed score breakdown
score_breakdown = RationaleService.generate_detailed_score_breakdown(match, query)

# Combine both for complete rationale
combined = f"{full_rationale}\n\n---\n\n{score_breakdown}"

# Format for HTML table cell
rationale_html = combined.replace('\n', '<br>').replace('|', '&#124;')
```

## Reference Script

The canonical implementation is: `tests/generate_report_sme.py`

## NEVER DO THIS

- ❌ One-line summaries like "Semantic match" or "Near-exact text"
- ❌ Abbreviated score tables without context
- ❌ Missing "What This Means" / "Action Required" / "Why This Happens" sections
- ❌ Rationale without score breakdown
- ❌ Rationale outside of expandable `<details>` tags

## ALWAYS DO THIS

- ✅ Full `RationaleService.generate_match_rationale()` output
- ✅ Full `RationaleService.generate_detailed_score_breakdown()` output  
- ✅ Combine both in expandable `<details>` section
- ✅ Score icons (🟢🟡🟠🔴) for quick visual scanning
- ✅ Table format with columns: # | Score | Company | Type | Rationale
