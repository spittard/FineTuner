---
description: Always use comprehensive detailed rationales in reports
---

# CRITICAL: Comprehensive Rationales Required

When generating ANY report or match output for SME review, ALWAYS use the **full comprehensive rationale** from `RationaleService`. 

## Required Rationale Sections

Every match rationale MUST include:

1. **Match Rationale:** - Detailed narrative explaining the match
2. **Why This Match Makes Sense:** - Justification for the match
3. **Why This Match Might Be Wrong:** - Potential issues/concerns
4. **Score Breakdown** - Full component breakdown (String, Semantic, Acronym, Location)
5. **Alternative Interpretations** - When applicable

## Implementation

```python
from finetuner.web.services.rationale_service import RationaleService

# Generate FULL rationale - never abbreviated
rationale = RationaleService.generate_match_rationale(
    query=query,
    company_name=match['name'],
    explanation=match,  # Pass full match data
    score=match['score']
)

# Also use detailed score breakdown
breakdown = RationaleService.generate_detailed_score_breakdown(match, query)
```

## NEVER DO THIS

- ❌ One-line summaries like "Semantic match" or "Near-exact text"
- ❌ Abbreviated score tables without context
- ❌ Missing "Why This Match Makes Sense" / "Might Be Wrong" sections

## ALWAYS DO THIS

- ✅ Full narrative rationale explaining the match
- ✅ Complete score breakdown with all components
- ✅ Context about why the match is ranked where it is
- ✅ Alternative interpretations when relevant
