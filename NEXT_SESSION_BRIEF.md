# MISSION BRIEFING: SME REPORTING & RATIONALES

> **CRITICAL DIRECTIVE:** The user ("Serge Nation" proxy) demands **EXTREMELY DETAILED, COMPREHENSIVE RATIONALES** for every single match. 
> 
> ❌ **NEVER** summarize or abbreviate rationales.
> ❌ **NEVER** hide score breakdowns.
> ✅ **ALWAYS** show "What This Means", "Action Required", "Why This Happens", and the FULL component table.

## Current State (As of Dec 29, 2025)

1.  **RPC Cache Server**: Running perfectly on port 9876. Latency is ~1ms (vs 10ms HTTP).
2.  **Report Format**: The "Serge Nation" format is locked in. 
    - **Gold Standard**: `serge_nation_showcase.md` (Contains the Likelihood stats & categorization).
    - **Structure**: Group by Frequency (High -> Low).
    - **Stats**: MUST include "Likelihood: ~X%" headers.
3.  **Rationale Engine**: `RationaleService.generate_match_rationale()` + `RationaleService.generate_detailed_score_breakdown()` MUST both be used.

## Immediate Next Steps (DO THIS FIRST)

1.  **Fix Location Boost Visibility**: 
    - *Problem:* Location boosts are happening (re-ranking works) but NOT showing in the breakdown table.
    - *Task:* Update `RationaleService.generate_detailed_score_breakdown` to explicitly check for and display `location_score` and `location_boost` if they exist in the match result.

2.  **Fix Frequency Boost Visibility**:
    - *Status:* Already visible in some reports, but verify it's CONSISTENT.

3.  **Generate Final Master Report**:
    - Run `tests/generate_report_sme.py` again after fixing the location visibility to get the "Perfect" report.

## Key Files
- `tests/generate_report_sme.py` -> The gathered report generator.
- `src/finetuner/web/services/rationale_service.py` -> The brain of the explanations.
- `.agent/workflows/comprehensive-rationales.md` -> The law for report formatting.

## User Persona
- Has zero tolerance for hidden logic.
- Wants to be able to explain *everything* to the end client.
- "How do I explain why this ranked #1?" is the driving question.
