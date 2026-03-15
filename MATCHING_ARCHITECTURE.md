# Company Matching System — Architecture & Improvement Roadmap

**Last updated:** 2026-02-22  
**Baseline:** `all-MiniLM-L6-v2` — 36/46 (78%) on canonical test suite  
**Test suite:** `test_model_comparison.py` | **Audit trail:** `model_comparison_results.json`  
**Active model config:** `model_config.json`

---

## Directive for All Future Agents and Developers

Read this entire document before touching any part of the pipeline. The system computes
six distinct scoring signals that interact. A change to one signal affects the balance of
all others. **Every proposed change must be evaluated holistically against all 11 categories
in `test_model_comparison.py` before implementation.** A fix that helps one category while
regressing another is not acceptable.

String pattern hacks, single-word regex penalties, and threshold tweaks targeting specific
failure cases are explicitly rejected. They have historically broken 2–3 other categories
for every one they fix.

The test suite is ground truth. Add cases for new failure patterns. Never remove cases.

---

## How the Scoring System Works — All Six Signals

The final score for every candidate is assembled from six distinct components. Understanding
all six — and where each one can fail — is prerequisite to any improvement work.

```
For each FAISS-retrieved candidate:

  name_score = (string_score       × 0.50)   [Signal 1: Lexical]
             + (sem_score_norm     × 0.25)   [Signal 2: Semantic]
             + (concept_alignment  × 0.25)   [Signal 3: Concept]

  if acronym_fidelity > 0.8 and query is short:
      name_score += (acronym_fidelity × 0.15)  [Signal 4: Acronym]

  Lexical boost tiers (applied to name_score):
      string_score ≥ 0.92  →  floor name_score to 0.95
      string_score ≥ 0.80  →  floor name_score to 0.90

  if use_location:
      if exact name match:   final = name_score + (location_score × 0.05)  [Signal 5: Location tie-breaker]
      if non-exact match:    final = (name_score × 0.80) + (location_score × 0.20)  [Signal 5: Location normalizer]

  final += (log(count) / log(max_count)) × 0.05 × name_score  [Signal 6: Frequency]
```

The RationaleService (`rationale_service.py`) surfaces all six signals to the user:
verdict banner, relationship classification, evidence bullets (one per signal), concept
"nutritional label", score breakdown table, and formula box. Every signal that is
wrong or noisy in scoring appears wrong or noisy in the rationale the user reads.

---

## Signal-by-Signal Analysis: What Works, What Fails, and Why

### Signal 1 — Lexical (string_score, 50%)

**Computed by:** Jaro-Winkler string similarity  
**RationaleService display:** "Name Similarity" evidence bullet, "High/Moderate Lexical Similarity" verdict reason

**What works well:** Near-exact matches, suffix variants (DermaQuest vs DermaQuest Inc),
single-character differences (Kruger vs Kroger), token reordering (JP Morgan Chase vs
JPMorgan Chase & Co). The existing lexical boost tiers (0.92/0.80 thresholds) correctly
ensure high-string-similarity candidates cannot be outranked by weaker semantic noise.

**Where it fails:**  
- **RC-2 (Noise):** "NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46" — the
  booking ID and date tokens reduce string similarity against "National Association of
  Landscape Architects" from what would be a reasonable match to a very low one. The
  string scoring is then misleading: it reports a "Weak" name similarity that is actually
  an artifact of query noise, not a genuine name mismatch.
- **RC-2 (Noise):** "NWACUHO - Northwest Association..." — the "NWACUHO - " prefix lowers
  Jaro-Winkler against "Northwest Association..." even though they refer to the same entity.
  The lexical boost tier partially compensates (≥0.80 threshold), but the noise still
  corrupts the score.

**Fix (Priority 3 — Query Preprocessor):** Strip noise tokens from the query before
string scoring. Use the cleaned query for FAISS embedding AND concept probing; retain the
original for Jaro-Winkler so the user-facing rationale reflects what they actually typed.

---

### Signal 2 — Semantic (sem_score_norm, 25%)

**Computed by:** FAISS cosine similarity, normalized against the best match in the retrieved set  
**RationaleService display:** "Semantic Link" evidence bullet, "High/Moderate Semantic Similarity" verdict reason

**Critical design detail — the normalization:**  
`sem_score_norm = raw_cosine / max_cosine_in_retrieved_set`  
This is a *relative* ranking within the retrieved pool, not an absolute quality measure.
The top-ranked FAISS result always gets sem_score_norm = 1.0 regardless of how good or
bad it actually is. If all retrieved candidates are poor matches, the least-poor one still
shows "EXCELLENT (100%)" semantic similarity. The rationale's "Semantic Link" bullet is
therefore always relative, never absolute.

**Where it fails:**  
- **RC-1 (Directional dilution):** "Northwest Association of Housing Officers" and
  "Southwest Association of Housing Officers" produce embeddings that are within 0.02
  cosine of each other. After normalization, both get sem_score_norm ≈ 1.0 and ≈ 0.98.
  The rationale correctly shows both as "EXCELLENT semantic link" — and both are, from
  the model's perspective. The model is not wrong; it is incapable of distinguishing them
  because the directional token is 1/9th of the mean-pooled vector.
- **RC-1 (Paraphrase model bias):** `paraphrase-MiniLM-L3-v2` was explicitly trained to
  maximize similarity for paraphrase pairs — sentences with the same meaning expressed
  differently. Northwest/Southwest variants are perfect paraphrases from the model's
  perspective. `all-MiniLM-L6-v2` (currently active) is a general-purpose model and
  partially mitigates this, but the architectural limitation persists.
- **RC-3 (Bi-encoder discrimination limit):** The model encodes query and candidate
  *independently*. It cannot reason about the relationship between two candidates
  relative to the query. "Hartford Hospital School of Nursing" vs "Hartford Community
  College School of Nursing" — both score ~0.87 against the query independently. The
  model cannot say which is more correct without seeing both together.

**Fixes:**  
- Priority 4 — Bi-encoder model upgrade (L6 now, MPNet when downloadable): better
  per-token context preservation
- Priority 5 — Cross-encoder re-ranking: sees the full (query, candidate) pair, resolves
  both RC-1 and RC-3 by replacing independent embedding comparison with bidirectional
  attention across both strings simultaneously

---

### Signal 3 — Concept Alignment (concept_alignment, 25%)

**Computed by:** Cosine correlation of concept anchor probe signatures  
**RationaleService display:** "Concept Alignment" evidence bullet, full "Concept Analysis" nutritional label (Geography/Industry/Structure/Nature)

**What the concept system actually does:**  
For each company embedding, it computes cosine similarity against 32 fixed anchor vectors
(the embeddings of words like "Ohio", "California", "Medical", "Corporate"). The concept
"signature" is this 32-element vector of similarities. Concept alignment between query
and candidate is the cosine similarity of their two signatures — how similar their
"concept nutritional labels" are.

**Where it fails — and why this is the second-most widespread problem:**

The 11 geographic anchors are: Pennsylvania, London, Canada, California, New York, Texas,
Chicago, Illinois, Ohio, Miami, Paris.

These anchors were chosen because they are recognizable geographic terms. But they are
**not representative of US corporate geography** — they are the locations that appear most
frequently in English-language text corpora that the underlying SentenceTransformer was
trained on. The model's training distribution heavily over-represents major metros.

Consequence: Any company name from a location NOT in this list (Alaska, Pacific Northwest,
Mountain West, rural Midwest, most of the South, etc.) will produce a concept signature
with spuriously high similarity to Ohio or California — not because of any real connection,
but because the model's embedding space has these locations as the nearest "geographic"
attractors for unfamiliar geographic contexts.

**The "Ohio 33.3%, California 29.2%" for Fairbanks, AK NWACUHO:**  
This is exactly this failure. The model has never seen enough "Fairbanks" or "Alaska" or
"Pacific Northwest housing officer" context to build a meaningful anchor. The concept
signature anchors to Ohio by default. This appears in the rationale as if it is
informative — but it is geographic noise presented as analysis.

**Scale of this problem:**  
In the 109-query SME control set, ~30–40% of queries with concept analysis show geographic
labels that do not correspond to the actual company location. "Ohio" appears for Oregon
companies, Illinois companies not related to GE, New Jersey companies. The concept
alignment component is contributing biased noise to 25% of name_score for a large fraction
of all queries.

**Industry anchors also fail at fine resolution:**  
The 11 industry anchors are coarse: Food, Medical, Technology, etc. "Seafood Nutrition
Partnership" and "Organic Food Partnership" both anchor to Food. "Hartford Hospital School
of Nursing" and "Hartford Hospital School of Allied Health" both anchor to Medical and
Education. The concept alignment cannot distinguish within a category.

**Fix (Priority 1 — Concept Anchor Redesign):**  
Replace the 11 city/state geographic anchors with US regional terms that are more
evenly distributed across the country:  
`["Pacific Northwest", "Pacific Southwest", "Mountain West", "Great Plains", "Midwest",
"Great Lakes", "Mid-Atlantic", "New England", "Southeast", "Gulf Coast", "Southwest",
"Alaska", "Hawaii", "Canada", "International"]`

Extend industry anchors with finer-grained terms: Healthcare, Energy, Hospitality,
Nonprofit, Government, Academic, Pharmaceutical, Agriculture.

This eliminates the Ohio/California bias and makes the concept analysis genuinely useful
for geographic disambiguation instead of actively misleading.

---

### Signal 4 — Acronym Fidelity (conditional, +15% max)

**Computed by:** Ratio of query acronym letters that match first letters of candidate words  
**RationaleService display:** "Acronym Expansion" match type classification, fidelity score in breakdown

**What works:** NIH → "National Institutes of Health" (all letters match), IBM → "International
Business Machines", GE → "General Electric", ABA → "American Bar Association" (L6 fixed this
from L3's failure). The +15% boost correctly surfaces exact acronym expansions above
semantic alternatives.

**Where it fails — RC-4 (Acronym retrieval gap):**  
The acronym fidelity boost only fires *after* FAISS retrieval. If the correct expansion
is not in the FAISS top-K retrieval set, the boost never has a chance to apply.

"PDMA" → "Product Development and Management Association" fails because:
1. FAISS embeds "PDMA" as a short 4-character string with weak semantic signal
2. "Product Development and Management Association" is semantically distant in the
   embedding space from the 4-letter token "PDMA"
3. FAISS does not retrieve it in the top-K
4. Acronym fidelity = 0.0 (expansion never reached)

The system already has Phase 0 acronym expansion logic (before FAISS search) for known
patterns, but it does not cover all acronyms in the reference index.

**Fix:** The acronym expansion Phase 0 pre-lookup (already in the codebase) needs to be
more aggressive — for short queries (≤6 chars), explicitly search the acronym index before
FAISS and inject any matches into the candidate pool regardless of FAISS retrieval.

---

### Signal 5 — Location (conditional, 5% or 20%)

**Computed by:** `calculate_location_score(query_city, query_state, candidate_city, candidate_state)`  
**RationaleService display:** "Location Data" evidence bullet (shows boost or "Neutral"), location boost in score breakdown

**The formula design flaw (RC-6 — highest frequency failure):**  
For non-exact matches: `final = (name_score × 0.80) + (location_score × 0.20)`

When `location_score = 0.0` — which happens when:
- The candidate has no city/state stored in the index, OR
- The candidate's state doesn't match the query's state

...the formula becomes `final = name_score × 0.80`. A 20% reduction in every non-exact
match score, applied uniformly to all candidates.

This is not a "location boost." It is a penalty that fires whenever location data is
absent or mismatched. Since ~80–90% of SME queries show "no boost warranted," the formula
is reducing every non-exact match score by 20% without any compensating benefit in the
vast majority of queries.

**The `calculate_location_score` missing-data problem:**  
```python
if not query_city or not target_city: return 0.0
```
Empty target city returns 0.0. This is treated as a confirmed mismatch rather than as
unknown. A correct company with no city stored in the reference index gets penalized.

**RC-8 — Geographic text in query names ignored:**  
The system only reads city/state from explicit database columns. It never extracts
geographic hints from the company name text. "Ohio University" typed as a query name has
"Ohio" in it — but the system does not use this when the explicit state field is empty.
Multiple "Ohio University" entries in the index (Columbus OH, Athens OH, others) cannot
be disambiguated without this hint.

**Fix (Priority 0 — Location Formula Redesign):**  
Replace the normalizing formula with a boost-only formula:
```
final = name_score + location_boost
where location_boost = location_score × 0.10  (if both sides have location data AND it matches)
      location_boost = 0.0                    (if either side has no location data — neutral, not penalized)
      location_boost = 0.0  (not negative)    (if location explicitly mismatches)
```
Candidates without location data are neutral, not penalized.

**Fix (Priority 2 — Geographic Text Extraction):**  
Extract geographic hints from query name text when explicit city/state fields are empty.
Use `RationaleService.analyze_geographic_context()` — this method already exists in the
codebase but is not wired into the scoring pipeline.

---

### Signal 6 — Frequency (conditional, +5% max)

**Computed by:** Log-scale of record count, scaled by name_score  
**RationaleService display:** "Entity Popularity" evidence bullet

**Status:** Works correctly. Companies that appear more frequently in the reference set
get a small confidence boost. This is correctly capped and does not dominate.

**One nuance:** The frequency boost is scaled by `name_score`, so it amplifies already-good
matches rather than rescuing poor ones. This is the right behavior.

**No changes recommended.**

---

### The Lexical Boost Tiers — A Design Trade-off

The two boost tiers (string_score ≥ 0.92 → floor 0.95; string_score ≥ 0.80 → floor 0.90)
are a deliberately aggressive measure to ensure near-literal matches consistently outrank
semantic noise.

**They work well** for their stated purpose: preventing a company with a very different name
from outranking the obvious match just because it has good semantic similarity scores.

**They create a tie-compression problem:** When two candidates BOTH have string_score ≥ 0.92
(which happens for directional variants like Northwest/Southwest), both get floored to
name_score = 0.95. The small difference that the semantic and concept signals would have
provided is erased. The tiebreaker then falls entirely to location and frequency — both
of which may be zero or noise.

**This is the final step in the NWACUHO failure chain:**
1. Semantic: ~identical (RC-1)
2. Concept: ~identical (RC-7, both anchor to Ohio)
3. String: both ≥ 0.92 → both floored to 0.95 ← **tie compression fires here**
4. Location: both = 0.0 (no AK companies in index, RC-6)
5. Frequency: arbitrary winner from database occurrence counts

**Fix:** The tiers are correct. The fix is to eliminate the conditions that cause the tiers
to fire on wrong candidates — i.e., fix the signals that feed into the tie (cross-encoder
re-ranking, concept anchor redesign). The tier logic itself should not be changed.

---

## Capabilities in RationaleService That Are Not Yet Used in Scoring

The `RationaleService` contains a significant set of analysis methods that are implemented
but not wired into the scoring pipeline. These represent opportunities to improve signal
quality without model changes:

| Method | What It Does | Relevant Failure |
|--------|-------------|-----------------|
| `analyze_geographic_context()` | Extracts geographic indicators from a name string | RC-8: could provide city/state hints from query name text |
| `analyze_phonetic_similarity()` / `get_soundex()` | Phonetic matching (Soundex algorithm) | International/transliteration name variants |
| `analyze_industry_context()` | Keyword-based industry classification | Could supplement or replace coarse concept anchor industry categories |
| `analyze_business_context()` | Detects business context keywords | Could improve match type classification |
| `is_ordinal_relationship()` | Detects "First" → "1st" type relationships | Edge case: conference series, annual events |
| `is_abbreviation_relationship()` | Detects abbreviation patterns | Complements acronym fidelity for non-acronym abbreviations |
| `generate_relative_positioning_explanation()` | Explains WHY a result ranks where it does vs adjacent results | Currently built but never called in any UI path |

Particularly: `analyze_geographic_context()` is the direct implementation needed for the
RC-8 fix (Priority 2). It already exists. It only needs to be called from `match_with_location()`
when explicit city/state fields are empty, and its output used to supplement the location
scoring.

---

## The Compound Failure: How All Six Signals Interact for NWACUHO

```
Query:     "NWACUHO - Northwest Association of College & University Housing Officers"
Location:  Fairbanks, AK
Expected:  "Northwest Association of College and University Housing Officers" at rank 1
Actual:    "Southwest Association College University Housing Officers" at rank 1
```

Tracing every signal:

| Signal | Northwest entry | Southwest entry | Verdict |
|--------|----------------|----------------|---------|
| **string_score** | ~0.93 (high — same phrase, different first word) | ~0.92 (slightly lower) | Both ≥ 0.92 tier → both floored to 0.95 |
| **sem_score_norm** | ~1.00 (top FAISS hit or close to it) | ~0.98 | After normalization, negligible difference |
| **concept_alignment** | ~0.72 (Ohio/California anchors, no AK signal) | ~0.72 (same anchors, same noise) | Tie — RC-7 fires |
| **name_score** | 0.95 (floored by tier) | 0.95 (floored by tier) | Identical |
| **location_score** | 0.0 (no AK in index) | 0.0 (no AK in index) | Both penalized equally — RC-6 fires |
| **final score** | 0.95 × 0.80 = 0.76 | 0.95 × 0.80 = 0.76 | Identical |
| **frequency boost** | depends on DB occurrence count | may be higher | **Arbitrary winner** |

The correct answer could only win at the frequency boost stage — if "Northwest" variants
happen to appear more often in the reference database than "Southwest" variants. This is
arbitrary and unreliable.

All three system failures (RC-1, RC-6, RC-7) must be addressed together. Fixing any
single one does not change the outcome because the other two maintain the tie.

---

## Proposed Architecture: Addressing All Six Signals

```
Raw Query
   │
   ▼
[Stage 0 — NEW: Query Intelligence]
   Addresses: Signal 1 (Lexical noise), Signal 5 (Location from name text)
   
   A. Noise extraction (fixes RC-2):
      Strip booking IDs, dates, meeting prefixes, conference labels from query.
      Clean query used for FAISS embedding and concept probing.
      Original query retained for Jaro-Winkler (string_score).
      
   B. Geographic text extraction (fixes RC-8):
      When explicit city/state fields are empty, call analyze_geographic_context()
      (already implemented in RationaleService) to extract city/state/region hints.
      "Ohio University" → state_hint = "OH"
      "Pacific Northwest Diabetes Research" → region_hint = "Pacific Northwest"
      "Chicago South Swim Club" → city_hint = "Chicago"
   │
   ▼
[Stage 1 — EXISTING, improved: FAISS Retrieval]
   Addresses: Signal 2 (Semantic), Signal 4 (Acronym retrieval gap)
   
   Over-retrieve: top-K × 3 to ensure correct candidate is in pool.
   Aggressive acronym pre-expansion for short queries (≤6 chars): inject
   acronym index matches before FAISS (already partially done in Phase 0).
   Model: all-MiniLM-L6-v2 now; all-mpnet-base-v2 when downloadable.
   │
   ▼
[Stage 2 — FIX RC-6 and RC-7: Revised Hybrid Scorer]
   Addresses: Signal 3 (Concept), Signal 5 (Location)

   A. Concept anchor redesign (fixes RC-7):
      Replace 11 city/state geographic anchors with 15 US regional terms:
      ["Pacific Northwest", "Pacific Southwest", "Mountain West", "Great Plains",
       "Midwest", "Great Lakes", "Mid-Atlantic", "New England", "Southeast",
       "Gulf Coast", "Southwest", "Alaska", "Hawaii", "Canada", "International"]
      Extend Industry anchors with finer-grained terms.
      Eliminates Ohio/California bias. Makes concept nutritional label genuinely informative.
      Requires CACHE_VERSION bump and full index rebuild.

   B. Location formula redesign (fixes RC-6):
      Replace normalizing formula with boost-only formula.
      Missing location data → neutral (0.0 boost, NOT a penalty).
      Confirmed location match → positive boost.
      Location mismatch → neutral (not negative).
   │
   ▼
[Stage 3 — NEW when model available: Cross-Encoder Re-ranking]
   Addresses: Signal 2 (Semantic directional dilution), Signal 3 (Fine-grained discrimination)
   
   Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22 MB)
   Sees full (query, candidate) string pair simultaneously.
   "Northwest" in query attends directly to "Northwest" vs "Southwest" in candidate.
   Re-ranks top-K × 3 down to top-K.
   Applied after Stage 2 scoring, before final result delivery.
   │
   ▼
[Stage 4 — EXISTING: Lexical Boost Tiers, Frequency]
   Signals 1, 6: Unchanged. These work correctly.
   The lexical boost tiers remain. Their tie-compression effect is resolved
   by Stages 2 and 3 ensuring that candidates reaching the tiers have
   genuinely differentiated scores before hitting the floor.
```

---

## Complete Failure-to-Fix Mapping

| Root Cause | Signal Affected | Priority | Fix |
|-----------|----------------|----------|-----|
| RC-1: Directional token dilution | Signal 2: Semantic | Priority 4 | Cross-encoder + MPNet |
| RC-2: Query noise | Signal 1: Lexical, Signal 2: Semantic | Priority 3 | Query Intelligence Preprocessor |
| RC-3: Bi-encoder discrimination limit | Signal 2: Semantic | Priority 4 | Cross-encoder re-ranking |
| RC-4: Alias/world knowledge gap | Signal 2, 3, 4 | Accept — no text similarity fix | Known limitation |
| RC-5: Candidate-subset scoring | Signal 1: Lexical | Low priority | Add reverse-overlap token check |
| **RC-6: Location formula penalizes absent data** | **Signal 5: Location** | **Priority 0** | **Boost-only formula** |
| **RC-7: Geographic anchor bias** | **Signal 3: Concept** | **Priority 1** | **Concept anchor redesign** |
| **RC-8: Geographic text not extracted** | **Signal 5: Location** | **Priority 2** | **Reuse existing analyze_geographic_context()** |

---

## Implementation Priority

### Priority 0 — Location Formula Redesign (RC-6)
Affects 80–90% of queries. Boost-only formula. No index rebuild required.  
File: `src/finetuner/core/matcher.py` lines 1550–1564

### Priority 1 — Concept Anchor Redesign (RC-7)
Affects 30–40% of queries. Removes geographic noise from 25% of name_score.  
File: `src/finetuner/core/matcher.py` lines 40–46, CACHE_VERSION constant.  
Requires full index rebuild (4.3M companies).

### Priority 2 — Geographic Text Extraction (RC-8)
Fixes score compression for exact-name disambiguation.  
Wire `RationaleService.analyze_geographic_context()` into `matcher.py` match_with_location().  
No index rebuild required.

### Priority 3 — Query Noise Preprocessor (RC-2)
Fixes 2–4 specific noise failure cases.  
File: `src/finetuner/utils/text_preprocessor.py` (new method), `matcher.py` call site.  
No index rebuild required.

### Priority 4 — Cross-Encoder Re-ranking (RC-1, RC-3)
Highest precision gain. Requires model download.  
File: `src/finetuner/core/matcher.py` (new rerank() method).  
Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 MB, needs manual download).

### Priority 5 — MPNet Model Switch
Better bi-encoder for RC-1. Requires full index rebuild and manual download.  
Do not implement until cross-encoder (Priority 4) is in place.  
File: `model_config.json` → `active_model: all-mpnet-base-v2`

---

## What Is Not Recommended

**Single-signal hacks:** Regex for "Northwest vs Southwest," threshold tweaks for specific
words, per-category weight adjustments. All have been tested and all produce net-zero or
negative improvement when evaluated across all 11 test categories.

**Increasing location weight:** RC-6 shows the current 20% weight is already too high
given sparse location data. Increasing it makes the penalty worse.

**Accepting Ohio/California concept labels as informative:** They are training-data bias
artifacts, not analysis. Displaying them misleads users and should be eliminated by
Priority 1.

**RC-4 (alias/rebrand knowledge):** AVIAKOMPANIYA SIBIR → S7 Airlines cannot be solved
with text similarity. Accept as a known limitation until an alias table is built.

---

## Model Reference

Active model configured in `model_config.json`. Changing models requires full FAISS index
rebuild. Old caches are preserved automatically (cache key includes model name).

| Model | Dims | Size | Score | Notes |
|-------|------|------|-------|-------|
| `paraphrase-MiniLM-L3-v2` | 384 | 67 MB | 32/46 (70%) | Original. Cached. Paraphrase-trained — treats NW/SW as synonyms. |
| `all-MiniLM-L6-v2` | 384 | 87 MB | 36/46 (78%) | **Active.** Cached. General-purpose. Same index size as L3. |
| `all-mpnet-base-v2` | 768 | 420 MB | ~40/46 est. | Needs manual download. 2× FAISS index. Best bi-encoder. |

---

## Evaluation Protocol (Mandatory Before Any Change)

```bash
python test_model_comparison.py --model-a paraphrase-MiniLM-L3-v2 --model-b all-MiniLM-L6-v2
```

Pass gate:
- Overall: ≥ 36/46
- Exact, Partial, Suffix, Person: must remain 100%
- No category regresses by more than 1 case
- Results saved to `model_comparison_results.json`

---

*This document supersedes all previous versions. Update it when architecture changes,
new failure patterns are identified, new models are evaluated, or priorities shift.*
