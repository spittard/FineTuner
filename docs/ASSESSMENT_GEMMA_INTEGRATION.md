# Assessment: Gemma Integration for Enhanced Matching

**Document Version:** 1.0
**Created:** 2026-04-26
**Focus Area:** Leveraging Google's Gemma LLM for Company Name Matching

---

## Executive Summary

This assessment explores how Google's Gemma family of open-source LLMs can enhance FineTuner's company name matching capabilities. Gemma offers several opportunities:

1. **Semantic Re-ranking** - Use Gemma to re-score ambiguous matches
2. **Entity Normalization** - Standardize company name variants
3. **Contextual Disambiguation** - Resolve industry/geographic ambiguity
4. **Explanation Generation** - Replace template-based rationales with natural language
5. **Query Understanding** - Parse complex/malformed queries

**Recommendation:** Implement Gemma as a secondary re-ranker for low-confidence matches (scores 75-92%), where the current hybrid approach struggles most.

---

## 1. Gemma Model Overview

### 1.1 Available Models (as of 2026)

| Model | Parameters | Context | Use Case | VRAM Required |
|-------|------------|---------|----------|---------------|
| Gemma 2 2B | 2B | 8K | Fast inference, edge | 4-6 GB |
| Gemma 2 9B | 9B | 8K | Balanced quality/speed | 12-18 GB |
| Gemma 2 27B | 27B | 8K | Highest quality | 32-48 GB |
| Gemma 3 (if released) | TBD | 32K+ | Latest capabilities | TBD |

### 1.2 Key Capabilities Relevant to Matching

- **Entity Recognition**: Understands company names, locations, industries
- **Semantic Similarity**: Can judge if two names refer to same entity
- **Reasoning**: Can explain why names match or don't match
- **Few-shot Learning**: Can be guided with examples in prompt
- **Instruction Following**: Responds well to structured prompts

---

## 2. Integration Opportunities

### 2.1 Opportunity 1: Semantic Re-ranking for Ambiguous Matches

**Problem:** Current hybrid scoring struggles with scores in the 75-92% range where lexical and semantic signals conflict.

**Solution:** Use Gemma to re-rank these ambiguous candidates.

```python
# gemma_reranker.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GemmaReranker:
    """Use Gemma to re-rank ambiguous match candidates."""

    def __init__(self, model_name: str = "google/gemma-2-9b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def rerank(self, query: str, candidates: list[dict],
               query_city: str = None, query_state: str = None) -> list[dict]:
        """
        Re-rank candidates using Gemma's semantic understanding.

        Args:
            query: Original search query
            candidates: List of {name, score, city, state} dicts
            query_city: Optional location filter
            query_state: Optional location filter

        Returns:
            Re-ranked candidates with gemma_score added
        """
        prompt = self._build_ranking_prompt(query, candidates, query_city, query_state)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,  # Low temp for consistency
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse rankings from response
        rankings = self._parse_rankings(response, candidates)

        return rankings

    def _build_ranking_prompt(self, query, candidates, city, state):
        """Build structured prompt for ranking task."""
        location_context = ""
        if city or state:
            location_context = f"\nThe user is looking for a company in {city or ''} {state or ''}."

        candidate_list = "\n".join([
            f"{i+1}. {c['name']} (Location: {c.get('city', 'Unknown')}, {c.get('state', 'Unknown')})"
            for i, c in enumerate(candidates[:10])
        ])

        return f"""You are an expert at matching company names. Given a search query and a list of candidate companies, rank them by how likely they are to be the company the user is looking for.

Search Query: "{query}"{location_context}

Candidate Companies:
{candidate_list}

Consider:
1. Exact name matches or slight variations (Inc, Corp, LLC differences)
2. Acronym expansions (IBM = International Business Machines)
3. Location relevance if provided
4. Industry context clues in the name
5. Common misspellings or abbreviations

Output your ranking as a numbered list from most to least likely match, with a confidence score (0-100) and brief reason:

Ranking:"""

    def _parse_rankings(self, response, candidates):
        """Parse Gemma's ranking response."""
        import re

        # Extract rankings with scores
        pattern = r'(\d+)\.\s*(.+?)\s*[-–]\s*(\d+)%?'
        matches = re.findall(pattern, response)

        ranked = []
        for rank, name_fragment, score in matches:
            # Match back to original candidates
            for c in candidates:
                if name_fragment.strip().lower() in c['name'].lower():
                    c_copy = c.copy()
                    c_copy['gemma_score'] = int(score) / 100
                    c_copy['gemma_rank'] = int(rank)
                    ranked.append(c_copy)
                    break

        # Add any candidates not ranked by Gemma
        ranked_names = {r['name'] for r in ranked}
        for c in candidates:
            if c['name'] not in ranked_names:
                c_copy = c.copy()
                c_copy['gemma_score'] = c['score'] * 0.8  # Penalize unranked
                ranked.append(c_copy)

        return sorted(ranked, key=lambda x: x['gemma_score'], reverse=True)
```

**Integration Point:**

```python
# In CompanyMatcher.match_with_location()

def match_with_location(self, query, city=None, state=None, top_k=10):
    # Phase 1-2: Existing hybrid matching
    results = self._hybrid_match(query, city, state, candidate_k=100)

    # Phase 3: Gemma re-ranking for ambiguous results
    if self.gemma_reranker and self._needs_reranking(results):
        ambiguous = [r for r in results if 0.75 <= r['score'] <= 0.92]

        if len(ambiguous) >= 3:
            reranked = self.gemma_reranker.rerank(
                query, ambiguous, city, state
            )
            results = self._merge_rankings(results, reranked)

    return results[:top_k]

def _needs_reranking(self, results):
    """Determine if Gemma re-ranking would help."""
    if len(results) < 2:
        return False

    # Check if top scores are close (ambiguous)
    top_score = results[0]['score']
    second_score = results[1]['score']

    return (top_score - second_score) < 0.05 and top_score < 0.95
```

### 2.2 Opportunity 2: Company Name Normalization

**Problem:** Same company appears with many variants (Inc, Corp, LLC, punctuation differences).

**Solution:** Use Gemma to normalize names at index time.

```python
class GemmaNormalizer:
    """Normalize company names to canonical form."""

    NORMALIZATION_PROMPT = """Normalize this company name to its canonical form.
Rules:
- Remove legal suffixes (Inc, Corp, LLC, Ltd) unless they distinguish companies
- Expand common abbreviations (Intl → International, Natl → National)
- Standardize punctuation and spacing
- Keep the most recognizable form of the name

Company: {name}
Normalized:"""

    def normalize(self, name: str) -> str:
        """Return normalized company name."""
        prompt = self.NORMALIZATION_PROMPT.format(name=name)

        response = self._generate(prompt, max_tokens=50)

        # Extract normalized name
        normalized = response.strip().split('\n')[0]

        return normalized if normalized else name

    def batch_normalize(self, names: list[str], batch_size: int = 32) -> list[str]:
        """Normalize multiple names efficiently."""
        # Use batched inference for efficiency
        ...
```

**Use at Index Build Time:**

```python
# In IndexBuilder.build()

def build(self, filepath: str, data: list[dict] = None):
    ...

    # Optional: Normalize names with Gemma
    if self.config.use_gemma_normalization:
        original_names = [d['Company Name'] for d in data]
        normalized_names = self.normalizer.batch_normalize(original_names)

        # Store both for matching
        self.name_to_normalized = dict(zip(original_names, normalized_names))
```

### 2.3 Opportunity 3: Contextual Disambiguation

**Problem:** "Northwest Medical" vs "Northwest Construction" both match "Northwest" with similar semantic scores.

**Solution:** Use Gemma to extract and compare business context.

```python
class GemmaContextExtractor:
    """Extract business context from company names."""

    EXTRACTION_PROMPT = """Extract business context from this company name.

Company: {name}

Provide:
- Industry (e.g., Healthcare, Construction, Finance, Technology)
- Type (e.g., Hospital, Clinic, Contractor, Bank, Software Company)
- Geographic Scope (Local, Regional, National, International)
- Any other distinguishing characteristics

Format as JSON:"""

    def extract_context(self, name: str) -> dict:
        """Extract structured context from company name."""
        prompt = self.EXTRACTION_PROMPT.format(name=name)
        response = self._generate(prompt, max_tokens=200)

        try:
            import json
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            return json.loads(response[start:end])
        except:
            return {}

    def compare_contexts(self, query_context: dict, candidate_context: dict) -> float:
        """Score context alignment between query and candidate."""
        score = 0.0

        if query_context.get('industry') == candidate_context.get('industry'):
            score += 0.4
        if query_context.get('type') == candidate_context.get('type'):
            score += 0.3
        if query_context.get('geographic_scope') == candidate_context.get('geographic_scope'):
            score += 0.2

        return score
```

### 2.4 Opportunity 4: Natural Language Explanations

**Problem:** Current rationales are template-based and can feel mechanical.

**Solution:** Use Gemma to generate natural, context-aware explanations.

```python
class GemmaRationaleGenerator:
    """Generate natural language match explanations."""

    RATIONALE_PROMPT = """Explain why "{candidate}" is a {quality} match for the search "{query}".

Match Details:
- Overall Score: {score}%
- Name Similarity: {string_score}%
- Semantic Similarity: {semantic_score}%
- Acronym Match: {is_acronym}
- Location Match: {location_match}

Write a brief, professional explanation (2-3 sentences) that a data entry clerk would find helpful. Focus on why this match makes sense or what concerns they should have."""

    def generate_rationale(self, query: str, candidate: str,
                          score_details: dict) -> str:
        """Generate natural language rationale."""
        quality = self._score_to_quality(score_details['score'])

        prompt = self.RATIONALE_PROMPT.format(
            query=query,
            candidate=candidate,
            quality=quality,
            score=round(score_details['score'] * 100),
            string_score=round(score_details.get('string_score', 0) * 100),
            semantic_score=round(score_details.get('semantic_score', 0) * 100),
            is_acronym="Yes" if score_details.get('acronym_fidelity', 0) > 0.7 else "No",
            location_match="Yes" if score_details.get('location_score', 0) > 0.5 else "No/Unknown"
        )

        return self._generate(prompt, max_tokens=150)

    def _score_to_quality(self, score: float) -> str:
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "strong"
        elif score >= 0.70:
            return "moderate"
        else:
            return "weak"
```

### 2.5 Opportunity 5: Query Understanding & Correction

**Problem:** Malformed queries, typos, and implicit context are hard to handle.

**Solution:** Use Gemma to understand and normalize queries before matching.

```python
class GemmaQueryProcessor:
    """Process and enhance search queries."""

    QUERY_PROMPT = """You are a company name search assistant. Analyze this search query and provide:
1. Corrected spelling (if needed)
2. Likely full name (if query is abbreviated or partial)
3. Inferred industry/type (if apparent)
4. Any alternative names to also search for

Query: "{query}"

Respond in JSON format:
{{
  "corrected_query": "...",
  "expanded_name": "...",
  "industry_hint": "...",
  "alternatives": ["...", "..."]
}}"""

    def process_query(self, query: str) -> dict:
        """Analyze and enhance a search query."""
        prompt = self.QUERY_PROMPT.format(query=query)
        response = self._generate(prompt, max_tokens=200)

        try:
            import json
            start = response.find('{')
            end = response.rfind('}') + 1
            result = json.loads(response[start:end])
            result['original_query'] = query
            return result
        except:
            return {'original_query': query, 'corrected_query': query}

    def should_expand_search(self, processed: dict) -> bool:
        """Determine if we should search alternatives too."""
        return (
            processed.get('corrected_query') != processed.get('original_query') or
            len(processed.get('alternatives', [])) > 0
        )
```

---

## 3. Implementation Approaches

### 3.1 Approach A: Local Inference with Ollama

**Pros:** No API costs, data stays local, low latency
**Cons:** Requires GPU, model quality limited by hardware

```python
# Using Ollama for local Gemma inference

import requests

class OllamaGemmaClient:
    """Local Gemma inference via Ollama."""

    def __init__(self, model: str = "gemma2:9b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1
                }
            }
        )
        return response.json()["response"]

# Setup: ollama pull gemma2:9b
```

### 3.2 Approach B: Quantized Models with llama.cpp

**Pros:** Runs on CPU, minimal VRAM, portable
**Cons:** Slower, some quality loss from quantization

```python
# Using llama-cpp-python for GGUF models

from llama_cpp import Llama

class LlamaCppGemmaClient:
    """Quantized Gemma inference via llama.cpp."""

    def __init__(self, model_path: str = "gemma-2-9b-Q4_K_M.gguf"):
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0  # CPU only, or set for GPU offload
        )

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion."""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            stop=["</s>", "\n\n"]
        )
        return output["choices"][0]["text"]
```

### 3.3 Approach C: Hybrid - Embeddings + Selective LLM

**Pros:** Best balance of speed and quality
**Cons:** More complex to implement

```python
class HybridGemmaIntegration:
    """Use Gemma selectively where it adds most value."""

    def __init__(self, config):
        self.config = config

        # Gemma for re-ranking and explanations only
        self.gemma = OllamaGemmaClient(model="gemma2:2b")  # Smaller model

        # Keep sentence-transformers for bulk embedding
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def match(self, query: str, top_k: int = 10, **kwargs) -> list:
        # Fast path: Use existing hybrid matching
        results = self.fast_match(query, top_k=50, **kwargs)

        # Slow path: Gemma re-ranking for ambiguous top results
        if self._is_ambiguous(results[:10]):
            results[:10] = self.gemma_rerank(query, results[:10])

        # Optional: Gemma explanations for top-3
        for r in results[:3]:
            r['gemma_rationale'] = self.gemma_explain(query, r)

        return results[:top_k]
```

---

## 4. Performance Considerations

### 4.1 Latency Impact

| Operation | Current | With Gemma 2B | With Gemma 9B |
|-----------|---------|---------------|---------------|
| Basic query | 50-100ms | 50-100ms (no change) | 50-100ms |
| Ambiguous query | 50-100ms | +200-500ms | +500-1500ms |
| Full explanation | N/A | +100-200ms | +300-600ms |
| Query preprocessing | N/A | +50-100ms | +150-300ms |

### 4.2 When to Use Gemma

**Always Use (negligible overhead):**
- Index-time name normalization (batched, offline)
- Precomputing context for top companies

**Selective Use (on-demand):**
- Re-ranking when top scores within 5% of each other
- When user requests detailed explanation
- For queries with detected typos/abbreviations

**Never Use:**
- Simple exact matches (score > 98%)
- High-confidence matches (score > 95%, clear top result)
- Batch processing without GPU resources

### 4.3 Caching Strategy

```python
class GemmaResultCache:
    """Cache Gemma results to avoid repeated inference."""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def get_or_compute(self, key: str, compute_fn, *args, **kwargs):
        """Return cached result or compute and cache."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['value']

        result = compute_fn(*args, **kwargs)
        self.cache[key] = {'value': result, 'timestamp': time.time()}

        return result

# Cache normalized names permanently
# Cache re-rankings for 1 hour
# Cache explanations for 24 hours
```

---

## 5. Quality Assessment

### 5.1 Expected Improvements

| Scenario | Current Accuracy | With Gemma | Improvement |
|----------|------------------|------------|-------------|
| Exact matches | 99%+ | 99%+ | None needed |
| Acronym expansion | 85-90% | 92-95% | +5-7% |
| Typo handling | 60-70% | 85-90% | +20-25% |
| Industry disambiguation | 75-80% | 88-92% | +10-15% |
| Location context | 80-85% | 90-93% | +8-10% |
| **Overall control set** | ~90% | ~94-96% | +4-6% |

### 5.2 Validation Plan

```python
# Gemma validation against control set

def validate_gemma_impact(control_set, matcher_without_gemma, matcher_with_gemma):
    """Compare accuracy with and without Gemma."""

    results = {
        'without_gemma': {'correct': 0, 'total': 0},
        'with_gemma': {'correct': 0, 'total': 0},
        'improvements': [],
        'regressions': []
    }

    for case in control_set:
        query = case['query']
        expected = case['expected_match']

        # Without Gemma
        matches_base = matcher_without_gemma.match(query, top_k=5)
        base_correct = matches_base[0]['name'] == expected
        results['without_gemma']['total'] += 1
        results['without_gemma']['correct'] += int(base_correct)

        # With Gemma
        matches_gemma = matcher_with_gemma.match(query, top_k=5)
        gemma_correct = matches_gemma[0]['name'] == expected
        results['with_gemma']['total'] += 1
        results['with_gemma']['correct'] += int(gemma_correct)

        # Track changes
        if gemma_correct and not base_correct:
            results['improvements'].append({
                'query': query,
                'expected': expected,
                'base_top': matches_base[0]['name'],
                'gemma_top': matches_gemma[0]['name']
            })
        elif base_correct and not gemma_correct:
            results['regressions'].append({
                'query': query,
                'expected': expected,
                'base_top': matches_base[0]['name'],
                'gemma_top': matches_gemma[0]['name']
            })

    return results
```

---

## 6. Recommended Implementation Plan

### Phase 1: Foundation (1-2 weeks)

1. **Setup Ollama + Gemma 2B** locally for development
2. **Implement GemmaReranker** for ambiguous matches
3. **Add configuration** to enable/disable Gemma features
4. **Benchmark latency** impact on typical queries

### Phase 2: Integration (2-3 weeks)

5. **Integrate selective re-ranking** into matcher pipeline
6. **Add caching layer** for Gemma results
7. **Implement GemmaRationaleGenerator** as optional enhancement
8. **Validate against control set** - ensure no regressions

### Phase 3: Optimization (2-3 weeks)

9. **Tune prompts** for best accuracy
10. **Evaluate Gemma 9B** vs 2B quality/speed tradeoff
11. **Implement batch processing** for index-time normalization
12. **Add query preprocessing** for typo detection

### Phase 4: Production (1-2 weeks)

13. **Deploy with feature flags** for gradual rollout
14. **Monitor latency and accuracy** metrics
15. **Collect user feedback** on explanation quality
16. **Iterate on prompts** based on failure cases

---

## 7. Resource Requirements

### Hardware (Recommended)

| Setup | Gemma Model | RAM | GPU VRAM | Notes |
|-------|-------------|-----|----------|-------|
| Minimal | 2B Q4 | 8 GB | None (CPU) | ~500ms/query |
| Development | 2B | 16 GB | 6 GB | ~200ms/query |
| Production | 9B | 32 GB | 16 GB | ~300ms/query |
| High-quality | 27B | 64 GB | 48 GB | ~600ms/query |

### Software Dependencies

```
# requirements-gemma.txt

# Option 1: Ollama (recommended for simplicity)
ollama  # System install

# Option 2: Direct transformers
transformers>=4.40.0
accelerate>=0.28.0
torch>=2.2.0

# Option 3: Quantized via llama.cpp
llama-cpp-python>=0.2.50
```

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Latency regression | User experience | Selective use, caching, async processing |
| Model hallucination | Wrong matches | Validate against known examples, constrain outputs |
| GPU unavailable | Feature disabled | Graceful fallback to existing matching |
| Prompt injection | Security | Sanitize inputs, limit model capabilities |
| Model size | Deployment complexity | Use quantized models, Ollama |

---

## 9. Conclusion & Recommendation

**Gemma integration can meaningfully improve FineTuner's matching accuracy, particularly for:**
- Ambiguous matches (75-92% score range)
- Acronym and abbreviation handling
- Typo tolerance
- Natural language explanations

**Recommended approach:**
1. Start with **Gemma 2B via Ollama** for low resource requirements
2. Use **selectively** for ambiguous matches only (not every query)
3. **Cache aggressively** to minimize repeated inference
4. **Validate thoroughly** against control set before production

**Expected outcome:**
- 4-6% improvement in overall accuracy
- Better user confidence in match explanations
- Graceful degradation when Gemma unavailable

The investment is worthwhile for improving edge cases without disrupting the fast path for high-confidence matches.

---

*End of Gemma Integration Assessment*
