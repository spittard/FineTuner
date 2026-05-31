# Proposal 3: Architecture & Developer Experience Improvements

**Document Version:** 1.0
**Created:** 2026-04-22
**Focus Area:** Code Organization, Testing, Observability, and Developer Productivity

---

## Executive Summary

This proposal focuses on improving the maintainability, testability, and operational visibility of the FineTuner codebase. The goal is to reduce technical debt, improve developer productivity, and enable confident deployments.

**Key Outcomes:**
- Clean separation of concerns with well-defined interfaces
- Comprehensive test coverage (>80%)
- Full observability with metrics, logging, and tracing
- Streamlined deployment pipeline
- Better documentation and onboarding experience

---

## 1. Code Refactoring: Single Responsibility

### 1.1 Problem

`CompanyMatcher` (1,850 lines) handles too many responsibilities:
- Index building
- Caching
- Searching
- Scoring
- Explanation generation

This makes testing difficult and changes risky.

### 1.2 Solution: Extract Focused Classes

```
src/finetuner/
├── core/
│   ├── __init__.py
│   ├── matcher.py          # Orchestrator only (200 lines)
│   ├── index_builder.py    # Index building logic
│   ├── searcher.py         # Search execution
│   ├── scorer.py           # Scoring algorithms
│   ├── cache_manager.py    # Cache operations
│   ├── vector_store.py     # FAISS wrapper (existing)
│   └── models/
│       ├── __init__.py
│       ├── match_result.py # Result dataclasses
│       ├── cache_info.py   # Cache metadata
│       └── config.py       # Configuration models
├── scoring/
│   ├── __init__.py
│   ├── string_scorer.py    # String similarity
│   ├── semantic_scorer.py  # Semantic scoring
│   ├── concept_scorer.py   # Concept alignment
│   ├── location_scorer.py  # Location matching
│   ├── acronym_scorer.py   # Acronym handling
│   └── composite.py        # Score combination
├── preprocessing/
│   ├── __init__.py
│   ├── text_processor.py   # Text cleaning
│   ├── acronym_engine.py   # Acronym detection
│   └── normalizers.py      # Location, name normalization
└── utils/
    ├── __init__.py
    ├── hashing.py          # Cache key generation
    └── progress.py         # Progress bar utilities
```

### 1.3 Refactored Matcher

```python
# core/matcher.py - Clean orchestrator

from dataclasses import dataclass
from typing import Optional, List

from finetuner.core.index_builder import IndexBuilder
from finetuner.core.searcher import Searcher
from finetuner.core.cache_manager import CacheManager
from finetuner.core.models import MatchResult, MatchConfig


@dataclass
class CompanyMatcher:
    """
    High-level orchestrator for company name matching.

    Delegates to specialized components for each responsibility.
    """

    config: MatchConfig
    index_builder: IndexBuilder
    searcher: Searcher
    cache_manager: CacheManager

    @classmethod
    def create(cls, config: Optional[MatchConfig] = None) -> 'CompanyMatcher':
        """Factory method to create fully configured matcher."""
        config = config or MatchConfig.default()

        return cls(
            config=config,
            index_builder=IndexBuilder(config),
            searcher=Searcher(config),
            cache_manager=CacheManager(config.cache_dir)
        )

    def build_index(self, filepath: str = None, data: List[dict] = None) -> bool:
        """Build or load index from cache."""
        # Check cache first
        cache_key = self.cache_manager.get_key_for_file(filepath)
        if self.cache_manager.exists(cache_key):
            return self._load_cached_index(cache_key)

        # Build new index
        index_data = self.index_builder.build(filepath=filepath, data=data)

        # Save to cache
        self.cache_manager.save(cache_key, index_data)

        # Initialize searcher
        self.searcher.initialize(index_data)

        return True

    def match(
        self,
        query: str,
        city: Optional[str] = None,
        state: Optional[str] = None,
        top_k: int = 10
    ) -> List[MatchResult]:
        """Execute match query."""
        if not self.searcher.is_ready():
            raise RuntimeError("Index not loaded. Call build_index() first.")

        return self.searcher.search(
            query=query,
            city=city,
            state=state,
            top_k=top_k
        )

    def explain(self, query: str, match_name: str) -> dict:
        """Get detailed explanation for a match."""
        return self.searcher.explain(query, match_name)
```

### 1.4 Scoring Components

```python
# scoring/composite.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ScoreComponents:
    """All score components for a match."""
    string_score: float
    semantic_score: float
    concept_alignment: float
    location_score: float
    acronym_fidelity: float
    lexical_boost: float
    popularity_boost: float

class Scorer(ABC):
    """Base class for scoring components."""

    @abstractmethod
    def score(self, query: str, candidate: str, **context) -> float:
        """Calculate score component."""
        pass

class CompositeScorer:
    """Combines multiple scorers with configurable weights."""

    def __init__(self, scorers: Dict[str, Scorer], weights: Dict[str, float]):
        self.scorers = scorers
        self.weights = weights

    def score(self, query: str, candidate: str, **context) -> ScoreComponents:
        """Calculate all score components."""
        components = {}

        for name, scorer in self.scorers.items():
            components[name] = scorer.score(query, candidate, **context)

        return ScoreComponents(**components)

    def combine(self, components: ScoreComponents) -> float:
        """Combine components into final score."""
        total = 0.0
        total += components.string_score * self.weights.get('string', 0.5)
        total += components.semantic_score * self.weights.get('semantic', 0.25)
        total += components.concept_alignment * self.weights.get('concept', 0.25)

        # Apply boosts
        total += components.lexical_boost
        total += components.popularity_boost

        # Location blend
        if components.location_score > 0:
            total = total * 0.7 + components.location_score * 0.3

        return min(1.0, total)
```

---

## 2. Configuration Management

### 2.1 Problem

Configuration is scattered across:
- `model_config.json`
- `tier_config.json`
- Hardcoded values in `matcher.py`
- Environment variables

### 2.2 Solution: Unified Configuration

```python
# core/models/config.py

from dataclasses import dataclass, field
from typing import Optional, Dict
import os
import json

@dataclass
class ModelConfig:
    """Embedding model configuration."""
    name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    normalize_embeddings: bool = True
    batch_size: int = 2048

@dataclass
class ScoringConfig:
    """Scoring weights and thresholds."""
    string_weight: float = 0.50
    semantic_weight: float = 0.25
    concept_weight: float = 0.25
    location_weight: float = 0.30  # When location provided

    lexical_boost_thresholds: Dict[float, float] = field(
        default_factory=lambda: {0.92: 0.95, 0.80: 0.90}
    )
    acronym_boost_factor: float = 0.15
    popularity_boost_max: float = 0.05

@dataclass
class TierConfig:
    """Score tier thresholds."""
    great: int = 98
    high: int = 93
    medium: int = 88

@dataclass
class CacheConfig:
    """Cache settings."""
    directory: str = "company_matcher_cache"
    version: str = "v4.1_location_decoupled"
    ttl_seconds: int = 86400 * 30  # 30 days

@dataclass
class SearchConfig:
    """Search parameters."""
    candidate_k: int = 1000  # FAISS retrieval count
    default_top_k: int = 10
    timeout_seconds: float = 120.0

@dataclass
class MatchConfig:
    """Complete matcher configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    tiers: TierConfig = field(default_factory=TierConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def default(cls) -> 'MatchConfig':
        """Load default configuration."""
        return cls()

    @classmethod
    def from_file(cls, path: str) -> 'MatchConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'MatchConfig':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**data.get('model', {})),
            scoring=ScoringConfig(**data.get('scoring', {})),
            tiers=TierConfig(**data.get('tiers', {})),
            cache=CacheConfig(**data.get('cache', {})),
            search=SearchConfig(**data.get('search', {}))
        )

    @classmethod
    def from_env(cls) -> 'MatchConfig':
        """Load configuration from environment variables."""
        config = cls.default()

        if model_name := os.getenv('FINETUNER_MODEL'):
            config.model.name = model_name
        if cache_dir := os.getenv('FINETUNER_CACHE_DIR'):
            config.cache.directory = cache_dir
        if candidate_k := os.getenv('FINETUNER_CANDIDATE_K'):
            config.search.candidate_k = int(candidate_k)

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
```

### 2.3 Unified Config File

```json
{
  "model": {
    "name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "normalize_embeddings": true,
    "batch_size": 2048
  },
  "scoring": {
    "string_weight": 0.50,
    "semantic_weight": 0.25,
    "concept_weight": 0.25,
    "location_weight": 0.30,
    "lexical_boost_thresholds": {"0.92": 0.95, "0.80": 0.90},
    "acronym_boost_factor": 0.15,
    "popularity_boost_max": 0.05
  },
  "tiers": {
    "great": 98,
    "high": 93,
    "medium": 88
  },
  "cache": {
    "directory": "company_matcher_cache",
    "version": "v4.2",
    "ttl_seconds": 2592000
  },
  "search": {
    "candidate_k": 1000,
    "default_top_k": 10,
    "timeout_seconds": 120.0
  }
}
```

---

## 3. Comprehensive Testing

### 3.1 Problem

Limited test coverage:
- No unit tests for scoring functions
- No integration tests for RPC
- No load/stress tests
- Control set is only validation mechanism

### 3.2 Solution: Test Strategy

```
tests/
├── unit/
│   ├── test_text_preprocessor.py
│   ├── test_string_scorer.py
│   ├── test_semantic_scorer.py
│   ├── test_concept_scorer.py
│   ├── test_location_scorer.py
│   ├── test_acronym_engine.py
│   ├── test_cache_manager.py
│   └── test_config.py
├── integration/
│   ├── test_matcher_integration.py
│   ├── test_rpc_server.py
│   ├── test_web_app.py
│   └── test_end_to_end.py
├── performance/
│   ├── test_query_latency.py
│   ├── test_throughput.py
│   └── test_memory_usage.py
├── fixtures/
│   ├── sample_companies.json
│   ├── control_set.json
│   └── edge_cases.json
├── conftest.py
└── pytest.ini
```

### 3.3 Unit Test Examples

```python
# tests/unit/test_string_scorer.py

import pytest
from finetuner.scoring.string_scorer import StringScorer

class TestStringScorer:
    """Unit tests for string similarity scoring."""

    @pytest.fixture
    def scorer(self):
        return StringScorer()

    def test_exact_match(self, scorer):
        """Exact match should score 1.0."""
        score = scorer.score("IBM", "IBM")
        assert score == 1.0

    def test_case_insensitive(self, scorer):
        """Matching should be case insensitive."""
        score = scorer.score("ibm", "IBM")
        assert score == 1.0

    def test_partial_overlap(self, scorer):
        """Partial word overlap should have moderate score."""
        score = scorer.score("American Airlines", "American Express")
        assert 0.3 < score < 0.7  # Shares "American"

    def test_no_overlap(self, scorer):
        """No overlap should have low score."""
        score = scorer.score("Apple", "Microsoft")
        assert score < 0.3

    def test_substring_match(self, scorer):
        """Substring should have high score."""
        score = scorer.score("Microsoft", "Microsoft Corporation")
        assert score > 0.8

    @pytest.mark.parametrize("query,target,min_score,max_score", [
        ("Bank of America", "Bank of America Corp", 0.85, 1.0),
        ("Wells Fargo", "Wells Fargo Bank", 0.80, 0.95),
        ("J.P. Morgan", "JPMorgan Chase", 0.50, 0.80),
        ("3M", "3M Company", 0.70, 0.95),
    ])
    def test_common_patterns(self, scorer, query, target, min_score, max_score):
        """Test common matching patterns."""
        score = scorer.score(query, target)
        assert min_score <= score <= max_score


# tests/unit/test_acronym_engine.py

import pytest
from finetuner.preprocessing.acronym_engine import AcronymEngine

class TestAcronymEngine:
    """Unit tests for acronym handling."""

    def test_generate_acronym(self):
        """Test acronym generation."""
        assert AcronymEngine.generate_acronym("International Business Machines") == "IBM"
        assert AcronymEngine.generate_acronym("American Bar Association") == "ABA"
        assert AcronymEngine.generate_acronym("National Aeronautics and Space Administration") == "NASA"

    def test_is_acronym(self):
        """Test acronym detection."""
        assert AcronymEngine.is_acronym("IBM") == True
        assert AcronymEngine.is_acronym("Ibm") == False
        assert AcronymEngine.is_acronym("I.B.M.") == True
        assert AcronymEngine.is_acronym("inc") == True  # Known abbreviation

    def test_acronym_fidelity(self):
        """Test fidelity scoring."""
        # Perfect match
        fidelity = AcronymEngine.calculate_fidelity("IBM", "International Business Machines")
        assert fidelity >= 0.95

        # Partial match
        fidelity = AcronymEngine.calculate_fidelity("IBM", "International Bank of Miami")
        assert 0.5 < fidelity < 0.9

        # No match
        fidelity = AcronymEngine.calculate_fidelity("IBM", "Apple Inc")
        assert fidelity < 0.3

    def test_two_letter_handling(self):
        """Test 2-letter acronym handling."""
        # Valid company acronym
        assert AcronymEngine.should_process_two_letter("GE") == True

        # State code (should not process)
        assert AcronymEngine.should_process_two_letter("NY") == False
        assert AcronymEngine.should_process_two_letter("CA") == False
```

### 3.4 Integration Tests

```python
# tests/integration/test_matcher_integration.py

import pytest
import tempfile
import json
from finetuner.core.matcher import CompanyMatcher

@pytest.fixture
def sample_data():
    """Create sample company data."""
    return [
        {"Company Name": "Apple Inc", "City": "Cupertino", "State": "CA", "Count": 100},
        {"Company Name": "Microsoft Corporation", "City": "Redmond", "State": "WA", "Count": 150},
        {"Company Name": "International Business Machines", "City": "Armonk", "State": "NY", "Count": 80},
        {"Company Name": "IBM", "City": "Armonk", "State": "NY", "Count": 50},
        {"Company Name": "American Bar Association", "City": "Chicago", "State": "IL", "Count": 30},
    ]

@pytest.fixture
def matcher_with_data(sample_data, tmp_path):
    """Create matcher with loaded data."""
    # Write sample data
    data_file = tmp_path / "companies.json"
    with open(data_file, 'w') as f:
        json.dump(sample_data, f)

    # Create and initialize matcher
    matcher = CompanyMatcher.create()
    matcher.build_index(filepath=str(data_file))

    return matcher

class TestMatcherIntegration:
    """Integration tests for full matching pipeline."""

    def test_exact_match_found(self, matcher_with_data):
        """Exact company name should be found."""
        results = matcher_with_data.match("Apple Inc", top_k=5)

        assert len(results) > 0
        assert results[0].name == "Apple Inc"
        assert results[0].score >= 0.99

    def test_acronym_expansion(self, matcher_with_data):
        """Acronym should expand to full name."""
        results = matcher_with_data.match("ABA", top_k=5)

        # Should find American Bar Association
        names = [r.name for r in results]
        assert "American Bar Association" in names

    def test_location_boost(self, matcher_with_data):
        """Location match should boost score."""
        results_no_loc = matcher_with_data.match("IBM", top_k=5)
        results_with_loc = matcher_with_data.match("IBM", city="Armonk", state="NY", top_k=5)

        # With location should have higher or equal score for matching location
        ibm_no_loc = next(r for r in results_no_loc if r.name == "IBM")
        ibm_with_loc = next(r for r in results_with_loc if r.name == "IBM")

        assert ibm_with_loc.score >= ibm_no_loc.score

    def test_semantic_similarity(self, matcher_with_data):
        """Semantically similar queries should find matches."""
        results = matcher_with_data.match("International Business Machines", top_k=5)

        # Should find IBM company
        names = [r.name for r in results]
        assert "International Business Machines" in names or "IBM" in names


# tests/integration/test_rpc_server.py

import pytest
import threading
import time
from finetuner.core.cache_rpc import run_server, connect, is_server_running

@pytest.fixture(scope="module")
def rpc_server():
    """Start RPC server for tests."""
    # Start server in background thread
    server_thread = threading.Thread(
        target=run_server,
        kwargs={'port': 19876},  # Use different port for tests
        daemon=True
    )
    server_thread.start()

    # Wait for server to be ready
    for _ in range(10):
        if is_server_running(port=19876):
            break
        time.sleep(0.5)

    yield

    # Server will be terminated when thread exits

class TestRPCServer:
    """Integration tests for RPC server."""

    def test_server_status(self, rpc_server):
        """Server should return status."""
        client = connect(port=19876)
        status = client.get_status()

        assert 'running' in status
        assert status['running'] == True

    def test_list_caches(self, rpc_server):
        """Should list available caches."""
        client = connect(port=19876)
        caches = client.list_available_caches()

        assert isinstance(caches, list)

    def test_search_without_cache(self, rpc_server):
        """Search without loaded cache should return error."""
        client = connect(port=19876)
        result = client.search("test query")

        # Should indicate no cache loaded
        assert 'error' in result or len(result.get('results', [])) == 0
```

### 3.5 Performance Tests

```python
# tests/performance/test_query_latency.py

import pytest
import time
import statistics

@pytest.fixture
def loaded_matcher():
    """Matcher with production-like data."""
    # Load from actual cache or test fixture
    ...

class TestQueryLatency:
    """Performance tests for query latency."""

    @pytest.mark.slow
    def test_p50_latency(self, loaded_matcher):
        """P50 latency should be under 50ms."""
        queries = ["IBM", "Microsoft", "Apple", "Bank of America", "Wells Fargo"]
        latencies = []

        for _ in range(100):
            for query in queries:
                start = time.perf_counter()
                loaded_matcher.match(query, top_k=10)
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)

        p50 = statistics.median(latencies)
        assert p50 < 50, f"P50 latency {p50:.1f}ms exceeds 50ms threshold"

    @pytest.mark.slow
    def test_p99_latency(self, loaded_matcher):
        """P99 latency should be under 200ms."""
        # Similar to above but check p99
        ...

    @pytest.mark.slow
    def test_cold_start_latency(self, tmp_path):
        """First query after load should be under 5s."""
        # Measure time from load to first query result
        ...
```

### 3.6 Pytest Configuration

```ini
# tests/pytest.ini

[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests

addopts =
    --verbose
    --tb=short
    --strict-markers
    -ra

filterwarnings =
    ignore::DeprecationWarning
```

---

## 4. Observability

### 4.1 Problem

No visibility into:
- Query patterns and failures
- System performance
- Cache hit rates
- Error rates

### 4.2 Solution: Metrics & Logging

```python
# observability/metrics.py

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable

# Query metrics
QUERY_TOTAL = Counter(
    'finetuner_queries_total',
    'Total number of queries',
    ['status', 'has_location']
)

QUERY_LATENCY = Histogram(
    'finetuner_query_latency_seconds',
    'Query latency in seconds',
    ['query_type'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

TOP_MATCH_SCORE = Histogram(
    'finetuner_top_match_score',
    'Score of top match',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.98, 1.0]
)

# Cache metrics
CACHE_HIT_TOTAL = Counter(
    'finetuner_cache_hits_total',
    'Number of cache hits',
    ['cache_type']
)

CACHE_MISS_TOTAL = Counter(
    'finetuner_cache_misses_total',
    'Number of cache misses',
    ['cache_type']
)

INDEX_SIZE = Gauge(
    'finetuner_index_size',
    'Number of companies in index'
)

MEMORY_USAGE_BYTES = Gauge(
    'finetuner_memory_usage_bytes',
    'Memory usage in bytes'
)

# System info
SYSTEM_INFO = Info(
    'finetuner_build',
    'Build information'
)


def track_query(func: Callable) -> Callable:
    """Decorator to track query metrics."""
    @wraps(func)
    def wrapper(self, query: str, *args, **kwargs):
        has_location = bool(kwargs.get('city') or kwargs.get('state'))

        start_time = time.perf_counter()
        try:
            result = func(self, query, *args, **kwargs)

            QUERY_TOTAL.labels(
                status='success',
                has_location=str(has_location)
            ).inc()

            if result:
                TOP_MATCH_SCORE.observe(result[0].score)

            return result

        except Exception as e:
            QUERY_TOTAL.labels(
                status='error',
                has_location=str(has_location)
            ).inc()
            raise

        finally:
            latency = time.perf_counter() - start_time
            QUERY_LATENCY.labels(
                query_type='with_location' if has_location else 'name_only'
            ).observe(latency)

    return wrapper


# observability/logging.py

import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    """JSON-structured logging for production environments."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_handler()

    def _setup_handler(self):
        """Configure JSON output handler."""
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_query(
        self,
        query: str,
        top_k: int,
        city: str = None,
        state: str = None,
        latency_ms: float = None,
        result_count: int = None,
        top_score: float = None
    ):
        """Log a search query."""
        self.logger.info(json.dumps({
            'event': 'query',
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'top_k': top_k,
            'city': city,
            'state': state,
            'latency_ms': latency_ms,
            'result_count': result_count,
            'top_score': top_score
        }))

    def log_cache_hit(self, cache_key: str, cache_type: str):
        """Log a cache hit."""
        self.logger.debug(json.dumps({
            'event': 'cache_hit',
            'timestamp': datetime.utcnow().isoformat(),
            'cache_key': cache_key[:16],
            'cache_type': cache_type
        }))

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error."""
        self.logger.error(json.dumps({
            'event': 'error',
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }))


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record):
        if isinstance(record.msg, str):
            try:
                # Already JSON
                json.loads(record.msg)
                return record.msg
            except json.JSONDecodeError:
                # Plain string, wrap it
                return json.dumps({
                    'level': record.levelname,
                    'message': record.msg,
                    'timestamp': datetime.utcnow().isoformat()
                })
        return str(record.msg)
```

### 4.3 Metrics Endpoint

```python
# web/app.py additions

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/health')
def health():
    """Health check endpoint."""
    status = search_service.get_status()

    if status.get('status') == 'ready':
        return jsonify({'status': 'healthy', **status}), 200
    else:
        return jsonify({'status': 'unhealthy', **status}), 503

@app.route('/ready')
def ready():
    """Readiness check endpoint."""
    status = search_service.get_status()

    if status.get('status') == 'ready':
        return '', 200
    else:
        return '', 503
```

---

## 5. API Versioning & Documentation

### 5.1 Problem

No API versioning or OpenAPI documentation.

### 5.2 Solution: Versioned API with OpenAPI

```python
# web/api/v1/__init__.py

from flask import Blueprint
from flask_restx import Api, Resource, fields

v1_bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')
api = Api(
    v1_bp,
    version='1.0',
    title='FineTuner API',
    description='Company name matching API',
    doc='/docs'
)

# Namespaces
search_ns = api.namespace('search', description='Search operations')
cache_ns = api.namespace('cache', description='Cache management')
feedback_ns = api.namespace('feedback', description='User feedback')

# Models
search_request = api.model('SearchRequest', {
    'query': fields.String(required=True, description='Company name to search'),
    'city': fields.String(description='Optional city filter'),
    'state': fields.String(description='Optional state filter'),
    'top_k': fields.Integer(default=10, description='Number of results')
})

match_result = api.model('MatchResult', {
    'rank': fields.Integer(description='Result rank'),
    'company_name': fields.String(description='Matched company name'),
    'score': fields.Float(description='Match score (0-1)'),
    'likeness_percent': fields.Float(description='Score as percentage'),
    'match_type': fields.String(description='Type of match'),
    'city': fields.String(description='Company city'),
    'state': fields.String(description='Company state'),
    'explanation': fields.Raw(description='Detailed scoring breakdown')
})

search_response = api.model('SearchResponse', {
    'success': fields.Boolean,
    'query': fields.String,
    'results': fields.List(fields.Nested(match_result)),
    'total_matches': fields.Integer,
    'latency_ms': fields.Float
})


@search_ns.route('/')
class SearchResource(Resource):
    """Search for company matches."""

    @search_ns.expect(search_request)
    @search_ns.marshal_with(search_response)
    @search_ns.response(200, 'Success')
    @search_ns.response(400, 'Invalid request')
    @search_ns.response(503, 'Service unavailable')
    def post(self):
        """
        Search for company name matches.

        Returns ranked list of matching companies with scores and explanations.
        """
        data = api.payload

        import time
        start = time.perf_counter()

        results = search_service.search(
            query=data['query'],
            city=data.get('city'),
            state=data.get('state'),
            top_k=data.get('top_k', 10)
        )

        latency = (time.perf_counter() - start) * 1000

        return {
            'success': True,
            'query': data['query'],
            'results': results,
            'total_matches': len(results),
            'latency_ms': latency
        }


@search_ns.route('/batch')
class BatchSearchResource(Resource):
    """Batch search for multiple queries."""

    batch_request = api.model('BatchSearchRequest', {
        'queries': fields.List(fields.Nested(search_request), required=True),
    })

    @search_ns.expect(batch_request)
    @search_ns.response(200, 'Success')
    def post(self):
        """
        Search for multiple company names in one request.

        More efficient than multiple single requests.
        """
        data = api.payload
        results = []

        for query_data in data['queries']:
            result = search_service.search(
                query=query_data['query'],
                city=query_data.get('city'),
                state=query_data.get('state'),
                top_k=query_data.get('top_k', 10)
            )
            results.append({
                'query': query_data['query'],
                'results': result
            })

        return {'success': True, 'batch_results': results}
```

---

## 6. CLI Improvements

### 6.1 Problem

Current CLI is basic and lacks common development commands.

### 6.2 Solution: Rich CLI with Click

```python
# cli.py

import click
import json
import sys

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """FineTuner - Company name matching CLI."""
    pass


@cli.command()
@click.argument('query')
@click.option('--city', '-c', help='City filter')
@click.option('--state', '-s', help='State filter')
@click.option('--top-k', '-k', default=10, help='Number of results')
@click.option('--output', '-o', type=click.Choice(['table', 'json']), default='table')
def search(query, city, state, top_k, output):
    """Search for company matches."""
    from finetuner.core.matcher import CompanyMatcher

    matcher = CompanyMatcher.create()
    # Load default cache...

    results = matcher.match(query, city=city, state=state, top_k=top_k)

    if output == 'json':
        click.echo(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        _print_table(results)


@cli.command()
@click.argument('filepath')
@click.option('--model', '-m', default='all-MiniLM-L6-v2', help='Embedding model')
@click.option('--work-dir', '-w', help='Working directory for checkpoints')
def build(filepath, model, work_dir):
    """Build index from company data file."""
    from finetuner.core.matcher import CompanyMatcher
    from finetuner.core.models.config import MatchConfig, ModelConfig

    config = MatchConfig.default()
    config.model.name = model

    matcher = CompanyMatcher.create(config)

    with click.progressbar(length=100, label='Building index') as bar:
        # Hook into progress callbacks
        matcher.build_index(filepath=filepath, work_dir=work_dir)

    click.echo(f"Index built successfully!")


@cli.command()
def serve():
    """Start the RPC server."""
    from finetuner.core.cache_rpc import run_server
    run_server()


@cli.command()
@click.option('--port', '-p', default=5000, help='Port to run on')
@click.option('--debug/--no-debug', default=False, help='Debug mode')
def web(port, debug):
    """Start the web application."""
    from finetuner.web.app import app
    app.run(port=port, debug=debug)


@cli.command()
def status():
    """Check server status."""
    from finetuner.core.cache_rpc import connect, is_server_running

    if not is_server_running():
        click.echo("RPC server is not running")
        sys.exit(1)

    client = connect()
    status = client.get_status()

    click.echo(f"Server Status: {'Running' if status['running'] else 'Stopped'}")
    click.echo(f"Loaded Caches: {status['loaded_caches']}")
    click.echo(f"Total Memory: {status['total_memory_mb']:.0f} MB")


@cli.group()
def cache():
    """Cache management commands."""
    pass


@cache.command('list')
def cache_list():
    """List available caches."""
    from finetuner.core.cache_rpc import connect

    client = connect()
    caches = client.list_available_caches()

    for cache in caches:
        status = "LOADED" if cache['loaded'] else "available"
        click.echo(f"  {cache['cache_key'][:16]}... [{status}] - {cache.get('num_companies', '?')} companies")


@cache.command('load')
@click.argument('cache_key')
def cache_load(cache_key):
    """Load a cache into memory."""
    from finetuner.core.cache_rpc import connect

    client = connect()
    if client.load_cache(cache_key):
        click.echo(f"Loaded: {cache_key}")
    else:
        click.echo(f"Failed to load: {cache_key}", err=True)
        sys.exit(1)


@cache.command('clear')
@click.option('--confirm', is_flag=True, help='Confirm deletion')
def cache_clear(confirm):
    """Clear all caches."""
    if not confirm:
        click.echo("Use --confirm to actually clear caches")
        return

    from finetuner.core.matcher import CompanyMatcher
    matcher = CompanyMatcher.create()
    matcher.cache_manager.clear_all(confirm_delete=True)
    click.echo("Caches cleared")


@cli.command()
@click.argument('control_set_file')
@click.option('--output', '-o', default='control_set_results.md', help='Output file')
def verify(control_set_file, output):
    """Verify against control set."""
    # Run control set verification
    ...


@cli.command()
@click.option('--queries', '-n', default=100, help='Number of queries')
@click.option('--threads', '-t', default=1, help='Number of threads')
def benchmark(queries, threads):
    """Run performance benchmark."""
    # Run benchmark
    ...


def _print_table(results):
    """Print results as formatted table."""
    click.echo(f"{'Rank':<5} {'Score':<8} {'Company Name':<50} {'Match Type':<15}")
    click.echo("-" * 80)

    for i, r in enumerate(results, 1):
        click.echo(f"{i:<5} {r.score*100:>6.1f}% {r.name[:48]:<50} {r.match_type:<15}")


if __name__ == '__main__':
    cli()
```

---

## 7. Docker & Deployment

### 7.1 Dockerfile

```dockerfile
# Dockerfile

FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ src/
COPY cli.py .

# Set environment
ENV PYTHONPATH=/app/src
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Default command
CMD ["python", "-m", "finetuner.core.cache_rpc", "--serve"]

# ----------------------------------------
# Web server image
FROM base as web

EXPOSE 5000
CMD ["python", "-m", "finetuner.web.app"]

# ----------------------------------------
# RPC server image
FROM base as rpc

EXPOSE 9876
CMD ["python", "-m", "finetuner.core.cache_rpc", "--serve"]
```

### 7.2 Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  rpc-server:
    build:
      context: .
      target: rpc
    volumes:
      - ./company_matcher_cache:/app/company_matcher_cache
      - ./models:/app/models
    ports:
      - "9876:9876"
    environment:
      - TRANSFORMERS_OFFLINE=1
    healthcheck:
      test: ["CMD", "python", "-c", "from finetuner.core.cache_rpc import is_server_running; exit(0 if is_server_running() else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3

  web:
    build:
      context: .
      target: web
    ports:
      - "5000:5000"
    depends_on:
      rpc-server:
        condition: service_healthy
    environment:
      - RPC_HOST=rpc-server
      - RPC_PORT=9876

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  grafana-data:
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)

1. **Extract scoring components** - Create focused scorer classes
2. **Unified configuration** - Single config file
3. **Basic unit tests** - Core scoring functions
4. **Structured logging** - JSON logs

### Phase 2: Testing & Quality (2-3 weeks)

5. **Integration tests** - Full matching pipeline
6. **Performance tests** - Latency benchmarks
7. **CI/CD pipeline** - GitHub Actions
8. **Code coverage** - Target 80%

### Phase 3: Observability (2-3 weeks)

9. **Prometheus metrics** - Query/cache/error metrics
10. **Grafana dashboards** - Visualizations
11. **Health endpoints** - /health, /ready
12. **Alerting rules** - Latency/error alerts

### Phase 4: Developer Experience (2-3 weeks)

13. **Rich CLI** - Click-based commands
14. **API documentation** - OpenAPI/Swagger
15. **Docker setup** - Multi-stage builds
16. **Development guide** - Contributing docs

---

## 9. Success Criteria

1. **Test coverage > 80%** for core modules
2. **P95 latency visible** in Grafana dashboard
3. **Zero manual deployment steps** (CI/CD)
4. **New developer productive in < 1 day**
5. **API documentation auto-generated**
6. **All errors logged with context**

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Refactoring breaks behavior | Regressions | Control set must pass before merge |
| Metric overhead | Latency increase | Benchmark with/without metrics |
| Docker image too large | Slow deploys | Multi-stage builds, slim base |
| Config migration | Breaking changes | Provide migration script |

---

*End of Proposal 3*
