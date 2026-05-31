# Proposal 1: Performance & Scalability Improvements

**Document Version:** 1.0
**Created:** 2026-04-22
**Focus Area:** Query Latency, Throughput, and Horizontal Scaling

---

## Executive Summary

This proposal addresses performance bottlenecks in the current FineTuner implementation. The goal is to reduce query latency from 50-300ms to sub-50ms while enabling horizontal scaling for high-throughput scenarios.

**Key Outcomes:**
- 5-10x improvement in query latency
- Support for 100+ concurrent queries
- Horizontal scaling via multiple RPC workers
- Batch query optimization

---

## 1. Vectorize Re-Ranking Operations

### 1.1 Problem

The current Phase 2 re-ranking iterates over 1,000 candidates with expensive per-candidate calculations:

```python
# Current: O(1000) loop with expensive operations
for j, i in enumerate(semantic_indices[0]):
    string_score = TextPreprocessor.calculate_string_similarity(query, candidate)  # Expensive
    concept_sig = self._get_concept_signature(cand_vec)  # Matrix multiply
    concept_alignment = self._calculate_signature_correlation(query_sig, cand_sig)
```

### 1.2 Solution

Vectorize operations using NumPy broadcasting:

```python
class CompanyMatcher:
    def _vectorized_rerank(self, query, candidate_indices, semantic_scores):
        """Vectorized re-ranking for all candidates at once."""
        n = len(candidate_indices)

        # 1. Batch concept signature calculation
        cand_vecs = self.vector_store.embeddings[candidate_indices]  # (n, 384)
        cand_sigs = np.dot(cand_vecs, self._anchor_vectors.T)  # (n, 31) - one matmul

        # 2. Concept alignment via broadcasting
        query_sig_expanded = self.query_sig.reshape(1, -1)  # (1, 31)
        concept_scores = np.sum(
            np.maximum(0, query_sig_expanded) * np.maximum(0, cand_sigs),
            axis=1
        )  # (n,)

        # 3. Normalize semantic scores
        sem_scores_norm = semantic_scores / semantic_scores[0] if semantic_scores[0] > 0 else semantic_scores

        # 4. Precompute string scores (see 1.3)
        string_scores = self._batch_string_similarity(query, candidate_indices)

        # 5. Combine with weights
        base_scores = (string_scores * 0.5) + (sem_scores_norm * 0.25) + (concept_scores * 0.25)

        return base_scores
```

### 1.3 Batch String Similarity

Create a vectorized string similarity using precomputed features:

```python
class TextPreprocessor:
    @classmethod
    def precompute_features(cls, names: List[str]) -> Dict:
        """Precompute features for all company names at index build time."""
        features = {
            'tokens': [],          # List[Set[str]]
            'token_weights': [],   # List[Dict[str, float]]
            'clean_names': [],     # List[str]
            'char_ngrams': [],     # List[Set[str]] for 3-grams
        }

        for name in names:
            clean = cls.clean_company_name(name)
            tokens = set(clean.split())
            weights = {t: cls.get_term_weight(t) for t in tokens}
            ngrams = set(clean[i:i+3] for i in range(len(clean)-2))

            features['tokens'].append(tokens)
            features['token_weights'].append(weights)
            features['clean_names'].append(clean)
            features['char_ngrams'].append(ngrams)

        return features

class CompanyMatcher:
    def _batch_string_similarity(self, query, candidate_indices):
        """Compute string similarity for all candidates using precomputed features."""
        query_clean = TextPreprocessor.clean_company_name(query)
        query_tokens = set(query_clean.split())
        query_ngrams = set(query_clean[i:i+3] for i in range(len(query_clean)-2))

        scores = np.zeros(len(candidate_indices))

        for i, idx in enumerate(candidate_indices):
            cand_tokens = self._precomputed_features['tokens'][idx]
            cand_ngrams = self._precomputed_features['char_ngrams'][idx]

            # Jaccard on tokens
            intersection = query_tokens & cand_tokens
            union = query_tokens | cand_tokens
            jaccard = len(intersection) / len(union) if union else 0

            # Ngram overlap (fast approximation of SequenceMatcher)
            ngram_overlap = len(query_ngrams & cand_ngrams) / len(query_ngrams | cand_ngrams) if query_ngrams else 0

            scores[i] = max(jaccard, ngram_overlap)

        return scores
```

### 1.4 Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Concept alignment | 1000 × matmul | 1 batched matmul | 100x |
| String similarity | 1000 × function calls | Vectorized | 10x |
| Overall re-rank | 50-200ms | 5-20ms | 10x |

---

## 2. GPU Acceleration for FAISS

### 2.1 Problem

FAISS CPU search is fast but becomes a bottleneck at high query volumes.

### 2.2 Solution

Add optional GPU acceleration:

```python
class VectorStore:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.gpu_resources = None

    def _check_gpu_available(self):
        try:
            import faiss.contrib.torch_utils  # Check for GPU FAISS
            return faiss.get_num_gpus() > 0
        except:
            return False

    def build_index(self, embeddings, use_ivf=False, nlist=100):
        """Build index with optional IVF clustering for faster search."""
        dim = embeddings.shape[1]

        if use_ivf and len(embeddings) > 100000:
            # Use IVF for large datasets
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        if self.use_gpu:
            self._move_to_gpu()

    def _move_to_gpu(self):
        """Move index to GPU."""
        import faiss.contrib.torch_utils
        self.gpu_resources = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
        print("[VectorStore] Index moved to GPU")
```

### 2.3 IVF Index for Approximate Search

For datasets >1M, use IVF clustering:

```python
# At build time
nlist = 1000  # Number of clusters
nprobe = 50   # Clusters to search (tradeoff: speed vs accuracy)

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)
index.add(embeddings)
index.nprobe = nprobe  # Set at search time
```

### 2.4 Expected Impact

| Configuration | Search Time (2.9M vectors) |
|---------------|---------------------------|
| IndexFlatIP CPU | ~20ms |
| IndexFlatIP GPU | ~2ms |
| IndexIVFFlat CPU (nprobe=50) | ~5ms |
| IndexIVFFlat GPU (nprobe=50) | ~0.5ms |

---

## 3. Multi-Worker RPC Server

### 3.1 Problem

Current RPC server is single-threaded, limiting throughput.

### 3.2 Solution

Implement a worker pool architecture:

```python
# cache_rpc_pool.py

import multiprocessing as mp
from typing import List

class CacheServerWorkerPool:
    """Multi-worker RPC server for horizontal scaling."""

    def __init__(self, num_workers: int = 4, cache_keys: List[str] = None):
        self.num_workers = num_workers
        self.cache_keys = cache_keys or []
        self.workers = []
        self.ports = list(range(9876, 9876 + num_workers))

    def start(self):
        """Start all workers."""
        for i, port in enumerate(self.ports):
            p = mp.Process(
                target=self._run_worker,
                args=(port, self.cache_keys),
                name=f"CacheWorker-{i}"
            )
            p.start()
            self.workers.append(p)

        print(f"[Pool] Started {self.num_workers} workers on ports {self.ports}")

    @staticmethod
    def _run_worker(port, cache_keys):
        """Run a single worker process."""
        from finetuner.core.cache_rpc import run_server
        run_server(port=port)

class LoadBalancedClient:
    """Client that distributes queries across workers."""

    def __init__(self, ports: List[int] = None):
        self.ports = ports or list(range(9876, 9880))
        self._index = 0
        self._lock = threading.Lock()

    def _get_next_port(self):
        """Round-robin port selection."""
        with self._lock:
            port = self.ports[self._index]
            self._index = (self._index + 1) % len(self.ports)
            return port

    def search(self, query: str, **kwargs):
        """Send query to next available worker."""
        port = self._get_next_port()
        client = connect(port=port)
        return client.search(query, **kwargs)

    def batch_search(self, queries: List[str], **kwargs):
        """Parallel batch search across workers."""
        from concurrent.futures import ThreadPoolExecutor

        def search_one(query):
            return self.search(query, **kwargs)

        with ThreadPoolExecutor(max_workers=len(self.ports)) as executor:
            results = list(executor.map(search_one, queries))

        return results
```

### 3.3 Kubernetes Deployment

```yaml
# k8s/cache-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finetuner-cache-server
spec:
  replicas: 4
  selector:
    matchLabels:
      app: finetuner-cache
  template:
    metadata:
      labels:
        app: finetuner-cache
    spec:
      containers:
      - name: cache-server
        image: finetuner:latest
        command: ["python", "-m", "finetuner.core.cache_rpc", "--serve"]
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: cache-data
          mountPath: /app/company_matcher_cache
      volumes:
      - name: cache-data
        persistentVolumeClaim:
          claimName: finetuner-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: finetuner-cache-lb
spec:
  type: LoadBalancer
  selector:
    app: finetuner-cache
  ports:
  - port: 9876
    targetPort: 9876
```

---

## 4. Query Result Caching

### 4.1 Problem

Identical queries recompute the full matching pipeline.

### 4.2 Solution

Add LRU caching for query results:

```python
from functools import lru_cache
import hashlib

class CompanyMatcher:
    def __init__(self, ...):
        ...
        self._query_cache = {}
        self._cache_max_size = 10000
        self._cache_ttl_seconds = 300  # 5 minutes

    def _get_cache_key(self, query: str, city: str, state: str, top_k: int) -> str:
        """Generate cache key for query."""
        content = f"{query.lower()}|{city or ''}|{state or ''}|{top_k}"
        return hashlib.md5(content.encode()).hexdigest()

    def match_with_location(self, query, city=None, state=None, top_k=10):
        """Match with caching."""
        cache_key = self._get_cache_key(query, city, state, top_k)

        # Check cache
        if cache_key in self._query_cache:
            entry = self._query_cache[cache_key]
            if time.time() - entry['timestamp'] < self._cache_ttl_seconds:
                return entry['results']

        # Compute results
        results = self._match_with_location_impl(query, city, state, top_k)

        # Store in cache
        self._query_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }

        # Evict old entries if needed
        if len(self._query_cache) > self._cache_max_size:
            self._evict_oldest_entries()

        return results
```

### 4.3 Redis for Distributed Caching

```python
import redis
import json

class DistributedQueryCache:
    """Redis-backed query cache for multi-worker deployments."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl_seconds = 300

    def get(self, cache_key: str):
        """Get cached results."""
        data = self.redis.get(f"finetuner:query:{cache_key}")
        if data:
            return json.loads(data)
        return None

    def set(self, cache_key: str, results):
        """Cache results."""
        self.redis.setex(
            f"finetuner:query:{cache_key}",
            self.ttl_seconds,
            json.dumps(results)
        )

    def invalidate_all(self):
        """Clear all cached queries."""
        keys = self.redis.keys("finetuner:query:*")
        if keys:
            self.redis.delete(*keys)
```

---

## 5. Asynchronous Query Processing

### 5.1 Problem

Web requests block waiting for search results.

### 5.2 Solution

Implement async query queue with WebSocket notifications:

```python
# async_search.py

import asyncio
from typing import Dict, Any
import uuid

class AsyncSearchQueue:
    """Async query queue with callback notifications."""

    def __init__(self, matcher):
        self.matcher = matcher
        self.queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
        self.callbacks: Dict[str, asyncio.Future] = {}

    async def submit(self, query: str, **kwargs) -> str:
        """Submit query and return job ID."""
        job_id = str(uuid.uuid4())

        await self.queue.put({
            'job_id': job_id,
            'query': query,
            'kwargs': kwargs
        })

        self.callbacks[job_id] = asyncio.get_event_loop().create_future()
        return job_id

    async def get_result(self, job_id: str, timeout: float = 30.0):
        """Wait for result."""
        try:
            return await asyncio.wait_for(
                self.callbacks[job_id],
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return {'error': 'Query timed out'}

    async def worker(self):
        """Process queries from queue."""
        while True:
            job = await self.queue.get()

            try:
                results = self.matcher.match_with_location(
                    job['query'],
                    **job['kwargs']
                )
                self.callbacks[job['job_id']].set_result({
                    'success': True,
                    'results': results
                })
            except Exception as e:
                self.callbacks[job['job_id']].set_result({
                    'error': str(e)
                })

            self.queue.task_done()
```

### 5.3 Flask-SocketIO Integration

```python
# app_async.py

from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

search_queue = AsyncSearchQueue(matcher)

@socketio.on('search')
async def handle_search(data):
    """Handle async search request."""
    job_id = await search_queue.submit(
        data['query'],
        city=data.get('city'),
        state=data.get('state'),
        top_k=data.get('top_k', 10)
    )

    # Send job ID immediately
    emit('job_submitted', {'job_id': job_id})

    # Wait for result and send
    result = await search_queue.get_result(job_id)
    emit('search_result', {'job_id': job_id, **result})
```

---

## 6. Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

1. **Vectorize concept alignment** - Batch matrix multiply
2. **Add query result caching** - LRU cache with TTL
3. **Precompute string features** - At index build time

### Phase 2: RPC Improvements (2-3 weeks)

4. **Multi-worker RPC pool** - Round-robin load balancing
5. **Connection pooling** - Reuse Pyro5 connections
6. **Batch search endpoint** - Single RPC call for multiple queries

### Phase 3: Advanced Optimization (3-4 weeks)

7. **GPU acceleration** - FAISS GPU support
8. **IVF indexing** - Approximate search for scale
9. **Redis caching** - Distributed query cache
10. **Async processing** - WebSocket notifications

---

## 7. Monitoring & Benchmarks

### 7.1 Key Metrics to Track

```python
# metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Query metrics
QUERY_COUNT = Counter('finetuner_queries_total', 'Total queries', ['status'])
QUERY_LATENCY = Histogram('finetuner_query_latency_seconds', 'Query latency')
CACHE_HIT_RATE = Gauge('finetuner_cache_hit_rate', 'Query cache hit rate')

# System metrics
INDEX_SIZE = Gauge('finetuner_index_size', 'Number of companies in index')
MEMORY_USAGE_MB = Gauge('finetuner_memory_mb', 'Memory usage in MB')
RPC_WORKERS = Gauge('finetuner_rpc_workers', 'Number of active RPC workers')
```

### 7.2 Benchmark Script

```python
# benchmark.py

import time
import statistics
from concurrent.futures import ThreadPoolExecutor

def benchmark_queries(queries, num_iterations=100, num_threads=1):
    """Benchmark query performance."""
    latencies = []

    def run_query(query):
        start = time.perf_counter()
        service.search(query)
        return time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_iterations):
            for query in queries:
                latency = executor.submit(run_query, query).result()
                latencies.append(latency * 1000)  # Convert to ms

    return {
        'p50': statistics.median(latencies),
        'p95': statistics.quantiles(latencies, n=20)[18],
        'p99': statistics.quantiles(latencies, n=100)[98],
        'avg': statistics.mean(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'throughput_qps': len(latencies) / sum(latencies) * 1000
    }
```

---

## 8. Expected Results

| Metric | Current | After Phase 1 | After Phase 3 |
|--------|---------|---------------|---------------|
| P50 Latency | 100ms | 30ms | 10ms |
| P99 Latency | 300ms | 100ms | 50ms |
| Max QPS (1 worker) | 10 | 30 | 100 |
| Max QPS (4 workers) | N/A | 100 | 400 |
| Memory per worker | 600MB | 600MB | 600MB |
| Cache hit rate | 0% | 30% | 50% |

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU not available | Falls back to CPU | Graceful degradation in code |
| Redis unavailable | No distributed caching | Local LRU cache fallback |
| IVF quality loss | Slightly lower recall | Tune nprobe, validate on control set |
| Worker failures | Reduced capacity | Health checks, auto-restart |

---

## 10. Success Criteria

1. **P50 latency < 30ms** on standard queries
2. **P99 latency < 100ms** under load
3. **100+ QPS** with 4 workers
4. **Zero regressions** on control set accuracy
5. **< 1GB RAM** per worker

---

*End of Proposal 1*
