from sentence_transformers import SentenceTransformer
import hashlib
import json
import os
import pickle
import re
import unicodedata
import numpy as np
from finetuner.utils.text_preprocessor import TextPreprocessor
from finetuner.core.vector_store import VectorStore

# Try to import tqdm for progress bars, fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback that just returns the iterable unchanged
    def tqdm(iterable, desc=None, total=None, unit=None, ncols=None, **kwargs):
        return iterable


def _to_loc_str(v):
    """Normalize location field to string for use as hash key. Handles list values from cache."""
    if isinstance(v, list):
        return (v[0] if v else "") or ""
    return str(v) if v else ""


def _iter_indices(val):
    """Yield integer indices from value that may be int, list of ints, or nested list (from pickle)."""
    if val is None:
        return
    if isinstance(val, (list, tuple)):
        for x in val:
            if isinstance(x, (list, tuple)):
                yield from _iter_indices(x)
            else:
                yield int(x)
    else:
        yield int(val)


def _legal_name_match_key(s) -> str:
    """
    Canonical key for a full legal name so duplicate offices (Unicode, spacing) map together.
    Used for: multi-office exact match, Phase 3, and is_this_exact when query vs row differ cosmetically.
    """
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = t.casefold().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _load_active_model() -> str:
    """Read active embedding model from model_config.json (project root)."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
                return cfg.get('active_model', 'all-MiniLM-L6-v2')
        except Exception:
            pass
    return 'all-MiniLM-L6-v2'


class CompanyMatcher:
    """
    Core engine for high-precision company name matching.
    
    ARCHITECTURE ROLES:
    - CompanyMatcher (this class): The reusable core logic for indexing and matching.
    - company_search.py: CLI utility that uses this engine for interactive/batch searches.
    - SearchService: Web service wrapper that provides an API for the web application.
    
    This separation ensures consistent matching across all interfaces while 
    avoiding logic redundancy.
    """
    # Cache version - increment this when logic changes to invalidate old caches.
    # Scoring-only changes (e.g. v4.3 unified single score) do not require a cache rebuild,
    # since FAISS/embeddings/locations are unchanged.
    CACHE_VERSION = "v4.2_unified_user_score"
    # For exact name matches, location_score is multiplied by this and added to name_score
    # when the query includes location — must outrank cross-office frequency noise (~2–5%).
    EXACT_MATCH_LOCATION_WEIGHT = 0.10
    # For exact+geo user-facing score: name * (FLOOR + (1-FLOOR)*location_score) — 1.0 only if both are perfect.
    EXACT_GEO_USER_FLOOR = 0.2

    # Query omitted city/state: never show near-100% on exact-name alone — multiple same-name
    # offices may exist; SMEs should not read ~99% as "this is the right site."
    # Set below tier_config "high" (default 0.95) so report tier stays Good until geo is verified.
    EXACT_WHEN_QUERY_HAS_NO_GEO_CAP = 0.94
    # Non-exact candidates on a geo-less query must stay strictly below the exact no-geo cap,
    # otherwise a near-name variant (e.g. "Vision Council of America") outranks the exact-name
    # row ("VISION AMERICA"). Held below EXACT_WHEN_QUERY_HAS_NO_GEO_CAP so the exact stays #1.
    NONEXACT_WHEN_QUERY_HAS_NO_GEO_CAP = 0.92

    # Semantic Anchors for Concept Probing (Nutritional Label)
    CONCEPT_ANCHORS = {
        "Geography": ["Pennsylvania", "London", "Canada", "California", "New York", "Texas", "Chicago", "Illinois", "Ohio", "Miami", "Paris"],
        "Industry": ["Automotive", "Medical", "Technology", "Construction", "Legal", "Food", "Finance", "Education", "Insurance", "Retail", "Manufacturing"],
        "Structure": ["Corporate", "Non-Profit", "Government", "Small Business"],
        "Nature": ["Global", "Local", "Industrial", "Consumer", "Professional"]
    }
    
    @staticmethod
    def _rank_and_user_facing_score(
        name_score: float,
        location_score: float,
        use_location: bool,
        is_this_exact: bool,
        freq_boost_val: float,
        elw: float,
        exact_geo_full_match: bool = False,
        query_geo_complete: bool = False,
    ) -> tuple:
        """
        Unified single user-facing score in [0, 1].
        Display, ranking, and API all use the same number.

        Components are clamped so no input can exceed 1.0:
        - name_score, location_score: clamped to [0, 1]
        - freq_boost_val: capped at 0.05 in the additive blend

        Score paths:
        - With query geo + exact name: max of multiplicative geo penalty and the
          hybrid additive blend, so an exact-name same-state branch never falls
          below the equivalent hybrid candidate.
        - With query geo + non-exact name: 0.8*name + 0.2*location + capped freq.
        - Without query geo: name + capped freq.

        100% gate: a score of 1.0 is allowed only when name is exactly equal AND
        city+state match exactly on both sides.

        If the query omits geography, every row with near-unity name component (exact flag
        or ``name_score`` ≥ 0.998) and no full geo match is capped at
        EXACT_WHEN_QUERY_HAS_NO_GEO_CAP; the ``name_score`` fallback covers legal-key edge
        cases where the exact flag did not latch.
        """
        ns = max(0.0, min(1.0, float(name_score)))
        ls = max(0.0, min(1.0, float(location_score)))
        fb_capped = min(max(0.0, float(freq_boost_val)), 0.05)

        if use_location:
            hybrid_blend = (ns * 0.8) + (ls * 0.2) + fb_capped
            if is_this_exact:
                fl = CompanyMatcher.EXACT_GEO_USER_FLOOR
                multiplicative = ns * (fl + (1.0 - fl) * ls)
                score = max(multiplicative, hybrid_blend)
            else:
                score = hybrid_blend
        else:
            score = ns + fb_capped

        score = max(0.0, min(1.0, score))

        # Query named both city and state: penalize candidates with no material
        # location agreement (ls ~ 0), so wrong-state / cross-country lexical
        # lookalikes (e.g. dba variant in CA when query is Durham, CT) cannot sit
        # at ~0.72 and bury the correct office in retrieval order.
        if use_location and query_geo_complete and ls < 0.01:
            score = min(score, 0.52)
        elif (
            use_location
            and query_geo_complete
            and not exact_geo_full_match
            and 0.36 <= ls <= 0.44
        ):
            # Same-state but wrong city vs query (TextPreprocessor returns ~0.40).
            # Prevents generic same-state offices (e.g. Hartford) from outranking
            # the stated city (Durham) when both appear for related names.
            score = min(score, 0.53)

        qualifies_for_100 = is_this_exact and exact_geo_full_match
        if not qualifies_for_100:
            # Linear rescale of [0.80, 1.0] -> [0.80, 0.999] preserves micro-ordering
            # within the gate band instead of collapsing every non-100% candidate to 0.999.
            if score >= 0.80:
                score = 0.80 + (score - 0.80) * (0.999 - 0.80) / (1.0 - 0.80)
            score = min(score, 0.999)

        # Cap when the query did not verify geography: do not rely only on ``is_this_exact``
        # (legal-key vs lower() edge cases left name_score at ~1.0 without setting the flag).
        if not use_location and not exact_geo_full_match:
            if is_this_exact or ns >= 0.998:
                # Exact name, unverified office: sub-100 but must outrank non-exact lookalikes.
                score = min(score, CompanyMatcher.EXACT_WHEN_QUERY_HAS_NO_GEO_CAP)
            else:
                # Non-exact on a geo-less query: hold strictly below the exact no-geo cap so an
                # exact-name match is never buried by a variant. Rescale into [0.80, cap] rather
                # than a flat clamp, so the runner-up cluster keeps distinct, ordered scores
                # instead of collapsing to one value (avoids re-creating tie mush).
                cap = CompanyMatcher.NONEXACT_WHEN_QUERY_HAS_NO_GEO_CAP
                if score > 0.80:
                    score = 0.80 + (score - 0.80) * (cap - 0.80) / (0.999 - 0.80)
                score = min(score, cap)

        return (score, score)

    @staticmethod
    def _apply_lexical_floor(name_score: float, floor: float, band: float = 0.0099) -> float:
        """Raise a strong-string match to a visibility ``floor`` WITHOUT flattening ties.

        The old logic clamped every below-floor candidate to exactly ``floor``, so several
        distinct candidates collapsed to one identical score and their true ordering was lost
        (top-5 "tie mush"). Here, below-floor candidates are mapped *monotonically* into
        ``[floor, floor + band)`` by their raw blend, so siblings keep distinct, correctly
        ordered scores while still clearing the floor. Candidates already at/above ``floor``
        are returned unchanged. Inflation is bounded by ``band`` (default <0.01).
        """
        ns = float(name_score)
        if ns >= floor:
            return ns
        frac = max(0.0, ns) / floor if floor > 0 else 0.0
        return floor + band * frac

    @staticmethod
    def _sanitize_candidate_loc(city, state) -> tuple:
        """Drop a bogus city that merely echoes the state (source-data defect).

        Many source rows carry ``city == state`` (e.g. city="UT" state="UT", or
        city="Utah" state="UT") which is not a real locality. Treating such a city as
        a verified locale lets it falsely satisfy city-level geo matching/display.
        When the normalized (or raw, case-folded) city equals the normalized state we
        blank the city and keep only the state, so scoring/display behave as state-only.
        Returns the (possibly cleaned) ``(city, state)`` strings.
        """
        c = str(city or "").strip()
        st = str(state or "").strip()
        if not c or not st:
            return c, st
        if c.upper() == st.upper():
            return "", st
        cn = TextPreprocessor.normalize_city(c)
        sn = TextPreprocessor.normalize_state(st)
        if cn and sn and cn == sn:
            return "", st
        return c, st

    @staticmethod
    def _ensure_top5_score_spread(results: list, min_spread: float = 0.001) -> None:
        """Bump top-5 ``score`` values (descending ranks) so max-min >= min_spread if needed."""
        if len(results) < 5:
            return
        top = results[:5]
        scores = [float(r.get("score") or 0) for r in top]
        if max(scores) - min(scores) >= min_spread:
            return
        step = (min_spread / 4.0) + 1e-10
        for idx, r in enumerate(top):
            r["score"] = min(1.0, float(r.get("score") or 0) + (4 - idx) * step)

    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)
        self.vector_store = VectorStore()
        
        self.original_company_names = []  # Store original names
        self.company_names = []  # Store preprocessed names for matching
        self.model_name = model_name  # Store the actual model name used
        
        # Location data storage (for location-aware matching)
        self.company_locations = []  # List of {"city": str, "state": str} per company
        self.company_counts = []  # List of record counts per company
        self.company_ids = []  # List of database IDs per company (for reference back to DB)
        self.acronym_index = {}  # Map of Acronym -> List of Indices
        self.has_location_data = False  # Flag to indicate if location data is loaded
        self.max_company_count = 0  # To be populated for frequency boost
        
        # Precomputed similarity caches (NEW - for performance)
        self.similarity_cache = {}  # Dict: (idx1, idx2) -> similarity_score
        self.acronym_cache = {}  # Dict: idx -> {'acronym': str, 'fidelity_scores': {idx: score}}
        
        # Persistence settings
        self.cache_dir = "company_matcher_cache"
        self.ensure_cache_dir()

        # Precompute anchor vectors once
        self._anchor_vectors = None
        self._anchor_names = []
        for cat, anchors in self.CONCEPT_ANCHORS.items():
            self._anchor_names.extend(anchors)
    
    def ensure_cache_dir(self):
        """Ensure the cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_key_from_file(self, filepath):
        """Generate cache key from file metadata (fast - no file loading needed)"""
        if not os.path.exists(filepath):
            return None
        
        # Get file metadata
        stat = os.stat(filepath)
        file_size = stat.st_size
        file_mtime = stat.st_mtime
        
        # Create hash from: filepath + size + mtime + model + cache version
        # This allows cache checking without loading 2.9M+ company names
        content = f"{os.path.abspath(filepath)}|{file_size}|{file_mtime}|{self.model_name}|{self.CACHE_VERSION}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cache_key(self, company_names):
        """Generate a unique cache key based on company names and model"""
        # Create a hash of the sorted company names, model name, and cache version
        sorted_names = sorted(company_names)
        content = "|".join(sorted_names) + "|" + self.model_name + "|" + self.CACHE_VERSION
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cache_paths(self, cache_key):
        """Get file paths for cached data"""
        base_path = os.path.join(self.cache_dir, cache_key)
        return {
            'embeddings': base_path + '_embeddings.npy',
            'index': base_path + '_index.faiss',
            'names': base_path + '_names.pkl',
            'metadata': base_path + '_metadata.pkl',
            'acronyms': base_path + '_acronyms.pkl'
        }
    
    def save_to_cache(self, cache_key, embeddings, index, company_names, original_names, 
                      locations=None, counts=None, ids=None):
        """Save embeddings, index, names, and optionally location data to cache"""
        try:
            import time
            cache_start = time.time()
            paths = self.get_cache_paths(cache_key)
            
            # Save vectors and index via VectorStore
            # Note: embeddings and index args are kept for signature compatibility but ignored
            # as we use self.vector_store
            self.vector_store.save(paths['embeddings'], paths['index'])
            
            # Save company names (and location data if available)
            print("      Saving company names to cache...", end=" ", flush=True)
            names_start = time.time()
            names_data = {
                'company_names': company_names,
                'original_company_names': original_names
            }
            # Include location data and IDs if provided
            if locations is not None:
                names_data['company_locations'] = locations
            if counts is not None:
                names_data['company_counts'] = counts
            if ids is not None:
                names_data['company_ids'] = ids
                
            # Save acronym index if available
            if self.acronym_index:
                names_data['acronym_index'] = self.acronym_index
            
            # Save precomputed caches (NEW - for performance)
            if self.similarity_cache:
                names_data['similarity_cache'] = self.similarity_cache
            if self.acronym_cache:
                names_data['acronym_cache'] = self.acronym_cache
                
            # Save fast lookup structures (NEW - to avoid recomputing on load)
            if hasattr(self, '_company_names_lower_set'):
                names_data['_company_names_lower_set'] = self._company_names_lower_set
            if hasattr(self, '_company_names_lower_to_index'):
                names_data['_company_names_lower_to_index'] = self._company_names_lower_to_index
            if hasattr(self, '_company_words_dict'):
                names_data['_company_words_dict'] = self._company_words_dict
            if hasattr(self, '_legal_nfc_to_indices') and self._legal_nfc_to_indices:
                names_data['_legal_nfc_to_indices'] = self._legal_nfc_to_indices
                
            with open(paths['names'], 'wb') as f:
                pickle.dump(names_data, f)
            print(f"[OK] ({time.time() - names_start:.1f}s)")
            
            # Save metadata
            print("      Saving metadata to cache...", end=" ", flush=True)
            meta_start = time.time()
            with open(paths['metadata'], 'wb') as f:
                pickle.dump({
                    'model_name': self.model_name,
                    'cache_key': cache_key,
                    'cache_version': self.CACHE_VERSION,
                    'num_companies': len(company_names),
                    'max_company_count': self.max_company_count,
                    'has_location_data': locations is not None,
                    'has_similarity_cache': bool(self.similarity_cache),
                    'has_acronym_cache': bool(self.acronym_cache)
                }, f)
            print(f"[OK] ({time.time() - meta_start:.1f}s)")
            
            cache_time = time.time() - cache_start
            print(f"   Cache saved successfully: {cache_key[:16]}... (total: {cache_time:.1f}s)")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_from_cache(self, cache_key):
        """Load embeddings, index, names, and location data from cache"""
        try:
            import time
            cache_start = time.time()
            paths = self.get_cache_paths(cache_key)
            
            # Check if all cache files exist
            # Note: We relax strict checking for acronyms or metadata if reusing old cache mostly works
            # But here we strictly require core files.
            # Acronyms are embedded in names.pkl, so check for core files.
            if not all(os.path.exists(p) for p in [paths['embeddings'], paths['index'], paths['names'], paths['metadata']]):
                return False
            
            # Load vectors and index via VectorStore
            if not self.vector_store.load(paths['embeddings'], paths['index']):
                return False
            
            # Load company names and location data
            print("      Loading company names from cache...", end=" ", flush=True)
            names_start = time.time()
            with open(paths['names'], 'rb') as f:
                names_data = pickle.load(f)
                self.company_names = names_data['company_names']
                self.original_company_names = names_data['original_company_names']
                
                # Load Acronym Index
                if 'acronym_index' in names_data:
                    self.acronym_index = names_data['acronym_index']
                else:
                    self.acronym_index = {}
                
                # Load precomputed caches (NEW - for performance)
                if 'similarity_cache' in names_data:
                    self.similarity_cache = names_data['similarity_cache']
                    print(f"\n      [PERF] Loaded {len(self.similarity_cache):,} precomputed similarity scores")
                else:
                    self.similarity_cache = {}
                
                if 'acronym_cache' in names_data:
                    self.acronym_cache = names_data['acronym_cache']
                    total_fidelity = sum(len(data.get('fidelity_scores', {})) for data in self.acronym_cache.values())
                    print(f"      [PERF] Loaded {total_fidelity:,} precomputed acronym fidelity scores")
                else:
                    self.acronym_cache = {}
                    
                # Load location data and IDs if available
                if 'company_locations' in names_data:
                    self.company_locations = names_data['company_locations']
                    self.has_location_data = True
                else:
                    self.company_locations = []
                    self.has_location_data = False
                if 'company_counts' in names_data:
                    self.company_counts = names_data['company_counts']
                else:
                    self.company_counts = []
                if 'company_ids' in names_data:
                    self.company_ids = names_data['company_ids']
                else:
                    self.company_ids = []
                    
                # Load fast lookup structures if available (NEW)
                if '_company_names_lower_set' in names_data:
                    self._company_names_lower_set = names_data['_company_names_lower_set']
                if '_company_names_lower_to_index' in names_data:
                    self._company_names_lower_to_index = names_data['_company_names_lower_to_index']
                if '_company_words_dict' in names_data:
                    self._company_words_dict = names_data['_company_words_dict']
                if '_legal_nfc_to_indices' in names_data:
                    self._legal_nfc_to_indices = names_data['_legal_nfc_to_indices']
                else:
                    self._legal_nfc_to_indices = None
            print(f"[OK] ({time.time() - names_start:.1f}s)")
            
            # Verify metadata
            print("      Verifying cache metadata...", end=" ", flush=True)
            meta_start = time.time()
            with open(paths['metadata'], 'rb') as f:
                metadata = pickle.load(f)
                if metadata.get('cache_version') != self.CACHE_VERSION:
                    print(f"[FAIL] (Cache version mismatch: {metadata.get('cache_version')} != {self.CACHE_VERSION})")
                    return False
                if metadata['model_name'] != self.model_name:
                    print("[FAIL] (Model name changed, cache invalid)")
                    return False
                self.max_company_count = metadata.get('max_company_count', 0)
            print(f"[OK] ({time.time() - meta_start:.1f}s)")
            
            # Create fast lookup sets for exact matching
            self._create_fast_lookup_sets()
            
            cache_time = time.time() - cache_start
            print(f"   Cache loaded successfully: {cache_key[:16]}... (total: {cache_time:.1f}s)")
            print(f"   Loaded {len(self.original_company_names):,} companies from cache")
            if self.has_location_data:
                print(f"   Location data: Available ({len(self.company_locations):,} entries)")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ensure_fast_lookup_sets(self):
        """Ensure fast lookup sets exist (useful for existing cached data)"""
        if not hasattr(self, '_company_names_lower_set') or not hasattr(self, '_company_words_dict'):
            print("Creating fast lookup sets for existing data...")
            self._create_fast_lookup_sets()
            return True
        return False

    def _create_acronym_index(self):
        """
        Creates a reverse index mapping acronyms to company indices.
        Example: "ABA" -> [105, 2099, 5001]
        """
        print("   Creating Acronym Index...")
        self.acronym_index = {}
        
        count = 0
        for i, name in enumerate(self.original_company_names):
            acronym = TextPreprocessor.generate_acronym(name)
            if acronym:
                if acronym not in self.acronym_index:
                    self.acronym_index[acronym] = []
                self.acronym_index[acronym].append(i)
                count += 1
                
        print(f"   [OK] Indexed {count:,} acronyms mapping to {len(self.acronym_index):,} unique keys")

    def _precompute_similarity_matrix(self):
        """
        Precompute pairwise string similarities for all companies.
        Only stores scores > 0.3 to save memory.
        This is the HIGHEST IMPACT optimization - eliminates 100+ expensive calls per query.
        """
        import time
        print("   Precomputing similarity matrix (this will take time but saves MASSIVE time later)...")
        start_time = time.time()
        
        n = len(self.original_company_names)
        self.similarity_cache = {}
        
        # Use threshold to only cache meaningful similarities
        threshold = 0.3
        cached_count = 0
        
        # Progress bar for precomputation
        total_comparisons = (n * (n - 1)) // 2  # Only compute upper triangle
        
        pbar = tqdm(total=total_comparisons, desc="   Computing similarities", unit="pairs", ncols=80, disable=not HAS_TQDM)
        
        for i in range(n):
            for j in range(i + 1, n):
                score = TextPreprocessor.calculate_string_similarity(
                    self.original_company_names[i],
                    self.original_company_names[j]
                )
                
                if score > threshold:
                    # Store both directions for O(1) lookup
                    self.similarity_cache[(i, j)] = score
                    self.similarity_cache[(j, i)] = score
                    cached_count += 1
                
                pbar.update(1)
        
        pbar.close()
        
        elapsed = time.time() - start_time
        print(f"   [OK] Precomputed {cached_count:,} similarity pairs (>{threshold}) in {elapsed:.1f}s")
        print(f"   [OK] Cache size: {len(self.similarity_cache):,} entries (saves ~{len(self.similarity_cache) * 0.001:.1f}s per query)")

    def _precompute_acronym_data(self):
        """
        Precompute acronym fidelity scores for all companies.
        For each company with an acronym, compute fidelity vs all other companies.
        Only stores scores > 0.3 to save memory.
        """
        import time
        print("   Precomputing acronym fidelity scores...")
        start_time = time.time()
        
        n = len(self.original_company_names)
        self.acronym_cache = {}
        
        threshold = 0.3
        total_fidelity_scores = 0
        companies_with_acronyms = 0
        
        # Progress bar
        pbar = tqdm(enumerate(self.original_company_names), desc="   Computing acronyms", total=n, unit="companies", ncols=80, disable=not HAS_TQDM)
        
        for idx, name in pbar:
            acronym = TextPreprocessor.generate_acronym(name)
            
            self.acronym_cache[idx] = {
                'acronym': acronym,
                'fidelity_scores': {}
            }
            
            if acronym:
                companies_with_acronyms += 1
                # Compute fidelity vs all other companies
                for other_idx, other_name in enumerate(self.original_company_names):
                    if idx != other_idx:
                        fidelity = TextPreprocessor.calculate_acronym_fidelity(acronym, other_name)
                        if fidelity > threshold:
                            self.acronym_cache[idx]['fidelity_scores'][other_idx] = fidelity
                            total_fidelity_scores += 1
        
        pbar.close()
        
        elapsed = time.time() - start_time
        print(f"   [OK] Precomputed acronym data for {companies_with_acronyms:,} companies in {elapsed:.1f}s")
        print(f"   [OK] Cached {total_fidelity_scores:,} fidelity scores (>{threshold})")

    def _get_cached_similarity(self, query, query_idx, candidate_idx):
        """
        Get similarity score from cache if available, otherwise compute it.
        
        Args:
            query: Query string (for fallback computation)
            query_idx: Index of query in original_company_names (None if not in database)
            candidate_idx: Index of candidate in original_company_names
            
        Returns:
            Similarity score between 0 and 1
        """
        # Try cache first if query is in our database
        if query_idx is not None and (query_idx, candidate_idx) in self.similarity_cache:
            return self.similarity_cache[(query_idx, candidate_idx)]
        
        # Fallback to computation for new queries not in cache
        candidate_name = self.original_company_names[candidate_idx]
        return TextPreprocessor.calculate_string_similarity(query, candidate_name)
    
    def _get_cached_acronym_fidelity(self, query, query_idx, candidate_idx):
        """
        Get acronym fidelity score from cache if available, otherwise compute it.
        
        Args:
            query: Query string (for fallback computation)
            query_idx: Index of query in original_company_names (None if not in database)
            candidate_idx: Index of candidate in original_company_names
            
        Returns:
            Acronym fidelity score between 0 and 1
        """
        # Try cache first if query is in our database
        if query_idx is not None and query_idx in self.acronym_cache:
            fidelity_scores = self.acronym_cache[query_idx].get('fidelity_scores', {})
            if candidate_idx in fidelity_scores:
                return fidelity_scores[candidate_idx]
        
        # Fallback to computation
        candidate_name = self.original_company_names[candidate_idx]
        acronym = TextPreprocessor.generate_acronym(query)
        if acronym:
            return TextPreprocessor.calculate_acronym_fidelity(acronym, candidate_name)
        return 0.0



    def _ensure_anchors_loaded(self):
        """Lazy load anchor vectors into GPU/CPU memory"""
        if self._anchor_vectors is None:
            self._anchor_vectors = self.model.encode(self._anchor_names, convert_to_numpy=True)
            # Normalize for fast dot-product similarity
            norms = np.linalg.norm(self._anchor_vectors, axis=1, keepdims=True)
            self._anchor_vectors = self._anchor_vectors / (norms + 1e-10)

    def _get_concept_signature(self, vector):
        """
        Convert a raw embedding into a 'Concept Signature' (Nutritional Label).
        
        Args:
            vector: Normalized vector of the company name
            
        Returns:
            numpy array of similarity scores against anchors
        """
        self._ensure_anchors_loaded()
        
        # Ensure vector is normalized and 2D
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        v_norm = vector / (np.linalg.norm(vector) + 1e-10)
        
        # Calculate dot product (cosine similarity since both are normalized)
        # Result is 1 x NumAnchors
        signature = np.dot(v_norm, self._anchor_vectors.T)[0]
        return signature

    def _calculate_signature_correlation(self, sig1, sig2):
        """
        Calculate how well two 'Nutritional Labels' align.
        Focuses on high-value spikes (Top 3 concepts).
        """
        # Pearson correlation or simple dot product of signatures
        # Sig1 and Sig2 are already similarity scores (-1 to 1)
        # We focus on the POSITIVE alignment (what they BOTH taste like)
        correlation = np.dot(np.maximum(0, sig1), np.maximum(0, sig2).T)
        
        # Normalize by magnitude of spikes
        mag = (np.linalg.norm(np.maximum(0, sig1)) * np.linalg.norm(np.maximum(0, sig2)))
        return float(correlation / (mag + 1e-10))

    def _ensure_legal_nfc_to_indices(self):
        """
        Map canonical legal-name key -> all row indices (multi-office, Unicode/whitespace variants).
        Required for Phase 3 to collect every same-legal row; optional when loaded from new names.pkl.
        """
        if getattr(self, "_legal_nfc_to_indices", None) and len(self._legal_nfc_to_indices) > 0:
            return
        n = len(self.original_company_names)
        if n == 0:
            self._legal_nfc_to_indices = {}
            return
        print("   Building legal-name key index (NFKC, multi-site rows)...", end=" ", flush=True)
        self._legal_nfc_to_indices = {}
        for i, name in enumerate(self.original_company_names):
            k = _legal_name_match_key(name)
            if k not in self._legal_nfc_to_indices:
                self._legal_nfc_to_indices[k] = []
            self._legal_nfc_to_indices[k].append(i)
        print(f"[OK] ({len(self._legal_nfc_to_indices):,} unique keys)")

    def _create_fast_lookup_sets(self):
        """Create fast lookup sets for exact matching (called after building index)"""
        # SKIP if already loaded from cache (but still build legal index if new/missing)
        if (hasattr(self, '_company_names_lower_set') and
            hasattr(self, '_company_names_lower_to_index') and
            hasattr(self, '_company_words_dict')):
            self._ensure_legal_nfc_to_indices()
            return

        print("Creating fast lookup sets for exact matching...")
        total = len(self.original_company_names)
        
        # Create lowercase sets for O(1) exact match lookup
        print("   Step 1/3: Creating lowercase lookup set...")
        self._company_names_lower_set = set()
        for name in tqdm(self.original_company_names, desc="   Lowercase set", total=total, unit="names", ncols=80, disable=not HAS_TQDM):
            self._company_names_lower_set.add(name.lower())
        
        # Create reverse lookup dictionary: lowercase_name -> list of indices (for O(1) index lookup)
        print("   Step 2/3: Creating reverse lookup dictionary...")
        self._company_names_lower_to_index = {}
        for i, name in enumerate(tqdm(self.original_company_names, desc="   Reverse lookup", total=total, unit="names", ncols=80, disable=not HAS_TQDM)):
            name_lower = name.lower()
            if name_lower not in self._company_names_lower_to_index:
                self._company_names_lower_to_index[name_lower] = []
            self._company_names_lower_to_index[name_lower].append(i)
        
        # Create word-based lookup for faster partial matching
        print("   Step 3/3: Creating word-based lookup dictionary...")
        self._company_words_dict = {}
        for i, name in enumerate(tqdm(self.original_company_names, desc="   Word lookup", total=total, unit="names", ncols=80, disable=not HAS_TQDM)):
            words = set(name.lower().split())
            for word in words:
                if word not in self._company_words_dict:
                    self._company_words_dict[word] = []
                self._company_words_dict[word].append(i)
        
        print(f"   [OK] Created fast lookup sets for {len(self.original_company_names):,} companies")
        self._ensure_legal_nfc_to_indices()

    def get_cache_info(self):
        """Get information about cached data"""
        if not os.path.exists(self.cache_dir):
            return "No cache directory found"
        
        cache_files = os.listdir(self.cache_dir)
        if not cache_files:
            return "Cache directory is empty"
        
        # Group files by cache key
        cache_groups = {}
        for file in cache_files:
            if '_' in file:
                key = file.split('_')[0]
                if key not in cache_groups:
                    cache_groups[key] = []
                cache_groups[key].append(file)
        
        info = f"Found {len(cache_groups)} cached datasets:\n"
        for key, files in cache_groups.items():
            if len(files) == 4:  # Complete cache
                info += f"  {key}: Complete cache\n"
            else:
                info += f"  {key}: Incomplete cache ({len(files)}/4 files)\n"
        
        return info
    
    def clear_cache(self, cache_key=None, confirm_delete=False):
        """
        Clear specific cache or all cache.
        
        SAFETY: Requires confirm_delete=True to actually delete files.
        This prevents accidental data loss.
        
        Args:
            cache_key: Specific cache key to clear, or None for all caches
            confirm_delete: Must be True to actually delete files
        """
        if not confirm_delete:
            raise ValueError(
                "Cache deletion requires confirm_delete=True. "
                "This is a safety measure to prevent accidental data loss. "
                "If you are sure you want to delete cache files, call: "
                "clear_cache(cache_key, confirm_delete=True)"
            )
            
        if not os.path.exists(self.cache_dir):
            print("No cache directory found")
            return
        
        if cache_key:
            # Clear specific cache
            paths = self.get_cache_paths(cache_key)
            for path in paths.values():
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Removed: {path}")
            print(f"Cleared cache: {cache_key}")
        else:
            # Clear all cache
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print("Cleared all cache")
    
    def is_index_ready(self):
        """Check if the index is ready for matching"""
        return (self.vector_store.is_ready() and 
                len(self.original_company_names) > 0)

    def preprocess(self, names):
        # Optional: normalize casing, strip punctuation, etc.
        return [name.strip().lower() for name in names]

    def build_index(self, company_names=None, filepath=None):
        """
        Build index from company names or filepath.
        If filepath is provided, uses file metadata for fast cache checking.
        """
        # Early return if index is already loaded - prevents duplicate loading
        if self.is_index_ready():
            print(f"[OK] Index already loaded with {len(self.original_company_names):,} companies - skipping rebuild")
            print(f"  Index status: Ready | Companies: {len(self.original_company_names):,} | Embeddings shape: {self.vector_store.embeddings.shape if self.vector_store.embeddings is not None else 'N/A'}")
            return True
        
        cache_key = None
        file_cache_key = None
        
        # Phase 1: Fast cache check using file metadata (if filepath provided)
        if filepath and os.path.exists(filepath):
            file_cache_key = self.get_cache_key_from_file(filepath)
            if file_cache_key and self.load_from_cache(file_cache_key):
                print(f"Using cached index (loaded from file metadata - no file loading needed!)")
                print(f"  Cache key: {file_cache_key[:16]}...")
                print(f"  Companies in cache: {len(self.original_company_names):,}")
                return True
        
        # Phase 2: Load company names if not provided
        if company_names is None:
            if filepath and os.path.exists(filepath):
                # Load from file
                import json
                print(f"Cache miss - loading companies from {filepath}...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                company_names = []
                for item in data:
                    if isinstance(item, dict) and "Company Name" in item:
                        company_names.append(item["Company Name"])
                print(f"Loaded {len(company_names):,} company names")
            else:
                raise ValueError("Must provide either company_names or filepath")
        
        # Phase 3: Generate content-based cache key (for validation)
        content_cache_key = self.get_cache_key(company_names)
        
        # Use file-based key if available and matches, otherwise use content-based
        if file_cache_key and file_cache_key == content_cache_key:
            cache_key = file_cache_key
            print(f"File-based cache key matches content-based key [OK]")
        else:
            cache_key = content_cache_key
            if file_cache_key:
                print(f"File-based cache key differs from content - using content-based key")
        
        # Try to load from cache with content-based key
        if self.load_from_cache(cache_key):
            print(f"Using cached index for {len(self.original_company_names)} companies")
            return True
        
        # Cache miss - build new index
        print(f"Building new index for {len(company_names):,} companies...")
        
        # Store original names
        self.original_company_names = company_names
        
        # Store preprocessed names for matching
        print("Preprocessing company names...")
        import time
        preprocess_start = time.time()
        self.company_names = []
        for name in tqdm(company_names, desc="   Preprocessing", total=len(company_names), unit="names", ncols=80, disable=not HAS_TQDM):
            self.company_names.append(name.strip().lower())
        preprocess_time = time.time() - preprocess_start
        print(f"   [OK] Preprocessed {len(self.company_names):,} names in {preprocess_time:.1f}s")
        
        # Generate embeddings with dramatically increased batch size for CPU speed
        print("Generating embeddings with optimized CPU batch processing...")
        
        # AGGRESSIVE optimization for sub-1-hour processing
        batch_size = 2048  # Reduced from 50000 to avoid stuck processes
        
        print(f"   CPU-optimized processing")
        print(f"   Batch size: {batch_size:,} (vs previous 32)")
        print(f"   Total companies: {len(company_names):,}")
        print(f"   Estimated batches: {(len(company_names) + batch_size - 1) // batch_size:,}")
        print(f"   Target: Complete in under 1 hour")
        print(f"   Normalization: DISABLED for speed")
        
        self.max_company_count = 0 # No frequency data in standard build_index
        
        # Memory optimization: clear any existing data
        import gc
        gc.collect()
        
        # Process in much larger batches for dramatic speed improvement
        embeddings_list = []
        total_batches = (len(company_names) + batch_size - 1) // batch_size
        
        import time
        overall_start = time.time()
        
        # Create progress bar for batch processing
        batch_range = range(0, len(company_names), batch_size)
        pbar = tqdm(batch_range, desc="   Generating embeddings", total=total_batches, unit="batch", ncols=80, disable=not HAS_TQDM)
        
        for i in pbar:
            batch_count = (i // batch_size) + 1
            batch_end = min(i + batch_size, len(company_names))
            batch_names = company_names[i:batch_end]
            
            # Update progress bar description with current batch info
            pbar.set_description(f"   Batch {batch_count:,}/{total_batches:,} (companies {i:,}-{batch_end:,})")
            
            start_time = time.time()
            
            # Process with ULTRA-speed optimizations
            try:
                batch_embeddings = self.model.encode(
                    batch_names, 
                    convert_to_numpy=True, 
                    normalize_embeddings=False,  # Disable normalization for speed
                    show_progress_bar=False
                )
                
                batch_time = time.time() - start_time
                pbar.set_postfix({"time": f"{batch_time:.1f}s", "companies": f"{len(batch_names):,}"})
                
            except Exception as e:
                pbar.write(f"      Error processing batch {batch_count}: {e}")
                pbar.write(f"      Retrying with smaller batch...")
                # Fallback to smaller batch size
                smaller_batch = batch_names[:len(batch_names)//2]
                batch_embeddings = self.model.encode(
                    smaller_batch, 
                    convert_to_numpy=True, 
                    normalize_embeddings=False,
                    show_progress_bar=False
                )
                pbar.write(f"      Smaller batch completed successfully")
            
            embeddings_list.append(batch_embeddings)
            
            # Memory cleanup every few batches
            if (i // batch_size) % 3 == 0:  # Every 3 batches
                gc.collect()
        
        pbar.close()
        overall_time = time.time() - overall_start
        print(f"   [OK] Embedding generation completed in {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
        
        # Combine all embeddings
        all_embeddings = np.vstack(embeddings_list)
        print(f"   Generated embeddings: {all_embeddings.shape}")
        
        # Build FAISS index via VectorStore
        self.vector_store.build_index(all_embeddings)
        
        # Create fast lookup sets for exact matching
        self._create_fast_lookup_sets()
        
        # Create Acronym Index
        self._create_acronym_index()
        
        # Precompute similarity matrix and acronym data (NEW - for performance)
        print("\n" + "="*60)
        print("PRECOMPUTING CACHES FOR INSTANT QUERY PERFORMANCE")
        print("="*60)
        self._precompute_similarity_matrix()
        self._precompute_acronym_data()
        print("="*60 + "\n")
        
        # Save to cache for future use
        print("Saving to cache...")
        
        # If we have a file-based key, use it for storage so we can fast-load next time
        if file_cache_key:
             cache_key = file_cache_key
             
        import time
        cache_start = time.time()
        self.save_to_cache(cache_key, None, None, self.company_names, self.original_company_names)
        cache_time = time.time() - cache_start
        print(f"   [OK] Cache saved in {cache_time:.1f}s")
        
        print(f"\n{'='*60}")
        print(f"[OK] Index built successfully! Ready to match {len(company_names):,} companies")
        print(f"{'='*60}\n")
        return True

    def add_companies(self, new_company_names):
        """Add new companies to existing index (incremental update)"""
        if not self.is_index_ready():
            print("Error: No existing index to update")
            return False
        
        # Check for duplicates
        existing_set = set(self.original_company_names)
        truly_new = [name for name in new_company_names if name not in existing_set]
        
        if not truly_new:
            print("No new companies to add")
            return True
        
        print(f"Adding {len(truly_new):,} new companies to existing index...")
        import time
        add_start = time.time()
        
        # Preprocess new names
        print("   Step 1/5: Preprocessing company names...")
        new_preprocessed = []
        for name in tqdm(truly_new, desc="   Preprocessing", unit="names", ncols=80, disable=not HAS_TQDM):
            new_preprocessed.append(name.strip().lower())
        print(f"   [OK] Preprocessed {len(new_preprocessed):,} names")
        
        # Generate embeddings for new companies with ULTRA-speed optimizations
        print("   Step 2/5: Generating embeddings for new companies...")
        embed_start = time.time()
        new_embeddings = self.model.encode(
            new_preprocessed, 
            convert_to_numpy=True, 
            normalize_embeddings=False,  # Disable normalization for speed
            show_progress_bar=HAS_TQDM
        )
        embed_time = time.time() - embed_start
        print(f"   [OK] Generated embeddings in {embed_time:.1f}s")
        
        # Add to existing arrays / index via VectorStore
        print("   Step 3/4: Updating VectorStore...")
        self.original_company_names.extend(truly_new)
        self.company_names.extend(new_preprocessed)
        
        self.vector_store.add_vectors(new_embeddings)
        
        # Update fast lookup sets for incremental updates
        print("   Step 4/5: Updating fast lookup sets...")
        self._create_fast_lookup_sets()
        
        add_time = time.time() - add_start
        print(f"\n   [OK] Successfully added {len(truly_new):,} companies in {add_time:.1f}s")
        print(f"   [OK] Total companies in index: {len(self.original_company_names):,}")
        
        # Update cache with new data
        print("   Updating cache...")
        cache_start = time.time()
        cache_key = self.get_cache_key(self.original_company_names)
        self.save_to_cache(cache_key, None, None, self.company_names, self.original_company_names)
        cache_time = time.time() - cache_start
        print(f"   [OK] Cache updated in {cache_time:.1f}s")
        
        return True

    # Text processing methods delegated to TextPreprocessor

    def match(self, query, top_k=10):
        """
        Hybrid Semantic + Lexical Matching (Retrieve & Re-rank)
        
        Delegates to match_with_location for consistent logic.
        """
        return self.match_with_location(query, city=None, state=None, top_k=top_k)

    def batch_match(self, queries, top_k=10, batch_size=32):
        """
        Batch version of match() - processes multiple queries efficiently
        Encodes all queries in batches for better performance
        NOTE: All queries go through full semantic search and re-ranking to show all potential matches
        
        Args:
            queries: List of query strings
            top_k: Number of top matches to return per query
            batch_size: Number of queries to encode at once
            
        Returns:
            List of results, one per query (same format as match())
        """
        if not queries:
            return []
        
        all_results = []
        
        # Encode all queries in batches (no early termination - we want to see all matches)
        query_vecs_dict = {}  # query_idx -> query_vec
        
        total_encode_batches = (len(queries) + batch_size - 1) // batch_size
        encode_batch_range = range(0, len(queries), batch_size)
        
        print(f"Encoding {len(queries):,} queries in {total_encode_batches:,} batches...")
        encode_pbar = tqdm(encode_batch_range, desc="   Encoding queries", total=total_encode_batches, unit="batch", ncols=80, disable=not HAS_TQDM)
        
        for batch_start in encode_pbar:
            batch_end = min(batch_start + batch_size, len(queries))
            batch_queries = queries[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            # Encode batch at once
            batch_vecs = self.model.encode(
                batch_queries, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Store vectors with their original query indices
            for local_idx, orig_idx in enumerate(batch_indices):
                query_vecs_dict[orig_idx] = batch_vecs[local_idx:local_idx+1]
            
            encode_pbar.set_postfix({"queries": f"{batch_end:,}/{len(queries):,}"})
        
        encode_pbar.close()
        print(f"   [OK] Encoded {len(queries):,} queries")
        
        # INCREASED retrieval limit (funnel) for higher precision on large datasets
        candidate_k = min(1000, len(self.original_company_names))
        
        print(f"Processing {len(queries):,} queries with semantic search and re-ranking...")
        match_pbar = tqdm(enumerate(queries), desc="   Matching queries", total=len(queries), unit="query", ncols=80, disable=not HAS_TQDM)
        
        for query_idx, query in match_pbar:
            query_lower = query.lower().strip()
            
            # Semantic search for this query
            query_vec = query_vecs_dict[query_idx]
            semantic_scores, semantic_indices = self.vector_store.search(query_vec, candidate_k)
            
            candidates = []
            
            # Normalize semantic scores
            if len(semantic_scores) > 0 and len(semantic_scores[0]) > 0:
                max_sem_score = float(semantic_scores[0][0])
            else:
                max_sem_score = 1.0
            
            # Re-ranking
            for j, i in enumerate(semantic_indices[0]):
                idx = int(i)
                company_name = self.original_company_names[idx]
                original_semantic_score = float(semantic_scores[0][j])
                
                # Normalize Vector Score
                sem_score_norm = original_semantic_score / max_sem_score if max_sem_score > 0 else 0
                
                # Get query index if query is in database (for cache lookup)
                query_idx = None
                if hasattr(self, '_company_names_lower_to_index') and query_lower in self._company_names_lower_to_index:
                    query_idx = self._company_names_lower_to_index[query_lower]
                
                # Calculate String Similarity (using cache if available)
                string_score = self._get_cached_similarity(query, query_idx, idx)
                
                # Exact Match Bonus
                if query_lower == company_name.lower():
                    string_score = 1.0
                
                # Check for acronym fidelity boost
                acronym_fidelity = 0.0
                query_acronym = TextPreprocessor.generate_acronym(query)
                if query_acronym and len(query) < 12:  # Query might be an acronym
                    # Check if candidate is a literal expansion of this acronym
                    acronym_fidelity = self._get_cached_acronym_fidelity(query, query_idx, idx)
                
                # --- CONCEPT PROBING ---
                concept_alignment = 0.0
                cand_sig = None
                query_sig = None
                
                # Check for cached anchors or ensure loaded
                self._ensure_anchors_loaded()
                
                if self._anchor_vectors is not None:
                     cand_vec = self.vector_store.embeddings[idx].reshape(1, -1)
                     cand_sig = self._get_concept_signature(cand_vec)
                     # Batch query vector access
                     q_vec = query_vecs_dict[query_idx]
                     query_sig = self._get_concept_signature(q_vec)
                     concept_alignment = self._calculate_signature_correlation(query_sig, cand_sig)

                # Weighted Combination: 50/25/25
                base_score = (string_score * 0.5) + (sem_score_norm * 0.25) + (concept_alignment * 0.25)
                name_score = base_score
                
                # BOOST for acronym fidelity
                if acronym_fidelity > 0.8 and len(query_acronym) > 2:
                    name_score = min(1.0, name_score + (acronym_fidelity * 0.15))
                
                # TOKEN COVERAGE CHECK
                q_tokens = set(TextPreprocessor.clean_company_name(query).split())
                t_tokens = set(TextPreprocessor.clean_company_name(company_name).split())
                is_full_overlap = q_tokens.issubset(t_tokens) if q_tokens else False
                
                # TIERED LEXICAL BOOSTS (Replacing hard floors for transparency)
                lexical_boost = 0.0
                if string_score >= 0.92 or is_full_overlap:
                    new_ns = CompanyMatcher._apply_lexical_floor(name_score, 0.95)
                    if new_ns > name_score:
                        lexical_boost = new_ns - name_score
                        name_score = new_ns
                elif string_score >= 0.80:
                    new_ns = CompanyMatcher._apply_lexical_floor(name_score, 0.90)
                    if new_ns > name_score:
                        lexical_boost = new_ns - name_score
                        name_score = new_ns
                
                final_score = name_score
                
                candidates.append({
                    "name": company_name,
                    "score": final_score,
                    "name_score": name_score,
                    "lexical_boost": lexical_boost,
                    "concept_alignment": concept_alignment,
                    "concept_signature": cand_sig.tolist() if cand_sig is not None else None,
                    "semantic_score": original_semantic_score,
                    "normalized_semantic_score": sem_score_norm,
                    "string_score": string_score,
                    "index": idx,
                    "match_type": "hybrid"
                })
            
            # Exact match override - ensure exact match is marked correctly
            if hasattr(self, '_company_names_lower_set') and query_lower in self._company_names_lower_set:
                # Use O(1) lookup if available
                if hasattr(self, '_company_names_lower_to_index'):
                    i = self._company_names_lower_to_index[query_lower]
                    name = self.original_company_names[i]
                else:
                    # Fallback to iteration if dictionary doesn't exist
                    for i, name in enumerate(self.original_company_names):
                        if name.lower() == query_lower:
                            break
                
                existing = next((c for c in candidates if c['index'] == i), None)
                if existing:
                    existing['score'] = 1.0
                    existing['match_type'] = "exact"
                else:
                    candidates.append({
                        "name": name,
                        "score": 1.0,
                        "semantic_score": 1.0,
                        "string_score": 1.0,
                        "index": i,
                        "match_type": "exact"
                    })
            
            # Sort by Final Score
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top_k
            results = candidates[:top_k]
            CompanyMatcher._ensure_top5_score_spread(results)
            all_results.append(results)
            
            match_pbar.set_postfix({"matches": f"{len(results)}/query"})
        
        match_pbar.close()
        print(f"   [OK] Processed {len(queries):,} queries, found matches for all")
        
        return all_results

    def explain_match(self, query, match_name):
        """Explanation based on the new hybrid logic"""
        query_clean = TextPreprocessor.clean_company_name(query)
        match_clean = TextPreprocessor.clean_company_name(match_name)
        
        q_tokens = set(query_clean.split())
        t_tokens = set(match_clean.split())
        overlap = q_tokens.intersection(t_tokens)
        
        # Find the specific match details from the last run if available
        match_details = None
        if hasattr(self, '_last_matches'):
             match_details = next((m for m in self._last_matches if m['name'] == match_name), None)

        explanation = {
            "query_tokens": list(q_tokens),
            "match_tokens": list(t_tokens),
            "overlap": list(overlap),
            "overlap_score": len(overlap) / max(len(q_tokens), 1) if q_tokens else 0,
            "match_type": match_details['match_type'] if match_details else "hybrid"
        }
        
        # Add hybrid scoring details if available
        if match_details:
            explanation["string_score"] = match_details.get("string_score", 0.0)
            explanation["semantic_score"] = match_details.get("semantic_score", 0.0)
            explanation["normalized_semantic_score"] = match_details.get("normalized_semantic_score", 0.0)
            explanation["concept_alignment"] = match_details.get("concept_alignment", 0.0)
            explanation["location_score"] = match_details.get("location_score", 0.0)
            explanation["final_score"] = match_details.get("score", 0.0)
            
            # Additional reporting fields
            explanation["lexical_boost"] = match_details.get("lexical_boost", 0.0)
            explanation["acronym_fidelity"] = match_details.get("acronym_fidelity", 0.0)
            explanation["concept_signature"] = match_details.get("concept_signature")
            explanation["location_boost"] = match_details.get("location_boost", 0.0)
            explanation["popularity_boost"] = match_details.get("popularity_boost", 0.0)
            explanation["match_city"] = match_details.get("city", "")
            explanation["match_state"] = match_details.get("state", "")
            explanation["count"] = match_details.get("count", 0)
        
        return explanation

    # ========================================================================
    # CHUNKED ENCODING (for checkpoint/resume during large rebuilds)
    # ========================================================================

    def encode_chunk_to_file(self, chunk_texts, path, batch_size=512):
        """Encode a chunk of texts and save embeddings to .npy. Uses sort-by-length for optimal padding."""
        if not chunk_texts:
            np.save(path, np.array([]).reshape(0, 384))
            return True
        n_texts = len(chunk_texts)
        sort_idx = sorted(range(n_texts), key=lambda i: len(chunk_texts[i]))
        sorted_text = [chunk_texts[i] for i in sort_idx]
        emb_list = []
        for i in range(0, n_texts, batch_size):
            batch = sorted_text[i : i + batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True,
                                    normalize_embeddings=False, show_progress_bar=False)
            emb_list.append(emb)
        sorted_all = np.vstack(emb_list)
        restore = np.empty(n_texts, dtype=np.int64)
        for new_pos, orig_pos in enumerate(sort_idx):
            restore[orig_pos] = new_pos
        embeddings = sorted_all[restore]
        try:
            np.save(path, embeddings)
        except OSError as e:
            raise OSError(
                f"Failed to save chunk to {path}: {e}\n"
                "Check disk space, antivirus, or use --work-dir on a different drive."
            ) from e
        return True

    def build_index_from_chunk_dir(self, work_dir, cache_key):
        """Load chunk .npy files, merge, build FAISS, and save cache. Expects company_* attrs to be set."""
        import re
        import glob as _glob
        chunks_dir = os.path.join(work_dir, "chunks")
        raw = _glob.glob(os.path.join(chunks_dir, "chunk_*.npy"))
        chunk_files = sorted(raw, key=lambda p: int(re.search(r"chunk_(\d+)", os.path.basename(p)).group(1)))
        if not chunk_files:
            print("   ERROR: No chunk files found in " + chunks_dir)
            return False
        print(f"   Loading {len(chunk_files)} chunks...")
        parts = [np.array(np.load(p, mmap_mode="r"), dtype=np.float32) for p in chunk_files]
        merged = np.vstack(parts)
        print(f"   Merged embeddings shape: {merged.shape}")
        self.vector_store.build_index(merged)
        self._create_fast_lookup_sets()
        self._create_acronym_index()
        print("   Saving to cache...")
        self.save_to_cache(cache_key, None, None, self.company_names, self.original_company_names,
                          locations=self.company_locations, counts=self.company_counts, ids=self.company_ids)
        return True

    # ========================================================================
    # LOCATION-AWARE MATCHING METHODS
    # ========================================================================
    
    def build_index_with_location(self, filepath=None, data=None, work_dir=None):
        """
        Build index from data that includes location information and IDs.
        
        Args:
            filepath: Path to JSON file with format:
                [{"ID": 123, "Company Name": "...", "City": "...", "State": "...", "Count": N}, ...]
            data: List of dicts with same format (alternative to filepath)
            work_dir: If set, use checkpointed chunk encoding (resume on restart)
            
        Returns:
            True if successful, False otherwise
        """
        import json
        import time
        
        # Early return if index is already loaded
        if self.is_index_ready() and self.has_location_data:
            print(f"[OK] Index with location already loaded - skipping rebuild")
            return True
        
        # OPTIMIZATION: Check file-based cache BEFORE loading data
        if filepath and os.path.exists(filepath):
            cache_key_file = self.get_cache_key_from_file(filepath) + "_loc"
            # We don't know the exact key yet because we haven't loaded names, 
            # but we can check if a cache exists for this file signature.
            # get_cache_key_from_file returns a hash of filepath+size+mtime.
            # Use that as the key.
            # NOTE: The existing logic uses get_cache_key(names) + "_loc".
            # If we want to support file-based, we need to save/load using file-based key OR 
            # verify if we can trust the file signature to map to the same content key.
            
            # Strategy: Try to load using file-based key directly.
            # If we succeed, we save significantly. 
            # BUT we need to ensure we save using this key too.
            # Or, we just use this key for loading.
            
            print(f"Checking cache for file: {os.path.basename(filepath)}")
            print(f"   Cache key: {cache_key_file}")
            
            if self.load_from_cache(cache_key_file):
                print(f"[OK] Fast load successful! Loaded from cache using file metadata.")
                return True
        
        # Load data (fallback)
        
        # Load data
        if data is None:
            if filepath and os.path.exists(filepath):
                print(f"Loading company data with location from {filepath}...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   Loaded {len(data):,} entries")
            else:
                raise ValueError("Must provide either data or filepath")
        
        # Extract company names, location data, and IDs
        print("Extracting company names, location data, and IDs...")
        company_names = []
        locations = []
        counts = []
        ids = []
        
        for item in tqdm(data, desc="   Processing", unit="entries", ncols=80, disable=not HAS_TQDM):
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
                locations.append({
                    "city": item.get("City", "").strip() if item.get("City") else "",
                    "state": item.get("State", "").strip() if item.get("State") else ""
                })
                counts.append(item.get("Count", 0))
                ids.append(item.get("ID", None))  # Database identifier
        
        print(f"   Extracted {len(company_names):,} companies with location data")
        has_ids = any(id is not None for id in ids)
        if has_ids:
            print(f"   Database IDs: Available")
        
        # Store location data and IDs
        self.company_locations = locations
        self.company_counts = counts
        self.company_ids = ids
        self.has_location_data = True
        self.max_company_count = max(counts) if counts else 0
        
        # --- LOCATION BAKING REMOVED ---
        # The baking logic below is removed to decouple location from core embeddings.
        # We now use only the company names for the semantic search space.
        baking_text = company_names
        
        # Generate cache key
        # If we have a file, use the file-based key for storage (matches our fast-load logic)
        if filepath and os.path.exists(filepath):
             cache_key = self.get_cache_key_from_file(filepath) + "_loc"
        else:
             cache_key = self.get_cache_key(company_names) + "_loc"
        
        # Try to load from cache (standard check using the selected key)
        # This catches cases where we just calculated the key (file or content) and it exists
        if self.load_from_cache(cache_key):
            print(f"Using cached index with location data")
            return True
        
        # Build index using existing method logic
        print(f"Building new index with location for {len(company_names):,} companies...")
        
        # Store original names
        self.original_company_names = company_names
        
        # Store preprocessed names
        print("Preprocessing company names...")
        preprocess_start = time.time()
        self.company_names = []
        for name in tqdm(company_names, desc="   Preprocessing", total=len(company_names), 
                        unit="names", ncols=80, disable=not HAS_TQDM):
            self.company_names.append(name.strip().lower())
        print(f"   [OK] Preprocessed in {time.time() - preprocess_start:.1f}s")
        
        # --- CHECKPOINTED ENCODING (when work_dir is set) ---
        if work_dir:
            chunks_dir = os.path.join(work_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            chunk_size = 20000
            total_chunks = (len(baking_text) + chunk_size - 1) // chunk_size
            # Pre-check: if all chunks exist, skip encoding entirely (resume path)
            existing = sum(1 for i in range(total_chunks)
                          if os.path.exists(os.path.join(chunks_dir, f"chunk_{i}.npy")))
            if existing == total_chunks:
                print(f"   RESUMING: All {total_chunks} chunks present — skipping encoding (no recompute)")
            else:
                print(f"   Checkpointed encoding: {existing}/{total_chunks} chunks done, ~{chunk_size:,} companies/chunk")
            for chunk_idx, start in enumerate(range(0, len(baking_text), chunk_size)):
                chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx}.npy")
                if os.path.exists(chunk_path):
                    print(f"   Skipping chunk {chunk_idx} (already encoded)")
                    continue
                chunk_texts = baking_text[start:start + chunk_size]
                print(f"   Encoding chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_texts):,} companies)...")
                self.encode_chunk_to_file(chunk_texts, chunk_path, batch_size=512)
            print("   All chunks encoded. Building index from chunks...")
            return self.build_index_from_chunk_dir(work_dir, cache_key)
        
        # Generate embeddings using multi-process pool for dramatic speedup
        print(f"Generating embeddings using multi-process pool...")
        overall_start = time.time()
        
        # Start a multi-process pool
        pool = self.model.start_multi_process_pool()
        
        # Chunk the data to show progress
        # 4.3M records is too big to wait for a single progress bar update at the very end
        chunk_size = 5000 
        total_chunks = (len(baking_text) + chunk_size - 1) // chunk_size
        
        embeddings_list = []
        
        print(f"   Splitting {len(baking_text):,} records into {total_chunks} chunks for visibility...")
        pbar = tqdm(range(0, len(baking_text), chunk_size), desc="   Multi-process Encoding", 
                   total=total_chunks, unit="chunk", ncols=80, disable=not HAS_TQDM)
        
        for i in pbar:
            chunk_batch = baking_text[i : i + chunk_size]
            
            # Encode chunk with multi-process (distributes this chunk across cores)
            chunk_emb = self.model.encode_multi_process(
                chunk_batch, 
                pool,
                batch_size=2048
            )
            embeddings_list.append(chunk_emb)
            
            # Force garbage collection to keep memory stable
            if (i // chunk_size) % 5 == 0:
                import gc
                gc.collect()
        
        pbar.close()
        
        # Stop the pool
        self.model.stop_multi_process_pool(pool)
        
        print(f"   [OK] Multi-process encoding completed in {time.time() - overall_start:.1f}s")
        
        # Combine embeddings
        self.embeddings = np.vstack(embeddings_list)
        
        # Build FAISS index via VectorStore
        print("Building FAISS index...")
        self.vector_store.build_index(self.embeddings)
        print(f"   [OK] FAISS index built")
        
        # Create fast lookup sets
        self._create_fast_lookup_sets()
        
        # Create Acronym Index
        self._create_acronym_index()
        
        # Precompute similarity matrix and acronym data (DISABLED - O(N^2) too slow for large datasets)
        # self._precompute_similarity_matrix()
        # self._precompute_acronym_data()
        
        # Save to cache with location data
        print("Saving to cache with location data...")
        self.save_to_cache(cache_key, None, None, 
                          self.company_names, self.original_company_names,
                          locations=self.company_locations, counts=self.company_counts,
                          ids=self.company_ids)
        
        print(f"\n{'='*60}")
        print(f"[OK] Index built with location data!")
        print(f"   Companies: {len(company_names):,}")
        print(f"   Location entries: {len(self.company_locations):,}")
        print(f"{'='*60}\n")
        return True

    # State abbreviation mappings (both directions)
    # Location methods moved to TextPreprocessor

    def match_with_location(self, query, city=None, state=None, top_k=10):
        """
        Match company name with optional location-based re-ranking.
        
        When no exact name match is found, uses city/state to boost
        scores of candidates in the same location.
        
        Args:
            query: Company name to search for
            city: Optional city for location matching
            state: Optional state for location matching  
            top_k: Number of results to return
            
        Returns:
            List of match results with location and count info
        """
        query_lower = query.lower().strip()
        q_legal = _legal_name_match_key(query)
        self._ensure_legal_nfc_to_indices()
        # Normalize city/state in case they come as lists (e.g. from DB or JSON)
        city = _to_loc_str(city) if city else ""
        state = _to_loc_str(state) if state else ""
        city = city.strip() or None
        state = state.strip() or None
        use_location = (city or state) and self.has_location_data
        
        candidates = []
        found_indices = set()
        
        # --- PHASE 0: ACRONYM EXPANSION ---
        # 1. Query is potential Acronym (e.g. "ABA") -> Look for full names
        # We check if query is short-ish, upper case or query_lower not in stop words
        if len(query) < 12 and hasattr(self, 'acronym_index'):
             # Try exact case first, then upper
             potential_acronym = query.strip()
             acronym_matches = None
             
             # IGNORE 2-letter acronyms in Phase 0 (too much noise from states/suffixes)
             if len(potential_acronym) > 2:
                 acronym_matches = self.acronym_index.get(potential_acronym)
             if not acronym_matches:
                 acronym_matches = self.acronym_index.get(potential_acronym.upper())
                 
             if acronym_matches:
                 for idx in _iter_indices(acronym_matches):
                     if idx not in found_indices:
                         company_name = self.original_company_names[idx]
                         fidelity = TextPreprocessor.calculate_acronym_fidelity(query, company_name)

                         query_vec_ac = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                         target_vec_ac = self.vector_store.embeddings[idx].reshape(1, -1)
                         target_vec_ac = target_vec_ac / (np.linalg.norm(target_vec_ac) + 1e-10)
                         sem_score_ac = float(np.dot(query_vec_ac, target_vec_ac.T)[0][0])

                         # Fix 3 (deferred): require minimum evidence to surface an
                         # acronym expansion at all. Below threshold, drop the candidate
                         # entirely and let Phase 1/2 retrieval decide.
                         if fidelity < 0.6 or sem_score_ac < 0.35:
                             continue

                         # Fix 3 (deferred): tighter formula caps at 0.85 instead of 0.99.
                         final_score_ac = min(
                             0.85,
                             0.50 + (fidelity * 0.30) + (max(0.0, sem_score_ac) * 0.20),
                         )

                         # Phase 0 acronym path does not compute a real name_score
                         # (no string/concept signal). Tag as low-confidence and cap at
                         # 0.79 so audit Gate C cannot fire on these rows.
                         ns_estimate = 0.0
                         if ns_estimate < 0.05:
                             final_score_ac = min(final_score_ac, 0.79)
                             mt_label = "acronym_expansion_low_conf"
                         else:
                             mt_label = "acronym_expansion"

                         s_cap = min(1.0, final_score_ac)
                         candidates.append({
                            "name": company_name,
                            "score": s_cap,
                            "name_score": ns_estimate,
                            "acronym_fidelity": fidelity,
                            "semantic_score": sem_score_ac,
                            "normalized_semantic_score": sem_score_ac,
                            "string_score": 0.5, # Dummy
                            "index": idx,
                            "match_type": mt_label,
                            "acronym_fwd": True,
                            "city": "", "state": "", "id": None, "count": 0, "location_score": 0.0
                         })
                         found_indices.add(idx)

        # 2. Query is Full Name (e.g. "American Bar Association") -> Look for Acronym (e.g. "ABA")
        generated_acronym = TextPreprocessor.generate_acronym(query)
        # IGNORE 2-letter acronyms in Phase 0 (too much noise from states/suffixes)
        if generated_acronym and len(generated_acronym) > 2 and hasattr(self, 'acronym_index'):
            # Check if this acronym exists as a company name
            if hasattr(self, '_company_names_lower_to_index'):
                val = self._company_names_lower_to_index.get(generated_acronym.lower())
                idx = val[0] if isinstance(val, list) and val else val
                if idx is not None:
                    idx = int(idx)
                    if idx not in found_indices:
                        company_name = self.original_company_names[idx]
                        # TIE-BREAKER: Use fidelity score and semantic check
                        # Note: For reverse, query is text and company_name is the acronym
                        fidelity = TextPreprocessor.calculate_acronym_fidelity(company_name, query)
                        
                        # FURTHER REDUCED BASE for reverse acronyms (very lossy/risky)
                        # Base (0.45) + (Fidelity * 0.20) + (Semantic * 0.10)
                        final_score_ac = 0.45 + (fidelity * 0.20) + (1.0 * 0.10)
                        
                        # Cap at 0.80 to ensure strong string matches win over reverse acronyms
                        final_score_ac = min(0.80, final_score_ac)
                        
                        # Verify it's actually the acronym we want (case sensitive-ish)
                        if company_name.strip() == generated_acronym:
                            candidates.append({
                                "name": company_name,
                                "score": final_score_ac,
                                "name_score": final_score_ac,
                                "acronym_fidelity": fidelity,
                                "semantic_score": 1.0,
                                "normalized_semantic_score": 1.0,
                                "string_score": 1.0,
                                "location_score": 0.0,
                                "index": idx,
                                "match_type": "acronym_reverse",
                                "acronym_rev": True,
                                "city": "", "state": "", "id": None, "count": 0
                            })
                            found_indices.add(idx)
        
        # --- PHASE 1: RETRIEVAL (Semantic Search) ---
        # INCREASED retrieval limit (funnel) for higher precision on large datasets
        candidate_k = min(1000, len(self.original_company_names))
        
        # RETRIEVAL (Semantic Search)
        # Location baking removed from query - we search by name only
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # GENERATE CONCEPT SIGNATURE FOR QUERY
        query_sig = self._get_concept_signature(query_vec)
            
        semantic_scores, semantic_indices = self.vector_store.search(query_vec, candidate_k)
        
        # candidates list already initialized in Phase 0
        max_sem_score = float(semantic_scores[0][0]) if len(semantic_scores[0]) > 0 else 1.0
        
        # Check for exact match first (any row whose name matches query's raw OR NFKC-legal form)
        is_exact_match = False
        if hasattr(self, '_company_names_lower_set') and query_lower in self._company_names_lower_set:
            is_exact_match = True
        elif self._legal_nfc_to_indices and q_legal in self._legal_nfc_to_indices:
            is_exact_match = True
        
        # --- PHASE 2: RE-RANKING (Weighted Scoring with Location) ---
        for j, i in enumerate(semantic_indices[0]):
            idx = int(i)
            
            # Skip if already found in Phase 0 (Acronyms)
            if idx in found_indices:
                continue
                
            company_name = self.original_company_names[idx]
            original_semantic_score = float(semantic_scores[0][j])
            
            # Normalize semantic score
            sem_score_norm = original_semantic_score / max_sem_score if max_sem_score > 0 else 0
            
            # Get query index if query is in database (for cache lookup)
            query_idx = None
            if hasattr(self, '_company_names_lower_to_index') and query_lower in self._company_names_lower_to_index:
                # Use the first index for common scores (string/acronym) as they won't vary by location
                val = self._company_names_lower_to_index[query_lower]
                query_idx = val[0] if isinstance(val, list) else val
            
            # Calculate string similarity (using cache if available)
            string_score = self._get_cached_similarity(query, query_idx, idx)

            # Token-reorder guard: when the bag of *content* tokens (raw split,
            # then stop-word stripped, but suffixes RETAINED so that "Corporation"
            # is treated as a content noun when it's not at the end of the name) is
            # identical but the surface order differs, the candidate is likely a
            # different entity (e.g. "Henry Linda" vs "Linda Henry",
            # "Corporation of Hamilton" vs "Hamilton Corporation"). Penalize 15%
            # AND suppress the lexical-floor bypass below.
            _token_reorder_demoted = False
            _corp_enterprise_demote = False
            if query_lower == company_name.lower():
                string_score = 1.0
            else:
                import re as _re_tk
                _STOP = {"the", "of", "and", "a", "an", "to", "for", "in", "on", "at", "by"}
                _q_raw = [t for t in _re_tk.findall(r"[a-z0-9]+", query.lower()) if t and t not in _STOP]
                _c_raw = [t for t in _re_tk.findall(r"[a-z0-9]+", company_name.lower()) if t and t not in _STOP]
                if (
                    _q_raw and _c_raw
                    and tuple(sorted(_q_raw)) == tuple(sorted(_c_raw))
                    and tuple(_q_raw) != tuple(_c_raw)
                    and min(len(_q_raw), len(_c_raw)) <= 4
                ):
                    string_score = string_score * 0.85
                    _token_reorder_demoted = True
                # Municipal-style "Corporation of X" vs unrelated "X Enterprises" (Gate B).
                _corp_gov = _re_tk.match(r"^corporation\s+of\s+([a-z0-9]+)$", query_lower)
                if _corp_gov:
                    _stem = _corp_gov.group(1)
                    _cn = company_name.lower()
                    if "corporation" not in _cn and _re_tk.match(
                        rf"^{_re_tk.escape(_stem)}\s+enterprises\b", _cn
                    ):
                        string_score = min(string_score, 0.68)
                        _token_reorder_demoted = True
                        _corp_enterprise_demote = True

            # Check for acronym fidelity boost
            acronym_fidelity = 0.0
            query_acronym = TextPreprocessor.generate_acronym(query)
            if query_acronym and len(query) < 12:  # Query might be an acronym
                # Check if candidate is a literal expansion of this acronym
                acronym_fidelity = self._get_cached_acronym_fidelity(query, query_idx, idx)
            
            # --- CONCEPT PROBING (The "Common Sense" Filter) ---
            target_vec = self.vector_store.embeddings[idx].reshape(1, -1)
            target_sig = self._get_concept_signature(target_vec)
            concept_alignment = self._calculate_signature_correlation(query_sig, target_sig)

            # Base score: 50% string, 25% semantic, 25% concept alignment
            # This balances literal characters, generalized meaning, and specific concept "flavor"
            name_score = (string_score * 0.5) + (sem_score_norm * 0.25) + (concept_alignment * 0.25)
            
            # BOOST for acronym fidelity
            if acronym_fidelity > 0.8 and len(query_acronym) > 2:
                name_score = min(1.0, name_score + (acronym_fidelity * 0.15))
            
            # TOKEN COVERAGE CHECK
            q_tokens = set(TextPreprocessor.clean_company_name(query).split())
            t_tokens = set(TextPreprocessor.clean_company_name(company_name).split())
            is_full_overlap = q_tokens.issubset(t_tokens) if q_tokens else False

            # TIERED LEXICAL BOOSTS (Ensures literal matches consistently outrank semantic noise).
            # Suppressed when token-reorder guard fired: a reordered short name is NOT
            # a "literal match" even if cleaned tokens overlap fully.
            lexical_boost = 0.0
            if not _token_reorder_demoted:
                if string_score >= 0.92 or is_full_overlap:
                    # Near-perfect lexical match OR 100% token coverage
                    new_ns = CompanyMatcher._apply_lexical_floor(name_score, 0.95)
                    if new_ns > name_score:
                        lexical_boost = new_ns - name_score
                        name_score = new_ns
                elif string_score >= 0.80:
                    # Strong lexical match - high priority
                    new_ns = CompanyMatcher._apply_lexical_floor(name_score, 0.90)
                    if new_ns > name_score:
                        lexical_boost = new_ns - name_score
                        name_score = new_ns

            if _corp_enterprise_demote:
                name_score = min(name_score, 0.78)
            
            # BOOST for high token coverage (all query words found in target)
            # (Merged into tiered logic above)
            
            # --- LOCATION SCORING ---
            location_score = 0.0
            target_city = ""
            target_state = ""
            record_count = 0
            
            if self.has_location_data and idx < len(self.company_locations):
                loc = self.company_locations[idx]
                target_city = _to_loc_str(loc.get("city", ""))
                target_state = _to_loc_str(loc.get("state", ""))
                target_city, target_state = CompanyMatcher._sanitize_candidate_loc(
                    target_city, target_state
                )
                
                if idx < len(self.company_counts):
                    record_count = self.company_counts[idx]
                
                # Always calculate location score when location is provided
                if use_location:
                    location_score = TextPreprocessor.calculate_location_score(
                        city, state, target_city, target_state
                    )
            
            # DEBUG
            # if j < 3:
            #      print(f"DEBUG: {company_name} | Raw={original_semantic_score} | Max={max_sem_score} | Norm={sem_score_norm}")
            
            # --- FINAL SCORE CALCULATION ---
            # Same legal name (incl. Unicode/spacing) must use exact+location path so wrong office cannot beat
            # right-geo "hybrid" on a near-identical string (multi-site CO-OP case).
            is_this_exact = (query_lower == company_name.lower()) or (
                q_legal == _legal_name_match_key(company_name)
            )
            
            elw = CompanyMatcher.EXACT_MATCH_LOCATION_WEIGHT
            # --- FREQUENCY BOOST (ordering only; excluded from user score when exact+full geo) ---
            # Unified multiplier (0.03) across exact and hybrid paths so the same legal
            # company gets the same freq tail regardless of which retrieval path it took.
            # Guards:
            # - Skip entirely when exact + full geo (location_score already decides).
            # - Skip when query has location but candidate has no location at all
            #   (prevents popular far-away rows from beating a same-state candidate).
            # - Cap at +0.015 when location_score < 0.4 (no state agreement) so freq
            #   cannot overcome a state match in the 0.88 cliff band.
            freq_boost_val = 0.0
            target_has_loc = bool((target_city or "").strip() or (target_state or "").strip())
            if use_location and city and state and is_this_exact:
                pass  # skip frequency — let location_score + tie-sort decide
            elif use_location and not target_has_loc:
                pass  # candidate is location-less while query has location: don't boost
            elif not use_location:
                pass  # query omitted geo: popularity must not reorder a company's offices.
                # Suppressing freq lets same-name rows tie on name_score so the bare/national
                # row wins the tie-break (SME rule: prefer the geo-less row when no geo asked).
            elif record_count > 1 and self.max_company_count > 0:
                import math
                freq_score = math.log1p(record_count) / math.log1p(self.max_company_count)
                freq_boost_val = (freq_score * 0.03 * name_score)
                if use_location and location_score < 0.4:
                    freq_boost_val = min(freq_boost_val, 0.015)

            q_city_norm = TextPreprocessor.normalize_city(city) if city else ""
            q_state_norm = TextPreprocessor.normalize_state(state) if state else ""
            t_city_norm = TextPreprocessor.normalize_city(target_city) if target_city else ""
            t_state_norm = TextPreprocessor.normalize_state(target_state) if target_state else ""
            query_has_location = bool((city or "").strip() or (state or "").strip())
            candidate_has_location = bool((target_city or "").strip() or (target_state or "").strip())
            exact_geo_full_match = (
                bool(q_city_norm and q_state_norm and t_city_norm and t_state_norm) and
                q_city_norm == t_city_norm and
                q_state_norm == t_state_norm
            )
            query_geo_complete = bool((city or "").strip() and (state or "").strip())
            _, final_score = CompanyMatcher._rank_and_user_facing_score(
                name_score,
                location_score,
                use_location,
                is_this_exact,
                freq_boost_val,
                elw,
                exact_geo_full_match=exact_geo_full_match,
                query_geo_complete=query_geo_complete,
            )
            
            # Get database ID if available
            record_id = None
            if self.company_ids and idx < len(self.company_ids):
                record_id = self.company_ids[idx]
            
            # Calculate explicit location boost value for reporting
            loc_boost_val = 0.0
            if use_location and location_score > 0:
                 if is_this_exact:
                     loc_boost_val = location_score * elw
                 else:
                     # For hybrid, it's weighted, not additive, but we can approximate the "boost" 
                     # relative to name score for reporting, OR just report the raw boost component if it was additive.
                     # However, the RationaleService expects an additive boost for display.
                     # In the hybrid formula: final = (name * 0.8) + (loc * 0.2)
                     # The "boost" is effectively how much location pulled it up (or down).
                     # But for simplicity and consistency with the "Bonus" concept in RationaleService,
                     # we'll report the weighted contribution of location.
                     loc_boost_val = location_score * 0.2
            
            candidates.append({
                "name": company_name,
                "id": record_id,
                "score": final_score,
                "name_score": name_score,
                "lexical_boost": lexical_boost,
                "acronym_fidelity": acronym_fidelity,
                "semantic_score": original_semantic_score,
                "normalized_semantic_score": sem_score_norm,
                "string_score": string_score,
                "concept_alignment": concept_alignment,
                "concept_signature": target_sig.tolist() if target_sig is not None else None,
                "location_score": location_score,
                "location_boost": loc_boost_val,
                "popularity_boost": freq_boost_val,
                "city": target_city,
                "state": target_state,
                "count": record_count,
                "index": idx,
                "match_type": "exact" if query_lower == company_name.lower() else "hybrid"
            })
        
        # --- PHASE 3: EXACT MATCH OVERRIDE (OPTIMIZED) ---
        # Find ALL companies with exact name match (there may be multiple in different locations)
        if is_exact_match and hasattr(self, '_company_names_lower_to_index'):
            # Pre-index existing candidates for fast duplicate check
            existing_lookup = {}
            for idx, c in enumerate(candidates):
                key = (c['name'], _to_loc_str(c.get('city', '')), _to_loc_str(c.get('state', '')))
                existing_lookup[key] = idx

            # All rows that share the same legal name (incl. Unicode/spacing variants across offices)
            exact_idx_set = set()
            exact_idx_set.update(_iter_indices(self._company_names_lower_to_index.get(query_lower, [])))
            nfc = getattr(self, "_legal_nfc_to_indices", None) or {}
            for i0 in nfc.get(q_legal, []):
                exact_idx_set.add(int(i0))
            for i in sorted(exact_idx_set):
                name = self.original_company_names[i]
                
                # Get location data and ID for this exact match
                exact_city = ""
                exact_state = ""
                exact_count = 0
                exact_id = None
                if self.has_location_data and i < len(self.company_locations):
                    loc = self.company_locations[i]
                    exact_city = _to_loc_str(loc.get("city", ""))
                    exact_state = _to_loc_str(loc.get("state", ""))
                    exact_city, exact_state = CompanyMatcher._sanitize_candidate_loc(
                        exact_city, exact_state
                    )
                    if i < len(self.company_counts):
                        exact_count = self.company_counts[i]
                if self.company_ids and i < len(self.company_ids):
                    exact_id = self.company_ids[i]
                
                # Calculate location score for this exact match
                exact_loc_score = 0.0
                if use_location:
                    exact_loc_score = TextPreprocessor.calculate_location_score(city, state, exact_city, exact_state)

                # Calculate frequency boost for this exact match as a tie-breaker.
                # Unified with the hybrid path (0.03 multiplier) and gated identically:
                # zero when query has location but candidate has none.
                import math
                exact_freq_boost = 0.0
                exact_target_has_loc = bool((exact_city or "").strip() or (exact_state or "").strip())
                if use_location and city and state:
                    exact_freq_boost = 0.0
                elif use_location and not exact_target_has_loc:
                    exact_freq_boost = 0.0
                elif (not use_location) and not exact_target_has_loc:
                    exact_freq_boost = 0.0
                elif exact_count > 1 and self.max_company_count > 0:
                    exact_freq_boost = (math.log1p(exact_count) / math.log1p(self.max_company_count)) * 0.03
                    if use_location and exact_loc_score < 0.4:
                        exact_freq_boost = min(exact_freq_boost, 0.015)

                elw = CompanyMatcher.EXACT_MATCH_LOCATION_WEIGHT
                q_city_norm = TextPreprocessor.normalize_city(city) if city else ""
                q_state_norm = TextPreprocessor.normalize_state(state) if state else ""
                t_city_norm = TextPreprocessor.normalize_city(exact_city) if exact_city else ""
                t_state_norm = TextPreprocessor.normalize_state(exact_state) if exact_state else ""
                query_has_location = bool((city or "").strip() or (state or "").strip())
                candidate_has_location = bool((exact_city or "").strip() or (exact_state or "").strip())
                exact_geo_full_match = (
                    bool(q_city_norm and q_state_norm and t_city_norm and t_state_norm) and
                    q_city_norm == t_city_norm and
                    q_state_norm == t_state_norm
                )
                query_geo_complete = bool((city or "").strip() and (state or "").strip())
                _, exact_user = CompanyMatcher._rank_and_user_facing_score(
                    1.0,
                    exact_loc_score,
                    use_location,
                    True,
                    exact_freq_boost,
                    elw,
                    exact_geo_full_match=exact_geo_full_match,
                    query_geo_complete=query_geo_complete,
                )
                # Deduplicate fast using the pre-indexed candidates
                key = (name, _to_loc_str(exact_city), _to_loc_str(exact_state))
                if key in existing_lookup:
                    existing = candidates[existing_lookup[key]]
                    existing['score'] = exact_user
                    existing['name_score'] = 1.0
                    existing['location_score'] = exact_loc_score
                    existing['location_boost'] = exact_loc_score * elw
                    existing['popularity_boost'] = exact_freq_boost
                    existing['match_type'] = "exact"
                    continue

                # Exact match not yet in Phase 1/2 list — append (e.g. below FAISS top-K)
                candidates.append({
                    "name": name,
                    "id": exact_id,
                    "score": exact_user,
                    "name_score": 1.0,
                    "semantic_score": 1.0,
                    "normalized_semantic_score": 1.0,
                    "string_score": 1.0,
                    "concept_alignment": 1.0,
                    "location_score": exact_loc_score,
                    "location_boost": exact_loc_score * elw,
                    "popularity_boost": exact_freq_boost,
                    "city": exact_city,
                    "state": exact_state,
                    "count": exact_count,
                    "index": i,
                    "match_type": "exact"
                })
                found_indices.add(i)
        
        # Sort by unified score; tie-break so exact rows with verifiable geography
        # outrank same-name rows missing city/state (SME-defensible when query is broad).
        def _candidate_has_geo(c):
            return 1 if (
                str(c.get("city") or "").strip() or str(c.get("state") or "").strip()
            ) else 0

        if use_location:
            candidates.sort(
                key=lambda x: (
                    round(x.get("score", 0.0), 6),
                    round(x.get("location_score", 0.0), 6),
                    _candidate_has_geo(x),
                    1 if x.get("match_type") == "exact" else 0,
                    round(x.get("concept_alignment", 0.0), 6),
                ),
                reverse=True,
            )
        else:
            # Query omitted city/state: within any 4dp score band, prefer bare (no geo)
            # rows over geo-tagged rows — SMEs read geo-first as "wrong HQ".
            # Using 4dp banding (matching the assessment tie-definition) so a geo row
            # scoring 0.954712 does not outrank a bare row at 0.954700 just because
            # they differ at 6dp but round to the same 4dp value.
            candidates.sort(
                key=lambda x: (
                    round(x.get("score", 0.0), 4),
                    1 - _candidate_has_geo(x),
                    round(x.get("score", 0.0), 6),
                    1 if x.get("match_type") == "exact" else 0,
                    round(x.get("concept_alignment", 0.0), 6),
                ),
                reverse=True,
            )

        # Dedup: collapse candidates whose (NFKC-casefold name, normalized city,
        # normalized state) triple is identical, keeping the first (highest-ranked).
        # Eliminates duplicate-row noise like "Mount Laurel" vs "Mount  Laurel".
        import unicodedata as _uni
        import re as _re
        def _legal_key(s):
            return _re.sub(r"\s+", " ", _uni.normalize("NFKC", str(s or "")).casefold().strip())
        def _loc_key(s):
            # Collapse internal whitespace so "Mount  Vernon" == "Mount Vernon" (matches the
            # audit's duplicate definition; prevents same (name, city, state) dupes in top-5).
            return _re.sub(r"\s+", " ", str(s or "").casefold().strip())
        _seen_keys = set()
        _deduped = []
        for _c in candidates:
            _name_k = _legal_key(_c.get("name", ""))
            _city_k = _loc_key(TextPreprocessor.normalize_city(_c.get("city", "") or ""))
            _state_k = _loc_key(TextPreprocessor.normalize_state(_c.get("state", "") or ""))
            _key = (_name_k, _city_k, _state_k)
            if _key in _seen_keys:
                continue
            _seen_keys.add(_key)
            _deduped.append(_c)
        candidates = _deduped

        # Return top_k
        results = candidates[:top_k]
        CompanyMatcher._ensure_top5_score_spread(results)

        # Make stored score non-increasing with final rank. The multi-key sort (esp.
        # bare-first within a 4dp score band on geo-less queries) can place a row with a
        # marginally lower raw score above one with a higher raw score; clamp so the
        # displayed % never contradicts the displayed order (score-sanity stays monotonic).
        for _k in range(1, len(results)):
            _prev = float(results[_k - 1].get("score") or 0.0)
            if float(results[_k].get("score") or 0.0) > _prev:
                results[_k]["score"] = _prev

        # Store for explanation
        self._last_matches = results
        
        return results
    
    def get_company_count(self, company_name):
        """
        Get the record count for a specific company.
        
        Args:
            company_name: Company name to look up
            
        Returns:
            Integer count or 0 if not found
        """
        if not self.has_location_data or not self.company_counts:
            return 0
        
        name_lower = company_name.lower()
        if hasattr(self, '_company_names_lower_to_index'):
            val = self._company_names_lower_to_index.get(name_lower)
            if val is not None:
                idx = val[0] if isinstance(val, list) else val
                if idx < len(self.company_counts):
                    return self.company_counts[idx]
        return 0
    
    def get_company_location(self, company_name):
        """
        Get the location for a specific company.
        
        Args:
            company_name: Company name to look up
            
        Returns:
            Dict with 'city' and 'state' keys, or empty dict if not found
        """
        if not self.has_location_data or not self.company_locations:
            return {"city": "", "state": ""}
        
        name_lower = company_name.lower()
        if hasattr(self, '_company_names_lower_to_index'):
            val = self._company_names_lower_to_index.get(name_lower)
            if val is not None:
                idx = val[0] if isinstance(val, list) else val
                if idx < len(self.company_locations):
                    return self.company_locations[idx]
        return {"city": "", "state": ""}