from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import hashlib
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
    # Cache version - increment this when logic changes to invalidate old caches
    CACHE_VERSION = "v4.0_location_baked"
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
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
            print(f"[OK] ({time.time() - names_start:.1f}s)")
            
            # Verify metadata
            print("      Verifying cache metadata...", end=" ", flush=True)
            meta_start = time.time()
            with open(paths['metadata'], 'rb') as f:
                metadata = pickle.load(f)
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



    def _create_fast_lookup_sets(self):
        """Create fast lookup sets for exact matching (called after building index)"""
        print("Creating fast lookup sets for exact matching...")
        total = len(self.original_company_names)
        
        # Create lowercase sets for O(1) exact match lookup
        print("   Step 1/3: Creating lowercase lookup set...")
        self._company_names_lower_set = set()
        for name in tqdm(self.original_company_names, desc="   Lowercase set", total=total, unit="names", ncols=80, disable=not HAS_TQDM):
            self._company_names_lower_set.add(name.lower())
        
        # Create reverse lookup dictionary: lowercase_name -> index (for O(1) index lookup)
        print("   Step 2/3: Creating reverse lookup dictionary...")
        self._company_names_lower_to_index = {}
        for i, name in enumerate(tqdm(self.original_company_names, desc="   Reverse lookup", total=total, unit="names", ncols=80, disable=not HAS_TQDM)):
            name_lower = name.lower()
            # Store first occurrence (in case of duplicates, which shouldn't happen)
            if name_lower not in self._company_names_lower_to_index:
                self._company_names_lower_to_index[name_lower] = i
        
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
    
    def clear_cache(self, cache_key=None):
        """Clear specific cache or all cache"""
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
        
        # Process all queries with full semantic search and re-ranking
        candidate_k = min(50, len(self.original_company_names))
        
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
                
                # Weighted Combination
                final_score = (string_score * 0.7) + (sem_score_norm * 0.3)
                
                # BOOST for acronym fidelity (literal expansions get significant boost)
                # FIX: Never boost 2-letter acronyms (too much noise from states/suffixes)
                if acronym_fidelity > 0.8 and len(query_acronym) > 2:  # Increased threshold and length check
                    # Add up to +0.15 boost for perfect acronym expansions
                    final_score = min(1.0, final_score + (acronym_fidelity * 0.15))
                
                candidates.append({
                    "name": company_name,
                    "score": final_score,
                    "semantic_score": original_semantic_score,
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
            explanation["location_score"] = match_details.get("location_score", 0.0)
            explanation["final_score"] = match_details.get("score", 0.0)
        
        return explanation

    # ========================================================================
    # LOCATION-AWARE MATCHING METHODS
    # ========================================================================
    
    def build_index_with_location(self, filepath=None, data=None):
        """
        Build index from data that includes location information and IDs.
        
        Args:
            filepath: Path to JSON file with format:
                [{"ID": 123, "Company Name": "...", "City": "...", "State": "...", "Count": N}, ...]
            data: List of dicts with same format (alternative to filepath)
            
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
        
        # --- LOCATION BAKING ---
        print("Baking location into company names for embeddings...")
        baking_text = []
        for i, name in enumerate(company_names):
            loc = locations[i]
            city = loc.get("city", "")
            state = loc.get("state", "")
            if city or state:
                # Format: "Name City State"
                baking_text.append(f"{name} {city} {state}".strip())
            else:
                baking_text.append(name)
        
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
                 for idx in acronym_matches:
                     if idx not in found_indices:
                         company_name = self.original_company_names[idx]
                         # TIE-BREAKER: Use fidelity score and semantic check
                         fidelity = TextPreprocessor.calculate_acronym_fidelity(query, company_name)
                         
                         # Get semantic score for quality check
                         query_vec_ac = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                         target_vec_ac = self.vector_store.embeddings[idx].reshape(1, -1)
                         target_vec_ac = target_vec_ac / (np.linalg.norm(target_vec_ac) + 1e-10)
                         sem_score_ac = float(np.dot(query_vec_ac, target_vec_ac.T)[0][0])
                         final_score_ac = min(0.99, 0.70 + (fidelity * 0.20) + (sem_score_ac * 0.10) if sem_score_ac > 0 else 0.70 + (fidelity * 0.20))
                         
                         candidates.append({
                             "name": company_name,
                             "score": min(1.0, final_score_ac),
                             "acronym_fidelity": fidelity,
                             "semantic_score": sem_score_ac,
                             "normalized_semantic_score": sem_score_ac,
                             "string_score": 0.5, # Dummy
                             "index": idx,
                             "match_type": "acronym_expansion",
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
                idx = self._company_names_lower_to_index.get(generated_acronym.lower())
                if idx is not None:
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
        candidate_k = min(50, len(self.original_company_names))
        
        # RETRIEVAL (Semantic Search)
        # Use location-baked query for semantic search if location is provided
        if use_location:
            bake_query = f"{query} {city or ''} {state or ''}".strip()
            query_vec = self.model.encode([bake_query], convert_to_numpy=True, normalize_embeddings=True)
        else:
            query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            
        semantic_scores, semantic_indices = self.vector_store.search(query_vec, candidate_k)
        
        # candidates list already initialized in Phase 0
        max_sem_score = float(semantic_scores[0][0]) if len(semantic_scores[0]) > 0 else 1.0
        
        # Check for exact match first
        is_exact_match = False
        if hasattr(self, '_company_names_lower_set') and query_lower in self._company_names_lower_set:
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
                query_idx = self._company_names_lower_to_index[query_lower]
            
            # Calculate string similarity (using cache if available)
            string_score = self._get_cached_similarity(query, query_idx, idx)
            
            # Exact match bonus
            if query_lower == company_name.lower():
                string_score = 1.0
            
            # Check for acronym fidelity boost
            acronym_fidelity = 0.0
            query_acronym = TextPreprocessor.generate_acronym(query)
            if query_acronym and len(query) < 12:  # Query might be an acronym
                # Check if candidate is a literal expansion of this acronym
                acronym_fidelity = self._get_cached_acronym_fidelity(query, query_idx, idx)
            
            # Base score: 70% string, 30% semantic
            name_score = (string_score * 0.7) + (sem_score_norm * 0.3)
            
            # BOOST for acronym fidelity (literal expansions get significant boost)
            # FIX: Never boost 2-letter acronyms (too much noise from states/suffixes)
            if acronym_fidelity > 0.8 and len(query_acronym) > 2:  # Increased threshold and length check
                # Add up to +0.15 boost for perfect acronym expansions
                name_score = min(1.0, name_score + (acronym_fidelity * 0.15))
            
            # BOOST for high token coverage (all query words found in target)
            if string_score >= 0.80: # Relaxed threshold to capture penalized lexical matches
                # Ensure literal overlap is prioritized over generic acronyms
                name_score = max(name_score, 0.90)
            
            # --- LOCATION SCORING ---
            location_score = 0.0
            target_city = ""
            target_state = ""
            record_count = 0
            
            if self.has_location_data and idx < len(self.company_locations):
                loc = self.company_locations[idx]
                target_city = loc.get("city", "")
                target_state = loc.get("state", "")
                
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
            # Check if this specific candidate is an exact name match
            is_this_exact = (query_lower == company_name.lower())
            
            if use_location:
                if is_this_exact:
                    # For exact name matches: location is a TIE-BREAKER
                    # Small boost (5%) to differentiate between same-name companies
                    final_score = name_score + (location_score * 0.05)
                else:
                    # For non-exact matches: 80% name, 20% location
                    final_score = (name_score * 0.8) + (location_score * 0.2)
            else:
                final_score = name_score
            
            # --- FREQUENCY BOOST ---
            if record_count > 0 and self.max_company_count > 0:
                import math
                # Logarithmic scale for frequency boost
                freq_score = math.log1p(record_count) / math.log1p(self.max_company_count)
                # Add up to +0.05 boost for popular companies (scaled by name score to avoid over-boosting weak matches)
                # NOTE: We allow this to slightly exceed 1.0 for sorting purposes; UI will cap if needed
                final_score = final_score + (freq_score * 0.05 * name_score)
            
            # Get database ID if available
            record_id = None
            if self.company_ids and idx < len(self.company_ids):
                record_id = self.company_ids[idx]
            
            candidates.append({
                "name": company_name,
                "id": record_id,
                "score": final_score,
                "name_score": name_score,
                "semantic_score": original_semantic_score,
                "normalized_semantic_score": sem_score_norm,
                "string_score": string_score,
                "location_score": location_score,
                "city": target_city,
                "state": target_state,
                "count": record_count,
                "index": idx,
                "match_type": "exact" if query_lower == company_name.lower() else "hybrid"
            })
        
        # --- PHASE 3: EXACT MATCH OVERRIDE ---
        # Find ALL companies with exact name match (there may be multiple in different locations)
        if is_exact_match:
            for i, name in enumerate(self.original_company_names):
                if name.lower() != query_lower:
                    continue
                
                # Get location data and ID for this exact match
                exact_city = ""
                exact_state = ""
                exact_count = 0
                exact_id = None
                if self.has_location_data and i < len(self.company_locations):
                    loc = self.company_locations[i]
                    exact_city = loc.get("city", "")
                    exact_state = loc.get("state", "")
                    if i < len(self.company_counts):
                        exact_count = self.company_counts[i]
                if self.company_ids and i < len(self.company_ids):
                    exact_id = self.company_ids[i]
                
                # Calculate location score for this exact match
                exact_loc_score = 0.0
                if use_location:
                    exact_loc_score = TextPreprocessor.calculate_location_score(city, state, exact_city, exact_state)

                # Calculate frequency boost for this exact match as a tie-breaker
                import math
                exact_freq_boost = 0.0
                if self.max_company_count > 0:
                    exact_freq_boost = (math.log1p(exact_count) / math.log1p(self.max_company_count)) * 0.02

                # Deduplicate within this loop to avoid adding identical exact matches
                # (e.g. multiple entries for same company with same/no location)
                is_duplicate = False
                for existing in candidates:
                    if existing['name'] == name and existing.get('city') == exact_city and existing.get('state') == exact_state:
                        # Update existing with exact score and type
                        existing['score'] = 1.0 + (exact_loc_score * 0.05) + exact_freq_boost
                        existing['name_score'] = 1.0
                        existing['location_score'] = exact_loc_score
                        existing['match_type'] = "exact"
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Exact match score = 1.0 + location boost (5%) + frequency boost (2%)
                    exact_final_score = 1.0 + (exact_loc_score * 0.05) + exact_freq_boost
                    
                    candidates.append({
                        "name": name,
                        "id": exact_id,
                        "score": exact_final_score,
                        "name_score": 1.0,
                        "semantic_score": 1.0,
                        "normalized_semantic_score": 1.0,
                        "string_score": 1.0,
                        "location_score": exact_loc_score,
                        "city": exact_city,
                        "state": exact_state,
                        "count": exact_count,
                        "index": i,
                        "match_type": "exact"
                    })
                    found_indices.add(i)
        
        # Sort by final score, with secondary priority to "exact" match types
        # This ensures that if an acronym expansion and an exact match both score 100%,
        # the exact match is always Rank 1.
        candidates.sort(key=lambda x: (round(x["score"], 4), 1 if x.get("match_type") == "exact" else 0), reverse=True)
        
        # Return top_k
        results = candidates[:top_k]
        
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
            idx = self._company_names_lower_to_index.get(name_lower)
            if idx is not None and idx < len(self.company_counts):
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
            idx = self._company_names_lower_to_index.get(name_lower)
            if idx is not None and idx < len(self.company_locations):
                return self.company_locations[idx]
        return {"city": "", "state": ""}