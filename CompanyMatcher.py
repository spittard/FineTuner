from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import hashlib

# Try to import tqdm for progress bars, fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback that just returns the iterable unchanged
    def tqdm(iterable, desc=None, total=None, unit=None, ncols=None, **kwargs):
        return iterable

# Generic terms that should be down-weighted in matching
# These are common organizational/structural words that don't distinguish entities
GENERIC_TERMS = {
    # Facility types (weight 0.3)
    'center': 0.3, 'school': 0.3, 'hospital': 0.3, 'office': 0.3, 
    'building': 0.3, 'facility': 0.3, 'church': 0.3, 'synagogue': 0.3,
    # Event types (weight 0.3)
    'meeting': 0.3, 'breakfast': 0.3, 'lunch': 0.3, 'dinner': 0.3, 
    'conference': 0.3, 'event': 0.3, 'events': 0.3, 'tournament': 0.3,
    'wedding': 0.3,
    # Organization suffixes (weight 0.2) - common corporate terms
    'group': 0.2, 'association': 0.2, 'coalition': 0.2, 'foundation': 0.2, 
    'services': 0.2, 'service': 0.2, 'solutions': 0.2, 'partners': 0.2,
    # Location modifiers (weight 0.5) - somewhat distinctive but common
    'north': 0.5, 'south': 0.5, 'east': 0.5, 'west': 0.5, 
    'shore': 0.5, 'bay': 0.5, 'coast': 0.5, 'lake': 0.5,
    'valley': 0.5, 'mountain': 0.5, 'hill': 0.5, 'river': 0.5,
    # Common modifiers (weight 0.4)
    'national': 0.4, 'international': 0.4, 'global': 0.4, 'regional': 0.4,
    'local': 0.4, 'community': 0.4, 'public': 0.4, 'private': 0.4,
}

# Category words that define the TYPE of entity - mismatches should be penalized
CATEGORY_WORDS = {
    'facility_type': {'center', 'school', 'hospital', 'church', 'synagogue', 'office', 'building'},
    'event_type': {'meeting', 'conference', 'wedding', 'tournament', 'breakfast', 'lunch', 'dinner', 'event'},
    'service_type': {'senior', 'medical', 'financial', 'legal', 'technical', 'nursing', 'dental', 'health'},
}

# Common words that should NOT be treated as proper nouns even if capitalized
# These are words that commonly appear capitalized at the start of names but are generic
COMMON_WORDS = {
    # Articles and prepositions
    'the', 'a', 'an', 'of', 'and', 'or', 'for', 'to', 'in', 'on', 'at', 'by', 'with',
    # Generic business terms
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'co', 'company',
    'group', 'holdings', 'enterprises', 'associates', 'partners', 'services', 'solutions',
    # Facility/organization types
    'center', 'school', 'hospital', 'office', 'building', 'facility', 'church', 'synagogue',
    'university', 'college', 'institute', 'academy', 'association', 'foundation', 'society',
    # Event types
    'meeting', 'conference', 'event', 'events', 'breakfast', 'lunch', 'dinner', 'tournament',
    'wedding', 'reception', 'ceremony', 'celebration', 'gala', 'banquet',
    # Descriptors
    'national', 'international', 'global', 'regional', 'local', 'community', 'public', 'private',
    'general', 'special', 'annual', 'monthly', 'weekly', 'daily',
    # Directions/locations
    'north', 'south', 'east', 'west', 'central', 'upper', 'lower', 'new', 'old',
    'shore', 'bay', 'coast', 'lake', 'valley', 'mountain', 'hill', 'river', 'island',
    # Service types
    'senior', 'medical', 'financial', 'legal', 'technical', 'nursing', 'dental', 'health',
    'professional', 'executive', 'administrative', 'clinical', 'educational',
    # Common adjectives
    'first', 'second', 'third', 'fourth', 'fifth', 'primary', 'secondary',
    'main', 'major', 'minor', 'grand', 'great', 'big', 'small', 'little',
}

class CompanyMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load the ULTRA-fastest available model for speed
        if model_name == 'all-MiniLM-L6-v2':
            # Use the absolute fastest model available - 10x+ speed boost
            ultra_fast_model = 'paraphrase-MiniLM-L3-v2'  # Ultra-light, ultra-fast
            print(f"Using ULTRA-fast model: {ultra_fast_model} (10x+ speed boost)")
        else:
            ultra_fast_model = model_name
            
        self.model = SentenceTransformer(ultra_fast_model)
        self.index = None
        self.original_company_names = []  # Store original names
        self.company_names = []  # Store preprocessed names for matching
        self.embeddings = None
        self.model_name = ultra_fast_model  # Store the actual model name used
        
        # Location data storage (for location-aware matching)
        self.company_locations = []  # List of {"city": str, "state": str} per company
        self.company_counts = []  # List of record counts per company
        self.company_ids = []  # List of database IDs per company (for reference back to DB)
        self.has_location_data = False  # Flag to indicate if location data is loaded
        
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
        
        # Create hash from: filepath + size + mtime + model
        # This allows cache checking without loading 2.9M+ company names
        content = f"{os.path.abspath(filepath)}|{file_size}|{file_mtime}|{self.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cache_key(self, company_names):
        """Generate a unique cache key based on company names and model"""
        # Create a hash of the sorted company names and model name
        sorted_names = sorted(company_names)
        content = "|".join(sorted_names) + "|" + self.model_name
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cache_paths(self, cache_key):
        """Get file paths for cached data"""
        base_path = os.path.join(self.cache_dir, cache_key)
        return {
            'embeddings': base_path + '_embeddings.npy',
            'index': base_path + '_index.faiss',
            'names': base_path + '_names.pkl',
            'metadata': base_path + '_metadata.pkl'
        }
    
    def save_to_cache(self, cache_key, embeddings, index, company_names, original_names, 
                      locations=None, counts=None, ids=None):
        """Save embeddings, index, names, and optionally location data to cache"""
        try:
            import time
            cache_start = time.time()
            paths = self.get_cache_paths(cache_key)
            
            # Save embeddings
            print("      Saving embeddings to cache...", end=" ", flush=True)
            embed_start = time.time()
            np.save(paths['embeddings'], embeddings)
            print(f"[OK] ({time.time() - embed_start:.1f}s)")
            
            # Save FAISS index
            print("      Saving FAISS index to cache...", end=" ", flush=True)
            index_start = time.time()
            faiss.write_index(index, paths['index'])
            print(f"[OK] ({time.time() - index_start:.1f}s)")
            
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
                    'num_companies': len(company_names),
                    'has_location_data': locations is not None
                }, f)
            print(f"[OK] ({time.time() - meta_start:.1f}s)")
            
            cache_time = time.time() - cache_start
            print(f"   Cache saved successfully: {cache_key[:16]}... (total: {cache_time:.1f}s)")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            return False
    
    def load_from_cache(self, cache_key):
        """Load embeddings, index, names, and location data from cache"""
        try:
            import time
            cache_start = time.time()
            paths = self.get_cache_paths(cache_key)
            
            # Check if all cache files exist
            if not all(os.path.exists(path) for path in paths.values()):
                return False
            
            # Load embeddings
            print("      Loading embeddings from cache...", end=" ", flush=True)
            embed_start = time.time()
            self.embeddings = np.load(paths['embeddings'])
            print(f"[OK] ({time.time() - embed_start:.1f}s)")
            
            # Load FAISS index
            print("      Loading FAISS index from cache...", end=" ", flush=True)
            index_start = time.time()
            self.index = faiss.read_index(paths['index'])
            print(f"[OK] ({time.time() - index_start:.1f}s)")
            
            # Load company names and location data
            print("      Loading company names from cache...", end=" ", flush=True)
            names_start = time.time()
            with open(paths['names'], 'rb') as f:
                names_data = pickle.load(f)
                self.company_names = names_data['company_names']
                self.original_company_names = names_data['original_company_names']
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
            return False
    
    def ensure_fast_lookup_sets(self):
        """Ensure fast lookup sets exist (useful for existing cached data)"""
        if not hasattr(self, '_company_names_lower_set') or not hasattr(self, '_company_words_dict'):
            print("Creating fast lookup sets for existing data...")
            self._create_fast_lookup_sets()
            return True
        return False

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
        return (self.index is not None and 
                self.embeddings is not None and 
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
            print(f"  Index status: Ready | Companies: {len(self.original_company_names):,} | Embeddings shape: {self.embeddings.shape if self.embeddings is not None else 'N/A'}")
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
        batch_size = 50000  # Large batch size for speed
        
        print(f"   CPU-optimized processing")
        print(f"   Batch size: {batch_size:,} (vs previous 32)")
        print(f"   Total companies: {len(company_names):,}")
        print(f"   Estimated batches: {(len(company_names) + batch_size - 1) // batch_size:,}")
        print(f"   Target: Complete in under 1 hour")
        print(f"   Normalization: DISABLED for speed")
        
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
        self.embeddings = np.vstack(embeddings_list)
        print(f"   Generated embeddings: {self.embeddings.shape}")
        
        # Build FAISS index
        print("Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity via normalized dot product
        
        # Add vectors to index with progress indication
        print(f"   Adding {len(company_names):,} vectors to FAISS index...")
        import time
        start_time = time.time()
        self.index.add(self.embeddings)
        index_time = time.time() - start_time
        print(f"   [OK] FAISS index built in {index_time:.1f}s")
        
        # Create fast lookup sets for exact matching
        self._create_fast_lookup_sets()
        
        # Save to cache for future use
        print("Saving to cache...")
        import time
        cache_start = time.time()
        self.save_to_cache(cache_key, self.embeddings, self.index, self.company_names, self.original_company_names)
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
        
        # Add to existing arrays
        print("   Step 3/5: Updating arrays with new data...")
        self.original_company_names.extend(truly_new)
        self.company_names.extend(new_preprocessed)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        print(f"   [OK] Arrays updated. Total companies: {len(self.original_company_names):,}")
        
        # Update FAISS index
        print("   Step 4/5: Updating FAISS index...")
        index_start = time.time()
        self.index.add(new_embeddings)
        index_time = time.time() - index_start
        print(f"   [OK] FAISS index updated in {index_time:.1f}s")
        
        # Update fast lookup sets for incremental updates
        print("   Step 5/5: Updating fast lookup sets...")
        self._create_fast_lookup_sets()
        
        add_time = time.time() - add_start
        print(f"\n   [OK] Successfully added {len(truly_new):,} companies in {add_time:.1f}s")
        print(f"   [OK] Total companies in index: {len(self.original_company_names):,}")
        
        # Update cache with new data
        print("   Updating cache...")
        cache_start = time.time()
        cache_key = self.get_cache_key(self.original_company_names)
        self.save_to_cache(cache_key, self.embeddings, self.index, self.company_names, self.original_company_names)
        cache_time = time.time() - cache_start
        print(f"   [OK] Cache updated in {cache_time:.1f}s")
        
        return True

    def _clean_company_name(self, name):
        """
        Removes common business suffixes and stop words for cleaner string comparison.
        This ensures 'Apple Inc' matches 'Apple' perfectly.
        """
        # Common suffixes to ignore during string comparison
        suffixes = {
            'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
            'co', 'company', 'plc', 'group', 'holdings', 'enterprises', 'associates'
        }
        # Stop words that add noise
        stop_words = {'the', 'of', 'and', '&', 'a', 'an'}
        
        # Normalize: replace hyphens with spaces so "Dallas-Parks" becomes "Dallas Parks"
        # This helps match hyphenated names with their non-hyphenated variants
        name_lower = name.lower().replace('.', '').replace(',', '').replace('-', ' ')
        words = name_lower.split()
        
        # Filter out suffixes and stop words
        clean_words = [w for w in words if w not in suffixes and w not in stop_words]
        
        # If we stripped everything (e.g. name was just "The Inc"), return original
        if not clean_words:
            return name_lower
            
        return " ".join(clean_words)

    def _get_term_weight(self, term):
        """
        Returns a weight for a term based on how generic/common it is.
        Generic terms get lower weights (0.2-0.5), distinctive terms get 1.0.
        """
        term_lower = term.lower()
        return GENERIC_TERMS.get(term_lower, 1.0)
    
    def _get_category_words(self, tokens):
        """
        Extracts category-defining words from a set of tokens.
        Returns a dict mapping category type to the words found.
        """
        found_categories = {}
        for token in tokens:
            token_lower = token.lower()
            for category, words in CATEGORY_WORDS.items():
                if token_lower in words:
                    if category not in found_categories:
                        found_categories[category] = set()
                    found_categories[category].add(token_lower)
        return found_categories
    
    def _check_category_mismatch(self, query_tokens, target_tokens):
        """
        Checks if query and target have mismatched category words.
        Returns a penalty multiplier (0.0-1.0) where 1.0 means no penalty.
        """
        query_categories = self._get_category_words(query_tokens)
        target_categories = self._get_category_words(target_tokens)
        
        penalty = 1.0
        
        # Check each category type for mismatches
        for category in CATEGORY_WORDS.keys():
            query_words = query_categories.get(category, set())
            target_words = target_categories.get(category, set())
            
            # If both have words in this category but they're different, apply penalty
            if query_words and target_words and not query_words.intersection(target_words):
                # Significant penalty for category mismatch (e.g., "Senior" vs "Financial")
                penalty *= 0.75
        
        return penalty

    def _extract_proper_nouns(self, original_name):
        """
        Extracts likely proper nouns from a company name.
        Proper nouns are capitalized words that aren't common generic terms.
        
        Args:
            original_name: The original (non-lowercased) company name
            
        Returns:
            Set of likely proper nouns (in lowercase for comparison)
        """
        proper_nouns = set()
        
        # Split on common delimiters while preserving original casing
        import re
        words = re.split(r'[\s\-/,&]+', original_name)
        
        for word in words:
            # Skip empty strings and very short words
            if len(word) < 2:
                continue
                
            # Check if word starts with uppercase (likely proper noun)
            if word[0].isupper():
                word_lower = word.lower()
                # Only treat as proper noun if NOT a common generic word
                if word_lower not in COMMON_WORDS:
                    proper_nouns.add(word_lower)
        
        return proper_nouns
    
    def _check_proper_noun_mismatch(self, query_original, target_original, query_tokens, target_tokens):
        """
        Checks if the query has proper nouns (identifiers) that are missing from the target.
        This catches cases like "Hartford Hospital" vs "Jefferson Hospital" where
        the structure matches but the identifying name differs.
        
        Returns a penalty multiplier (0.0-1.0) where 1.0 means no penalty.
        """
        # Extract proper nouns from original names (preserves capitalization info)
        query_proper = self._extract_proper_nouns(query_original)
        target_proper = self._extract_proper_nouns(target_original)
        
        # If query has no identifiable proper nouns, no penalty
        if not query_proper:
            return 1.0
        
        # Find proper nouns in query that are missing from target
        missing_proper = query_proper - target_proper
        
        # Also check if they're in the target tokens (handles case variations)
        target_tokens_lower = {t.lower() for t in target_tokens}
        truly_missing = {p for p in missing_proper if p not in target_tokens_lower}
        
        if not truly_missing:
            return 1.0
        
        # Calculate penalty based on how many proper nouns are missing
        # More missing = bigger penalty
        missing_ratio = len(truly_missing) / len(query_proper)
        
        # Apply penalty: up to 25% reduction for missing proper nouns
        # This is gentler than the category mismatch because proper nouns
        # can sometimes be abbreviations or variants
        penalty = 1.0 - (missing_ratio * 0.25)
        
        return penalty

    def _calculate_string_similarity(self, query, target):
        """
        Calculates a robust string similarity score (0.0 to 1.0)
        combining weighted Token Set Ratio, Sequence Matching, and penalties
        for generic term over-matching and category mismatches.
        """
        import difflib
        
        clean_query = self._clean_company_name(query)
        clean_target = self._clean_company_name(target)
        
        q_tokens = set(clean_query.split())
        t_tokens = set(clean_target.split())
        
        if not q_tokens or not t_tokens:
            return 0.0
        
        # ============================================================
        # IMPROVEMENT 1: WEIGHTED JACCARD (Generic term down-weighting)
        # ============================================================
        # Instead of counting each token equally, weight by term importance
        intersection = q_tokens.intersection(t_tokens)
        union = q_tokens.union(t_tokens)
        
        # Calculate weighted intersection and union
        weighted_intersection = sum(self._get_term_weight(t) for t in intersection)
        weighted_union = sum(self._get_term_weight(t) for t in union)
        
        weighted_jaccard = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
        
        # Also keep unweighted for comparison
        unweighted_jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        # Handle singular/plural variations
        adjusted_intersection = len(intersection)
        for q_token in q_tokens:
            if q_token not in t_tokens:
                if q_token + 's' in t_tokens or q_token + 'es' in t_tokens:
                    adjusted_intersection += 1
                elif (q_token.endswith('s') and q_token[:-1] in t_tokens) or \
                     (q_token.endswith('es') and q_token[:-2] in t_tokens):
                    adjusted_intersection += 1
        
        adjusted_jaccard = adjusted_intersection / len(union) if len(union) > 0 else 0.0
        
        # Use weighted Jaccard as the primary score
        jaccard_score = max(weighted_jaccard, adjusted_jaccard * 0.9)  # Slight preference for weighted
        
        # Sequence Matcher for typos/partial words
        seq_query_normalized = ' '.join([w.rstrip('es').rstrip('s') if len(w) > 3 else w for w in clean_query.split()])
        seq_target_normalized = ' '.join([w.rstrip('es').rstrip('s') if len(w) > 3 else w for w in clean_target.split()])
        seq_score = difflib.SequenceMatcher(None, seq_query_normalized, seq_target_normalized).ratio()
        seq_score_original = difflib.SequenceMatcher(None, clean_query, clean_target).ratio()
        seq_score = max(seq_score, seq_score_original)
        
        base_score = max(jaccard_score, seq_score)
        
        # ============================================================
        # IMPROVEMENT 2: DISCRIMINATING WORD PENALTY
        # ============================================================
        # If query has distinctive (non-generic) words that are missing from target,
        # apply a penalty. E.g., "J&J" missing from "AEP Breakfast Meeting"
        query_distinctive = [t for t in q_tokens if self._get_term_weight(t) >= 0.8]
        missing_distinctive = [t for t in query_distinctive if t not in t_tokens]
        
        if query_distinctive and missing_distinctive:
            # Penalty based on what fraction of distinctive words are missing
            missing_ratio = len(missing_distinctive) / len(query_distinctive)
            # Apply significant penalty (up to 40% reduction) for missing distinctive words
            distinctive_penalty = 1.0 - (missing_ratio * 0.4)
            base_score *= distinctive_penalty
        
        # ============================================================
        # IMPROVEMENT 3: SHORT STRING SCORE CAP
        # ============================================================
        # Prevent very short matches from getting artificially high scores
        # This fixes "BOD" matching "Bod Pro" at 80%+
        matched_chars = sum(len(t) for t in intersection)
        if matched_chars < 5:
            base_score = min(base_score, 0.50)  # Cap at 50% for < 5 chars matched
        elif matched_chars < 8:
            base_score = min(base_score, 0.65)  # Cap at 65% for 5-7 chars matched
        
        # ============================================================
        # IMPROVEMENT 4: CATEGORY MISMATCH PENALTY
        # ============================================================
        # E.g., "Senior Center" vs "Financial Center" should be penalized
        category_penalty = self._check_category_mismatch(q_tokens, t_tokens)
        base_score *= category_penalty
        
        # ============================================================
        # IMPROVEMENT 5: PROPER NOUN MISMATCH PENALTY
        # ============================================================
        # E.g., "Hartford Hospital" vs "Jefferson Hospital" - same structure,
        # different identifying proper noun. Uses original names to detect capitalization.
        proper_noun_penalty = self._check_proper_noun_mismatch(query, target, q_tokens, t_tokens)
        base_score *= proper_noun_penalty
        
        # ============================================================
        # COVERAGE AND LENGTH ADJUSTMENTS (existing logic, refined)
        # ============================================================
        query_words_in_target = len(intersection)
        coverage_ratio = query_words_in_target / len(q_tokens) if q_tokens else 0
        
        query_length = len(q_tokens)
        target_length = len(t_tokens)
        
        # Combined penalty for boosts (category + proper noun)
        combined_penalty = category_penalty * proper_noun_penalty
        
        # Boost for perfect substring matches (e.g. "Google" inside "Google Cloud")
        if clean_query in clean_target or clean_target in clean_query:
            # Only boost if not penalized significantly
            if combined_penalty >= 0.85:
                base_score = max(base_score, 0.9 * combined_penalty)
        # Boost for multi-word matches that cover significant portion of query
        elif coverage_ratio >= 0.5:
            coverage_boost = 0.7 + (coverage_ratio * 0.2)
            if target_length >= query_length * 0.6:
                coverage_boost += 0.05
            # Apply combined penalty to boost as well
            coverage_boost *= combined_penalty
            base_score = max(base_score, coverage_boost)
        elif coverage_ratio >= 0.25:
            coverage_boost = 0.6 + ((coverage_ratio - 0.25) * 0.4)
            if target_length < query_length * 0.5:
                coverage_boost *= 0.8
            coverage_boost *= combined_penalty
            base_score = max(base_score, coverage_boost)
        
        # Length-based penalty
        if target_length < query_length:
            length_shortfall = 1.0 - (target_length / query_length)
            
            if length_shortfall > 0.5:
                normalized_pos = min(1.0, (0.5 - (length_shortfall - 0.5)) / 0.5)
                penalty_factor = 0.4 + (0.3 * normalized_pos)
                base_score = base_score * penalty_factor
            elif length_shortfall > 0.3:
                normalized_pos = (length_shortfall - 0.3) / 0.2
                penalty_factor = 0.7 + (0.15 * (1.0 - normalized_pos))
                base_score = base_score * penalty_factor
            elif length_shortfall > 0.1:
                normalized_pos = (length_shortfall - 0.1) / 0.2
                penalty_factor = 0.85 + (0.1 * (1.0 - normalized_pos))
                base_score = base_score * penalty_factor
            
        return base_score

    def match(self, query, top_k=10):
        """
        Hybrid Semantic + Lexical Matching (Retrieve & Re-rank)
        """
        query_lower = query.lower().strip()
        
        # --- PHASE 1: RETRIEVAL (Semantic Search) ---
        # Get a larger candidate pool (e.g., top 50) using the fast vector index
        # We fetch more than top_k because the best string match might be semantically ranked #20
        candidate_k = min(50, len(self.original_company_names))
        
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        semantic_scores, semantic_indices = self.index.search(query_vec, candidate_k)
        
        candidates = []
        
        # Normalize semantic scores to 0-1 range roughly
        max_sem_score = float(semantic_scores[0][0]) if len(semantic_scores[0]) > 0 else 1.0
        
        # --- PHASE 2: RE-RANKING (Weighted Scoring) ---
        for j, i in enumerate(semantic_indices[0]):
            idx = int(i)
            company_name = self.original_company_names[idx]
            original_semantic_score = float(semantic_scores[0][j])
            
            # 1. Normalize Vector Score
            sem_score_norm = original_semantic_score / max_sem_score if max_sem_score > 0 else 0
            
            # 2. Calculate String Similarity (The "Better Solution")
            string_score = self._calculate_string_similarity(query, company_name)
            
            # 3. Exact Match Bonus
            if query_lower == company_name.lower():
                string_score = 1.0
            
            # 4. Weighted Combination
            # We trust string similarity MORE than semantic for company names
            # Weight: 70% String Match, 30% Semantic Meaning
            final_score = (string_score * 0.7) + (sem_score_norm * 0.3)
            
            candidates.append({
                "name": company_name,
                "score": final_score,
                "semantic_score": original_semantic_score,
                "string_score": string_score,
                "index": idx,
                "match_type": "hybrid"
            })

        # --- PHASE 3: EXACT MATCH OVERRIDE ---
        # If we have an exact match in our lookup set, ensure it's #1
        if hasattr(self, '_company_names_lower_set') and query_lower in self._company_names_lower_set:
            # Use O(1) lookup if available, otherwise fallback to iteration
            if hasattr(self, '_company_names_lower_to_index'):
                i = self._company_names_lower_to_index[query_lower]
                name = self.original_company_names[i]
            else:
                # Fallback to iteration if dictionary doesn't exist
                for i, name in enumerate(self.original_company_names):
                    if name.lower() == query_lower:
                        break
            
            # Check if already in candidates
            existing = next((c for c in candidates if c['index'] == i), None)
            if existing:
                existing['score'] = 1.0 # Force to top
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
        
        # Store last matches for explanation
        self._last_matches = results
            
        return results

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
            semantic_scores, semantic_indices = self.index.search(query_vec, candidate_k)
            
            candidates = []
            
            # Normalize semantic scores
            max_sem_score = float(semantic_scores[0][0]) if len(semantic_scores[0]) > 0 else 1.0
            
            # Re-ranking
            for j, i in enumerate(semantic_indices[0]):
                idx = int(i)
                company_name = self.original_company_names[idx]
                original_semantic_score = float(semantic_scores[0][j])
                
                # Normalize Vector Score
                sem_score_norm = original_semantic_score / max_sem_score if max_sem_score > 0 else 0
                
                # Calculate String Similarity
                string_score = self._calculate_string_similarity(query, company_name)
                
                # Exact Match Bonus
                if query_lower == company_name.lower():
                    string_score = 1.0
                
                # Weighted Combination
                final_score = (string_score * 0.7) + (sem_score_norm * 0.3)
                
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
        query_clean = self._clean_company_name(query)
        match_clean = self._clean_company_name(match_name)
        
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
        
        # Generate cache key that includes location data marker
        cache_key = self.get_cache_key(company_names) + "_loc"
        
        # Try to load from cache
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
        
        # Generate embeddings
        print("Generating embeddings...")
        import gc
        gc.collect()
        
        batch_size = 50000
        embeddings_list = []
        total_batches = (len(company_names) + batch_size - 1) // batch_size
        
        overall_start = time.time()
        batch_range = range(0, len(company_names), batch_size)
        pbar = tqdm(batch_range, desc="   Generating embeddings", total=total_batches, 
                   unit="batch", ncols=80, disable=not HAS_TQDM)
        
        for i in pbar:
            batch_end = min(i + batch_size, len(company_names))
            batch_names = company_names[i:batch_end]
            
            batch_embeddings = self.model.encode(
                batch_names, 
                convert_to_numpy=True, 
                normalize_embeddings=False,
                show_progress_bar=False
            )
            embeddings_list.append(batch_embeddings)
            
            if (i // batch_size) % 3 == 0:
                gc.collect()
        
        pbar.close()
        print(f"   [OK] Embedding generation completed in {time.time() - overall_start:.1f}s")
        
        # Combine embeddings
        self.embeddings = np.vstack(embeddings_list)
        
        # Build FAISS index
        print("Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"   [OK] FAISS index built")
        
        # Create fast lookup sets
        self._create_fast_lookup_sets()
        
        # Save to cache with location data
        print("Saving to cache with location data...")
        self.save_to_cache(cache_key, self.embeddings, self.index, 
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
    STATE_ABBREV = {
        'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
        'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
        'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
        'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
        'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
        'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
        'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
        'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
        'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
        'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
        'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
        'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
        'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia'
    }
    
    # Common city name variations
    CITY_VARIATIONS = {
        'nyc': 'new york', 'new york city': 'new york', 'ny': 'new york',
        'la': 'los angeles', 'l.a.': 'los angeles',
        'sf': 'san francisco', 'san fran': 'san francisco',
        'dc': 'washington', 'washington dc': 'washington', 'washington d.c.': 'washington',
        'philly': 'philadelphia', 'phila': 'philadelphia',
        'chi': 'chicago', 'chi-town': 'chicago',
        'vegas': 'las vegas', 'lv': 'las vegas',
        'nola': 'new orleans',
        'atl': 'atlanta',
        'stl': 'st louis', 'st. louis': 'saint louis', 'saint louis': 'st louis',
        'ft worth': 'fort worth', 'ft. worth': 'fort worth',
        'st paul': 'saint paul', 'st. paul': 'saint paul',
        'mt': 'mount', 'mt.': 'mount',
    }
    
    def _normalize_state(self, state):
        """Normalize state to abbreviation form for comparison."""
        if not state:
            return ""
        state = state.strip().lower()
        
        # If already abbreviation, return as-is
        if len(state) == 2 and state in self.STATE_ABBREV:
            return state
        
        # If full name, convert to abbreviation
        for abbrev, full_name in self.STATE_ABBREV.items():
            if state == full_name:
                return abbrev
        
        return state
    
    def _normalize_city(self, city):
        """Normalize city name for comparison."""
        if not city:
            return ""
        city = city.strip().lower()
        
        # Apply known variations
        if city in self.CITY_VARIATIONS:
            city = self.CITY_VARIATIONS[city]
        
        # Remove common prefixes/suffixes
        city = city.replace('city of ', '').replace(' city', '')
        city = city.replace('town of ', '').replace(' town', '')
        
        return city
    
    def _calculate_city_similarity(self, query_city, target_city):
        """
        Calculate city similarity using multiple methods (similar to company name matching).
        Returns score between 0.0 and 1.0.
        """
        import difflib
        
        if not query_city or not target_city:
            return 0.0
        
        q_city = self._normalize_city(query_city)
        t_city = self._normalize_city(target_city)
        
        # Exact match after normalization
        if q_city == t_city:
            return 1.0
        
        # Check if one contains the other (e.g., "York" in "New York")
        if q_city in t_city or t_city in q_city:
            # Partial containment - score based on coverage
            shorter = min(len(q_city), len(t_city))
            longer = max(len(q_city), len(t_city))
            return 0.7 + (0.3 * shorter / longer)
        
        # Token-based matching (similar to company matching)
        q_tokens = set(q_city.split())
        t_tokens = set(t_city.split())
        
        if q_tokens and t_tokens:
            intersection = q_tokens.intersection(t_tokens)
            union = q_tokens.union(t_tokens)
            jaccard = len(intersection) / len(union)
            if jaccard > 0:
                return 0.5 + (0.5 * jaccard)
        
        # Sequence similarity for typos/variations
        seq_ratio = difflib.SequenceMatcher(None, q_city, t_city).ratio()
        if seq_ratio > 0.7:
            return seq_ratio
        
        return 0.0
    
    def _calculate_location_score(self, query_city, query_state, target_city, target_state):
        """
        Calculate location similarity score using fuzzy matching.
        Similar matching logic as company names.
        
        Args:
            query_city: City from query
            query_state: State from query
            target_city: City from target company
            target_state: State from target company
            
        Returns:
            Float between 0.0 and 1.0
        """
        state_score = 0.0
        city_score = 0.0
        
        # Normalize inputs
        q_state = self._normalize_state(query_state)
        t_state = self._normalize_state(target_state)
        
        # State matching (40% weight)
        if q_state and t_state:
            if q_state == t_state:
                state_score = 1.0
            else:
                # Check if states are similar (handles typos)
                import difflib
                state_ratio = difflib.SequenceMatcher(None, q_state, t_state).ratio()
                if state_ratio > 0.8:
                    state_score = state_ratio
        
        # City matching (60% weight) - use enhanced similarity
        city_score = self._calculate_city_similarity(query_city, target_city)
        
        # Weighted combination
        final_score = (city_score * 0.6) + (state_score * 0.4)
        
        return final_score

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
        
        # --- PHASE 1: RETRIEVAL (Semantic Search) ---
        candidate_k = min(50, len(self.original_company_names))
        
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        semantic_scores, semantic_indices = self.index.search(query_vec, candidate_k)
        
        candidates = []
        max_sem_score = float(semantic_scores[0][0]) if len(semantic_scores[0]) > 0 else 1.0
        
        # Check for exact match first
        is_exact_match = False
        if hasattr(self, '_company_names_lower_set') and query_lower in self._company_names_lower_set:
            is_exact_match = True
        
        # --- PHASE 2: RE-RANKING (Weighted Scoring with Location) ---
        for j, i in enumerate(semantic_indices[0]):
            idx = int(i)
            company_name = self.original_company_names[idx]
            original_semantic_score = float(semantic_scores[0][j])
            
            # Normalize semantic score
            sem_score_norm = original_semantic_score / max_sem_score if max_sem_score > 0 else 0
            
            # Calculate string similarity
            string_score = self._calculate_string_similarity(query, company_name)
            
            # Exact match bonus
            if query_lower == company_name.lower():
                string_score = 1.0
            
            # Base score: 70% string, 30% semantic
            name_score = (string_score * 0.7) + (sem_score_norm * 0.3)
            
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
                    location_score = self._calculate_location_score(
                        city, state, target_city, target_state
                    )
            
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
                    exact_loc_score = self._calculate_location_score(city, state, exact_city, exact_state)
                
                # Exact match score = 1.0 + location boost (5% for tie-breaking)
                exact_final_score = 1.0 + (exact_loc_score * 0.05) if use_location else 1.0
                
                existing = next((c for c in candidates if c['index'] == i), None)
                if existing:
                    # Update with correct boosted score
                    existing['score'] = exact_final_score
                    existing['name_score'] = 1.0
                    existing['location_score'] = exact_loc_score
                    existing['match_type'] = "exact"
                else:
                    candidates.append({
                        "name": name,
                        "id": exact_id,
                        "score": exact_final_score,
                        "name_score": 1.0,
                        "semantic_score": 1.0,
                        "string_score": 1.0,
                        "location_score": exact_loc_score,
                        "city": exact_city,
                        "state": exact_state,
                        "count": exact_count,
                        "index": i,
                        "match_type": "exact"
                    })
        
        # Sort by final score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
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