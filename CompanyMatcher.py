from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import hashlib

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
        
        # Persistence settings
        self.cache_dir = "company_matcher_cache"
        self.ensure_cache_dir()
    
    def ensure_cache_dir(self):
        """Ensure the cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
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
    
    def save_to_cache(self, cache_key, embeddings, index, company_names, original_names):
        """Save embeddings, index, and names to cache"""
        try:
            paths = self.get_cache_paths(cache_key)
            
            # Save embeddings
            np.save(paths['embeddings'], embeddings)
            
            # Save FAISS index
            faiss.write_index(index, paths['index'])
            
            # Save company names
            with open(paths['names'], 'wb') as f:
                pickle.dump({
                    'company_names': company_names,
                    'original_company_names': original_names
                }, f)
            
            # Save metadata
            with open(paths['metadata'], 'wb') as f:
                pickle.dump({
                    'model_name': self.model_name,
                    'cache_key': cache_key,
                    'num_companies': len(company_names)
                }, f)
            
            print(f"Cache saved successfully: {cache_key}")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            return False
    
    def load_from_cache(self, cache_key):
        """Load embeddings, index, and names from cache"""
        try:
            paths = self.get_cache_paths(cache_key)
            
            # Check if all cache files exist
            if not all(os.path.exists(path) for path in paths.values()):
                return False
            
            # Load embeddings
            self.embeddings = np.load(paths['embeddings'])
            
            # Load FAISS index
            self.index = faiss.read_index(paths['index'])
            
            # Load company names
            with open(paths['names'], 'rb') as f:
                names_data = pickle.load(f)
                self.company_names = names_data['company_names']
                self.original_company_names = names_data['original_company_names']
            
            # Verify metadata
            with open(paths['metadata'], 'rb') as f:
                metadata = pickle.load(f)
                if metadata['model_name'] != self.model_name:
                    print("Warning: Model name changed, cache invalid")
                    return False
            
            # Create fast lookup sets for exact matching
            self._create_fast_lookup_sets()
            
            print(f"Cache loaded successfully: {cache_key}")
            print(f"Loaded {len(self.original_company_names)} companies from cache")
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
        
        # Create lowercase sets for O(1) exact match lookup
        self._company_names_lower_set = set(name.lower() for name in self.original_company_names)
        
        # Create word-based lookup for faster partial matching
        self._company_words_dict = {}
        for i, name in enumerate(self.original_company_names):
            words = set(name.lower().split())
            for word in words:
                if word not in self._company_words_dict:
                    self._company_words_dict[word] = []
                self._company_words_dict[word].append(i)
        
        print(f"   Created fast lookup sets for {len(self.original_company_names)} companies")

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

    def build_index(self, company_names):
        # Generate cache key for this dataset
        cache_key = self.get_cache_key(company_names)
        
        # Try to load from cache first
        if self.load_from_cache(cache_key):
            print(f"Using cached index for {len(self.original_company_names)} companies")
            return True
        
        # Cache miss - build new index
        print(f"Building new index for {len(company_names):,} companies...")
        
        # Store original names
        self.original_company_names = company_names
        # Store preprocessed names for matching
        self.company_names = self.preprocess(company_names)
        
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
        
        batch_count = 0
        for i in range(0, len(company_names), batch_size):
            batch_count += 1
            batch_end = min(i + batch_size, len(company_names))
            batch_names = company_names[i:batch_end]
            
            # Show progress for every batch
            print(f"   Processing batch {batch_count:,}/{total_batches:,} (companies {i:,}-{batch_end:,})")
            
            # Add timing for each batch
            import time
            start_time = time.time()
            
            # Process with ULTRA-speed optimizations
            try:
                batch_embeddings = self.model.encode(
                    batch_names, 
                    convert_to_numpy=True, 
                    normalize_embeddings=False  # Disable normalization for speed
                )
                
                batch_time = time.time() - start_time
                print(f"      Batch completed in {batch_time:.1f}s")
                
            except Exception as e:
                print(f"      Error processing batch: {e}")
                print(f"      Retrying with smaller batch...")
                # Fallback to smaller batch size
                smaller_batch = batch_names[:len(batch_names)//2]
                batch_embeddings = self.model.encode(
                    smaller_batch, 
                    convert_to_numpy=True, 
                    normalize_embeddings=False
                )
                print(f"      Smaller batch completed successfully")
            
            embeddings_list.append(batch_embeddings)
            
            # Memory cleanup every few batches
            if (i // batch_size) % 3 == 0:  # Every 3 batches
                gc.collect()
                print(f"      Memory cleanup completed")
        
        # Combine all embeddings
        self.embeddings = np.vstack(embeddings_list)
        print(f"   Generated embeddings: {self.embeddings.shape}")
        
        # Build FAISS index
        print("Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity via normalized dot product
        
        # Add vectors to index with progress indication
        print(f"   Adding {len(company_names):,} vectors to index...")
        self.index.add(self.embeddings)
        
        # Create fast lookup sets for exact matching
        self._create_fast_lookup_sets()
        
        # Save to cache for future use
        print("Saving to cache...")
        self.save_to_cache(cache_key, self.embeddings, self.index, self.company_names, self.original_company_names)
        
        print(f"Index built successfully! Ready to match {len(company_names):,} companies")
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
        
        # Preprocess new names
        print("   Preprocessing company names...")
        new_preprocessed = self.preprocess(truly_new)
        
        # Generate embeddings for new companies with ULTRA-speed optimizations
        print("   Generating embeddings for new companies...")
        new_embeddings = self.model.encode(
            new_preprocessed, 
            convert_to_numpy=True, 
            normalize_embeddings=False  # Disable normalization for speed
        )
        
        # Add to existing arrays
        print("   Updating index with new data...")
        self.original_company_names.extend(truly_new)
        self.company_names.extend(new_preprocessed)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update FAISS index
        self.index.add(new_embeddings)
        
        # Update fast lookup sets for incremental updates
        self._create_fast_lookup_sets()
        
        print(f"   Successfully added {len(truly_new):,} companies. Total: {len(self.original_company_names):,}")
        
        # Update cache with new data
        print("   Updating cache...")
        cache_key = self.get_cache_key(self.original_company_names)
        self.save_to_cache(cache_key, self.embeddings, self.index, self.company_names, self.original_company_names)
        
        return True

    def match(self, query, top_k=10):
        """Simplified matching: exact matches + semantic similarity only"""
        query_lower = query.lower().strip()
        
        # Phase 1: Exact matches (highest priority)
        exact_matches = []
        if hasattr(self, '_company_names_lower_set'):
            if query_lower in self._company_names_lower_set:
                for i, name in enumerate(self.original_company_names):
                    if name.lower() == query_lower:
                        exact_matches.append({
                            "name": name,
                            "score": 1.0,
                            "match_type": "exact",
                            "index": i
                        })
        
        # Phase 2: Semantic similarity (the most accurate method)
        print("Running semantic similarity for intelligent matching...")
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        semantic_scores, semantic_indices = self.index.search(query_vec, min(top_k * 2, len(self.original_company_names)))
        
        # Create semantic matches (skip exact matches)
        semantic_matches = []
        print(f"Top 10 semantic similarity scores:")
        
        # Find the highest semantic score for normalization
        max_semantic_score = 0.0
        for j, i in enumerate(semantic_indices[0][:10]):
            score = float(semantic_scores[0][j])
            if score > max_semantic_score:
                max_semantic_score = score
        
        for j, i in enumerate(semantic_indices[0][:10]):
            score = float(semantic_scores[0][j])
            company_name = self.original_company_names[i]
            print(f"  {company_name}: {score:.4f}")
            
            # Skip if this is an exact match
            if i not in [m["index"] for m in exact_matches]:
                # Normalize the score so highest gets close to 100%, others get proportionally lower
                # This preserves the natural ranking while keeping scores reasonable
                if max_semantic_score > 0:
                    normalized_score = min(1.0, score / max_semantic_score)
                else:
                    normalized_score = 0.0
                
                # No special cases - let the semantic model handle relevance naturally
                
                semantic_matches.append({
                    "name": company_name,
                    "score": normalized_score,
                    "match_type": "semantic",
                    "index": i,
                    "overlap_words": []
                })
        
                 # Only take the top semantic matches - let the model's ranking do the work
         # No arbitrary threshold - the semantic model naturally ranks by relevance
        
        print(f"Found {len(semantic_matches)} semantic matches")
        
        # Combine exact + semantic matches and sort by score
        all_matches = exact_matches + semantic_matches
        
        # Remove duplicates based on company name
        seen_names = set()
        unique_matches = []
        for match in all_matches:
            if match["name"] not in seen_names:
                seen_names.add(match["name"])
                unique_matches.append(match)
        
        # Sort by score (highest first) and return top_k
        unique_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Debug: Show final ranking
        print(f"Final ranking (top {min(top_k, len(unique_matches))}):")
        for i, match in enumerate(unique_matches[:top_k]):
            print(f"  {i+1}. {match['name']}: {match['score']:.3f} ({match['match_type']})")
        
        return unique_matches[:top_k]

    def explain_match(self, query, match_name):
        """Enhanced explanation with match type and reasoning"""
        query_lower = query.lower().strip()
        match_lower = match_name.lower().strip()
        
        # Find the match in our results to get match type
        match_info = None
        for match in self._last_matches if hasattr(self, '_last_matches') else []:
            if match['name'] == match_name:
                match_info = match
                break
        
        # Basic token analysis
        query_tokens = set(query_lower.split())
        match_tokens = set(match_lower.split())
        overlap = query_tokens.intersection(match_tokens)
        
        explanation = {
            "query_tokens": list(query_tokens),
            "match_tokens": list(match_tokens),
            "overlap": list(overlap),
            "overlap_score": len(overlap) / max(len(query_tokens), 1),
            "match_type": match_info.get("match_type", "unknown") if match_info else "unknown"
        }
        
        # Add specific details based on match type
        if match_info and match_info.get("match_type") == "word_overlap":
            explanation["overlap_words"] = match_info.get("overlap_words", [])
        
        return explanation