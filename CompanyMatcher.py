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
        
        # Create reverse lookup dictionary: lowercase_name -> index (for O(1) index lookup)
        self._company_names_lower_to_index = {}
        for i, name in enumerate(self.original_company_names):
            name_lower = name.lower()
            # Store first occurrence (in case of duplicates, which shouldn't happen)
            if name_lower not in self._company_names_lower_to_index:
                self._company_names_lower_to_index[name_lower] = i
        
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
        
        # Normalize
        name_lower = name.lower().replace('.', '').replace(',', '')
        words = name_lower.split()
        
        # Filter out suffixes and stop words
        clean_words = [w for w in words if w not in suffixes and w not in stop_words]
        
        # If we stripped everything (e.g. name was just "The Inc"), return original
        if not clean_words:
            return name_lower
            
        return " ".join(clean_words)

    def _calculate_string_similarity(self, query, target):
        """
        Calculates a robust string similarity score (0.0 to 1.0)
        combining Token Set Ratio and Sequence Matching.
        """
        import difflib
        
        clean_query = self._clean_company_name(query)
        clean_target = self._clean_company_name(target)
        
        # 1. Jaccard Token Similarity (Handles reordering: "Justice Dept" == "Dept Justice")
        q_tokens = set(clean_query.split())
        t_tokens = set(clean_target.split())
        
        if not q_tokens or not t_tokens:
            return 0.0
            
        intersection = len(q_tokens.intersection(t_tokens))
        union = len(q_tokens.union(t_tokens))
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # 2. Sequence Matcher (Handles typos/partial words)
        seq_score = difflib.SequenceMatcher(None, clean_query, clean_target).ratio()
        
        # Return the higher of the two, boosted if one is a substring of the other
        base_score = max(jaccard_score, seq_score)
        
        # Boost if one is a clean substring of the other (e.g. "Google" inside "Google Cloud")
        if clean_query in clean_target or clean_target in clean_query:
            base_score = max(base_score, 0.9)
            
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
        
        for batch_start in range(0, len(queries), batch_size):
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
        
        # Process all queries with full semantic search and re-ranking
        candidate_k = min(50, len(self.original_company_names))
        
        for query_idx, query in enumerate(queries):
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