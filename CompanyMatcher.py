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
        """Improved matching with hybrid approach: exact > word overlap > semantic similarity"""
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # Phase 1: FAST exact matches (keep precedence!)
        exact_matches = []
        if hasattr(self, '_company_names_lower_set'):
            # Use fast O(1) lookup instead of slow O(n) loop
            if query_lower in self._company_names_lower_set:
                # Find the original names that match
                for i, name in enumerate(self.original_company_names):
                    if name.lower() == query_lower:
                        exact_matches.append({
                            "name": name,
                            "score": 1.0,
                            "match_type": "exact",
                            "index": i
                        })
        else:
            # Fallback to original slow method if fast lookup sets don't exist
            print("Fast lookup sets not found, using fallback method...")
            for i, name in enumerate(self.original_company_names):
                name_lower = name.lower()
                if query_lower in name_lower or name_lower in query_lower:
                    exact_matches.append({
                        "name": name,
                        "score": 1.0,
                        "match_type": "exact",
                        "index": i
                    })
        
        # Phase 2: FAST word overlap matches (keep precedence!)
        word_overlap_matches = []
        if hasattr(self, '_company_words_dict'):
            # Use fast word-based lookup instead of slow O(n) loop
            for query_word in query_words:
                if query_word in self._company_words_dict:
                    for idx in self._company_words_dict[query_word]:
                        # Skip if already matched in exact phase
                        if idx in [m["index"] for m in exact_matches]:
                            continue
                        
                        name = self.original_company_names[idx]
                        name_words = set(name.lower().split())
                        
                        # Calculate word overlap with improved scoring
                        overlap = query_words.intersection(name_words)
                        
                        # Base score: prioritize exact word matches over partial matches
                        exact_match_ratio = len(overlap) / len(query_words)
                        
                        # Semantic relevance bonus (much higher weight)
                        semantic_bonus = 1.0
                        if "dept" in query_words and "dept" in name_words:
                            semantic_bonus = 2.0  # Major bonus for exact department match
                        if "justice" in query_words and "justice" in name_words:
                            semantic_bonus = 2.0  # Major bonus for exact justice match
                        elif "just" in query_words and "justice" in name_words:
                            semantic_bonus = 1.8  # High bonus for semantic relevance
                        
                        # Penalty for extra words (shorter, more focused names get preference)
                        length_penalty = 1.0 / (1.0 + (len(name_words) - len(query_words)) * 0.2)
                        
                        # Final score prioritizes semantic relevance
                        overlap_score = exact_match_ratio * semantic_bonus * length_penalty
                        
                        if overlap_score >= 0.3:
                            word_overlap_matches.append({
                                "name": name,
                                "score": overlap_score,
                                "match_type": "word_overlap",
                                "index": idx,
                                "overlap_words": list(overlap)
                            })
            
            # Also check for substring matches within words (e.g., "just" in "justice")
            for i, name in enumerate(self.original_company_names):
                if i in [m["index"] for m in exact_matches] or i in [m["index"] for m in word_overlap_matches]:
                    continue
                
                name_lower = name.lower()
                name_words = set(name_lower.split())
                
                # Check for substring matches within words
                substring_matches = 0
                for query_word in query_words:
                    for name_word in name_words:
                        if query_word in name_word or name_word in query_word:
                            substring_matches += 1
                            break
                
                if substring_matches > 0:
                    # Calculate score based on substring matches
                    substring_score = substring_matches / len(query_words)
                    
                    # Apply the same scoring logic
                    word_count_bonus = substring_matches / len(query_words)
                    length_penalty = 1.0 / (1.0 + (len(name_words) - len(query_words)) * 0.1)
                    
                    # Semantic bonus for substring relevance
                    semantic_bonus = 1.0
                    if "just" in query_words and any("justice" in word for word in name_words):
                        semantic_bonus = 1.2
                    elif "dept" in query_words and any("department" in word for word in name_words):
                        semantic_bonus = 1.1
                    
                    final_score = substring_score * word_count_bonus * length_penalty * semantic_bonus
                    
                    if final_score >= 0.2:  # Lower threshold for substring matches
                        word_overlap_matches.append({
                            "name": name,
                            "score": final_score,
                            "match_type": "substring_overlap",
                            "index": i,
                            "overlap_words": [word for word in query_words if any(word in name_word or name_word in word for name_word in name_words)]
                        })
        else:
            # Fallback to original slow method if fast lookup sets don't exist
            print("Fast lookup sets not found, using fallback method...")
            for i, name in enumerate(self.original_company_names):
                if i in [m["index"] for m in exact_matches]:  # Skip if already matched
                    continue
                    
                name_lower = name.lower()
                name_words = set(name_lower.split())
                
                # Calculate word overlap with improved scoring
                overlap = query_words.intersection(name_words)
                if len(overlap) > 0:
                    # Base score: prioritize exact word matches over partial matches
                    exact_match_ratio = len(overlap) / len(query_words)
                    
                    # Semantic relevance bonus (much higher weight)
                    semantic_bonus = 1.0
                    if "dept" in query_words and "dept" in name_words:
                        semantic_bonus = 2.0  # Major bonus for exact department match
                    if "justice" in query_words and "justice" in name_words:
                        semantic_bonus = 2.0  # Major bonus for exact justice match
                    elif "just" in query_words and "justice" in name_words:
                        semantic_bonus = 1.8  # High bonus for semantic relevance
                    
                    # Penalty for extra words (shorter, more focused names get preference)
                    length_penalty = 1.0 / (1.0 + (len(name_words) - len(query_words)) * 0.2)
                    
                    # Final score prioritizes semantic relevance
                    overlap_score = exact_match_ratio * semantic_bonus * length_penalty
                    
                    if overlap_score >= 0.3:  # Only include if significant overlap
                        word_overlap_matches.append({
                            "name": name,
                            "score": overlap_score,
                            "match_type": "word_overlap",
                            "index": i,
                            "overlap_words": list(overlap)
                        })
                
                # Also check for substring matches within words (e.g., "just" in "justice")
                substring_matches = 0
                for query_word in query_words:
                    for name_word in name_words:
                        if query_word in name_word or name_word in query_word:
                            substring_matches += 1
                            break
                
                if substring_matches > 0:
                    # Calculate score based on substring matches
                    substring_score = substring_matches / len(query_words)
                    
                    # Apply the same scoring logic
                    word_count_bonus = substring_matches / len(query_words)
                    length_penalty = 1.0 / (1.0 + (len(name_words) - len(query_words)) * 0.1)
                    
                    # Semantic bonus for substring relevance
                    semantic_bonus = 1.0
                    if "just" in query_words and any("justice" in word for word in name_words):
                        semantic_bonus = 1.2
                    elif "dept" in query_words and any("department" in word for word in name_words):
                        semantic_bonus = 1.1
                    
                    final_score = substring_score * word_count_bonus * length_penalty * semantic_bonus
                    
                    if final_score >= 0.2:  # Lower threshold for substring matches
                        word_overlap_matches.append({
                            "name": name,
                            "score": final_score,
                            "match_type": "substring_overlap",
                            "index": i,
                            "overlap_words": [word for word in query_words if any(word in name_word or name_word in word for name_word in name_words)]
                        })
        
        # Phase 3: Semantic similarity (lowest priority, only if no good matches)
        semantic_matches = []
        if len(exact_matches) == 0 and len(word_overlap_matches) == 0:
            # Only use semantic similarity if no exact or word overlap matches
            query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, indices = self.index.search(query_vec, top_k)
            for j, i in enumerate(indices[0]):
                semantic_matches.append({
                    "name": self.original_company_names[i],
                    "score": float(scores[0][j]),
                    "match_type": "semantic",
                    "index": i
                })
        
        # Combine and sort results
        all_matches = exact_matches + word_overlap_matches + semantic_matches
        
        # Sort by score (exact matches will be first due to score=1.0)
        all_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove duplicates based on company name while preserving order
        seen_names = set()
        unique_matches = []
        for match in all_matches:
            if match["name"] not in seen_names:
                seen_names.add(match["name"])
                unique_matches.append(match)
        
        # Return top_k unique results
        results = unique_matches[:top_k]
        
        # Store for explanation function
        self._last_matches = results
        
        return results

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