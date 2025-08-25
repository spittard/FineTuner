from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import hashlib

class CompanyMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load pretrained model
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.original_company_names = []  # Store original names
        self.company_names = []  # Store preprocessed names for matching
        self.embeddings = None
        self.model_name = model_name
        
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
            
            print(f"Cache loaded successfully: {cache_key}")
            print(f"Loaded {len(self.original_company_names)} companies from cache")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return False
    
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
                info += f"  ✓ {key}: Complete cache\n"
            else:
                info += f"  ⚠ {key}: Incomplete cache ({len(files)}/4 files)\n"
        
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
        print(f"Building new index for {len(company_names)} companies...")
        
        # Store original names
        self.original_company_names = company_names
        # Store preprocessed names for matching
        self.company_names = self.preprocess(company_names)
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.model.encode(self.company_names, convert_to_numpy=True, normalize_embeddings=True)
        
        # Build FAISS index
        print("Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity via normalized dot product
        self.index.add(self.embeddings)
        
        # Save to cache for future use
        print("Saving to cache...")
        self.save_to_cache(cache_key, self.embeddings, self.index, self.company_names, self.original_company_names)
        
        print("Index built successfully!")
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
        
        print(f"Adding {len(truly_new)} new companies to existing index...")
        
        # Preprocess new names
        new_preprocessed = self.preprocess(truly_new)
        
        # Generate embeddings for new companies
        new_embeddings = self.model.encode(new_preprocessed, convert_to_numpy=True, normalize_embeddings=True)
        
        # Add to existing arrays
        self.original_company_names.extend(truly_new)
        self.company_names.extend(new_preprocessed)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update FAISS index
        self.index.add(new_embeddings)
        
        print(f"Successfully added {len(truly_new)} companies. Total: {len(self.original_company_names)}")
        
        # Update cache with new data
        cache_key = self.get_cache_key(self.original_company_names)
        self.save_to_cache(cache_key, self.embeddings, self.index, self.company_names, self.original_company_names)
        
        return True

    def match(self, query, top_k=10):
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, top_k)
        results = [
            {"name": self.original_company_names[i], "score": float(scores[0][j])}
            for j, i in enumerate(indices[0])
        ]
        return results

    def explain_match(self, query, match_name):
        # Optional: show token overlap or keyword match
        query_tokens = set(query.lower().split())
        match_tokens = set(match_name.lower().split())
        overlap = query_tokens.intersection(match_tokens)
        return {
            "query_tokens": query_tokens,
            "match_tokens": match_tokens,
            "overlap": overlap,
            "overlap_score": len(overlap) / max(len(query_tokens), 1)
        }