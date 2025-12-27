import faiss
import numpy as np
import os
import time

class VectorStore:
    """
    Manages the FAISS index and embeddings storage.
    """
    def __init__(self):
        self.index = None
        self.embeddings = None
        
    def build_index(self, embeddings):
        """
        Builds a fresh FAISS index from the provided embeddings.
        """
        print("Building FAISS index...")
        start_time = time.time()
        
        self.embeddings = embeddings
        dim = embeddings.shape[1]
        
        # optimized for inner product (cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        print(f"   [OK] FAISS index built in {time.time() - start_time:.1f}s")
        
    def add_vectors(self, new_embeddings):
        """
        Adds new vectors to the existing index and embeddings array.
        """
        if self.index is None:
            self.build_index(new_embeddings)
            return

        print("   Updating FAISS index...")
        start_time = time.time()
        
        # Update embeddings array
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Add to FAISS index
        self.index.add(new_embeddings)
        
        print(f"   [OK] FAISS index updated in {time.time() - start_time:.1f}s")
        
    def search(self, query_vec, k):
        """
        Searches the index for the k nearest neighbors.
        """
        if self.index is None:
            raise ValueError("Index not initialized")
        return self.index.search(query_vec, k)
    
    def save(self, embeddings_path, index_path):
        """
        Saves the embeddings and index to disk.
        """
        if self.embeddings is None or self.index is None:
            raise ValueError("No data to save")
            
        print("      Saving embeddings to cache...", end=" ", flush=True)
        start_time = time.time()
        np.save(embeddings_path, self.embeddings)
        print(f"[OK] ({time.time() - start_time:.1f}s)")
        
        print("      Saving FAISS index to cache...", end=" ", flush=True)
        start_time = time.time()
        faiss.write_index(self.index, index_path)
        print(f"[OK] ({time.time() - start_time:.1f}s)")
        
    def load(self, embeddings_path, index_path):
        """
        Loads the embeddings and index from disk.
        """
        if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
            return False
            
        print("      Loading embeddings from cache...", end=" ", flush=True)
        start_time = time.time()
        self.embeddings = np.load(embeddings_path)
        print(f"[OK] ({time.time() - start_time:.1f}s)")
        
        print("      Loading FAISS index from cache...", end=" ", flush=True)
        start_time = time.time()
        self.index = faiss.read_index(index_path)
        print(f"[OK] ({time.time() - start_time:.1f}s)")
        
        return True

    def is_ready(self):
        """
        Returns True if the index and embeddings are loaded and ready.
        """
        return self.index is not None and self.embeddings is not None and len(self.embeddings) > 0
    
    @property
    def size(self):
        """Returns the number of vectors in the store."""
        if self.embeddings is None:
            return 0
        return len(self.embeddings)
