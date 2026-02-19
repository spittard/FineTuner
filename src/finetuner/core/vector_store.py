import faiss
import numpy as np
import os
import time


class _LazyEmbeddingProxy:
    """
    Stands in for the numpy embeddings array when loading from cache.
    
    The FAISS index already stores all vectors internally, so we avoid loading
    the .npy file (which can be 6+ GB) a second time. This proxy exposes just
    enough of the ndarray interface (len, shape) to satisfy code that inspects
    the embeddings without triggering a full disk read.
    
    If actual array access is ever needed (e.g. for index rebuilding), the full
    array will be loaded from disk lazily on first access.
    """
    def __init__(self, path: str):
        self._path = path
        self._array = None

    def _ensure_loaded(self):
        if self._array is None:
            print(f"[VectorStore] Lazy-loading embeddings from {self._path} ...")
            self._array = np.load(self._path)

    def __len__(self):
        # Read shape from the npy header without loading the full array
        return _read_npy_len(self._path)

    @property
    def nbytes(self):
        # For a memory estimate: the .npy file size minus the small header is
        # essentially equal to the raw array bytes, which is what callers need.
        try:
            return os.path.getsize(self._path)
        except Exception:
            return 0

    @property
    def shape(self):
        self._ensure_loaded()
        return self._array.shape

    def __getitem__(self, key):
        self._ensure_loaded()
        return self._array[key]

    def __bool__(self):
        return True


def _read_npy_len(path: str) -> int:
    """Read the first dimension of a .npy file from its header (no full load).
    
    .npy format layout:
      6 bytes  - magic string b'\\x93NUMPY'
      1 byte   - major version
      1 byte   - minor version
      2 bytes  - header_len (uint16 LE) for version 1.x
      4 bytes  - header_len (uint32 LE) for version 2.x
      header_len bytes - header dict as ASCII/latin1
    """
    try:
        with open(path, 'rb') as f:
            f.read(6)  # skip magic '\x93NUMPY'
            major_ver = int.from_bytes(f.read(1), 'little')
            f.read(1)  # skip minor version
            if major_ver == 1:
                header_len = int.from_bytes(f.read(2), 'little')
            else:
                header_len = int.from_bytes(f.read(4), 'little')
            header = f.read(header_len).decode('latin1')
            import ast
            d = ast.literal_eval(header.strip().rstrip(',').strip())
            return d['shape'][0]
    except Exception:
        return 0


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
        
        # Resolve proxy to real array before stacking
        existing = self.get_embeddings_array()
        self.embeddings = np.vstack([existing, new_embeddings])
        
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

        embeddings_to_save = self.get_embeddings_array()
            
        print("      Saving embeddings to cache...", end=" ", flush=True)
        start_time = time.time()
        np.save(embeddings_path, embeddings_to_save)
        print(f"[OK] ({time.time() - start_time:.1f}s)")
        
        print("      Saving FAISS index to cache...", end=" ", flush=True)
        start_time = time.time()
        faiss.write_index(self.index, index_path)
        print(f"[OK] ({time.time() - start_time:.1f}s)")
        
    def load(self, embeddings_path, index_path):
        """
        Loads the embeddings and index from disk.
        
        For large indexes, uses FAISS memory-mapping (IO_FLAG_MMAP) so the index
        file is accessed from disk on demand rather than fully loaded into RAM.
        This reduces memory usage for multi-million-entry indexes from ~13 GB to
        ~600 MB (names/metadata only).
        
        The numpy embeddings array is intentionally NOT loaded — faiss.IndexFlatIP
        already stores all vectors internally, so loading the .npy file would be
        redundant (and double the RAM usage).
        """
        if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
            return False
        
        # Skip loading the numpy embeddings — the FAISS index already contains
        # all vectors. We set a sentinel so is_ready() still works.
        self.embeddings = _LazyEmbeddingProxy(embeddings_path)
        
        print("      Loading FAISS index from cache...", end=" ", flush=True)
        start_time = time.time()
        # Use memory-mapping so the OS loads pages on demand rather than copying
        # the full file into RAM. Critical for large (6+ GB) indexes.
        try:
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            print(f"[OK, mmap] ({time.time() - start_time:.1f}s)")
        except Exception:
            # Fallback to standard load if mmap is not supported on this platform
            self.index = faiss.read_index(index_path)
            print(f"[OK] ({time.time() - start_time:.1f}s)")
        
        return True

    def is_ready(self):
        """
        Returns True if the index and embeddings are loaded and ready.
        """
        return self.index is not None and self.embeddings is not None and len(self.embeddings) > 0

    def get_embeddings_array(self):
        """
        Return the full numpy embeddings array, loading from disk if needed.
        Prefer this over direct self.embeddings access for rebuild/save operations.
        """
        if isinstance(self.embeddings, _LazyEmbeddingProxy):
            self.embeddings._ensure_loaded()
            return self.embeddings._array
        return self.embeddings
    
    @property
    def size(self):
        """Returns the number of vectors in the store."""
        if self.embeddings is None:
            return 0
        return len(self.embeddings)
