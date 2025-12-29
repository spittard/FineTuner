"""
Cache Index Server - Persistent in-memory cache management.

Provides a central server to hold all caches/indexes in memory for faster
iteration on reporting and diagnostics. Supports hot-reload and file watching.

SAFETY: This module NEVER deletes files without explicit user confirmation.
"""

import os
import time
import threading
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

# Try to import watchdog for file monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    Observer = None
    FileSystemEventHandler = object


@dataclass
class CacheInfo:
    """Information about a loaded cache."""
    cache_key: str
    source_file: Optional[str]
    loaded_at: datetime
    num_companies: int
    has_location_data: bool
    memory_estimate_mb: float
    cache_version: str
    model_name: str
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class RebuildTask:
    """A pending rebuild task."""
    cache_key: str
    source_file: str
    triggered_at: datetime
    reason: str
    status: str = "pending"  # pending, in_progress, completed, failed
    error: Optional[str] = None


class SourceFileHandler(FileSystemEventHandler if HAS_WATCHDOG else object):
    """Handles file system events for watched source files."""
    
    def __init__(self, server: 'CacheIndexServer', source_path: str, cache_key: str):
        self.server = server
        self.source_path = source_path
        self.cache_key = cache_key
        self.last_modified = time.time()
        self.debounce_seconds = 2.0  # Wait before triggering rebuild
        
    def on_modified(self, event):
        if not isinstance(event, FileModifiedEvent):
            return
        if os.path.abspath(event.src_path) != os.path.abspath(self.source_path):
            return
            
        # Debounce rapid modifications
        now = time.time()
        if now - self.last_modified < self.debounce_seconds:
            return
        self.last_modified = now
        
        print(f"[CacheServer] Source file modified: {self.source_path}")
        self.server.queue_rebuild(self.cache_key, self.source_path, "source_file_modified")


class CacheIndexServer:
    """
    Persistent in-memory cache server for company matching indexes.
    
    Key features:
    - Holds multiple caches in memory for instant access
    - Watches source files for changes and triggers rebuilds
    - Hot-reload capability without server restart
    - NEVER deletes cache files without explicit confirmation
    
    Example usage:
        server = CacheIndexServer()
        server.load_cache('e1894e93a84bbc84a9ec980508a5fec4_loc')
        matcher = server.get_matcher('e1894e93a84bbc84a9ec980508a5fec4_loc')
        results = matcher.match('IBM', top_k=10)
    """
    
    def __init__(self, cache_dir: str = 'company_matcher_cache'):
        self.cache_dir = cache_dir
        self.loaded_caches: Dict[str, Any] = {}  # cache_key -> CompanyMatcher
        self.cache_info: Dict[str, CacheInfo] = {}  # cache_key -> CacheInfo
        self.rebuild_queue: List[RebuildTask] = []
        self.rebuild_callbacks: List[Callable[[RebuildTask], None]] = []
        
        # File watching
        self.watchers: Dict[str, Any] = {}  # source_path -> (observer, handler)
        self.source_to_cache: Dict[str, str] = {}  # source_path -> cache_key
        
        # Threading
        self._lock = threading.RLock()
        self._rebuild_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Lazy import to avoid circular dependencies
        self._matcher_class = None
        
    def _get_matcher_class(self):
        """Lazy import of CompanyMatcher to avoid circular dependencies."""
        if self._matcher_class is None:
            from finetuner.core.matcher import CompanyMatcher
            self._matcher_class = CompanyMatcher
        return self._matcher_class
    
    def start(self):
        """Start the server (background rebuild thread)."""
        self._running = True
        self._rebuild_thread = threading.Thread(target=self._process_rebuild_queue, daemon=True)
        self._rebuild_thread.start()
        print(f"[CacheServer] Started. Cache directory: {self.cache_dir}")
        
    def stop(self):
        """Stop the server and release resources."""
        self._running = False
        
        # Stop all file watchers
        for source_path, (observer, handler) in list(self.watchers.items()):
            observer.stop()
            observer.join(timeout=2.0)
        self.watchers.clear()
        
        # Wait for rebuild thread
        if self._rebuild_thread and self._rebuild_thread.is_alive():
            self._rebuild_thread.join(timeout=5.0)
            
        print("[CacheServer] Stopped.")
    
    def list_available_caches(self) -> List[Dict[str, Any]]:
        """List all cache files available on disk."""
        if not os.path.exists(self.cache_dir):
            return []
            
        # Group files by cache key
        cache_files = {}
        for filename in os.listdir(self.cache_dir):
            # Extract cache key (everything before the first underscore suffix)
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                key_part = parts[0]
                suffix = parts[1]
                
                # Handle location caches (key_loc_suffix)
                if key_part.endswith('_loc'):
                    base_key = key_part
                else:
                    # Check if this is a location cache by looking for _loc_ pattern
                    if '_loc_' in filename:
                        # e.g., abc123_loc_embeddings.npy
                        base_key = filename.split('_loc_')[0] + '_loc'
                    else:
                        base_key = key_part
                        
                if base_key not in cache_files:
                    cache_files[base_key] = []
                cache_files[base_key].append(filename)
        
        result = []
        for cache_key, files in cache_files.items():
            # Try to load metadata
            metadata_file = os.path.join(self.cache_dir, f"{cache_key}_metadata.pkl")
            info = {
                'cache_key': cache_key,
                'files': files,
                'complete': len(files) >= 4,  # embeddings, index, names, metadata
                'loaded': cache_key in self.loaded_caches
            }
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    info['num_companies'] = metadata.get('num_companies', 0)
                    info['has_location_data'] = metadata.get('has_location_data', False)
                    info['model_name'] = metadata.get('model_name', 'unknown')
                    info['cache_version'] = metadata.get('cache_version', 'unknown')
                except Exception as e:
                    info['metadata_error'] = str(e)
                    
            result.append(info)
            
        return sorted(result, key=lambda x: x.get('num_companies', 0), reverse=True)
    
    def load_cache(self, cache_key: str, model_name: str = 'paraphrase-MiniLM-L3-v2') -> bool:
        """
        Load a cache into memory.
        
        Args:
            cache_key: The cache key to load
            model_name: Model name for the matcher (must match cache)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        with self._lock:
            if cache_key in self.loaded_caches:
                print(f"[CacheServer] Cache already loaded: {cache_key}")
                self.cache_info[cache_key].last_accessed = datetime.now()
                return True
                
            print(f"[CacheServer] Loading cache: {cache_key}...")
            start_time = time.time()
            
            try:
                CompanyMatcher = self._get_matcher_class()
                matcher = CompanyMatcher(model_name=model_name)
                
                if not matcher.load_from_cache(cache_key):
                    print(f"[CacheServer] Failed to load cache: {cache_key}")
                    return False
                    
                self.loaded_caches[cache_key] = matcher
                
                # Create cache info
                memory_mb = self._estimate_memory_usage(matcher)
                self.cache_info[cache_key] = CacheInfo(
                    cache_key=cache_key,
                    source_file=None,  # Will be set if watching
                    loaded_at=datetime.now(),
                    num_companies=len(matcher.original_company_names),
                    has_location_data=matcher.has_location_data,
                    memory_estimate_mb=memory_mb,
                    cache_version=matcher.CACHE_VERSION,
                    model_name=matcher.model_name
                )
                
                load_time = time.time() - start_time
                print(f"[CacheServer] ✓ Loaded {cache_key} ({len(matcher.original_company_names):,} companies, {memory_mb:.0f}MB) in {load_time:.1f}s")
                return True
                
            except Exception as e:
                print(f"[CacheServer] Error loading cache {cache_key}: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def unload_cache(self, cache_key: str) -> bool:
        """
        Unload a cache from memory.
        
        NOTE: This does NOT delete the cache files from disk.
        
        Args:
            cache_key: The cache key to unload
            
        Returns:
            True if unloaded successfully
        """
        with self._lock:
            if cache_key not in self.loaded_caches:
                print(f"[CacheServer] Cache not loaded: {cache_key}")
                return False
                
            del self.loaded_caches[cache_key]
            if cache_key in self.cache_info:
                del self.cache_info[cache_key]
                
            # Stop any watchers for this cache
            for source_path, cached_key in list(self.source_to_cache.items()):
                if cached_key == cache_key:
                    self.unwatch_source(source_path)
                    
            print(f"[CacheServer] Unloaded cache: {cache_key} (files preserved on disk)")
            return True
    
    def get_matcher(self, cache_key: str) -> Optional[Any]:
        """
        Get a matcher instance for a loaded cache.
        
        Args:
            cache_key: The cache key
            
        Returns:
            CompanyMatcher instance if loaded, None otherwise
        """
        with self._lock:
            matcher = self.loaded_caches.get(cache_key)
            if matcher and cache_key in self.cache_info:
                self.cache_info[cache_key].last_accessed = datetime.now()
            return matcher
    
    def watch_source(self, source_path: str, cache_key: str) -> bool:
        """
        Watch a source file for changes and trigger rebuilds.
        
        Args:
            source_path: Path to the source file (e.g., companies_with_location.json)
            cache_key: The cache key to associate with this source
            
        Returns:
            True if watching started successfully
        """
        if not HAS_WATCHDOG:
            print("[CacheServer] WARNING: watchdog not installed, file watching disabled")
            print("             Install with: pip install watchdog")
            return False
            
        abs_path = os.path.abspath(source_path)
        
        if not os.path.exists(abs_path):
            print(f"[CacheServer] Source file not found: {abs_path}")
            return False
            
        if abs_path in self.watchers:
            print(f"[CacheServer] Already watching: {abs_path}")
            return True
            
        # Create observer and handler
        handler = SourceFileHandler(self, abs_path, cache_key)
        observer = Observer()
        observer.schedule(handler, os.path.dirname(abs_path), recursive=False)
        observer.start()
        
        self.watchers[abs_path] = (observer, handler)
        self.source_to_cache[abs_path] = cache_key
        
        # Update cache info
        if cache_key in self.cache_info:
            self.cache_info[cache_key].source_file = abs_path
            
        print(f"[CacheServer] Watching: {abs_path} -> cache:{cache_key}")
        return True
    
    def unwatch_source(self, source_path: str) -> bool:
        """Stop watching a source file."""
        abs_path = os.path.abspath(source_path)
        
        if abs_path not in self.watchers:
            return False
            
        observer, handler = self.watchers[abs_path]
        observer.stop()
        observer.join(timeout=2.0)
        
        del self.watchers[abs_path]
        if abs_path in self.source_to_cache:
            del self.source_to_cache[abs_path]
            
        print(f"[CacheServer] Stopped watching: {abs_path}")
        return True
    
    def queue_rebuild(self, cache_key: str, source_file: str, reason: str):
        """Queue a cache rebuild task."""
        with self._lock:
            # Check if already queued
            for task in self.rebuild_queue:
                if task.cache_key == cache_key and task.status == "pending":
                    print(f"[CacheServer] Rebuild already queued for: {cache_key}")
                    return
                    
            task = RebuildTask(
                cache_key=cache_key,
                source_file=source_file,
                triggered_at=datetime.now(),
                reason=reason
            )
            self.rebuild_queue.append(task)
            print(f"[CacheServer] Rebuild queued: {cache_key} ({reason})")
    
    def hot_reload(self, cache_key: str) -> bool:
        """
        Hot-reload a cache from disk (if updated externally).
        
        Args:
            cache_key: The cache key to reload
            
        Returns:
            True if reloaded successfully
        """
        with self._lock:
            if cache_key not in self.loaded_caches:
                return self.load_cache(cache_key)
                
            old_matcher = self.loaded_caches[cache_key]
            model_name = old_matcher.model_name
            
            # Temporarily remove
            del self.loaded_caches[cache_key]
            
            # Reload
            if self.load_cache(cache_key, model_name):
                print(f"[CacheServer] Hot-reloaded: {cache_key}")
                return True
            else:
                # Restore old on failure
                self.loaded_caches[cache_key] = old_matcher
                print(f"[CacheServer] Hot-reload failed, restored previous: {cache_key}")
                return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        with self._lock:
            total_memory = sum(info.memory_estimate_mb for info in self.cache_info.values())
            return {
                'running': self._running,
                'cache_dir': self.cache_dir,
                'loaded_caches': len(self.loaded_caches),
                'total_memory_mb': total_memory,
                'watched_files': len(self.watchers),
                'pending_rebuilds': len([t for t in self.rebuild_queue if t.status == 'pending']),
                'caches': {
                    key: {
                        'num_companies': info.num_companies,
                        'has_location_data': info.has_location_data,
                        'memory_mb': info.memory_estimate_mb,
                        'loaded_at': info.loaded_at.isoformat(),
                        'last_accessed': info.last_accessed.isoformat()
                    }
                    for key, info in self.cache_info.items()
                }
            }
    
    def _estimate_memory_usage(self, matcher) -> float:
        """Estimate memory usage of a matcher in MB."""
        import sys
        
        total_bytes = 0
        
        # Embeddings (numpy array)
        if matcher.vector_store.embeddings is not None:
            total_bytes += matcher.vector_store.embeddings.nbytes
        
        # Company names (rough estimate)
        if matcher.original_company_names:
            avg_len = sum(len(n) for n in matcher.original_company_names[:1000]) / min(1000, len(matcher.original_company_names))
            total_bytes += int(avg_len * len(matcher.original_company_names) * 2)  # *2 for both lists
            
        # Similarity cache
        if matcher.similarity_cache:
            total_bytes += len(matcher.similarity_cache) * 24  # tuple key + float value
            
        # Acronym cache
        if matcher.acronym_cache:
            total_bytes += len(matcher.acronym_cache) * 100  # rough estimate
            
        return total_bytes / (1024 * 1024)
    
    def _process_rebuild_queue(self):
        """Background thread to process rebuild queue."""
        while self._running:
            task = None
            
            with self._lock:
                for t in self.rebuild_queue:
                    if t.status == 'pending':
                        t.status = 'in_progress'
                        task = t
                        break
            
            if task:
                print(f"[CacheServer] Processing rebuild: {task.cache_key}")
                try:
                    self._do_rebuild(task)
                    task.status = 'completed'
                    print(f"[CacheServer] Rebuild completed: {task.cache_key}")
                except Exception as e:
                    task.status = 'failed'
                    task.error = str(e)
                    print(f"[CacheServer] Rebuild failed: {task.cache_key} - {e}")
                    
                # Notify callbacks
                for callback in self.rebuild_callbacks:
                    try:
                        callback(task)
                    except:
                        pass
            else:
                # No pending tasks, sleep
                time.sleep(1.0)
    
    def _do_rebuild(self, task: RebuildTask):
        """Execute a rebuild task."""
        # This would rebuild the cache from source
        # For now, just hot-reload from existing cache
        # Full rebuild would need to call matcher.build_index_with_location()
        
        CompanyMatcher = self._get_matcher_class()
        matcher = CompanyMatcher()
        
        # Build new index from source file
        # NOTE: This creates a NEW cache, preserving the old one
        import json
        print(f"[CacheServer] Building new index from: {task.source_file}")
        
        with open(task.source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Use build_index_with_location for full rebuild
        if hasattr(matcher, 'build_index_with_location'):
            matcher.build_index_with_location(filepath=task.source_file, data=data)
        else:
            company_names = [item['Company Name'] for item in data if 'Company Name' in item]
            matcher.build_index(company_names, filepath=task.source_file)
        
        # Get the new cache key
        new_cache_key = matcher.get_cache_key_from_file(task.source_file)
        if new_cache_key:
            new_cache_key += '_loc'
            
        # Update loaded cache
        with self._lock:
            if task.cache_key in self.loaded_caches:
                del self.loaded_caches[task.cache_key]
            if new_cache_key:
                self.loaded_caches[new_cache_key] = matcher
                self.cache_info[new_cache_key] = CacheInfo(
                    cache_key=new_cache_key,
                    source_file=task.source_file,
                    loaded_at=datetime.now(),
                    num_companies=len(matcher.original_company_names),
                    has_location_data=matcher.has_location_data,
                    memory_estimate_mb=self._estimate_memory_usage(matcher),
                    cache_version=matcher.CACHE_VERSION,
                    model_name=matcher.model_name
                )


# Convenience function for interactive use
def create_server(cache_dir: str = 'company_matcher_cache', autostart: bool = True) -> CacheIndexServer:
    """Create and optionally start a cache server."""
    server = CacheIndexServer(cache_dir=cache_dir)
    if autostart:
        server.start()
    return server
