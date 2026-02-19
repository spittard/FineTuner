"""
Cache Index Server RPC API using Pyro5.

Provides efficient RPC for multi-process access to the CacheIndexServer.
Much faster than HTTP for local IPC (~1ms vs ~10ms latency).

Usage:
    Server: python -m finetuner.core.cache_rpc --serve
    Client: 
        from finetuner.core.cache_rpc import connect
        server = connect()
        results = server.search("IBM")
"""

import os
import sys
import threading
from typing import Optional, List, Dict, Any

# Use locally cached models — no HuggingFace network calls needed
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Pyro5 for efficient RPC
try:
    import Pyro5.api
    import Pyro5.server
    HAS_PYRO = True
except ImportError:
    HAS_PYRO = False
    print("WARNING: Pyro5 not installed. Install with: pip install Pyro5")

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Server URI configuration
PYRO_HOST = "localhost"
PYRO_PORT = 9876
PYRO_NAME = "finetuner.cache_server"


if HAS_PYRO:
    @Pyro5.api.expose
    class CacheServerRPC:
        """
        RPC wrapper for CacheIndexServer.
        
        Exposes cache server methods via Pyro5 for efficient multi-process access.
        """
        
        def __init__(self):
            from finetuner.core.cache_server import CacheIndexServer
            self._server = CacheIndexServer()
            self._server.start()
            self._lock = threading.RLock()
            
        def get_status(self) -> Dict[str, Any]:
            """Get server status."""
            return self._server.get_status()
        
        def list_available_caches(self) -> List[Dict[str, Any]]:
            """List all caches available on disk."""
            return self._server.list_available_caches()
        
        def list_loaded_caches(self) -> List[str]:
            """List currently loaded cache keys."""
            return list(self._server.loaded_caches.keys())
        
        def load_cache(self, cache_key: str, model_name: str = 'paraphrase-MiniLM-L3-v2') -> bool:
            """Load a cache into memory."""
            with self._lock:
                return self._server.load_cache(cache_key, model_name)
        
        def unload_cache(self, cache_key: str) -> bool:
            """Unload cache from memory (keeps files on disk)."""
            with self._lock:
                return self._server.unload_cache(cache_key)
        
        def hot_reload(self, cache_key: str) -> bool:
            """Hot-reload a cache from disk."""
            with self._lock:
                return self._server.hot_reload(cache_key)
        
        def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
            """Get info about a loaded cache."""
            info = self._server.cache_info.get(cache_key)
            if info:
                return {
                    'cache_key': info.cache_key,
                    'num_companies': info.num_companies,
                    'has_location_data': info.has_location_data,
                    'memory_mb': info.memory_estimate_mb,
                    'model_name': info.model_name,
                    'cache_version': info.cache_version,
                    'loaded_at': info.loaded_at.isoformat(),
                    'last_accessed': info.last_accessed.isoformat()
                }
            return None
        
        def search(self, query: str, cache_key: Optional[str] = None, 
                   top_k: int = 10, city: Optional[str] = None, 
                   state: Optional[str] = None) -> Dict[str, Any]:
            """
            Execute a search using a loaded cache.
            
            Args:
                query: Search query
                cache_key: Specific cache to use (uses first loaded if None)
                top_k: Number of results
                city: Location filter
                state: Location filter
                
            Returns:
                Dict with query, cache_key, and results list
            """
            # Get matcher
            if cache_key:
                matcher = self._server.get_matcher(cache_key)
                if not matcher:
                    return {'error': f'Cache not loaded: {cache_key}'}
            else:
                if not self._server.loaded_caches:
                    return {'error': 'No caches loaded'}
                cache_key = list(self._server.loaded_caches.keys())[0]
                matcher = self._server.loaded_caches[cache_key]
            
            # Execute search
            if city or state:
                results = matcher.match_with_location(query, city=city, state=state, top_k=top_k)
            else:
                results = matcher.match(query, top_k=top_k)
            
            return {
                'query': query,
                'cache_key': cache_key,
                'results': results
            }
        
        def batch_search(self, queries: List[str], cache_key: Optional[str] = None,
                         top_k: int = 10) -> List[Dict[str, Any]]:
            """
            Execute multiple searches efficiently.
            
            Args:
                queries: List of search queries
                cache_key: Specific cache to use
                top_k: Number of results per query
                
            Returns:
                List of search results (same format as search())
            """
            results = []
            for query in queries:
                results.append(self.search(query, cache_key, top_k))
            return results
        
        def watch_source(self, source_path: str, cache_key: str) -> bool:
            """Start watching a source file for changes."""
            return self._server.watch_source(source_path, cache_key)
        
        def shutdown(self):
            """Gracefully shutdown the server."""
            self._server.stop()


def run_server(host: str = PYRO_HOST, port: int = PYRO_PORT):
    """Run the Pyro5 RPC server."""
    if not HAS_PYRO:
        print("ERROR: Pyro5 not installed. Install with: pip install Pyro5")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Cache Index Server (Pyro5 RPC)")
    print(f"{'='*60}")
    print(f"Starting on pyro://{host}:{port}/{PYRO_NAME}")
    print(f"\nMethods available:")
    print(f"  get_status()                  - Server health")
    print(f"  list_available_caches()       - List caches on disk")
    print(f"  list_loaded_caches()          - List loaded caches")
    print(f"  load_cache(key)               - Load cache into memory")
    print(f"  unload_cache(key)             - Unload from memory")
    print(f"  search(query, ...)            - Execute search")
    print(f"  batch_search(queries, ...)    - Batch search")
    print(f"{'='*60}\n")
    
    # Create the daemon
    daemon = Pyro5.server.Daemon(host=host, port=port)
    
    # Create and register the server
    server = CacheServerRPC()
    uri = daemon.register(server, PYRO_NAME)
    
    print(f"Server URI: {uri}")
    print("Press Ctrl+C to stop...")
    
    try:
        daemon.requestLoop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        daemon.close()


def connect(host: str = PYRO_HOST, port: int = PYRO_PORT) -> Any:
    """
    Connect to the cache server.
    
    Returns:
        Proxy object with same methods as CacheServerRPC
        
    Example:
        server = connect()
        server.load_cache("e1894e93a84bbc84a9ec980508a5fec4_loc")
        results = server.search("IBM", top_k=10)
    """
    if not HAS_PYRO:
        raise ImportError("Pyro5 not installed. Install with: pip install Pyro5")
    
    uri = f"PYRO:{PYRO_NAME}@{host}:{port}"
    return Pyro5.api.Proxy(uri)


def is_server_running(host: str = PYRO_HOST, port: int = PYRO_PORT) -> bool:
    """Check if the cache server is running."""
    try:
        with connect(host, port) as server:
            server.get_status()
            return True
    except:
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Cache Index Server RPC')
    parser.add_argument('--serve', action='store_true', help='Start the server')
    parser.add_argument('--host', default=PYRO_HOST, help=f'Host (default: {PYRO_HOST})')
    parser.add_argument('--port', type=int, default=PYRO_PORT, help=f'Port (default: {PYRO_PORT})')
    parser.add_argument('--status', action='store_true', help='Check server status')
    parser.add_argument('--list', action='store_true', help='List available caches')
    parser.add_argument('--load', type=str, help='Load a cache by key')
    parser.add_argument('--search', type=str, help='Search query')
    
    args = parser.parse_args()
    
    if args.serve:
        run_server(args.host, args.port)
    elif args.status:
        try:
            server = connect(args.host, args.port)
            print(server.get_status())
        except Exception as e:
            print(f"Server not running: {e}")
    elif args.list:
        server = connect(args.host, args.port)
        for cache in server.list_available_caches():
            print(f"  {cache['cache_key']}: {cache.get('num_companies', '?')} companies")
    elif args.load:
        server = connect(args.host, args.port)
        if server.load_cache(args.load):
            print(f"Loaded: {args.load}")
        else:
            print(f"Failed to load: {args.load}")
    elif args.search:
        server = connect(args.host, args.port)
        result = server.search(args.search)
        for r in result.get('results', [])[:5]:
            print(f"  {r['name']}: {r['score']:.3f}")
    else:
        parser.print_help()
