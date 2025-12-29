"""
Cache Index Server Client.

A simple client library to connect to the Cache Index Server API.
"""

import requests
from typing import Dict, List, Any, Optional


class CacheClient:
    """
    Client for the Cache Index Server API.
    
    Example usage:
        client = CacheClient("http://localhost:8765")
        
        # Check server status
        status = client.get_status()
        
        # List available caches
        caches = client.list_caches()
        
        # Load a cache
        client.load_cache("e1894e93a84bbc84a9ec980508a5fec4_loc")
        
        # Search
        results = client.search("IBM", top_k=10)
    """
    
    def __init__(self, base_url: str = "http://localhost:8765", timeout: int = 300):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the cache server
            timeout: Request timeout in seconds (default 300 for slow operations)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request."""
        response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a POST request."""
        response = requests.post(
            f"{self.base_url}{endpoint}", 
            json=data or {},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return self._get('/status')
    
    def list_caches(self) -> Dict[str, Any]:
        """List all available caches."""
        return self._get('/caches')
    
    def get_cache(self, cache_key: str) -> Dict[str, Any]:
        """Get information about a specific cache."""
        return self._get(f'/caches/{cache_key}')
    
    def load_cache(self, cache_key: str, model_name: str = 'paraphrase-MiniLM-L3-v2') -> Dict[str, Any]:
        """Load a cache into memory."""
        return self._post(f'/caches/{cache_key}/load', {'model_name': model_name})
    
    def unload_cache(self, cache_key: str) -> Dict[str, Any]:
        """
        Unload a cache from memory.
        NOTE: This does NOT delete cache files from disk.
        """
        return self._post(f'/caches/{cache_key}/unload')
    
    def reload_cache(self, cache_key: str) -> Dict[str, Any]:
        """Hot-reload a cache from disk."""
        return self._post(f'/caches/{cache_key}/reload')
    
    def watch_source(self, source_path: str, cache_key: str) -> Dict[str, Any]:
        """Start watching a source file for changes."""
        return self._post('/watch', {'source_path': source_path, 'cache_key': cache_key})
    
    def search(
        self, 
        query: str, 
        cache_key: Optional[str] = None,
        top_k: int = 10,
        city: Optional[str] = None,
        state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a search.
        
        Args:
            query: Search query
            cache_key: Specific cache to use (optional, uses first loaded if not specified)
            top_k: Number of results to return
            city: Filter by city
            state: Filter by state
            
        Returns:
            Search results
        """
        data = {
            'query': query,
            'top_k': top_k
        }
        if cache_key:
            data['cache_key'] = cache_key
        if city:
            data['city'] = city
        if state:
            data['state'] = state
            
        return self._post('/search', data)
    
    def list_rebuilds(self) -> Dict[str, Any]:
        """List pending and completed rebuild tasks."""
        return self._get('/rebuild')
    
    def is_server_available(self) -> bool:
        """Check if the server is available."""
        try:
            self.get_status()
            return True
        except:
            return False


# Convenience function
def connect(url: str = "http://localhost:8765") -> CacheClient:
    """Create a cache client connection."""
    return CacheClient(url)
