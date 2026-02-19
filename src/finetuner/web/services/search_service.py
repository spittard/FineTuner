
import os
import time
from typing import Optional
from finetuner.web.services.rationale_service import RationaleService

# Import RPC client
try:
    from finetuner.core.cache_rpc import connect, is_server_running
    HAS_RPC = True
except ImportError:
    HAS_RPC = False
    print("WARNING: cache_rpc not available. Install Pyro5 with: pip install Pyro5")


class SearchService:
    """
    Service for orchestrating company searches via RPC cache server.
    
    Uses the RPC cache server for efficient multi-process access to company data.
    The RPC server must be running: python -m finetuner.core.cache_rpc --serve
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._rpc_client = None
        self._last_status_check = 0
        self._cached_status = None
        self._initialized = True
        
    def _get_rpc_client(self):
        """Get a fresh RPC client connection for each request.
        
        Note: Pyro5 proxies are not thread-safe, so we create a new one
        for each request instead of caching.
        """
        if not HAS_RPC:
            raise Exception("RPC client not available. Install Pyro5.")
        
        return connect()
    
    def _ensure_server_running(self) -> bool:
        """Check if RPC server is running."""
        if not HAS_RPC:
            return False
        return is_server_running()

    def search(self, query: str, top_k: int = 10, city: Optional[str] = None, state: Optional[str] = None):
        """Perform search via RPC cache server."""
        if not self._ensure_server_running():
            raise Exception("RPC cache server not running. Start with: python -m finetuner.core.cache_rpc --serve")
        
        client = self._get_rpc_client()
        
        # Allow up to 120s for searches — first query on a 4M+ entry index
        # involves FAISS retrieval + full re-ranking and can take 20-60s.
        if hasattr(client, '_pyroTimeout'):
            client._pyroTimeout = 120.0
            
        # Call RPC search
        print(f"Searching via RPC for: {query}")
        if city or state:
            print(f"   Location filter: city='{city}', state='{state}'")
        
        rpc_result = client.search(query, cache_key=None, top_k=top_k, city=city, state=state)
        
        if 'error' in rpc_result:
            raise Exception(rpc_result['error'])
        
        matches = rpc_result.get('results', [])
        print(f"Found {len(matches)} matches from RPC server")
        
        # Format results for display
        results = []
        for i, match in enumerate(matches, 1):
            score = match.get('score', 0)
            name = match.get('name', match.get('company_name', ''))
            
            # Build explanation dict from RPC match data
            # The RPC result has all fields in a flat structure
            explanation = {
                'string_score': match.get('string_score', 0.0),
                'semantic_score': match.get('semantic_score', 0.0),
                'normalized_semantic_score': match.get('normalized_semantic_score', 0.0),
                'acronym_fidelity': match.get('acronym_fidelity', 0.0),
                'concept_alignment': match.get('concept_alignment', 0.0),
                'lexical_boost': match.get('lexical_boost', 0.0),
                'location_boost': match.get('location_boost', 0.0),
                'popularity_boost': match.get('popularity_boost', 0.0),
                'location_score': match.get('location_score', 0.0),
                'match_type': match.get('match_type', 'hybrid'),
                'city': match.get('city', ''),
                'state': match.get('state', ''),
                'count': match.get('count', 0),
                'concept_signature': match.get('concept_signature'),
                'name_score': match.get('name_score', 0.0),
            }
            
            rationale = RationaleService.generate_match_rationale(query, name, explanation, score)
            concise_rationale = RationaleService.generate_concise_rationale(query, name, explanation, score)
            
            result_entry = {
                'rank': i,
                'company_name': name,
                'likeness_percent': round(score * 100, 1),
                'match_rationale': rationale,
                'concise_rationale': concise_rationale,
                'raw_score': score,
                'explanation_details': {
                    'query_tokens': list(explanation.get('query_tokens', [])),
                    'match_tokens': list(explanation.get('match_tokens', [])),
                    'overlap_tokens': list(explanation.get('overlap', [])),
                    'overlap_score': explanation.get('overlap_score', 0.0),
                    'string_score': explanation.get('string_score', match.get('string_score', 0.0)),
                    'semantic_score': explanation.get('semantic_score', match.get('semantic_score', 0.0)),
                    'normalized_semantic_score': explanation.get('normalized_semantic_score', match.get('normalized_semantic_score', 0.0)),
                    'acronym_fidelity': explanation.get('acronym_fidelity', match.get('acronym_fidelity', 0.0)),
                    'concept_alignment': explanation.get('concept_alignment', match.get('concept_alignment', 0.0)),
                    'lexical_boost': explanation.get('lexical_boost', match.get('lexical_boost', 0.0)),
                    'location_boost': explanation.get('location_boost', match.get('location_boost', 0.0)),
                    'popularity_boost': explanation.get('popularity_boost', match.get('popularity_boost', 0.0)),
                    'concept_signature': explanation.get('concept_signature'),
                    'match_type': match.get('match_type', explanation.get('match_type', 'hybrid'))
                }
            }
            
            # Add top-level fields for convenience
            result_entry['string_score'] = result_entry['explanation_details']['string_score']
            result_entry['semantic_score'] = result_entry['explanation_details']['semantic_score']
            result_entry['normalized_semantic_score'] = result_entry['explanation_details']['normalized_semantic_score']
            result_entry['acronym_fidelity'] = result_entry['explanation_details']['acronym_fidelity']
            result_entry['concept_alignment'] = result_entry['explanation_details']['concept_alignment']
            result_entry['lexical_boost'] = result_entry['explanation_details']['lexical_boost']
            result_entry['location_boost'] = result_entry['explanation_details']['location_boost']
            result_entry['popularity_boost'] = result_entry['explanation_details']['popularity_boost']
            result_entry['match_type'] = result_entry['explanation_details']['match_type']
            
            # Add location and count data if available
            if 'city' in match:
                result_entry['city'] = match.get('city', '')
            if 'state' in match:
                result_entry['state'] = match.get('state', '')
            if 'count' in match:
                result_entry['record_count'] = match.get('count', 0)
            if 'location_score' in match:
                result_entry['location_score'] = round(match.get('location_score', 0) * 100, 1)
            if 'name_score' in match:
                result_entry['name_score'] = round(match.get('name_score', 0) * 100, 1)
            
            results.append(result_entry)
        
        return results

    def clear_cache(self):
        """Clear cache - not applicable for RPC mode."""
        # In RPC mode, cache is managed by the server
        return True

    def get_status(self):
        """Get current service status from RPC server."""
        current_time = time.time()
        
        # Cache status for 2 seconds to avoid hammering the server
        if self._cached_status and current_time - self._last_status_check < 2:
            return self._cached_status
        
        if not HAS_RPC:
            return {
                'status': 'not_ready',
                'error': 'RPC client not available',
                'message': 'Install Pyro5 with: pip install Pyro5'
            }
        
        if not self._ensure_server_running():
            return {
                'status': 'not_ready',
                'error': 'RPC server not running',
                'message': 'Start server with: python -m finetuner.core.cache_rpc --serve'
            }
        
        try:
            client = self._get_rpc_client()
            rpc_status = client.get_status()
            loaded_caches = client.list_loaded_caches()
            
            # Get companies count from first loaded cache
            companies_loaded = 0
            has_location_data = False
            if loaded_caches:
                cache_info = client.get_cache_info(loaded_caches[0])
                if cache_info:
                    companies_loaded = cache_info.get('num_companies', 0)
                    has_location_data = cache_info.get('has_location_data', False)
            
            self._cached_status = {
                'status': 'ready' if loaded_caches else 'no_cache',
                'companies_loaded': companies_loaded,
                'last_updated': current_time,
                'has_location_data': has_location_data,
                'loaded_caches': list(loaded_caches),
                'message': f'Ready with {companies_loaded:,} companies (RPC)' if loaded_caches else 'No cache loaded'
            }
            self._last_status_check = current_time
            return self._cached_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'message': f'RPC error: {e}'
            }

    def get_cache_info(self):
        """Get cache debugging info from RPC server."""
        if not self._ensure_server_running():
            return None
        
        try:
            client = self._get_rpc_client()
            loaded_caches = client.list_loaded_caches()
            
            cache_infos = {}
            for cache_key in loaded_caches:
                cache_infos[cache_key] = client.get_cache_info(cache_key)
            
            return {
                'loaded_caches': loaded_caches,
                'cache_infos': cache_infos,
                'available_caches': client.list_available_caches()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def load_cache(self, cache_key: str) -> bool:
        """Load a specific cache via RPC."""
        if not self._ensure_server_running():
            return False
        
        try:
            client = self._get_rpc_client()
            return client.load_cache(cache_key)
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
    
    def list_available_caches(self):
        """List available caches from RPC server."""
        if not self._ensure_server_running():
            return []
        
        try:
            client = self._get_rpc_client()
            return client.list_available_caches()
        except Exception as e:
            print(f"Failed to list caches: {e}")
            return []
