"""
Cache Index Server HTTP API.

Provides a REST API for interacting with the CacheIndexServer.
Uses Flask for the HTTP server.

SAFETY: No endpoints delete files. Unload only removes from memory.
"""

import os
import sys
from flask import Flask, request, jsonify
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from finetuner.core.cache_server import CacheIndexServer, create_server

app = Flask(__name__)
server: Optional[CacheIndexServer] = None


def get_server() -> CacheIndexServer:
    """Get or create the cache server instance."""
    global server
    if server is None:
        server = create_server(autostart=True)
    return server


@app.route('/status', methods=['GET'])
def status():
    """Get server status and health."""
    try:
        s = get_server()
        return jsonify(s.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/caches', methods=['GET'])
def list_caches():
    """List all available caches (both loaded and on disk)."""
    try:
        s = get_server()
        return jsonify({
            'available': s.list_available_caches(),
            'loaded': list(s.loaded_caches.keys())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/caches/<cache_key>', methods=['GET'])
def get_cache(cache_key: str):
    """Get information about a specific cache."""
    try:
        s = get_server()
        
        if cache_key in s.cache_info:
            info = s.cache_info[cache_key]
            return jsonify({
                'cache_key': cache_key,
                'loaded': True,
                'num_companies': info.num_companies,
                'has_location_data': info.has_location_data,
                'memory_mb': info.memory_estimate_mb,
                'model_name': info.model_name,
                'cache_version': info.cache_version,
                'loaded_at': info.loaded_at.isoformat(),
                'last_accessed': info.last_accessed.isoformat()
            })
        else:
            # Check if exists on disk
            available = s.list_available_caches()
            for c in available:
                if c['cache_key'] == cache_key:
                    return jsonify({
                        'cache_key': cache_key,
                        'loaded': False,
                        **c
                    })
            return jsonify({'error': 'Cache not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/caches/<cache_key>/load', methods=['POST'])
def load_cache(cache_key: str):
    """Load a cache into memory."""
    try:
        s = get_server()
        data = request.get_json(force=True, silent=True) or {}
        model_name = data.get('model_name', 'paraphrase-MiniLM-L3-v2')
        
        success = s.load_cache(cache_key, model_name=model_name)
        
        if success:
            return jsonify({
                'status': 'loaded',
                'cache_key': cache_key,
                'info': {
                    'num_companies': s.cache_info[cache_key].num_companies,
                    'has_location_data': s.cache_info[cache_key].has_location_data,
                    'memory_mb': s.cache_info[cache_key].memory_estimate_mb
                }
            })
        else:
            return jsonify({'status': 'failed', 'error': 'Failed to load cache'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/caches/<cache_key>/unload', methods=['POST'])
def unload_cache(cache_key: str):
    """
    Unload a cache from memory.
    NOTE: This does NOT delete cache files from disk.
    """
    try:
        s = get_server()
        success = s.unload_cache(cache_key)
        
        if success:
            return jsonify({
                'status': 'unloaded',
                'cache_key': cache_key,
                'files_preserved': True  # Emphasize that files are NOT deleted
            })
        else:
            return jsonify({'status': 'not_loaded', 'message': 'Cache was not loaded'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/caches/<cache_key>/reload', methods=['POST'])
def reload_cache(cache_key: str):
    """Hot-reload a cache from disk."""
    try:
        s = get_server()
        success = s.hot_reload(cache_key)
        
        if success:
            return jsonify({
                'status': 'reloaded',
                'cache_key': cache_key,
                'info': {
                    'num_companies': s.cache_info[cache_key].num_companies,
                    'has_location_data': s.cache_info[cache_key].has_location_data
                }
            })
        else:
            return jsonify({'status': 'failed', 'error': 'Failed to reload cache'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/watch', methods=['POST'])
def watch_source():
    """Start watching a source file for changes."""
    try:
        s = get_server()
        data = request.get_json()
        
        if not data or 'source_path' not in data or 'cache_key' not in data:
            return jsonify({'error': 'Missing source_path or cache_key'}), 400
            
        success = s.watch_source(data['source_path'], data['cache_key'])
        
        if success:
            return jsonify({
                'status': 'watching',
                'source_path': data['source_path'],
                'cache_key': data['cache_key']
            })
        else:
            return jsonify({'status': 'failed', 'error': 'Failed to start watching'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Execute a search using a loaded cache.
    
    Request body:
    {
        "query": "IBM",
        "cache_key": "...",  // optional, uses first loaded cache if not specified
        "top_k": 10,
        "city": null,
        "state": null
    }
    """
    try:
        s = get_server()
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query'}), 400
            
        query = data['query']
        top_k = data.get('top_k', 10)
        city = data.get('city')
        state = data.get('state')
        cache_key = data.get('cache_key')
        
        # Get matcher
        if cache_key:
            matcher = s.get_matcher(cache_key)
            if not matcher:
                return jsonify({'error': f'Cache not loaded: {cache_key}'}), 404
        else:
            # Use first loaded cache
            if not s.loaded_caches:
                return jsonify({'error': 'No caches loaded'}), 400
            cache_key = list(s.loaded_caches.keys())[0]
            matcher = s.loaded_caches[cache_key]
        
        # Execute search
        if city or state:
            results = matcher.match_with_location(query, city=city, state=state, top_k=top_k)
        else:
            results = matcher.match(query, top_k=top_k)
        
        return jsonify({
            'query': query,
            'cache_key': cache_key,
            'results': results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/rebuild', methods=['GET'])
def list_rebuilds():
    """List pending and completed rebuild tasks."""
    try:
        s = get_server()
        tasks = []
        for task in s.rebuild_queue:
            tasks.append({
                'cache_key': task.cache_key,
                'source_file': task.source_file,
                'triggered_at': task.triggered_at.isoformat(),
                'reason': task.reason,
                'status': task.status,
                'error': task.error
            })
        return jsonify({'tasks': tasks})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_api(host: str = '0.0.0.0', port: int = 8765, debug: bool = False):
    """Run the API server."""
    print(f"\n{'='*60}")
    print("Cache Index Server API")
    print(f"{'='*60}")
    print(f"Starting on http://{host}:{port}")
    print(f"\nEndpoints:")
    print(f"  GET  /status              - Server health")
    print(f"  GET  /caches              - List available caches")
    print(f"  GET  /caches/<key>        - Get cache info")
    print(f"  POST /caches/<key>/load   - Load cache into memory")
    print(f"  POST /caches/<key>/unload - Unload from memory (keeps files)")
    print(f"  POST /caches/<key>/reload - Hot-reload cache")
    print(f"  POST /search              - Execute search")
    print(f"  POST /watch               - Watch source file")
    print(f"  GET  /rebuild             - List rebuild tasks")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_api()
