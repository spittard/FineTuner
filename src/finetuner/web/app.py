from flask import Flask, render_template, request, jsonify
from finetuner.web.services.rationale_service import RationaleService
import json
import os
import time

from finetuner.web.services.search_service import SearchService

# Global service instance
search_service = SearchService()

app = Flask(__name__)

# Enable auto-reloading for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def index():
    """Main page with search form"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle company search requests with optional location filtering"""
    try:
        # Get search query
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Please enter a company name to search for.'}), 400
        
        # Get number of results (default to 10)
        top_k = int(request.form.get('top_k', 10))
        
        # Get optional location parameters
        city = request.form.get('city', '').strip() or None
        state = request.form.get('state', '').strip() or None
        
        # Perform search using service (RPC-based)
        results = search_service.search(query, top_k=top_k, city=city, state=state)
        
        # Get status for location data info
        status = search_service.get_status()
        
        response_data = {
            'success': True,
            'query': query,
            'results': results,
            'total_matches': len(results),
            'location_filter_used': bool(city or state),
            'has_location_data': status.get('has_location_data', False)
        }
        
        # Include location filter in response if used
        if city:
            response_data['filter_city'] = city
        if state:
            response_data['filter_state'] = state
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(f"Search error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Search failed: {str(e)}'
        }), 500

@app.route('/reload', methods=['POST'])
def reload_data():
    """Force reload of company data (not applicable in RPC mode)"""
    try:
        status = search_service.get_status()
        if status.get('status') == 'ready':
            return jsonify({
                'success': True,
                'message': f"Data available via RPC. {status.get('companies_loaded', 0):,} companies loaded.",
                'companies_loaded': status.get('companies_loaded', 0)
            })
        else:
            return jsonify({
                'success': False,
                'error': status.get('error', 'RPC server not ready')
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Reload failed: {str(e)}'
        }), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache (managed by RPC server in RPC mode)"""
    try:
        search_service.clear_cache()
        status = search_service.get_status()
        return jsonify({
            'success': True,
            'message': f"Cache managed by RPC server. {status.get('companies_loaded', 0):,} companies available.",
            'companies_loaded': status.get('companies_loaded', 0)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Cache clear failed: {str(e)}'
        }), 500

@app.route('/cache-info', methods=['GET'])
def get_cache_info():
    """Get cache information for debugging consistency issues"""
    try:
        cache_info = search_service.get_cache_info()
        if cache_info is None:
            return jsonify({
                'success': False,
                'error': 'No cache info available'
            }), 400
        
        return jsonify({
            'success': True,
            **cache_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Cache info failed: {str(e)}'
        }), 500


@app.route('/status')
def status():
    """Check if company data is loaded via RPC"""
    return jsonify(search_service.get_status())

if __name__ == '__main__':
    # Enable auto-reloading for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
