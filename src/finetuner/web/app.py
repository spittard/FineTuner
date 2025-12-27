from flask import Flask, render_template, request, jsonify
from finetuner.core.matcher import CompanyMatcher
from finetuner.web.services.rationale_service import RationaleService
import json
import os
import time

# Try to import tqdm for progress bars
from finetuner.web.services.search_service import SearchService

# Global service instance
search_service = SearchService()

app = Flask(__name__)

# Enable auto-reloading for development
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Helper functions for enhanced rationale generation
# Helper functions for enhanced rationale generation have been moved to RationaleService



@app.route('/')
def index():
    """Main page with search form"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle company search requests with optional location filtering"""
    try:
        # Load company data if not already loaded
        search_service.load_company_data()
        
        # Get search query
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Please enter a company name to search for.'}), 400
        
        # Get number of results (default to 10)
        top_k = int(request.form.get('top_k', 10))
        
        # Get optional location parameters
        city = request.form.get('city', '').strip() or None
        state = request.form.get('state', '').strip() or None
        
        # Perform search using service
        results = search_service.search(query, top_k=top_k, city=city, state=state)
        
        response_data = {
            'success': True,
            'query': query,
            'results': results,
            'total_matches': len(results),
            'location_filter_used': bool(city or state),
            'has_location_data': search_service.matcher.has_location_data if search_service.matcher else False
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
    """Force reload of company data"""
    try:
        if search_service.load_company_data(force_reload=True):
            loaded_count = len(search_service.matcher.original_company_names) if search_service.matcher else 0
            return jsonify({
                'success': True,
                'message': f'Data reloaded successfully. {loaded_count} companies loaded.',
                'companies_loaded': loaded_count
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to reload company data'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Reload failed: {str(e)}'
        }), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache and force fresh data loading - ensures CLI/webapp consistency"""
    try:
        if search_service.clear_cache():
            loaded_count = len(search_service.matcher.original_company_names) if search_service.matcher else 0
            return jsonify({
                'success': True,
                'message': f'Cache cleared and data reloaded. {loaded_count} companies loaded.',
                'companies_loaded': loaded_count
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to reload company data after cache clear'
            }), 500
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
                'error': 'No matcher initialized'
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
    """Check if company data is loaded"""
    return jsonify(search_service.get_status())

if __name__ == '__main__':
    # Enable auto-reloading for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
