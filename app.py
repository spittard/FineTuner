from flask import Flask, render_template, request, jsonify
from CompanyMatcher import CompanyMatcher
import json
import os
import time

app = Flask(__name__)

# Enable auto-reloading for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global variable to store the matcher instance
matcher = None
company_data_loaded = False
last_data_check = 0
data_check_interval = 5  # Check for data changes every 5 seconds

def load_company_data(force_reload=False):
    """Load company data and initialize the matcher"""
    global matcher, company_data_loaded, last_data_check
    
    current_time = time.time()
    
    # Check if we need to reload data (either forced or time-based)
    if not force_reload and company_data_loaded and matcher is not None:
        # Check if training_data.json has been modified
        if current_time - last_data_check < data_check_interval:
            return True
        
        try:
            # Check if file modification time has changed
            if os.path.exists('training_data.json'):
                file_mtime = os.path.getmtime('training_data.json')
                if hasattr(matcher, '_last_file_mtime') and matcher._last_file_mtime == file_mtime:
                    last_data_check = current_time
                    return True
        except:
            pass
    
    try:
        # Check if training_data.json exists
        if not os.path.exists('training_data.json'):
            return False
        
        # Load company names from the dataset
        with open('training_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        company_names = []
        for item in data:
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
        
        if not company_names:
            return False
        
        # Initialize CompanyMatcher
        matcher = CompanyMatcher()
        matcher.build_index(company_names)
        
        # Store file modification time for change detection
        try:
            matcher._last_file_mtime = os.path.getmtime('training_data.json')
        except:
            matcher._last_file_mtime = 0
        
        company_data_loaded = True
        last_data_check = current_time
        
        return True
        
    except Exception as e:
        print(f"Error loading company data: {e}")
        return False

@app.route('/')
def index():
    """Main page with search form"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle company search requests"""
    try:
        # Load company data if not already loaded
        if not load_company_data():
            return jsonify({
                'error': 'Company data not available. Please ensure training_data.json exists.'
            }), 500
        
        # Get search query
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Please enter a company name to search for.'}), 400
        
        # Get number of results (default to 10)
        top_k = int(request.form.get('top_k', 10))
        
        # Perform search
        matches = matcher.match(query, top_k=top_k)
        
        # Format results for display
        results = []
        for i, match in enumerate(matches, 1):
            # Generate match rationale based on the explanation
            explanation = matcher.explain_match(query, match['name'])
            rationale = generate_match_rationale(query, match['name'], explanation, match['score'])
            
            results.append({
                'rank': i,
                'company_name': match['name'],
                'likeness_percent': round(match['score'] * 100, 1),
                'match_rationale': rationale
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total_matches': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Search failed: {str(e)}'
        }), 500

@app.route('/reload', methods=['POST'])
def reload_data():
    """Force reload of company data"""
    try:
        if load_company_data(force_reload=True):
            return jsonify({
                'success': True,
                'message': f'Data reloaded successfully. {len(matcher.original_company_names)} companies loaded.',
                'companies_loaded': len(matcher.original_company_names)
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

def generate_match_rationale(query, company_name, explanation, score):
    """Generate a human-readable match rationale"""
    query_lower = query.lower()
    company_lower = company_name.lower()
    
    # Check for exact matches
    if query_lower == company_lower:
        return "Exact name match"
    
    # Check for starts with
    if company_lower.startswith(query_lower):
        return f"Company name starts with '{query}'"
    
    # Check for contains
    if query_lower in company_lower:
        return f"Company name contains '{query}'"
    
    # Check for word overlap
    query_words = set(query_lower.split())
    company_words = set(company_lower.split())
    overlap = query_words.intersection(company_words)
    
    if overlap:
        overlap_words = ', '.join(overlap)
        return f"Shared words: {overlap_words}"
    
    # Check for semantic similarity
    if score > 0.8:
        return "High semantic similarity"
    elif score > 0.6:
        return "Moderate semantic similarity"
    elif score > 0.4:
        return "Low semantic similarity"
    else:
        return "Minimal similarity"

@app.route('/status')
def status():
    """Check if company data is loaded"""
    if load_company_data():
        return jsonify({
            'status': 'ready',
            'companies_loaded': len(matcher.original_company_names) if matcher else 0,
            'last_updated': last_data_check
        })
    else:
        return jsonify({
            'status': 'not_ready',
            'error': 'Company data not available'
        })

if __name__ == '__main__':
    # Enable auto-reloading for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
