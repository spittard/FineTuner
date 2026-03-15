from flask import Flask, render_template, request, jsonify
from finetuner.web.services.rationale_service import RationaleService
import json
import os
import time

from finetuner.web.services.search_service import SearchService

# Paths to plugging data files (project root, 4 levels up from this file)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
PLUGGING_RECORDS_JSON = os.path.join(_PROJECT_ROOT, 'plugging_records.json')
PLUGGING_MATCHES_JSON = os.path.join(_PROJECT_ROOT, 'plugging_matches.json')
TIER_CONFIG_JSON      = os.path.join(_PROJECT_ROOT, 'tier_config.json')

_DEFAULT_TIER_CONFIG = {'high': 85, 'medium': 70}

def _load_tier_config():
    if os.path.exists(TIER_CONFIG_JSON):
        try:
            with open(TIER_CONFIG_JSON, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            h = int(cfg.get('high',   _DEFAULT_TIER_CONFIG['high']))
            m = int(cfg.get('medium', _DEFAULT_TIER_CONFIG['medium']))
            if 1 <= m < h <= 100:
                return {'high': h, 'medium': m}
        except Exception:
            pass
    return dict(_DEFAULT_TIER_CONFIG)

def _save_tier_config(cfg):
    with open(TIER_CONFIG_JSON, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)

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


# ── Plugging Records routes ───────────────────────────────────────────────────

@app.route('/plugging')
def plugging():
    """Plugging Records browser page"""
    return render_template('plugging.html')


@app.route('/api/plugging-records')
def get_plugging_records():
    """Return list of plugging records, annotated with best_score/tier from pre-computed matches."""
    if not os.path.exists(PLUGGING_RECORDS_JSON):
        return jsonify({'error': f'plugging_records.json not found at {PLUGGING_RECORDS_JSON}'}), 404

    with open(PLUGGING_RECORDS_JSON, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # Build a lookup of best pre-computed scores keyed by row_id (string for JSON safety)
    precomputed = {}
    if os.path.exists(PLUGGING_MATCHES_JSON):
        try:
            with open(PLUGGING_MATCHES_JSON, 'r', encoding='utf-8') as f:
                matches_data = json.load(f)
            for entry in matches_data:
                rid = str(entry.get('row_id', ''))
                best_matches = entry.get('matches', [])
                if best_matches:
                    best_score = best_matches[0].get('score', 0)
                    precomputed[rid] = round(best_score * 100, 1)
        except Exception:
            pass  # Missing / corrupt — just skip pre-computed scores

    cfg = _load_tier_config()
    def tier(pct):
        if pct is None:
            return 'unknown'
        if pct >= cfg['high']:
            return 'High'
        if pct >= cfg['medium']:
            return 'Medium'
        return 'Low'

    out = []
    for rec in records:
        rid = str(rec.get('ID', rec.get('row_id', '')))
        score_pct = precomputed.get(rid)
        out.append({
            'row_id':    rid,
            'company':   rec.get('Company Name', rec.get('Company', rec.get('query_company', ''))),
            'city':      rec.get('City', rec.get('query_city', '')),
            'state':     rec.get('State', rec.get('query_state', '')),
            'best_score': score_pct,
            'tier':      tier(score_pct),
        })

    return jsonify({'records': out, 'total': len(out)})


@app.route('/api/plugging-match', methods=['POST'])
def plugging_match():
    """Live-match a single plugging record against the RPC index."""
    try:
        data = request.get_json(force=True) or {}
        company = (data.get('company') or '').strip()
        city    = (data.get('city')    or '').strip() or None
        state   = (data.get('state')   or '').strip() or None
        top_k   = int(data.get('top_k', 5))

        if not company:
            return jsonify({'error': 'company is required'}), 400

        results = search_service.search(company, top_k=top_k, city=city, state=state)
        return jsonify({'success': True, 'results': results})

    except Exception as exc:
        import traceback
        print(f"plugging_match error: {exc}\n{traceback.format_exc()}")
        return jsonify({'error': str(exc)}), 500


@app.route('/api/tier-config', methods=['GET'])
def get_tier_config():
    """Return current tier thresholds."""
    return jsonify(_load_tier_config())


@app.route('/api/tier-config', methods=['POST'])
def set_tier_config():
    """Update tier thresholds. Expects JSON {high: int, medium: int}."""
    data = request.get_json(force=True) or {}
    try:
        h = int(data['high'])
        m = int(data['medium'])
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'high and medium must be integers'}), 400
    if not (1 <= m < h <= 100):
        return jsonify({'error': f'Must satisfy 1 ≤ medium ({m}) < high ({h}) ≤ 100'}), 400
    cfg = {'high': h, 'medium': m}
    _save_tier_config(cfg)
    return jsonify({'success': True, 'config': cfg})


if __name__ == '__main__':
    # Enable auto-reloading for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
