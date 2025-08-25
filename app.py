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

# Helper functions for enhanced rationale generation
def is_ordinal_relationship(word1, word2):
    """Check if two words are ordinal number variations"""
    ordinals = {
        "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
        "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
        "nineteenth": "19th", "twentieth": "20th", "twenty-first": "21st", "twenty-second": "22nd"
    }
    
    # Check direct mapping
    if word1 in ordinals and word2 == ordinals[word1]:
        return True
    if word2 in ordinals and word1 == ordinals[word2]:
        return True
    
    # Check numeric patterns
    if word1.isdigit() and word2.endswith(('st', 'nd', 'rd', 'th')):
        return True
    if word2.isdigit() and word1.endswith(('st', 'nd', 'rd', 'th')):
        return True
    
    return False

def get_numeric_ordinal(word):
    """Convert word ordinal to numeric form"""
    ordinals = {
        "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
        "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
        "nineteenth": "19th", "twentieth": "20th"
    }
    return ordinals.get(word, word)

def get_word_ordinal(word):
    """Convert numeric ordinal to word form"""
    ordinals = {
        "11th": "eleventh", "12th": "twelfth", "13th": "thirteenth", "14th": "fourteenth",
        "15th": "fifteenth", "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
        "19th": "nineteenth", "20th": "twentieth"
    }
    return ordinals.get(word, word)

def is_abbreviation_relationship(word1, word2):
    """Check if one word is an abbreviation of another"""
    if len(word1) < len(word2) and word1 in word2:
        return True
    if len(word2) < len(word1) and word2 in word1:
        return True
    return False

def is_contraction_relationship(word1, word2):
    """Check if words are contractions of each other"""
    # Common contractions
    contractions = {
        "cant": "cannot", "dont": "do not", "wont": "will not", "isnt": "is not",
        "arent": "are not", "wasnt": "was not", "werent": "were not", "hasnt": "has not",
        "havent": "have not", "hadnt": "had not", "doesnt": "does not", "didnt": "did not"
    }
    
    if word1 in contractions and word2 == contractions[word1]:
        return True
    if word2 in contractions and word1 == contractions[word2]:
        return True
    
    return False

def is_plural_relationship(word1, word2):
    """Check if words are plural/singular forms of each other"""
    if word1.endswith('s') and word1[:-1] == word2:
        return True
    if word2.endswith('s') and word2[:-1] == word1:
        return True
    return False

def is_word_variation(word1, word2):
    """Check for common word variations"""
    # Common variations
    variations = [
        ("info", "information"), ("tech", "technical"), ("assoc", "association"),
        ("corp", "corporation"), ("co", "company"), ("inc", "incorporated"),
        ("ltd", "limited"), ("intl", "international"), ("mgmt", "management")
    ]
    
    for var1, var2 in variations:
        if (word1 == var1 and word2 == var2) or (word1 == var2 and word2 == var1):
            return True
    
    return False

def get_variation_type(word1, word2):
    """Get the type of word variation"""
    if len(word1) < len(word2):
        return "abbreviation/expansion pair"
    elif len(word2) < len(word1):
        return "abbreviation/expansion pair"
    else:
        return "synonym pair"

def analyze_phonetic_similarity(query, company_name):
    """Analyze phonetic similarity between query and company name"""
    # Simple phonetic analysis - could be enhanced with more sophisticated algorithms
    query_sound = get_simple_phonetic(query.lower())
    company_sound = get_simple_phonetic(company_name.lower())
    
    if query_sound == company_sound:
        return "Identical phonetic representation"
    elif query_sound in company_sound or company_sound in query_sound:
        return "Partial phonetic overlap detected"
    
    return None

def get_simple_phonetic(text):
    """Get a simple phonetic representation of text"""
    # Basic phonetic simplification
    phonetic = text.replace('ph', 'f').replace('ck', 'k').replace('qu', 'kw')
    phonetic = ''.join(c for c in phonetic if c.isalpha())
    return phonetic

def analyze_industry_context(query, company_name):
    """Analyze industry context and business terminology"""
    industry_keywords = {
        'tech': ['technology', 'software', 'hardware', 'digital', 'computer'],
        'finance': ['bank', 'financial', 'investment', 'insurance', 'credit'],
        'healthcare': ['medical', 'health', 'hospital', 'clinic', 'pharmaceutical'],
        'retail': ['store', 'shop', 'market', 'retail', 'commerce'],
        'manufacturing': ['manufacturing', 'industrial', 'factory', 'production', 'machinery']
    }
    
    query_industry = None
    company_industry = None
    
    for industry, keywords in industry_keywords.items():
        if any(keyword in query.lower() for keyword in keywords):
            query_industry = industry
        if any(keyword in company_name.lower() for keyword in keywords):
            company_industry = industry
    
    if query_industry and company_industry:
        if query_industry == company_industry:
            return f"Both in {query_industry} industry - strong industry alignment"
        else:
            return f"Different industries: {query_industry} vs {company_industry}"
    
    return None

def analyze_geographic_context(query, company_name):
    """Analyze geographic context and location indicators"""
    geographic_indicators = [
        'national', 'international', 'global', 'worldwide', 'regional',
        'local', 'state', 'city', 'county', 'district'
    ]
    
    query_geo = [word for word in query.lower().split() if word in geographic_indicators]
    company_geo = [word for word in company_name.lower().split() if word in geographic_indicators]
    
    if query_geo and company_geo:
        if query_geo == company_geo:
            return f"Same geographic scope: {', '.join(query_geo)}"
        else:
            return f"Different geographic scope: {', '.join(query_geo)} vs {', '.join(company_geo)}"
    
    return None

def get_score_breakdown(score):
    """Provide detailed breakdown of the semantic score"""
    if score > 0.9:
        return "Exceptional (90%+) - Nearly perfect semantic match"
    elif score > 0.8:
        return "Excellent (80-89%) - Very strong semantic relationship"
    elif score > 0.7:
        return "Very Good (70-79%) - Strong semantic relationship"
    elif score > 0.6:
        return "Good (60-69%) - Moderate semantic relationship"
    elif score > 0.5:
        return "Fair (50-59%) - Some semantic relationship"
    elif score > 0.4:
        return "Poor (40-49%) - Weak semantic relationship"
    elif score > 0.3:
        return "Very Poor (30-39%) - Very weak semantic relationship"
    elif score > 0.2:
        return "Minimal (20-29%) - Minimal semantic relationship"
    else:
        return "Negligible (<20%) - No meaningful semantic relationship"

def get_search_quality_metrics(query, company_name, score):
    """Provide search quality metrics and recommendations"""
    metrics = []
    
    # Query length analysis
    query_length = len(query.split())
    if query_length < 2:
        metrics.append("Short query - consider adding more context")
    elif query_length > 5:
        metrics.append("Long query - may be too specific")
    else:
        metrics.append("Optimal query length")
    
    # Company name length analysis
    company_length = len(company_name.split())
    if company_length > 8:
        metrics.append("Long company name - may contain extra details")
    
    # Score confidence
    if score > 0.7:
        metrics.append("High confidence match")
    elif score > 0.5:
        metrics.append("Medium confidence match")
    else:
        metrics.append("Low confidence match - consider refining search")
    
    return ' | '.join(metrics)

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
                'match_rationale': rationale,
                'raw_score': match['score'],
                'explanation_details': {
                    'query_tokens': list(explanation['query_tokens']),
                    'match_tokens': list(explanation['match_tokens']),
                    'overlap_tokens': list(explanation['overlap']),
                    'overlap_score': explanation['overlap_score']
                }
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
    """Generate a detailed, step-by-step match rationale explaining each phase with linguistic analysis"""
    query_lower = query.lower()
    company_lower = company_name.lower()
    
    # Phase 1: Exact Match Check
    if query_lower == company_lower:
        return "🎯 EXACT MATCH: Query and company name are identical (100% match)"
    
    # Phase 2: Prefix Match Check
    if company_lower.startswith(query_lower):
        return f"🔍 PREFIX MATCH: Company name starts with '{query}' - perfect beginning match"
    
    # Phase 3: Substring Match Check
    if query_lower in company_lower:
        return f"📍 SUBSTRING MATCH: Company name contains '{query}' as a continuous sequence"
    
    # Phase 4: Word-by-Word Analysis with Linguistic Transformations
    query_words = set(query_lower.split())
    company_words = set(company_lower.split())
    overlap = query_words.intersection(company_words)
    
    if overlap:
        overlap_words = sorted(overlap)
        non_overlap_query = sorted(query_words - overlap)
        non_overlap_company = sorted(company_words - overlap)
        
        # Calculate detailed statistics
        total_query_words = len(query_words)
        total_company_words = len(company_words)
        overlap_count = len(overlap)
        overlap_percentage = (overlap_count / max(total_query_words, total_company_words)) * 100
        
        rationale = f"📊 WORD OVERLAP ANALYSIS:\n"
        rationale += f"   • Shared words ({overlap_count}): {', '.join(overlap_words)}\n"
        
        if non_overlap_query:
            rationale += f"   • Query-specific words ({len(non_overlap_query)}): {', '.join(non_overlap_query)}\n"
        if non_overlap_company:
            rationale += f"   • Company-specific words ({len(non_overlap_company)}): {', '.join(non_overlap_company)}\n"
        
        rationale += f"   • Overlap ratio: {overlap_count}/{max(total_query_words, total_company_words)} ({overlap_percentage:.1f}%)\n"
        rationale += f"   • Query word count: {total_query_words} | Company word count: {total_company_words}"
        
        return rationale
    
    # Phase 5: Advanced Linguistic Relationship Analysis
    linguistic_relationships = []
    transformation_details = []
    
    for q_word in query_words:
        for c_word in company_words:
            # Check for exact matches (already handled above)
            if q_word == c_word:
                continue
                
            # Check for ordinal number transformations
            if is_ordinal_relationship(q_word, c_word):
                relationship_type = "ordinal transformation"
                if q_word in ["eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"]:
                    numeric_form = get_numeric_ordinal(q_word)
                    transformation_details.append(f"'{q_word}' → '{numeric_form}' (ordinal number)")
                elif c_word in ["11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th"]:
                    word_form = get_word_ordinal(c_word)
                    transformation_details.append(f"'{c_word}' ← '{word_form}' (ordinal number)")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for abbreviation relationships
            elif is_abbreviation_relationship(q_word, c_word):
                relationship_type = "abbreviation/expansion"
                if len(q_word) < len(c_word):
                    transformation_details.append(f"'{q_word}' is abbreviation of '{c_word}'")
                else:
                    transformation_details.append(f"'{c_word}' is abbreviation of '{q_word}'")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for contraction relationships
            elif is_contraction_relationship(q_word, c_word):
                relationship_type = "contraction"
                if "'" in q_word:
                    transformation_details.append(f"'{q_word}' is contraction of '{c_word}'")
                else:
                    transformation_details.append(f"'{c_word}' is contraction of '{q_word}'")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for plural/singular relationships
            elif is_plural_relationship(q_word, c_word):
                relationship_type = "plural/singular"
                if q_word.endswith('s') and not c_word.endswith('s'):
                    transformation_details.append(f"'{q_word}' is plural of '{c_word}'")
                else:
                    transformation_details.append(f"'{c_word}' is plural of '{q_word}'")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for common word variations
            elif is_word_variation(q_word, c_word):
                relationship_type = "word variation"
                variation_type = get_variation_type(q_word, c_word)
                transformation_details.append(f"'{q_word}' and '{c_word}' are {variation_type}")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
    
    if linguistic_relationships:
        relationship_text = ', '.join(linguistic_relationships[:3])  # Limit to first 3
        if len(linguistic_relationships) > 3:
            relationship_text += f" (+{len(linguistic_relationships)-3} more relationships)"
        
        rationale = f"🔗 ADVANCED LINGUISTIC ANALYSIS:\n"
        rationale += f"   • Found {len(linguistic_relationships)} linguistic relationships: {relationship_text}\n"
        
        if transformation_details:
            rationale += f"   • Transformations identified:\n"
            for detail in transformation_details[:5]:  # Limit to first 5 details
                rationale += f"     - {detail}\n"
            if len(transformation_details) > 5:
                rationale += f"     ... and {len(transformation_details)-5} more transformations\n"
        
        rationale += f"   • Semantic similarity score: {score:.3f}"
        
        return rationale
    
    # Phase 6: Semantic Similarity Analysis
    rationale = f"🧠 SEMANTIC SIMILARITY ANALYSIS:\n"
    rationale += f"   • Raw semantic score: {score:.3f}\n"
    
    if score > 0.8:
        rationale += f"   • Classification: HIGH SIMILARITY\n"
        rationale += f"   • Interpretation: Very strong conceptual relationship between query and company\n"
        rationale += f"   • Confidence: High confidence in semantic match"
    elif score > 0.6:
        rationale += f"   • Classification: MODERATE SIMILARITY\n"
        rationale += f"   • Interpretation: Good conceptual relationship, some semantic overlap\n"
        rationale += f"   • Confidence: Moderate confidence in semantic match"
    elif score > 0.4:
        rationale += f"   • Classification: LOW SIMILARITY\n"
        rationale += f"   • Interpretation: Weak conceptual relationship, minimal semantic overlap\n"
        rationale += f"   • Confidence: Low confidence in semantic match"
    elif score > 0.2:
        rationale += f"   • Classification: VERY LOW SIMILARITY\n"
        rationale += f"   • Interpretation: Minimal conceptual relationship, likely coincidental\n"
        rationale += f"   • Confidence: Very low confidence in semantic match"
    else:
        rationale += f"   • Classification: MINIMAL SIMILARITY\n"
        rationale += f"   • Interpretation: No meaningful conceptual relationship\n"
        rationale += f"   • Confidence: No confidence in semantic match - likely random result"
    
    # Add explanation details if available
    if 'explanation_details' in locals():
        rationale += f"\n   • Query tokens: {', '.join(sorted(explanation.get('query_tokens', [])))}\n"
        rationale += f"   • Company tokens: {', '.join(sorted(explanation.get('match_tokens', [])))}\n"
        rationale += f"   • Token overlap: {', '.join(sorted(explanation.get('overlap', [])))}\n"
        rationale += f"   • Overlap score: {explanation.get('overlap_score', 0):.3f}"
    
    return rationale

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
