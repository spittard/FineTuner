from flask import Flask, render_template, request, jsonify
from CompanyMatcher import CompanyMatcher
import json
import os
import time

app = Flask(__name__)

# Enable auto-reloading for development
app.config['TEMPLATES_AUTO_RELOAD'] = False
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

def analyze_business_context(query, company_name):
    """Analyze business context and corporate terminology"""
    business_keywords = {
        'corporate': ['corp', 'corporation', 'incorporated', 'inc', 'llc', 'ltd', 'limited'],
        'partnership': ['partners', 'partnership', 'associates', 'assoc'],
        'holding': ['holdings', 'holding', 'group', 'enterprises', 'ventures'],
        'international': ['intl', 'international', 'global', 'worldwide'],
        'regional': ['regional', 'national', 'local', 'state', 'city'],
        'technology': ['tech', 'technology', 'digital', 'software', 'systems'],
        'financial': ['financial', 'finance', 'capital', 'investment', 'funds'],
        'consulting': ['consulting', 'consultants', 'advisory', 'services']
    }
    
    query_business = None
    company_business = None
    
    for business_type, keywords in business_keywords.items():
        if any(keyword in query.lower() for keyword in keywords):
            query_business = business_type
        if any(keyword in company_name.lower() for keyword in keywords):
            company_business = business_type
    
    if query_business and company_business:
        if query_business == company_business:
            return f"Both {business_type} entities - strong business structure alignment"
        else:
            return f"Different business structures: {query_business} vs {company_business}"
    
    return None

def analyze_word_origins(query, company_name):
    """Analyze word origins and etymology patterns"""
    origin_patterns = {
        'latin': ['corp', 'inc', 'ltd', 'assoc', 'intl', 'mgmt'],
        'greek': ['tech', 'info', 'sys', 'auto', 'bio', 'geo'],
        'french': ['enterprise', 'venture', 'capital', 'finance'],
        'german': ['holdings', 'group', 'werk', 'industrie'],
        'italian': ['banca', 'farmacia', 'ristorante'],
        'spanish': ['banco', 'farmacia', 'restaurante']
    }
    
    query_origins = []
    company_origins = []
    
    for origin, words in origin_patterns.items():
        if any(word in query.lower() for word in words):
            query_origins.append(origin)
        if any(word in company_name.lower() for word in words):
            company_origins.append(origin)
    
    if query_origins and company_origins:
        common_origins = set(query_origins) & set(company_origins)
        if common_origins:
            return f"Shared linguistic origins: {', '.join(common_origins)}"
        else:
            return f"Different linguistic origins: {', '.join(query_origins)} vs {', '.join(company_origins)}"
    
    return None

def get_enhanced_phonetic_similarity(query, company_name):
    """Enhanced phonetic analysis with multiple algorithms"""
    # Basic phonetic
    basic_phonetic = analyze_phonetic_similarity(query, company_name)
    
    # Soundex-like analysis
    query_soundex = get_soundex(query.lower())
    company_soundex = get_soundex(company_name.lower())
    
    if query_soundex == company_soundex:
        return "Identical phonetic codes (Soundex)"
    elif basic_phonetic:
        return f"{basic_phonetic} | Soundex codes: {query_soundex} vs {company_soundex}"
    
    return f"Soundex codes: {query_soundex} vs {company_soundex}"

def get_soundex(text):
    """Generate Soundex phonetic code for text"""
    # Simplified Soundex implementation
    soundex_map = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    
    # Remove non-alphabetic characters
    text = ''.join(c for c in text if c.isalpha())
    if not text:
        return "0000"
    
    # First letter
    result = text[0].upper()
    
    # Convert remaining letters to codes
    for char in text[1:]:
        code = soundex_map.get(char.lower(), '')
        if code and code != result[-1]:
            result += code
    
    # Pad to 4 characters
    result = result.ljust(4, '0')
    return result[:4]

def get_comprehensive_word_analysis(query, company_name):
    """Get comprehensive word analysis with minimal performance impact"""
    analysis = {}
    
    # Word length analysis
    query_words = query.lower().split()
    company_words = company_name.lower().split()
    
    analysis['query_stats'] = {
        'word_count': len(query_words),
        'avg_word_length': sum(len(w) for w in query_words) / len(query_words) if query_words else 0,
        'longest_word': max(query_words, key=len) if query_words else '',
        'shortest_word': min(query_words, key=len) if query_words else ''
    }
    
    analysis['company_stats'] = {
        'word_count': len(company_words),
        'avg_word_length': sum(len(w) for w in company_words) / len(company_words) if company_words else 0,
        'longest_word': max(company_words, key=len) if company_words else '',
        'shortest_word': min(company_words, key=len) if company_words else ''
    }
    
    # Character analysis
    analysis['character_analysis'] = {
        'query_chars': len(query.replace(' ', '')),
        'company_chars': len(company_name.replace(' ', '')),
        'query_vowels': sum(1 for c in query.lower() if c in 'aeiou'),
        'company_vowels': sum(1 for c in company_name.lower() if c in 'aeiou')
    }
    
    return analysis

def load_company_data(force_reload=False):
    """Load company data and initialize the matcher"""
    global matcher, company_data_loaded, last_data_check
    
    # Early return if data is already loaded and we don't need to force reload
    if not force_reload and company_data_loaded and matcher is not None:
        return True
    
    current_time = time.time()
    
    # Prevent multiple rapid calls to this function
    if not force_reload and company_data_loaded and matcher is not None:
        # Check if companies.json has been modified
        if current_time - last_data_check < data_check_interval:
            return True
        
        try:
            # Check if file modification time has changed
            if os.path.exists('companies.json'):
                file_mtime = os.path.getmtime('companies.json')
                if hasattr(matcher, '_last_file_mtime') and matcher._last_file_mtime == file_mtime:
                    last_data_check = current_time
                    return True
        except:
            pass
    
    # Add a guard to prevent multiple simultaneous loads
    if hasattr(load_company_data, '_loading') and load_company_data._loading:
        print("Already loading company data, skipping...")
        return company_data_loaded
    
    load_company_data._loading = True
    
    try:
        # Check if companies.json exists
        if not os.path.exists('companies.json'):
            load_company_data._loading = False
            return False
        
        # Load company names from the dataset
        print("📁 Loading company data from companies.json...")
        with open('companies.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   Found {len(data):,} total entries in file")
        
        company_names = []
        print("   Extracting company names...")
        for i, item in enumerate(data):
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
            
            # Show progress every 100,000 entries
            if (i + 1) % 100000 == 0:
                print(f"   Processed {i + 1:,} entries...")
        
        if not company_names:
            load_company_data._loading = False
            return False
        
        print(f"   ✓ Extracted {len(company_names):,} company names")
        
        # Initialize CompanyMatcher with EXACTLY the same parameters as CLI
        # CLI uses: CompanyMatcher(model_name=args.matcher_model) with default 'all-MiniLM-L6-v2'
        print("🤖 Initializing CompanyMatcher...")
        matcher = CompanyMatcher(model_name='all-MiniLM-L6-v2')
        
        # Build index using the same logic as CLI
        print(f"🔨 Building company matching index with {len(company_names):,} companies...")
        matcher.build_index(company_names)
        
        # Store file modification time for change detection
        try:
            matcher._last_file_mtime = os.path.getmtime('companies.json')
        except:
            matcher._last_file_mtime = 0
        
        company_data_loaded = True
        last_data_check = current_time
        
        print(f"🎉 SUCCESS: Loaded {len(company_names):,} company name entries")
        print(f"🚀 Webapp is now ready for company matching!")
        load_company_data._loading = False
        return True
        
    except Exception as e:
        print(f"Error loading company data: {e}")
        load_company_data._loading = False
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
                'error': 'Company data not available. Please ensure companies.json exists.'
            }), 500
        
        # Get search query
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Please enter a company name to search for.'}), 400
        
        # Get number of results (default to 10)
        top_k = int(request.form.get('top_k', 10))
        
        # Perform search using EXACTLY the same logic as CLI
        print(f"Searching for companies matching: {query}")
        matches = matcher.match(query, top_k=top_k)
        print(f"Found {len(matches)} matches")
        
        # Format results for display
        results = []
        for i, match in enumerate(matches, 1):
            print(f"Processing match {i}: {match['name']}")
            # Generate match rationale based on the explanation
            explanation = matcher.explain_match(query, match['name'])
            print(f"Explanation generated for match {i}")
            rationale = generate_match_rationale(query, match['name'], explanation, match['score'])
            print(f"Rationale generated for match {i}")
            
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
        
        # Log the search results for debugging consistency
        print(f"\nTop {len(matches)} matches for '{query}':")
        print("-" * 60)
        for i, match in enumerate(matches, 1):
            score_percent = match['score'] * 100
            print(f"{i:2d}. {match['name']:<40} {score_percent:5.1f}%")
        print("-" * 60)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total_matches': len(results)
        })
        
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

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache and force fresh data loading - ensures CLI/webapp consistency"""
    try:
        global matcher, company_data_loaded
        
        if matcher is not None:
            # Clear the cache for this matcher
            cache_key = matcher.get_cache_key(matcher.original_company_names)
            matcher.clear_cache(cache_key)
            print(f"Cleared cache: {cache_key}")
        
        # Reset state
        matcher = None
        company_data_loaded = False
        
        # Force reload of data
        if load_company_data(force_reload=True):
            return jsonify({
                'success': True,
                'message': f'Cache cleared and data reloaded. {len(matcher.original_company_names)} companies loaded.',
                'companies_loaded': len(matcher.original_company_names)
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
        if matcher is None:
            return jsonify({
                'success': False,
                'error': 'No matcher initialized'
            }), 400
        
        cache_info = matcher.get_cache_info()
        cache_key = matcher.get_cache_key(matcher.original_company_names) if matcher.original_company_names else None
        
        return jsonify({
            'success': True,
            'cache_info': cache_info,
            'current_cache_key': cache_key,
            'companies_loaded': len(matcher.original_company_names) if matcher.original_company_names else 0,
            'model_name': matcher.model_name if matcher else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Cache info failed: {str(e)}'
        }), 500

def generate_match_rationale(query, company_name, explanation, score):
    """Generate data-entry-clerk-focused explanations that are practical and actionable"""
    query_lower = query.lower()
    company_lower = company_name.lower()
    
    # Phase 1: Exact Match Check
    if query_lower == company_lower:
        return "🎯 PERFECT MATCH\n\n**What This Means:**\nThis is exactly the same company name you're looking for.\n\n**Action Required:**\n• Use this match - no further checking needed\n• This is 100% the same company\n\n**Why This Happens:**\n• Someone entered the company name exactly as it appears in your system\n• This is the ideal scenario for data entry"
    
    # Phase 2: Prefix Match Check
    if company_lower.startswith(query_lower):
        return f"🔍 PREFIX MATCH\n\n**What This Means:**\nThis company name starts with '{query}' and has additional information added.\n\n**Action Required:**\n• This is likely the same company with extra details\n• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')\n• If yes, use this match\n\n**Why This Happens:**\n• Someone entered just the core company name\n• Your system has the full legal name\n• Common in business databases where legal names include extra terms"
    
    # Phase 3: Substring Match Check
    if query_lower in company_lower:
        return f"📍 SUBSTRING MATCH\n\n**What This Means:**\nThis company name contains '{query}' somewhere within it.\n\n**Action Required:**\n• This is likely the same company\n• Check if the surrounding words make sense\n• If yes, use this match\n\n**Why This Happens:**\n• Someone entered a partial company name\n• Your system has the complete name\n• Common when people remember only part of a company name"
    
    # Phase 4: Word-by-Word Analysis
    query_words = set(query_lower.split())
    company_words = set(company_lower.split())
    overlap = query_words.intersection(company_words)
    
    if overlap:
        overlap_words = sorted(overlap)
        non_overlap_query = sorted(query_words - overlap)
        non_overlap_company = sorted(company_words - overlap)
        
        # Calculate statistics
        total_query_words = len(query_words)
        total_company_words = len(company_words)
        overlap_count = len(overlap)
        overlap_percentage = (overlap_count / max(total_query_words, total_company_words)) * 100
        
        rationale = f"📊 WORD OVERLAP MATCH\n\n**What This Means:**\n{overlap_count} word(s) match exactly between your search and this company.\n\n**Matching Words:**\n• {', '.join(overlap_words)}\n"
        
        if non_overlap_query:
            rationale += f"\n**Your Search Also Includes:**\n• {', '.join(non_overlap_query)}\n"
        if non_overlap_company:
            rationale += f"\n**Company Name Also Includes:**\n• {', '.join(non_overlap_company)}\n"
        
        rationale += f"\n**Match Strength:**\n• {overlap_percentage:.0f}% word overlap\n"
        
        if overlap_percentage > 50:
            rationale += f"• This is a STRONG match - likely the same company\n"
            rationale += f"• Action: Use this match with high confidence\n"
        elif overlap_percentage > 25:
            rationale += f"• This is a MODERATE match - worth investigating\n"
            rationale += f"• Action: Check if this makes business sense\n"
        else:
            rationale += f"• This is a WEAK match - may be coincidental\n"
            rationale += f"• Action: Verify carefully before using\n"
        
        rationale += f"\n**Why This Happens:**\n• Company names often have multiple words\n• Some words are more important than others\n• Business names can vary in how they're written"
        
        return rationale
    
    # Phase 5: Linguistic Relationship Analysis
    linguistic_relationships = []
    transformation_details = []
    practical_examples = []
    
    for q_word in query_words:
        for c_word in company_words:
            if q_word == c_word:
                continue
                
            # Check for ordinal number transformations
            if is_ordinal_relationship(q_word, c_word):
                relationship_type = "ordinal transformation"
                if q_word in ["eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"]:
                    numeric_form = get_numeric_ordinal(q_word)
                    transformation_details.append(f"'{q_word}' → '{numeric_form}' (ordinal number)")
                    practical_examples.append(f"Someone wrote '{q_word}' but your system has '{numeric_form}'")
                elif c_word in ["11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th"]:
                    word_form = get_word_ordinal(c_word)
                    transformation_details.append(f"'{c_word}' ← '{word_form}' (ordinal number)")
                    practical_examples.append(f"Your system has '{c_word}' but someone wrote '{word_form}'")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for abbreviation relationships
            elif is_abbreviation_relationship(q_word, c_word):
                relationship_type = "abbreviation/expansion"
                if len(q_word) < len(c_word):
                    transformation_details.append(f"'{q_word}' is abbreviation of '{c_word}'")
                    practical_examples.append(f"Someone used the short form '{q_word}' instead of '{c_word}'")
                else:
                    transformation_details.append(f"'{c_word}' is abbreviation of '{q_word}'")
                    practical_examples.append(f"Your system has the short form '{c_word}' but someone wrote '{q_word}'")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for contraction relationships
            elif is_contraction_relationship(q_word, c_word):
                relationship_type = "contraction"
                if "'" in q_word:
                    transformation_details.append(f"'{q_word}' is contraction of '{c_word}'")
                    practical_examples.append(f"Someone used '{q_word}' instead of '{c_word}'")
                else:
                    transformation_details.append(f"'{c_word}' is contraction of '{q_word}'")
                    practical_examples.append(f"Your system has '{c_word}' but someone wrote '{q_word}'")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for plural/singular relationships
            elif is_plural_relationship(q_word, c_word):
                relationship_type = "plural/singular"
                if q_word.endswith('s') and not c_word.endswith('s'):
                    transformation_details.append(f"'{q_word}' is plural of '{c_word}'")
                    practical_examples.append(f"Someone used '{q_word}' instead of '{c_word}'")
                else:
                    transformation_details.append(f"'{c_word}' is plural of '{q_word}'")
                    practical_examples.append(f"Your system has '{c_word}' but someone wrote '{q_word}'")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
            
            # Check for common word variations
            elif is_word_variation(q_word, c_word):
                relationship_type = "word variation"
                variation_type = get_variation_type(q_word, c_word)
                transformation_details.append(f"'{q_word}' and '{c_word}' are {variation_type}")
                practical_examples.append(f"Someone used '{q_word}' instead of '{c_word}' (same meaning, different form)")
                linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
    
    if linguistic_relationships:
        unique_relationships = list(set(linguistic_relationships))
        unique_transformations = list(set(transformation_details))
        unique_examples = list(set(practical_examples))
        
        rationale = f"🔗 WORD VARIATION MATCH\n\n**What This Means:**\nThe company names use different forms of the same words.\n\n**Key Differences Found:**\n"
        for detail in unique_transformations[:3]:
            rationale += f"• {detail}\n"
        
        rationale += f"\n**Real-World Examples:**\n"
        for example in unique_examples[:2]:
            rationale += f"• {example}\n"
        
        rationale += f"\n**Action Required:**\n• This is likely the same company\n• The differences are just how words are written\n• Use this match with confidence\n\n**Why This Happens:**\n• People write company names differently\n• Abbreviations are common in business\n• Numbers can be written as words or digits\n• This is normal in data entry"
        
        return rationale
    
    # Phase 6: Semantic Similarity Analysis
    rationale = f"🧠 MEANING-BASED MATCH\n\n**What This Means:**\nThe system found a match based on the meaning of the words, not exact spelling.\n\n**Match Confidence:**\n"
    
    if score > 0.8:
        rationale += f"• VERY HIGH confidence ({score:.0%})\n"
        rationale += f"• Action: This is almost certainly the same company\n"
        rationale += f"• Use this match with high confidence\n"
    elif score > 0.6:
        rationale += f"• HIGH confidence ({score:.0%})\n"
        rationale += f"• Action: This is likely the same company\n"
        rationale += f"• Use this match, but double-check\n"
    elif score > 0.4:
        rationale += f"• MEDIUM confidence ({score:.0%})\n"
        rationale += f"• Action: This might be the same company\n"
        rationale += f"• Investigate further before using\n"
    elif score > 0.2:
        rationale += f"• LOW confidence ({score:.0%})\n"
        rationale += f"• Action: This is probably not the same company\n"
        rationale += f"• Don't use this match\n"
    else:
        rationale += f"• VERY LOW confidence ({score:.0%})\n"
        rationale += f"• Action: This is almost certainly not the same company\n"
        rationale += f"• Don't use this match\n"
    
    rationale += f"\n**Why This Happens:**\n• Sometimes company names sound similar but aren't the same\n• The system looks at word meanings, not just spelling\n• This helps catch variations you might miss manually\n\n**Data Entry Tip:**\n• High confidence matches are usually safe to use\n• Medium confidence matches need manual verification\n• Low confidence matches should be rejected"
    
    return rationale

@app.route('/status')
def status():
    """Check if company data is loaded"""
    global matcher, company_data_loaded, last_data_check
    
    # Check if currently loading
    if hasattr(load_company_data, '_loading') and load_company_data._loading:
        return jsonify({
            'status': 'loading',
            'message': 'Building company matching index...',
            'progress': 'indexing'
        })
    
    if load_company_data():
        return jsonify({
            'status': 'ready',
            'companies_loaded': len(matcher.original_company_names) if matcher else 0,
            'last_updated': last_data_check,
            'message': f'Ready with {len(matcher.original_company_names):,} companies' if matcher else 'Ready'
        })
    else:
        return jsonify({
            'status': 'not_ready',
            'error': 'Company data not available',
            'message': 'Please ensure companies.json exists and is accessible'
        })

if __name__ == '__main__':
    # Enable auto-reloading for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
