import difflib
import re

class TextPreprocessor:
    # Generic terms that should be down-weighted in matching
    GENERIC_TERMS = {
        # Facility types (weight 0.3)
        'center': 0.3, 'school': 0.3, 'hospital': 0.3, 'office': 0.3, 
        'building': 0.3, 'facility': 0.3, 'church': 0.3, 'synagogue': 0.3,
        # Event types (weight 0.3)
        'meeting': 0.3, 'breakfast': 0.3, 'lunch': 0.3, 'dinner': 0.3, 
        'conference': 0.3, 'event': 0.3, 'events': 0.3, 'tournament': 0.3,
        'wedding': 0.3,
        # Organization suffixes (weight 0.2) - common corporate terms
        'group': 0.2, 'association': 0.2, 'coalition': 0.2, 'foundation': 0.2, 
        'services': 0.2, 'service': 0.2, 'solutions': 0.2, 'partners': 0.2,
        'pa': 0.2, 'pc': 0.2,
        # Location modifiers (weight 0.5) - somewhat distinctive but common
        'north': 0.5, 'south': 0.5, 'east': 0.5, 'west': 0.5, 
        'shore': 0.5, 'bay': 0.5, 'coast': 0.5, 'lake': 0.5,
        'valley': 0.5, 'mountain': 0.5, 'hill': 0.5, 'river': 0.5,
        # Common modifiers (weight 0.4)
        'national': 0.4, 'international': 0.4, 'global': 0.4, 'regional': 0.4,
        'local': 0.4, 'community': 0.4, 'public': 0.4, 'private': 0.4,
    }

    # Category words that define the TYPE of entity
    CATEGORY_WORDS = {
        'facility_type': {'center', 'school', 'hospital', 'church', 'synagogue', 'office', 'building'},
        'event_type': {'meeting', 'conference', 'wedding', 'tournament', 'breakfast', 'lunch', 'dinner', 'event'},
        'service_type': {'senior', 'medical', 'financial', 'legal', 'technical', 'nursing', 'dental', 'health'},
    }

    # Common words that should NOT be treated as proper nouns
    COMMON_WORDS = {
        # Articles and prepositions
        'the', 'a', 'an', 'of', 'and', 'or', 'for', 'to', 'in', 'on', 'at', 'by', 'with',
        # Generic business terms
        'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'co', 'company',
        'group', 'holdings', 'enterprises', 'associates', 'partners', 'services', 'solutions',
        'pa', 'pc',
        # Facility/organization types
        'center', 'school', 'hospital', 'office', 'building', 'facility', 'church', 'synagogue',
        'university', 'college', 'institute', 'academy', 'association', 'foundation', 'society',
        # Event types
        'meeting', 'conference', 'event', 'events', 'breakfast', 'lunch', 'dinner', 'tournament',
        'wedding', 'reception', 'ceremony', 'celebration', 'gala', 'banquet',
        # Descriptors
        'national', 'international', 'global', 'regional', 'local', 'community', 'public', 'private',
        'general', 'special', 'annual', 'monthly', 'weekly', 'daily',
        # Directions/locations
        'north', 'south', 'east', 'west', 'central', 'upper', 'lower', 'new', 'old',
        'shore', 'bay', 'coast', 'lake', 'valley', 'mountain', 'hill', 'river', 'island',
        # Service types
        'senior', 'medical', 'financial', 'legal', 'technical', 'nursing', 'dental', 'health',
        'professional', 'executive', 'administrative', 'clinical', 'educational',
        # Common adjectives
        'first', 'second', 'third', 'fourth', 'fifth', 'primary', 'secondary',
        'main', 'major', 'minor', 'grand', 'great', 'big', 'small', 'little',
    }

    # State abbreviation mappings
    STATE_ABBREV = {
        'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
        'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
        'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
        'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
        'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
        'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
        'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
        'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
        'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
        'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
        'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
        'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
        'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia'
    }
    
    # Common city name variations
    CITY_VARIATIONS = {
        'nyc': 'new york', 'new york city': 'new york', 'ny': 'new york',
        'la': 'los angeles', 'l.a.': 'los angeles',
        'sf': 'san francisco', 'san fran': 'san francisco',
        'dc': 'washington', 'washington dc': 'washington', 'washington d.c.': 'washington',
        'philly': 'philadelphia', 'phila': 'philadelphia',
        'chi': 'chicago', 'chi-town': 'chicago',
        'vegas': 'las vegas', 'lv': 'las vegas',
        'nola': 'new orleans',
        'atl': 'atlanta',
        'stl': 'st louis', 'st. louis': 'saint louis', 'saint louis': 'st louis',
        'ft worth': 'fort worth', 'ft. worth': 'fort worth',
        'st paul': 'saint paul', 'st. paul': 'saint paul',
        'mt': 'mount', 'mt.': 'mount',
    }

    @staticmethod
    def clean_company_name(name):
        """Removes common business suffixes and stop words."""
        if not name: return ""
        suffixes = {
            'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
            'co', 'company', 'plc', 'group', 'holdings', 'enterprises', 'associates',
            'pa', 'pc'
        }
        stop_words = {'the', 'of', 'and', '&', 'a', 'an'}
        
        name_lower = name.lower().replace('.', '').replace(',', '').replace('-', ' ')
        words = name_lower.split()
        
        clean_words = [w for w in words if w not in suffixes and w not in stop_words]
        
        if not clean_words:
            return name_lower
            
        return " ".join(clean_words)

    @classmethod
    def get_term_weight(cls, term):
        """Returns a weight for a term based on how generic/common it is."""
        term_lower = term.lower()
        return cls.GENERIC_TERMS.get(term_lower, 1.0)
    
    @classmethod
    def get_category_words(cls, tokens):
        """Extracts category-defining words from a set of tokens."""
        found_categories = {}
        for token in tokens:
            token_lower = token.lower()
            for category, words in cls.CATEGORY_WORDS.items():
                if token_lower in words:
                    if category not in found_categories:
                        found_categories[category] = set()
                    found_categories[category].add(token_lower)
        return found_categories
    
    @classmethod
    def generate_acronym(cls, text):
        """
        Generates an acronym from a given text (e.g. 'American Bar Association' -> 'ABA').
        Considers only capitalized words in the input.
        Returns None if text has fewer than 2 words.
        """
        if not text: return None
        
        # Clean text first
        clean_text = text.replace('.', '').replace(',', '').replace('-', ' ')
        words = clean_text.split()
        
        if len(words) < 2: return None
        
        # Collect first chars of eligible words
        acronym_chars = []
        for word in words:
            # Skip very short generic words unless capitalized (e.g. 'Of' vs 'of')
            # Actually, proper acronyms usually take first letter of all significant words
            if word.lower() in ['the', 'of', 'and', 'for', 'in', 'at', 'by', 'to']:
                continue
            
            if word: 
                acronym_chars.append(word[0].upper())
        
        if len(acronym_chars) < 2: return None
        
        return "".join(acronym_chars)

    @classmethod
    def calculate_acronym_fidelity(cls, acronym, text):
        """
        Calculates how "faithfully" a company name expands an acronym.
        Returns a score between 0.0 and 1.0.
        
        1.0: Perfect Expansion (Each acronym char is the first letter of a distinct significant word)
        0.7: Overlapping Match (Letters match word-starts, but some words contain multiple acronym letters)
        0.3: Partial/Internal Match (Letters match inside words or out of order)
        """
        if not acronym or not text:
            return 0.0
            
        acronym = acronym.upper()
        # Clean and tokenize
        clean_text = text.replace('.', '').replace(',', '').replace('-', ' ')
        words = [w for w in clean_text.split() if w.lower() not in ['the', 'of', 'and', 'for', 'in', 'at', 'by', 'to']]
        
        if not words:
            return 0.0
            
        # Target fidelity: One letter per word
        word_starts = [w[0].upper() for w in words if w]
        word_starts_str = "".join(word_starts)
        
        # Scenario 1: Exact Match of Word Starts (Perfect Expansion)
        # e.g. "IBM" vs "International Business Machines"
        if word_starts_str == acronym:
            # Check for collisions: Do any of the words contain other characters of the acronym?
            # e.g. searching "IBM" and finding "IBMA" as the first word
            collision = False
            for i, word in enumerate(words):
                if i >= len(acronym): break
                word_upper = word.upper()
                current_char = acronym[i]
                other_chars = acronym[:i] + acronym[i+1:]
                
                if len(word_upper) > 1:
                    # Only check for SUBSEQUENT characters in the acronym
                    # e.g. If we are at 'I', check if this word also contains 'B' or 'M'
                    for char in acronym[i+1:]:
                        if char in word_upper[1:]:
                            collision = True
                            break
                if collision: break
            
            if not collision:
                return 1.0
            else:
                return 0.70 # Penalize if words overlap with acronym
            
        # Scenario 2: Acronym is a prefix of Word Starts (Clean Expansion with extra words)
        # e.g. "IBM" vs "International Business Machines Corporation"
        if word_starts_str.startswith(acronym):
            return 0.95
            
        # Scenario 3: Word Starts contain Acronym as subsequence
        # e.g. "IBM" vs "International Bureau of Management" (Bureau of -> BM)
        if acronym in word_starts_str:
            return 0.90
            
        # Scenario 4: "Overlapping" Match (IBMA -> IBM)
        # Check if any single word contains more than one letter of the acronym as a sequence
        # e.g. "IBM" matching "IBMA" where "IBMA" has I, B, M
        for word in words:
            word_upper = word.upper()
            if len(acronym) > 1 and acronym[:2] in word_upper:
                # If the word itself contains the first two or more letters, it's a collision
                # e.g. searching "IBM" and finding "IBMA..."
                return 0.60
            
        # Scenario 5: Word contains the acronym as a prefix but is longer
        # e.g. "IBM" vs "IBMA"
        if len(words) >= 1:
            first_word = words[0].upper()
            if first_word.startswith(acronym) and len(first_word) > len(acronym):
                return 0.65

        # Scenario 6: Fuzzy word-start matching
        # Check how many characters in acronym can be mapped to word starts in order
        matched_chars = 0
        word_idx = 0
        for char in acronym:
            found = False
            while word_idx < len(word_starts):
                if word_starts[word_idx] == char:
                    matched_chars += 1
                    word_idx += 1
                    found = True
                    break
                word_idx += 1
            if not found:
                break
                
        fidelity = matched_chars / len(acronym)
        return fidelity * 0.4 # Scale down for partial out-of-order or skipped words

    @classmethod
    def check_category_mismatch(cls, query_tokens, target_tokens):
        """Checks if query and target have mismatched category words."""
        query_categories = cls.get_category_words(query_tokens)
        target_categories = cls.get_category_words(target_tokens)
        
        penalty = 1.0
        for category in cls.CATEGORY_WORDS.keys():
            query_words = query_categories.get(category, set())
            target_words = target_categories.get(category, set())
            if query_words and target_words and not query_words.intersection(target_words):
                penalty *= 0.75
        return penalty

    @classmethod
    def extract_proper_nouns(cls, original_name):
        """Extracts likely proper nouns from a company name."""
        proper_nouns = set()
        words = re.split(r'[\s\-/,&]+', original_name)
        for word in words:
            if len(word) < 2: continue
            if word[0].isupper():
                word_lower = word.lower()
                if word_lower not in cls.COMMON_WORDS:
                    proper_nouns.add(word_lower)
        return proper_nouns
    
    @classmethod
    def check_proper_noun_mismatch(cls, query_original, target_original, query_tokens, target_tokens):
        """Checks if the query has proper nouns missing from the target."""
        query_proper = cls.extract_proper_nouns(query_original)
        target_proper = cls.extract_proper_nouns(target_original)
        
        if not query_proper: return 1.0
        
        missing_proper = query_proper - target_proper
        target_tokens_lower = {t.lower() for t in target_tokens}
        truly_missing = {p for p in missing_proper if p not in target_tokens_lower}
        
        if not truly_missing: return 1.0
        
        missing_ratio = len(truly_missing) / len(query_proper)
        penalty = 1.0 - (missing_ratio * 0.25)
        return penalty

    @classmethod
    def calculate_string_similarity(cls, query, target):
        """Calculates robust string similarity score."""
        clean_query = cls.clean_company_name(query)
        clean_target = cls.clean_company_name(target)
        
        q_tokens = set(clean_query.split())
        t_tokens = set(clean_target.split())
        
        if not q_tokens or not t_tokens: return 0.0
        
        intersection = q_tokens.intersection(t_tokens)
        union = q_tokens.union(t_tokens)
        
        # Weighted Jaccard
        weighted_intersection = sum(cls.get_term_weight(t) for t in intersection)
        weighted_union = sum(cls.get_term_weight(t) for t in union)
        weighted_jaccard = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
        
        # Adjusted Jaccard (singular/plural)
        adjusted_intersection = len(intersection)
        for q_token in q_tokens:
            if q_token not in t_tokens:
                if q_token + 's' in t_tokens or q_token + 'es' in t_tokens:
                    adjusted_intersection += 1
                elif (q_token.endswith('s') and q_token[:-1] in t_tokens) or \
                     (q_token.endswith('es') and q_token[:-2] in t_tokens):
                    adjusted_intersection += 1
        adjusted_jaccard = adjusted_intersection / len(union) if len(union) > 0 else 0.0
        
        jaccard_score = max(weighted_jaccard, adjusted_jaccard * 0.9)
        
        # Sequence Matcher
        seq_query_normalized = ' '.join([w.rstrip('es').rstrip('s') if len(w) > 3 else w for w in clean_query.split()])
        seq_target_normalized = ' '.join([w.rstrip('es').rstrip('s') if len(w) > 3 else w for w in clean_target.split()])
        seq_score = difflib.SequenceMatcher(None, seq_query_normalized, seq_target_normalized).ratio()
        seq_score_original = difflib.SequenceMatcher(None, clean_query, clean_target).ratio()
        seq_score = max(seq_score, seq_score_original)
        
        base_score = max(jaccard_score, seq_score)
        
        # Distinctive Word Penalty
        query_distinctive = [t for t in q_tokens if cls.get_term_weight(t) >= 0.8]
        missing_distinctive = [t for t in query_distinctive if t not in t_tokens]
        if query_distinctive and missing_distinctive:
            missing_ratio = len(missing_distinctive) / len(query_distinctive)
            distinctive_penalty = 1.0 - (missing_ratio * 0.4)
            base_score *= distinctive_penalty
        
        # Short String Score Cap
        matched_chars = sum(len(t) for t in intersection)
        if matched_chars < 5:
            base_score = min(base_score, 0.50)
        elif matched_chars < 8:
            base_score = min(base_score, 0.65)
            
        # Penalties
        category_penalty = cls.check_category_mismatch(q_tokens, t_tokens)
        base_score *= category_penalty
        
        proper_noun_penalty = cls.check_proper_noun_mismatch(query, target, q_tokens, t_tokens)
        base_score *= proper_noun_penalty
        
        # Coverage/Length Adjustments
        query_words_in_target = len(intersection)
        coverage_ratio = query_words_in_target / len(q_tokens) if q_tokens else 0
        query_length = len(q_tokens)
        target_length = len(t_tokens)
        combined_penalty = category_penalty * proper_noun_penalty
        
        if clean_query in clean_target or clean_target in clean_query:
            if combined_penalty >= 0.85:
                base_score = max(base_score, 0.9 * combined_penalty)
        elif coverage_ratio >= 0.5:
            coverage_boost = 0.7 + (coverage_ratio * 0.2)
            if target_length >= query_length * 0.6: coverage_boost += 0.05
            coverage_boost *= combined_penalty
            base_score = max(base_score, coverage_boost)
        elif coverage_ratio >= 0.25:
            coverage_boost = 0.6 + ((coverage_ratio - 0.25) * 0.4)
            if target_length < query_length * 0.5: coverage_boost *= 0.8
            coverage_boost *= combined_penalty
            base_score = max(base_score, coverage_boost)
            
        # BIAS FIX: Penalize ANY deviation from query length (shorter OR longer)
        # This prevents short strings (abbreviations) from jumping to the top unfairly
        word_diff = abs(target_length - query_length)
        
        # INCREASE PENALTY for very short targets matching longer queries
        if target_length < query_length and target_length == 1:
            length_penalty = 1.0 / (1.0 + word_diff * 0.25) # More aggressive penalty
        else:
            length_penalty = 1.0 / (1.0 + word_diff * 0.1)
            
        base_score *= length_penalty
            
        return base_score

    @classmethod
    def normalize_state(cls, state):
        """Normalize state to abbreviation."""
        if not state: return ""
        state = state.strip().lower()
        if len(state) == 2 and state in cls.STATE_ABBREV: return state
        for abbrev, full_name in cls.STATE_ABBREV.items():
            if state == full_name: return abbrev
        return state
    
    @classmethod
    def normalize_city(cls, city):
        """Normalize city name."""
        if not city: return ""
        city = city.strip().lower()
        if city in cls.CITY_VARIATIONS: city = cls.CITY_VARIATIONS[city]
        city = city.replace('city of ', '').replace(' city', '')
        city = city.replace('town of ', '').replace(' town', '')
        return city
    
    @classmethod
    def calculate_city_similarity(cls, query_city, target_city):
        """Calculate city similarity score."""
        if not query_city or not target_city: return 0.0
        q_city = cls.normalize_city(query_city)
        t_city = cls.normalize_city(target_city)
        
        if q_city == t_city: return 1.0
        if q_city in t_city or t_city in q_city:
            shorter = min(len(q_city), len(t_city))
            longer = max(len(q_city), len(t_city))
            return 0.7 + (0.3 * shorter / longer)
            
        q_tokens = set(q_city.split())
        t_tokens = set(t_city.split())
        if q_tokens and t_tokens:
            intersection = q_tokens.intersection(t_tokens)
            union = q_tokens.union(t_tokens)
            jaccard = len(intersection) / len(union)
            if jaccard > 0: return 0.5 + (0.5 * jaccard)
            
        seq_ratio = difflib.SequenceMatcher(None, q_city, t_city).ratio()
        if seq_ratio > 0.7: return seq_ratio
        return 0.0

    @classmethod
    def calculate_location_score(cls, query_city, query_state, target_city, target_state):
        """Calculate location similarity score."""
        q_state = cls.normalize_state(query_state)
        t_state = cls.normalize_state(target_state)
        
        state_score = 0.0
        if q_state and t_state:
            if q_state == t_state: state_score = 1.0
            else:
                state_ratio = difflib.SequenceMatcher(None, q_state, t_state).ratio()
                if state_ratio > 0.8: state_score = state_ratio
                
        city_score = cls.calculate_city_similarity(query_city, target_city)
        return (city_score * 0.6) + (state_score * 0.4)
