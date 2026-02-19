import re
import logging

logger = logging.getLogger(__name__)

class RationaleService:
    """
    Service for generating human-readable match rationales and explanations.
    Focuses on providing practical, actionable feedback for data entry clerks.
    """
    
    @staticmethod
    def generate_match_rationale(query, company_name, explanation, score):
        """
        Generate match rationale explaining WHY a candidate is a good or bad match.
        
        Structure:
        1. Verdict Banner - Single line classification
        2. Match Classification - What type of match
        3. Evidence - Score breakdown with interpretations
        4. Concept Analysis - Industry/category breakdown (if available)
        """
        query_lower = query.lower()
        company_lower = company_name.lower()
        
        # Extract all scores
        string_score = explanation.get('string_score', 0.0)
        sem_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
        concept_align = explanation.get('concept_alignment', 0.0)
        fidelity = explanation.get('acronym_fidelity', 0.0)
        loc_boost = explanation.get('location_boost', 0.0)
        pop_boost = explanation.get('popularity_boost', 0.0)
        match_type = explanation.get('match_type', 'hybrid')
        
        # Calculate final score percentage
        score_pct = round(score * 100)
        
        # --- VERDICT BANNER ---
        verdict = RationaleService._get_verdict_banner(score_pct, string_score, sem_score, query_lower, company_lower, match_type)
        
        # --- MATCH CLASSIFICATION ---
        classification = RationaleService._get_match_classification(query, company_name, match_type, fidelity)
        
        # --- EVIDENCE TABLE ---
        evidence = RationaleService._build_evidence_section(explanation, query, company_name)
        
        # --- CONCEPT ANALYSIS ---
        concept_section = ""
        concept_sig = explanation.get('concept_signature')
        if concept_sig:
            concept_section = RationaleService._format_concept_analysis(concept_sig)
        
        # Combine all sections
        rationale = f"{verdict}<br><br>{classification}<br><br>{evidence}"
        if concept_section:
            rationale += f"<br>{concept_section}"
            
        return rationale
    
    @staticmethod
    def _get_verdict_banner(score_pct, string_score, sem_score, query_lower, company_lower, match_type):
        """Generate single-line verdict banner: [ICON] [STRENGTH] MATCH ([PERCENT]%) - [REASON]"""
        # Determine verdict level
        if score_pct >= 95 or query_lower == company_lower:
            icon = "✅"
            level = "EXCELLENT"
            color = "#00ff00"
        elif score_pct >= 80:
            icon = "✅"
            level = "STRONG"
            color = "#00cc00"
        elif score_pct >= 60:
            icon = "⚠️"
            level = "MODERATE"
            color = "#ffaa00"
        elif score_pct >= 40:
            icon = "⚠️"
            level = "WEAK"  
            color = "#ff6600"
        else:
            icon = "❌"
            level = "POOR"
            color = "#ff0000"
        
        # Generate reason based on what drove the score
        if query_lower == company_lower:
            reason = "Exact Name Match"
        elif match_type == 'acronym_expansion':
            reason = "Acronym Expansion"
        elif string_score >= 0.85:
            reason = "High Lexical Similarity"
        elif sem_score >= 0.85:
            reason = "High Semantic Similarity"
        elif string_score >= 0.7:
            reason = "Moderate Lexical Similarity"
        elif sem_score >= 0.7:
            reason = "Moderate Semantic Similarity"
        else:
            reason = "Partial Composite Match"
            
        return f"<div style='border: 1px solid {color}; border-left: 10px solid {color}; padding: 15px; background: rgba(0,0,0,0.1); border-radius: 4px;'>" \
               f"<span style='font-size:1.4em; font-weight:bold; color:{color};'>{icon} {level} MATCH ({score_pct}%)</span><br>" \
               f"<span style='color:#eee; font-size:1.1em;'>{reason}</span></div>"
    
    @staticmethod
    def _get_match_classification(query, company_name, match_type, fidelity):
        """Explain the nature of the match relationship."""
        query_lower = query.lower()
        company_lower = company_name.lower()
        
        if query_lower == company_lower:
            return f"<b>Relationship:</b> This is an <b>Exact Identity Match</b>. The query and candidate name are character-identical, representing a perfect lexical link."
        
        if match_type in ['acronym_expansion', 'acronym_reverse']:
            quality = "high" if fidelity >= 0.9 else "moderate"
            return f"<b>Relationship:</b> This is an <b>Acronym Expansion</b>. The system identified '{query}' as a {quality} fidelity match for the initials of '{company_name}'."
        
        if company_lower.startswith(query_lower) or query_lower in company_lower:
            return f"<b>Relationship:</b> This is a <b>Lexical Substring Match</b>. The query appears as a direct fragment within the candidate name, suggesting a strong partial identity."
        
        # Check word overlap
        query_words = set(query_lower.split())
        company_words = set(company_lower.split())
        overlap = query_words.intersection(company_words)
        
        if overlap:
            return f"<b>Relationship:</b> This is a <b>Hybrid Word Overlap</b>. The system detected shared keywords ('{', '.join(sorted(overlap))}') despite differences in overall string structure."
        
        return f"<b>Relationship:</b> This is a <b>Pure Semantic Match</b>. There is no direct text overlap; the connection is based entirely on the underlying business context and meaning."
    
    @staticmethod
    def _build_evidence_section(explanation, query, company_name):
        """Build the evidence section with descriptive interpretations for each component."""
        string_score = explanation.get('string_score', 0.0)
        sem_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
        concept_align = explanation.get('concept_alignment', 0.0)
        loc_boost = explanation.get('location_boost', 0.0)
        pop_boost = explanation.get('popularity_boost', 0.0)
        record_count = explanation.get('count', 0)
        city = explanation.get('city', '')
        state = explanation.get('state', '')
        
        def pick_badge(val):
            if val >= 0.9: return "🟢 EXCELLENT"
            if val >= 0.7: return "🟢 GOOD"
            if val >= 0.5: return "🟡 MODERATE"
            if val >= 0.3: return "🟠 FAIR"
            return "🔴 WEAK"

        evidence = "<b>Evidence Analysis:</b><br>"
        
        # Name Similarity
        ns_reason = "identical strings" if string_score >= 1.0 else "strong character overlap" if string_score >= 0.85 else "partial character alignment"
        evidence += f"• <b>Name Similarity:</b> {pick_badge(string_score)} ({string_score:.0%}) — Based on {ns_reason}.<br>"
        
        # Semantic Link
        sl_reason = "synonymous concepts" if sem_score >= 0.85 else "strong contextual link" if sem_score >= 0.7 else "moderate meaning-based connection"
        evidence += f"• <b>Semantic Link:</b> {pick_badge(sem_score)} ({sem_score:.0%}) — Detected via {sl_reason}.<br>"
        
        # Concept Alignment
        if concept_align > 0.1:
            ca_reason = "highly aligned industries" if concept_align >= 0.85 else "related business categories"
            evidence += f"• <b>Concept Alignment:</b> {pick_badge(concept_align)} ({concept_align:.0%}) — Reflects {ca_reason}.<br>"
        
        # Location
        location_str = ", ".join(filter(None, [city, state]))
        if loc_boost > 0:
            evidence += f"• <b>Location Match:</b> 🟢 <b>+{loc_boost*100:.1f}% Boost</b> — Geographic criteria confirmed in {location_str}.<br>"
        elif location_str:
            evidence += f"• <b>Location Data:</b> ⚪ <b>Neutral</b> — Found {location_str} but no boost was warranted.<br>"
        
        # Popularity
        if pop_boost > 0:
            evidence += f"• <b>Entity Popularity:</b> 🟢 <b>+{pop_boost*100:.1f}% Boost</b> — Higher confidence due to {record_count:,} occurrences in master set.<br>"
        elif record_count > 1:
            evidence += f"• <b>Entity Frequency:</b> ⚪ <b>Neutral</b> — Found {record_count} occurrences, which is common but not dominant.<br>"
            
        return evidence
    
    @staticmethod
    def generate_concise_rationale(query, company_name, explanation, score):
        """Generate a 1-line concise summary."""
        query_lower = query.lower()
        company_lower = company_name.lower()
        score_pct = round(score * 100)
        
        # Determine verdict
        if score_pct >= 90 or query_lower == company_lower:
            verdict = "Excellent Match"
        elif score_pct >= 75:
            verdict = "Strong Match"
        elif score_pct >= 50:
            verdict = "Moderate Match"
        else:
            verdict = "Weak Match"
            
        return f"{verdict} ({score_pct}%) — Based on {explanation.get('match_type', 'hybrid')} analysis."





        

    # Helper methods (made static)
    @staticmethod
    def is_ordinal_relationship(word1, word2):
        """Check if two words are ordinal number variations"""
        ordinals = {
            "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
            "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
            "nineteenth": "19th", "twentieth": "20th", "twenty-first": "21st", "twenty-second": "22nd"
        }
        
        if word1 in ordinals and word2 == ordinals[word1]:
            return True
        if word2 in ordinals and word1 == ordinals[word2]:
            return True
        
        if word1.isdigit() and word2.endswith(('st', 'nd', 'rd', 'th')):
            return True
        if word2.isdigit() and word1.endswith(('st', 'nd', 'rd', 'th')):
            return True
        
        return False

    @staticmethod
    def get_numeric_ordinal(word):
        """Convert word ordinal to numeric form"""
        ordinals = {
            "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
            "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
            "nineteenth": "19th", "twentieth": "20th"
        }
        return ordinals.get(word, word)

    @staticmethod
    def get_word_ordinal(word):
        """Convert numeric ordinal to word form"""
        ordinals = {
            "11th": "eleventh", "12th": "twelfth", "13th": "thirteenth", "14th": "fourteenth",
            "15th": "fifteenth", "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
            "19th": "nineteenth", "20th": "twentieth"
        }
        return ordinals.get(word, word)

    @staticmethod
    def is_abbreviation_relationship(word1, word2):
        """Check if one word is an abbreviation of another"""
        if len(word1) < len(word2) and word1 in word2:
            return True
        if len(word2) < len(word1) and word2 in word1:
            return True
        return False

    @staticmethod
    def is_contraction_relationship(word1, word2):
        """Check if words are contractions of each other"""
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

    @staticmethod
    def is_plural_relationship(word1, word2):
        """Check if words are plural/singular forms of each other"""
        if word1.endswith('s') and word1[:-1] == word2:
            return True
        if word2.endswith('s') and word2[:-1] == word1:
            return True
        return False

    @staticmethod
    def is_word_variation(word1, word2):
        """Check for common word variations"""
        variations = [
            ("info", "information"), ("tech", "technical"), ("assoc", "association"),
            ("corp", "corporation"), ("co", "company"), ("inc", "incorporated"),
            ("ltd", "limited"), ("intl", "international"), ("mgmt", "management")
        ]
        
        for var1, var2 in variations:
            if (word1 == var1 and word2 == var2) or (word1 == var2 and word2 == var1):
                return True
        
        return False

    @staticmethod
    def get_variation_type(word1, word2):
        """Get the type of word variation"""
        if len(word1) < len(word2):
            return "abbreviation/expansion pair"
        elif len(word2) < len(word1):
            return "abbreviation/expansion pair"
        else:
            return "synonym pair"

    @staticmethod
    def analyze_phonetic_similarity(query, company_name):
        """Analyze phonetic similarity between query and company name"""
        query_sound = RationaleService.get_simple_phonetic(query.lower())
        company_sound = RationaleService.get_simple_phonetic(company_name.lower())
        
        if query_sound == company_sound:
            return "Identical phonetic representation"
        elif query_sound in company_sound or company_sound in query_sound:
            return "Partial phonetic overlap detected"
        
        return None

    @staticmethod
    def get_simple_phonetic(text):
        """Get a simple phonetic representation of text"""
        phonetic = text.replace('ph', 'f').replace('ck', 'k').replace('qu', 'kw')
        phonetic = ''.join(c for c in phonetic if c.isalpha())
        return phonetic

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def analyze_business_context(query, company_name):
        """Analyze business context and corporate terminology"""
        business_keywords = {
            'religious': ['church', 'synagogue', 'temple', 'ministry', 'messianic', 'assemblies', 'catholic'],
            'non_profit': ['association', 'foundation', 'club', 'society', 'coalition', 'initiative', 'center', 'charity'],
            'technology': ['tech', 'technology', 'digital', 'software', 'systems'],
            'financial': ['financial', 'finance', 'capital', 'investment', 'funds'],
            'consulting': ['consulting', 'consultants', 'advisory', 'services'],
            'healthcare': ['health', 'medical', 'hospital', 'clinic', 'care', 'nursing'],
            'corporate': ['corp', 'corporation', 'incorporated', 'inc', 'llc', 'ltd', 'limited'],
            'partnership': ['partners', 'partnership', 'associates', 'assoc'],
            'international': ['intl', 'international', 'global', 'worldwide']
        }
        
        def get_contexts(text):
            found = set()
            text_lower = text.lower()
            for b_type, keywords in business_keywords.items():
                if any(k in text_lower for k in keywords):
                    found.add(b_type)
            
            # Refine Priority: Specific trumps Generic
            if 'religious' in found:
                found.discard('non_profit') # Religious usually implies NP, but let's be specific
                found.discard('corporate')
            if 'non_profit' in found:
                found.discard('corporate') # e.g. "Association Inc" -> Just Association context
            if 'healthcare' in found:
                found.discard('corporate')
            
            return found

        q_ctx = get_contexts(query)
        c_ctx = get_contexts(company_name)
        
        common = q_ctx.intersection(c_ctx)
        
        if common:
            # Pick the most specific one to mention
            priority_order = ['religious', 'healthcare', 'non_profit', 'technology', 'financial', 'consulting']
            for p in priority_order:
                if p in common:
                    return f"Both have {p} indicators - strong alignment"
            return f"Both have {'/'.join(common)} indicators"
            
        if q_ctx and c_ctx:
            return f"Different contexts: {', '.join(q_ctx)} vs {', '.join(c_ctx)}"
        
        if q_ctx:
             return f"Query indicates {', '.join(q_ctx)} context"
        if c_ctx:
             return f"Match indicates {', '.join(c_ctx)} context"
        
        return None

    @staticmethod
    def get_visual_indicator(score):
        """Get visual strength indicator (circle)"""
        if score >= 0.8:
            return "🟢"
        elif score >= 0.5:
            return "🟡"
        else:
            return "🔴"

    @staticmethod
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

    @staticmethod
    def get_enhanced_phonetic_similarity(query, company_name):
        """Enhanced phonetic analysis with multiple algorithms"""
        # Basic phonetic
        basic_phonetic = RationaleService.analyze_phonetic_similarity(query, company_name)
        
        # Soundex-like analysis
        query_soundex = RationaleService.get_soundex(query.lower())
        company_soundex = RationaleService.get_soundex(company_name.lower())
        
        if query_soundex == company_soundex:
            return "Identical phonetic codes (Soundex)"
        elif basic_phonetic:
            return f"{basic_phonetic} | Soundex codes: {query_soundex} vs {company_soundex}"
        
        return f"Soundex codes: {query_soundex} vs {company_soundex}"

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def generate_detailed_score_breakdown(match_data, query):
        """
        Generate comprehensive score breakdown showing all components.
        
        Args:
            match_data: Dictionary containing all match information including scores
            query: Original query string
            
        Returns:
            Formatted string with complete score breakdown
        """
        breakdown = "## Complete Score Breakdown\n\n"
        
        # Extract all score components
        final_score = match_data.get('raw_score', match_data.get('score', 0.0))
        string_score = match_data.get('string_score', 0.0)
        semantic_score_raw = match_data.get('semantic_score', 0.0)
        semantic_score_norm = match_data.get('normalized_semantic_score', semantic_score_raw)
        acronym_fidelity = match_data.get('acronym_fidelity', 0.0)
        location_score = match_data.get('location_score', 0.0)
        name_score = match_data.get('name_score', 0.0)
        
        # Score components table
        breakdown += "| Component | Raw Value | Weight | Contribution |\n"
        breakdown += "|-----------|-----------|--------|-------------|\n"
        
        # String similarity
        string_contrib = string_score * 0.5
        breakdown += f"| String Similarity | {string_score:.4f} | 50% | {string_contrib:.4f} |\n"
        
        # Semantic similarity
        sem_contrib = semantic_score_norm * 0.25
        breakdown += f"| Semantic Similarity (Normalized) | {semantic_score_norm:.4f} | 25% | {sem_contrib:.4f} |\n"
        
        # Concept Alignment
        concept_align = match_data.get('concept_alignment', 0.0)
        concept_contrib = concept_align * 0.25
        breakdown += f"| Concept Alignment (Scanner) | {concept_align:.4f} | 25% | {concept_contrib:.4f} |\n"
        
        breakdown += f"| Semantic Similarity (Raw) | {semantic_score_raw:.4f} | - | - |\n"
        
        # Base name score
        # Use name_score from match_data if available (it accounts for exact matches and tiered overrides)
        base_score = match_data.get('name_score', string_contrib + sem_contrib)
        
        # Account for tiered overrides in the breakdown
        if match_data.get('match_type') == 'exact' and base_score < 1.0:
             base_score = 1.0
             
        breakdown += f"| **Base Score (Name)** | **{base_score:.4f}** | - | - |\n"
        
        # Acronym fidelity boost
        if acronym_fidelity > 0.0:
            # Check if this was a Phase 0 acronym expansion or a boost
            if match_data.get('match_type') == 'acronym_expansion' or match_data.get('match_type') == 'acronym_reverse':
                 breakdown += f"| Acronym Fidelity contribution | {acronym_fidelity:.4f} | Built-in | (Included in Base) |\n"
            else:
                 acronym_boost = acronym_fidelity * 0.15
                 # Check if it was actually applied (if base_score + boost > base_score)
                 breakdown += f"| Acronym Fidelity Boost | {acronym_fidelity:.4f} | 15% max | +{acronym_boost:.4f} |\n"
        
        # Location boost
        loc_boost_val = match_data.get('location_boost', 0.0)
        if loc_boost_val > 0.0:
            breakdown += f"| Location Context Boost | {location_score:.4f} | 5-20% | +{loc_boost_val:.4f} |\n"
        
        # Frequency boost - Explicit Visualization
        pop_boost = match_data.get('popularity_boost', 0.0)
        record_count = match_data.get('count', 0)
        if pop_boost > 0.0 or record_count > 1:
            breakdown += f"| Frequency Impact | {record_count:,} records | ~2-5% | +{pop_boost:.4f} |\n"
        
        # Final score
        breakdown += f"| **FINAL SCORE** | **{final_score:.4f}** | - | **{final_score*100:.1f}%** |\n\n"
        
        # Formula explanation
        breakdown += "### Score Calculation Formula\n\n"
        breakdown += "```\n"
        breakdown += "Base Score = (String Sim × 0.50) + (Semantic Sim × 0.25) + (Concept Alignment × 0.25)\n"
        
        if acronym_fidelity > 0.0:
            breakdown += f"Acronym Boost = Acronym Fidelity × 0.15 = {acronym_fidelity:.4f} × 0.15 = {acronym_fidelity * 0.15:.4f}\n"
        
        if location_score > 0.0:
            breakdown += f"Location Boost = Location Score × 0.05 = {location_score:.4f} × 0.05 = {location_score * 0.05:.4f}\n"
        
        breakdown += f"\nFinal Score = Base Score"
        if acronym_fidelity > 0.0:
            breakdown += " + Acronym Boost"
        if location_score > 0.0:
            breakdown += " + Location Boost"
        breakdown += f" = {final_score:.4f}\n"
        breakdown += "```\n\n"
        
        # Component analysis
        breakdown += "### Component Analysis\n\n"
        
        if string_score >= 0.95:
            breakdown += "- **String Similarity (EXCELLENT):** Nearly perfect lexical match - words align very closely\n"
        elif string_score >= 0.80:
            breakdown += "- **String Similarity (VERY GOOD):** Strong lexical match - most words align well\n"
        elif string_score >= 0.60:
            breakdown += "- **String Similarity (GOOD):** Moderate lexical match - significant word overlap\n"
        elif string_score >= 0.40:
            breakdown += "- **String Similarity (FAIR):** Some lexical similarity - partial word overlap\n"
        else:
            breakdown += "- **String Similarity (WEAK):** Low lexical match - minimal word overlap\n"
        
        if semantic_score_norm >= 0.90:
            breakdown += "- **Semantic Similarity (EXCELLENT):** Very strong meaning-based connection\n"
        elif semantic_score_norm >= 0.70:
            breakdown += "- **Semantic Similarity (VERY GOOD):** Strong meaning-based connection\n"
        elif semantic_score_norm >= 0.50:
            breakdown += "- **Semantic Similarity (GOOD):** Moderate meaning-based connection\n"
        elif semantic_score_norm >= 0.30:
            breakdown += "- **Semantic Similarity (FAIR):** Some meaning-based connection\n"
        else:
            breakdown += "- **Semantic Similarity (WEAK):** Weak meaning-based connection\n"
        
        if acronym_fidelity > 0.0:
            if acronym_fidelity >= 0.90:
                breakdown += f"- **Acronym Fidelity (EXCELLENT):** {acronym_fidelity:.2f} - Highly likely literal acronym expansion\n"
            elif acronym_fidelity >= 0.70:
                breakdown += f"- **Acronym Fidelity (GOOD):** {acronym_fidelity:.2f} - Probable acronym expansion\n"
            else:
                breakdown += f"- **Acronym Fidelity (MODERATE):** {acronym_fidelity:.2f} - Possible acronym connection\n"
        
        if location_score > 0.0:
            if location_score >= 0.90:
                breakdown += f"- **Location Match (EXCELLENT):** {location_score:.2f} - Strong geographic match\n"
            elif location_score >= 0.50:
                breakdown += f"- **Location Match (GOOD):** {location_score:.2f} - Good geographic alignment\n"
            else:
                breakdown += f"- **Location Match (PARTIAL):** {location_score:.2f} - Some geographic relevance\n"
        
        return breakdown

    @staticmethod
    def generate_relative_positioning_explanation(current_match, match_above, match_below, rank):
        """
        Generate explanation of why this match is ranked where it is relative to others.
        
        Args:
            current_match: Current match data dictionary
            match_above: Match ranked above (or None if rank 1)
            match_below: Match ranked below (or None if last)
            rank: Current rank position (1-indexed)
            
        Returns:
            Formatted string explaining relative positioning
        """
        explanation = f"## Relative Positioning Analysis (Rank #{rank})\n\n"
        
        current_score = current_match.get('raw_score', current_match.get('score', 0.0))
        current_name = current_match.get('company_name', 'Unknown')
        
        # Compare with match above
        if match_above:
            above_score = match_above.get('raw_score', match_above.get('score', 0.0))
            above_name = match_above.get('company_name', 'Unknown')
            score_diff = above_score - current_score
            
            explanation += f"### Why Ranked Below #{rank-1}: \"{above_name}\"\n\n"
            explanation += f"**Score Difference:** {score_diff:.4f} ({score_diff*100:.2f} percentage points)\n\n"
            
            # Identify key differentiators
            differentiators = []
            
            # String score comparison
            current_string = current_match.get('string_score', 0.0)
            above_string = match_above.get('string_score', 0.0)
            if abs(above_string - current_string) > 0.05:
                diff = above_string - current_string
                differentiators.append(f"String Similarity: {above_string:.4f} vs {current_string:.4f} (Δ {diff:+.4f})")
            
            # Semantic score comparison
            current_sem = current_match.get('normalized_semantic_score', current_match.get('semantic_score', 0.0))
            above_sem = match_above.get('normalized_semantic_score', match_above.get('semantic_score', 0.0))
            if abs(above_sem - current_sem) > 0.05:
                diff = above_sem - current_sem
                differentiators.append(f"Semantic Similarity: {above_sem:.4f} vs {current_sem:.4f} (Δ {diff:+.4f})")
            
            # Acronym fidelity comparison
            current_acro = current_match.get('acronym_fidelity', 0.0)
            above_acro = match_above.get('acronym_fidelity', 0.0)
            if current_acro > 0.0 or above_acro > 0.0:
                if abs(above_acro - current_acro) > 0.01:
                    diff = above_acro - current_acro
                    differentiators.append(f"Acronym Fidelity: {above_acro:.4f} vs {current_acro:.4f} (Δ {diff:+.4f})")
            
            # Location score comparison
            current_loc = current_match.get('location_score', 0.0)
            above_loc = match_above.get('location_score', 0.0)
            if current_loc > 0.0 or above_loc > 0.0:
                if abs(above_loc - current_loc) > 0.01:
                    diff = above_loc - current_loc
                    differentiators.append(f"Location Score: {above_loc:.4f} vs {current_loc:.4f} (Δ {diff:+.4f})")
            
            # Concept alignment comparison
            current_ca = current_match.get('concept_alignment', 0.0)
            above_ca = match_above.get('concept_alignment', 0.0)
            if abs(above_ca - current_ca) > 0.05:
                diff = above_ca - current_ca
                differentiators.append(f"Concept Alignment: {above_ca:.4f} vs {current_ca:.4f} (Δ {diff:+.4f})")
            
            if differentiators:
                explanation += "**Key Differentiators:**\n"
                for diff in differentiators:
                    explanation += f"- {diff}\n"
            else:
                explanation += "**Key Differentiators:** Scores are very similar - minor differences across components\n"
            
            explanation += "\n"
        else:
            explanation += "### Top Ranked Match\n\n"
            explanation += "This is the highest-scoring match for this query.\n\n"
        
        # Compare with match below
        if match_below:
            below_score = match_below.get('raw_score', match_below.get('score', 0.0))
            below_name = match_below.get('company_name', 'Unknown')
            score_diff = current_score - below_score
            
            explanation += f"### Why Ranked Above #{rank+1}: \"{below_name}\"\n\n"
            explanation += f"**Score Advantage:** {score_diff:.4f} ({score_diff*100:.2f} percentage points)\n\n"
            
            # Identify key advantages
            advantages = []
            
            # String score comparison
            current_string = current_match.get('string_score', 0.0)
            below_string = match_below.get('string_score', 0.0)
            if abs(current_string - below_string) > 0.05:
                diff = current_string - below_string
                advantages.append(f"String Similarity: {current_string:.4f} vs {below_string:.4f} (Δ {diff:+.4f})")
            
            # Semantic score comparison
            current_sem = current_match.get('normalized_semantic_score', current_match.get('semantic_score', 0.0))
            below_sem = match_below.get('normalized_semantic_score', match_below.get('semantic_score', 0.0))
            if abs(current_sem - below_sem) > 0.05:
                diff = current_sem - below_sem
                advantages.append(f"Semantic Similarity: {current_sem:.4f} vs {below_sem:.4f} (Δ {diff:+.4f})")
            
            # Acronym fidelity comparison
            current_acro = current_match.get('acronym_fidelity', 0.0)
            below_acro = match_below.get('acronym_fidelity', 0.0)
            if current_acro > 0.0 or below_acro > 0.0:
                if abs(current_acro - below_acro) > 0.01:
                    diff = current_acro - below_acro
                    advantages.append(f"Acronym Fidelity: {current_acro:.4f} vs {below_acro:.4f} (Δ {diff:+.4f})")
            
            # Location score comparison
            current_loc = current_match.get('location_score', 0.0)
            below_loc = match_below.get('location_score', 0.0)
            if current_loc > 0.0 or below_loc > 0.0:
                if abs(current_loc - below_loc) > 0.01:
                    diff = current_loc - below_loc
                    advantages.append(f"Location Score: {current_loc:.4f} vs {below_loc:.4f} (Δ {diff:+.4f})")
            
            if advantages:
                explanation += "**Key Advantages:**\n"
                for adv in advantages:
                    explanation += f"- {adv}\n"
            else:
                explanation += "**Key Advantages:** Scores are very similar - minor advantages across components\n"
            
            explanation += "\n"
        
        return explanation

    @staticmethod
    def _format_concept_analysis(signature):
        """
        Format the raw concept signature into a 'Nutritional Label' HTML block.
        
        Args:
            signature: List of floats (the concept signature)
            
        Returns:
            HTML string
        """
        if not signature:
            return ""
            
        # Anchor mapping (MUST MATCH CompanyMatcher.CONCEPT_ANCHORS)
        anchors = {
            "Geography": ["Pennsylvania", "London", "Canada", "California", "New York", "Texas", "Chicago", "Illinois", "Ohio", "Miami", "Paris"],
            "Industry": ["Automotive", "Medical", "Technology", "Construction", "Legal", "Food", "Finance", "Education", "Insurance", "Retail", "Manufacturing"],
            "Structure": ["Corporate", "Non-Profit", "Government", "Small Business"],
            "Nature": ["Global", "Local", "Industrial", "Consumer", "Professional"]
        }
        
        # Flattened list of names for index lookup
        anchor_names = []
        for cat in anchors.values():
            anchor_names.extend(cat)
            
        if len(signature) != len(anchor_names):
            return f"<br><i>[Concept Analysis Unavailable: Signature length mismatch {len(signature)} vs {len(anchor_names)}]</i><br>"
            
        # Find top matches in each category
        html = "<br><b>Concept Analysis:</b><br>"
        
        offset = 0
        for category, names in anchors.items():
            category_scores = []
            for i, name in enumerate(names):
                category_scores.append((name, signature[offset + i]))
            offset += len(names)
            
            # Filter for meaningful matches (> 0.25 similarity)
            top_matches = sorted([s for s in category_scores if s[1] > 0.25], key=lambda x: x[1], reverse=True)
            
            if top_matches:
                items_html = []
                for name, score in top_matches:
                    percent = score * 100
                    items_html.append(f"✅ {name} {percent:.1f}%")
                
                html += f"• {category}: {', '.join(items_html)}<br>"
        
        # Simple Insight logic based on top category
        all_matches = sorted([(n, s) for n, s in zip(anchor_names, signature) if s > 0.3], key=lambda x: x[1], reverse=True)
        if all_matches:
            top_name, _ = all_matches[0]
            insight = f"The model detects a strong '{top_name}' influence in the company's semantic vector."
            if "London" in top_name or "Canada" in top_name:
                insight = "The vector is strongly pulled toward geographic anchors, resolving potential ambiguity."
            elif "Automotive" in top_name:
                insight = "Confirms the model's inherent knowledge of industry concepts."
                
            html += f"• Insight: {insight}<br>"
            
        return html
