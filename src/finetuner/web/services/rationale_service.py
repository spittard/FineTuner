import re
import logging
from finetuner.core.llm_service import LLMService

logger = logging.getLogger(__name__)

class RationaleService:
    """
    Service for generating human-readable match rationales and explanations.
    Focuses on providing practical, actionable feedback for data entry clerks.
    """
    
    @staticmethod
    def generate_match_rationale(query, company_name, explanation, score):
        """Generate data-entry-clerk-focused explanations that are practical and actionable"""
        query_lower = query.lower()
        company_lower = company_name.lower()
        
        # Phase 0: Acronym Match Check
        match_type = explanation.get('match_type', 'hybrid')
        if match_type in ['acronym_expansion', 'acronym_reverse']:
            rationale = f"ACRONYM MATCH<br><br><b>What This Means:</b><br>The system identified a direct link between an exact acronym and its full company name.<br><br><b>Match Type:</b><br>• {match_type.replace('_', ' ').title()}<br>"
            
            # Score details for acronyms
            string_score = explanation.get('string_score', 0.0)
            sem_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
            fidelity = explanation.get('acronym_fidelity', 0.0)
            
            rationale += f"<br><b>Score Breakdown:</b><br>• Expansion Quality: {fidelity:.2f}<br>• Lexical match: {string_score:.2f}<br>• Semantic link: {sem_score:.4f}<br>"
            
            if fidelity >= 0.9:
                rationale += f"<br><b>Action Required:</b><br>• This is a HIGH CONFIDENCE acronym expansion<br>• Highly likely to be correct"
            else:
                rationale += f"<br><b>Action Required:</b><br>• Verify if the acronym '{query}' correctly represents '{company_name}'<br>• Expansion quality is moderate"
            
            rationale = RationaleService._append_location_frequency_analysis(rationale, explanation, explanation.get('match_city', ''), query, company_name)
            return rationale

        # Phase 1: Exact Match Check
        if query_lower == company_lower:
            rationale = "PERFECT MATCH<br><br><b>What This Means:</b><br>This is exactly the same company name you're looking for.<br><br><b>Action Required:</b><br>• Use this match - no further checking needed<br>• This is 100% the same company<br><br><b>Why This Happens:</b><br>• Someone entered the company name name exactly as it appears in your system<br>• This is the ideal scenario for data entry"
            rationale = RationaleService._append_location_frequency_analysis(rationale, explanation, explanation.get('match_city', ''), query, company_name)
            return rationale
        
        # Phase 2: Prefix Match Check
        if company_lower.startswith(query_lower):
            rationale = f"PREFIX MATCH<br><br><b>What This Means:</b><br>This company name starts with '{query}' and has additional information added.<br><br><b>Action Required:</b><br>• This is likely the same company with extra details<br>• Check if the additional words are just descriptive (like 'Inc', 'LLC', 'Corp')<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered just the core company name<br>• Your system has the full legal name<br>• Common in business databases where legal names include extra terms"
            rationale = RationaleService._append_location_frequency_analysis(rationale, explanation, explanation.get('match_city', ''), query, company_name)
            return rationale
        
        # Phase 3: Substring Match Check
        if query_lower in company_lower:
            rationale = f"SUBSTRING MATCH<br><br><b>What This Means:</b><br>This company name contains '{query}' somewhere within it.<br><br><b>Action Required:</b><br>• This is likely the same company<br>• Check if the surrounding words make sense<br>• If yes, use this match<br><br><b>Why This Happens:</b><br>• Someone entered a partial company name<br>• Your system has the complete name<br>• Common when people remember only part of a company name"
            rationale = RationaleService._append_location_frequency_analysis(rationale, explanation, explanation.get('match_city', ''), query, company_name)
            return rationale
        
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
            
            # Check for Full Query Coverage
            all_query_words_matched = (total_query_words > 0 and overlap_count == total_query_words)
            
            if all_query_words_matched:
                rationale = f"ALL WORDS MATCHED<br><br><b>What This Means:</b><br>Every word in your search '{query}' was found in this company name.<br><br><b>Matching Words:</b><br>• {', '.join(overlap_words)}<br>"
            else:
                rationale = f"WORD OVERLAP MATCH<br><br><b>What This Means:</b><br>{overlap_count} word(s) match exactly between your search and this company.<br><br><b>Matching Words:</b><br>• {', '.join(overlap_words)}<br>"
            
            if non_overlap_query:
                rationale += f"<br><b>Your Search Also Includes:</b><br>• {', '.join(non_overlap_query)}<br>"
            if non_overlap_company:
                rationale += f"<br><b>Company Name Also Includes:</b><br>• {', '.join(non_overlap_company)}<br>"
            
            rationale += f"<br><b>Match Strength:</b><br>• {overlap_percentage:.0f}% word overlap<br>"
            
            if overlap_percentage > 50:
                rationale += f"• This is a STRONG match - likely the same company<br>"
                rationale += f"• Action: Use this match with high confidence<br>"
            elif overlap_percentage > 25:
                rationale += f"• This is a MODERATE match - worth investigating<br>"
                rationale += f"• Action: Check if this makes business sense<br>"
            else:
                rationale += f"• This is a WEAK match - may be coincidental<br>"
                rationale += f"• Action: Verify carefully before using<br>"
            
            # Score details for overlap
            string_score = explanation.get('string_score', 0.0)
            sem_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
            rationale += f"<br><b>Score Breakdown:</b><br>"
            rationale += f"• Lexical Similarity: {string_score:.4f} (Weight: 70%)<br>"
            rationale += f"• Semantic Similarity: {sem_score:.4f} (Weight: 30%)<br>"
            
            loc_score = explanation.get('location_score', 0.0)
            if loc_score > 0.01:
                rationale += f"• Location Bonus: +{loc_score:.4f}<br>"

            record_count = explanation.get('count', 0)
            if record_count > 10:
                rationale += f"• Popularity: High record frequency ({record_count} occurrences)<br>"

            rationale += f"<br><b>Why This Happens:</b><br>• Company names often have multiple words<br>• Some words are more important than others<br>• Business names can vary in how they're written"
            
            rationale = RationaleService._append_location_frequency_analysis(rationale, explanation, explanation.get('match_city', ''), query, company_name)
            return rationale
        
        # Phase 5: Linguistic Relationship Analysis
        linguistic_relationships = []
        transformation_details = []
        practical_examples = []
        
        query_words_list = query_lower.split()
        company_words_list = company_lower.split()

        for q_word in query_words_list:
            for c_word in company_words_list:
                if q_word == c_word:
                    continue
                    
                # Check for ordinal number transformations
                if RationaleService.is_ordinal_relationship(q_word, c_word):
                    relationship_type = "ordinal transformation"
                    if q_word in ["eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"]:
                        numeric_form = RationaleService.get_numeric_ordinal(q_word)
                        transformation_details.append(f"'{q_word}' → '{numeric_form}' (ordinal number)")
                        practical_examples.append(f"Someone wrote '{q_word}' but your system has '{numeric_form}'")
                    elif c_word in ["11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th"]:
                        word_form = RationaleService.get_word_ordinal(c_word)
                        transformation_details.append(f"'{c_word}' ← '{word_form}' (ordinal number)")
                        practical_examples.append(f"Your system has '{c_word}' but someone wrote '{word_form}'")
                    linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
                
                # Check for abbreviation relationships
                elif RationaleService.is_abbreviation_relationship(q_word, c_word):
                    relationship_type = "abbreviation/expansion"
                    if len(q_word) < len(c_word):
                        transformation_details.append(f"'{q_word}' is abbreviation of '{c_word}'")
                        practical_examples.append(f"Someone used the short form '{q_word}' instead of '{c_word}'")
                    else:
                        transformation_details.append(f"'{c_word}' is abbreviation of '{q_word}'")
                        practical_examples.append(f"Your system has the short form '{c_word}' but someone wrote '{q_word}'")
                    linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
                
                # Check for contraction relationships
                elif RationaleService.is_contraction_relationship(q_word, c_word):
                    relationship_type = "contraction"
                    if "'" in q_word:
                        transformation_details.append(f"'{q_word}' is contraction of '{c_word}'")
                        practical_examples.append(f"Someone used '{q_word}' instead of '{c_word}'")
                    else:
                        transformation_details.append(f"'{c_word}' is contraction of '{q_word}'")
                        practical_examples.append(f"Your system has '{c_word}' but someone wrote '{q_word}'")
                    linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
                
                # Check for plural/singular relationships
                elif RationaleService.is_plural_relationship(q_word, c_word):
                    relationship_type = "plural/singular"
                    if q_word.endswith('s') and not c_word.endswith('s'):
                        transformation_details.append(f"'{q_word}' is plural of '{c_word}'")
                        practical_examples.append(f"Someone used '{q_word}' instead of '{c_word}'")
                    else:
                        transformation_details.append(f"'{c_word}' is plural of '{q_word}'")
                        practical_examples.append(f"Your system has '{c_word}' but someone wrote '{q_word}'")
                    linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")
                
                # Check for common word variations
                elif RationaleService.is_word_variation(q_word, c_word):
                    relationship_type = "word variation"
                    variation_type = RationaleService.get_variation_type(q_word, c_word)
                    transformation_details.append(f"'{q_word}' and '{c_word}' are {variation_type}")
                    practical_examples.append(f"Common variation between '{q_word}' and '{c_word}'")
                    linguistic_relationships.append(f"'{q_word}' ↔ '{c_word}' ({relationship_type})")

        if linguistic_relationships:
            rationale = "LINGUISTIC MATCH<br><br><b>What This Means:</b><br>The names look different but are linguistically related.<br><br><b>Key Relationships Found:</b><br>"
            for rel in linguistic_relationships:
                rationale += f"• {rel}<br>"
            
            if transformation_details:
                rationale += "<br><b>Details:</b><br>"
                for det in transformation_details:
                    rationale += f"• {det}<br>"
            
            if practical_examples:
                rationale += "<br><b>Real-World Scenario:</b><br>"
                for ex in practical_examples:
                    rationale += f"• {ex}<br>"
            
            rationale += f"<br><b>Action Required:</b><br>• Verify if this variation makes sense<br>• Likely the same company"
            rationale = RationaleService._append_location_frequency_analysis(rationale, explanation, explanation.get('match_city', ''), query, company_name)
            return rationale

        # Phase 6: Phonetic Match Check
        phonetic_analysis = RationaleService.analyze_phonetic_similarity(query, company_name)
        if phonetic_analysis:
            rationale = f"PHONETIC MATCH<br><br><b>What This Means:</b><br>The names sound similar when spoken aloud, even if spelled differently.<br><br><b>Analysis:</b><br>• {phonetic_analysis}<br><br><b>Action Required:</b><br>• Say both names out loud<br>• If they sound the same, it's likely a match<br><br><b>Why This Happens:</b><br>• Names are often entered by listening to someone speak<br>• typos can result in phonetically similar words"
            rationale = RationaleService._append_location_frequency_analysis(rationale, explanation, explanation.get('match_city', ''), query, company_name)
            return rationale
        
        # Phase 7: Semantic/Contextual Fallback
        industry_context = RationaleService.analyze_industry_context(query, company_name)
        geographic_context = RationaleService.analyze_geographic_context(query, company_name)
        
        rationale = f"SEMANTIC MATCH (Score: {score:.2f})<br><br><b>What This Means:</b><br>The AI model found a meaning-based connection, but no direct word overlap.<br>"
        
        if industry_context:
            rationale += f"<br><b>Industry Context:</b><br>• {industry_context}<br>"
        
        if geographic_context:
            rationale += f"<br><b>Geographic Context:</b><br>• {geographic_context}<br>"
            
        # Add Score Breakdown Section
        string_score = explanation.get('string_score', 0.0)
        sem_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
        loc_score = explanation.get('location_score', 0.0)
        
        score_details = f"<br><b>Score Breakdown:</b><br>"
        score_details += f"• Lexical Similarity: {string_score:.4f} (Weight: 70%)<br>"
        score_details += f"• Semantic Similarity: {sem_score:.4f} (Weight: 30%)<br>"
        
        if loc_score > 0.01:
            score_details += f"• Location Bonus: +{loc_score:.4f}<br>"

        record_count = explanation.get('count', 0)
        if record_count > 0:
            score_details += f"• Popularity Boost: Log-weighted frequency<br>"

        score_breakdown_text = RationaleService.get_score_breakdown(score)
        rationale += score_details
        rationale += f"<br><b>Confidence Level:</b><br>• {score_breakdown_text}<br><br><b>Action Required:</b><br>• This is a LOWER confidence match<br>• CAREFULLY verify if these companies are actually related<br>• Check address and other details"
        
        # --- PHASE 8: Location & Frequency Analysis (Common for ALL match types) ---
        # Note: We append this to the rationale for every match type. 
        # Ideally, we should refactor to have a common exit point, but for now we append here 
        # and we must also update the return statements in Phases 0-6 to include this.
        # Actually, let's create a helper method to append this and call it before returning in all phases.
        
        return rationale

    @staticmethod
    def _append_location_frequency_analysis(rationale, explanation, match_city, query=None, company_name=None):
        """Helper to append Semantic, Location & Frequency analysis to rationale"""
        # --- NEW: Semantic Analysis ---
        sem_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
        
        # Determine Visual Indicator
        if sem_score >= 0.8:
            indicator = "🟢"
        elif sem_score >= 0.5:
            indicator = "🟡"
        else:
            indicator = "🔴"
            
        rationale += "<br><br><b>Semantic Analysis:</b><br>"
        rationale += f"• <b>Strength:</b> {indicator} {sem_score:.2f}<br>"
        
        # Determine Base Meaning
        if sem_score >= 0.9:
            meaning = "The model detects a very strong meaning-based connection."
        elif sem_score >= 0.7:
             meaning = "The model detects a strong meaning-based connection."
        elif sem_score >= 0.5:
             meaning = "The model detects a moderate meaning-based connection."
        else:
             meaning = "The model detects a weak or incidental connection."

        # Add Specific Context if available
        context_notes = []
        if query and company_name:
            biz_ctx = RationaleService.analyze_business_context(query, company_name)
            ind_ctx = RationaleService.analyze_industry_context(query, company_name)
            
            if biz_ctx:
                context_notes.append(biz_ctx)
            if ind_ctx:
                context_notes.append(ind_ctx)

        # Add LLM-powered analysis if enabled/available
        # --- NEW: Tiered Evaluation ---
        # Skip LLM for:
        # 1. 100% Exact Matches (Lexical is enough)
        # 2. Very high confidence string matches (>0.95)
        # 3. Very low semantic score (<0.2) where it's clearly not a match
        
        string_score = explanation.get('string_score', 0.0)
        sem_score = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
        fidelity = explanation.get('acronym_fidelity', 0.0)
        
        should_use_llm = True
        # Only skip LLM if we have high confidence in BOTH lexical and semantic similarity
        # If string is high but semantic is low (or vice versa), we want the AI to resolve the discrepancy.
        if string_score > 0.98 and sem_score > 0.85:
            should_use_llm = False
        
        # Still skip for very low significance cases
        if string_score < 0.1 and sem_score < 0.1:
            should_use_llm = False

        # However, always use LLM for Acronyms if fidelity is not perfect
        if fidelity > 0.2 and fidelity < 0.95:
            should_use_llm = True
            
        llm_context = None
        if should_use_llm:
            llm_context = LLMService.analyze_semantic_context(query, company_name)
            
        if llm_context:
            rationale += f"• <b>AI Insight:</b> {llm_context}<br>"

        if context_notes:
            rationale += f"• <b>Meaning:</b> {meaning}<br>"
            for note in context_notes:
                rationale += f"• <b>Context:</b> {note}<br>"
        else:
            rationale += f"• <b>Meaning:</b> {meaning}<br>"

        loc_boost = explanation.get('location_boost', 0.0)
        pop_boost = explanation.get('popularity_boost', 0.0)
        record_count = explanation.get('count', 0)
        
        rationale += "<br><b>Location & Frequency Analysis:</b><br>"
        
        # Location Impact
        # Location Impact
        loc_score = explanation.get('location_score', 0.0)
        
        if loc_boost > 0:
            if loc_score >= 0.9:
                match_desc = f"The record's location ({match_city}) matches the city and state requirements."
            elif loc_score >= 0.4: # State match threshold
                match_desc = f"The record's location ({match_city}) is in the same region/state, providing a partial boost."
            else:
                match_desc = f"The record's location ({match_city}) provides a minor geographic boost."
                
            rationale += f"• <b>Location:</b> <span style='color: green;'>Positive Impact (+{loc_boost*100:.2f}%)</span>. {match_desc}<br>"
        elif match_city and not loc_boost:
             rationale += f"• <b>Location:</b> No impact. Location present but did not boost score (likely loose match or ignored).<br>"
        else:
             rationale += f"• <b>Location:</b> N/A (No location context used).<br>"
             
        # Frequency Impact
        if pop_boost > 0:
            if record_count > 10:
                rationale += f"• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+{pop_boost*100:.2f}%)</span>. This is a high-frequency record ({record_count:,} occurrences), suggesting it is a well-known entity.<br>"
            else:
                rationale += f"• <b>Frequency:</b> <span style='color: green;'>Positive Impact (+{pop_boost*100:.2f}%)</span>. Frequency boost applied ({record_count} occurrences).<br>"
        elif record_count > 10:
             # High count but maybe no boost (e.g. maxed out or not configured?) - usually implies boost if configured
             rationale += f"• <b>Frequency:</b> Neutral. Record appears {record_count:,} times (High), but popularity boost was minimal or capped.<br>"
        elif record_count > 1:
             rationale += f"• <b>Frequency:</b> Neutral. Record appears {record_count} times, but not enough to trigger significant popularity boost (>10 needed).<br>"
        else:
             rationale += f"• <b>Frequency:</b> Neutral (Single occurrence).<br>"
             
        return rationale
        
    @staticmethod
    def get_short_summary(query, company_name, explanation):
        """Returns a single line summary of why this matched"""
        query_l = query.lower().strip()
        comp_l = company_name.lower().strip()
        mtype = explanation.get('match_type', 'hybrid')
        
        if query_l == comp_l:
            return "Perfect character-for-character match."
        
        if mtype == 'acronym_expansion':
            return f"Detected as a literal expansion of acronym '{query.upper()}'."
        if mtype == 'acronym_reverse':
            return f"Matched based on generated acronym '{company_name.upper()}'."
        
        if comp_l.startswith(query_l):
            return "Direct prefix match (target contains extra trailing words)."
        
        if query_l in comp_l:
            return "Substring match (target contains query text)."
            
        fidelity = explanation.get('acronym_fidelity', 0.0)
        if fidelity > 0.8:
            return f"Strong acronym pattern detected ({fidelity:.2f} fidelity)."
            
        overlap = explanation.get('overlap_score', 0.0)
        if overlap > 0.7:
            return "High word-for-word overlap."
            
        sem = explanation.get('normalized_semantic_score', explanation.get('semantic_score', 0.0))
        if sem > 0.85:
            return "Matched via strong semantic/conceptual similarity."
            
        return "Hybrid match based on combined lexical and semantic features."

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
        string_contrib = string_score * 0.7
        breakdown += f"| String Similarity | {string_score:.4f} | 70% | {string_contrib:.4f} |\n"
        
        # Semantic similarity
        sem_contrib = semantic_score_norm * 0.3
        breakdown += f"| Semantic Similarity (Normalized) | {semantic_score_norm:.4f} | 30% | {sem_contrib:.4f} |\n"
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
        breakdown += "Base Score = (String Similarity × 0.70) + (Semantic Similarity × 0.30)\n"
        
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
