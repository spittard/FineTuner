
import os
import time
import json
from finetuner.core.matcher import CompanyMatcher
from finetuner.web.services.rationale_service import RationaleService

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None, unit=None, ncols=None, **kwargs):
        return iterable

class SearchService:
    """
    Service for orchestrating company searches, managing the matcher instance,
    and handling data loading/caching.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.matcher = None
        self.company_data_loaded = False
        self.last_data_check = 0
        self.data_check_interval = 5
        self._loading = False
        self._initialized = True
        
    def load_company_data(self, force_reload=False, model_name='paraphrase-MiniLM-L3-v2', filename=None):
        """Load company data and initialize the matcher"""
        
        # If no filename provided, try to find the best available file
        if filename is None:
            if os.path.exists('companies_with_location.json'):
                filename = 'companies_with_location.json'
            else:
                filename = 'companies.json'
        
        # Early return if data is already loaded and we don't need to force reload
        if not force_reload and self.company_data_loaded and self.matcher is not None:
            # If current model matches requested model, we're good
            if self.matcher.model_name == model_name:
                return True
        
        current_time = time.time()
        
        # Prevent multiple rapid calls to this function
        if not force_reload and self.company_data_loaded and self.matcher is not None:
            if self.matcher.model_name == model_name:
                # Check if companies.json has been modified
                if current_time - self.last_data_check < self.data_check_interval:
                    return True
            
            try:
                # Check if file has been modified
                if os.path.exists(filename):
                    file_mtime = os.path.getmtime(filename)
                    if hasattr(self.matcher, '_last_file_mtime') and self.matcher._last_file_mtime == file_mtime:
                        self.last_data_check = current_time
                        return True
            except:
                pass
        
        # Add a guard to prevent multiple simultaneous loads
        if self._loading:
            print("Already loading company data, skipping...")
            return self.company_data_loaded
        
        self._loading = True
        
        try:
            if not os.path.exists(filename):
                print(f"Error: {filename} not found")
                self._loading = False
                return False
            
            # Step 1: Initialize CompanyMatcher first
            if self.matcher is None or self.matcher.model_name != model_name:
                print(f"Initializing CompanyMatcher with model: {model_name}...")
                self.matcher = CompanyMatcher(model_name=model_name)
            
            # Step 2: FAST CACHE CHECK - Try to load from cache using file metadata
            # This avoids loading the 176MB+ JSON file if we already have it indexed
            file_cache_key_reg = self.matcher.get_cache_key_from_file(filename)
            file_cache_key_loc = file_cache_key_reg + "_loc" if file_cache_key_reg else None
            
            # Try location cache first (more feature-rich)
            if file_cache_key_loc and self.matcher.load_from_cache(file_cache_key_loc):
                print(f"   [OK] Fast load successful! Loaded location-aware index from cache.")
                self.company_data_loaded = True
                self.last_data_check = current_time
                self._loading = False
                return True
                
            # Try regular cache
            if file_cache_key_reg and self.matcher.load_from_cache(file_cache_key_reg):
                print(f"   [OK] Fast load successful! Loaded regular index from cache.")
                self.company_data_loaded = True
                self.last_data_check = current_time
                self._loading = False
                return True
            
            # Step 2.5: FALLBACK - Try to find any existing cache with matching model
            # This handles cases where file was modified after cache creation
            print(f"   File-based cache lookup failed, searching for compatible caches...")
            cache_dir = 'company_matcher_cache'
            if os.path.exists(cache_dir):
                import glob
                metadata_files = glob.glob(os.path.join(cache_dir, '*_metadata.pkl'))
                # Sort by file size (largest first) to try the most complete cache
                metadata_files.sort(key=lambda x: os.path.getsize(x.replace('_metadata.pkl', '_names.pkl')) if os.path.exists(x.replace('_metadata.pkl', '_names.pkl')) else 0, reverse=True)
                
                for mf in metadata_files:
                    try:
                        import pickle
                        with open(mf, 'rb') as f:
                            metadata = pickle.load(f)
                        
                            # Check if model matches and it's large enough (if we're trying for a specific file)
                            if metadata.get('model_name') == self.matcher.model_name:
                                num_companies = metadata.get('num_companies', 0)
                                
                                # If we're loading a specific large file, don't settle for a significantly smaller cache
                                if filename == 'companies_with_location.json' and num_companies < 4000000:
                                    print(f"   Skipping cache {cache_key}: too small ({num_companies:,} < 4M)")
                                    continue
                                    
                                cache_key = metadata.get('cache_key')
                                has_loc = metadata.get('has_location_data', False)
                                
                                print(f"   Found compatible cache: {cache_key} ({num_companies:,} companies, location={has_loc})")
                                
                                if self.matcher.load_from_cache(cache_key):
                                    print(f"   [OK] Successfully loaded from compatible cache!")
                                    self.company_data_loaded = True
                                    self.last_data_check = current_time
                                    self._loading = False
                                    return True
                    except Exception as e:
                        continue
            
            # Step 3: CACHE MISS - Load from JSON file
            print(f"Cache miss or reload forced - loading company data from {filename}...")
            load_start = time.time()
            
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   Found {len(data):,} total entries in file")
            
            # Check if data includes location information (City, State, Count)
            has_location_data = False
            if data and isinstance(data[0], dict):
                sample = data[0]
                has_location_data = 'City' in sample or 'State' in sample or 'Count' in sample
            
            if has_location_data:
                print("   Location data detected (City/State/Count) - using location-aware loading")
            
            company_names = []
            print("   Extracting company names...")
            
            # Use progress bar for extraction
            for i, item in enumerate(tqdm(data, desc="   Extracting", total=len(data), unit="entries", ncols=80, disable=not HAS_TQDM)):
                if isinstance(item, dict) and "Company Name" in item:
                    company_names.append(item["Company Name"])
            
            if not company_names:
                print("   [FAIL] No company names found in data")
                self._loading = False
                return False
            
            load_time = time.time() - load_start
            print(f"   [OK] Extracted {len(company_names):,} company names in {load_time:.1f}s")
            
            # Step 4: Build or update index
            # Build index - use location-aware method if data has location info
            if has_location_data:
                print(f"Building company matching index with location data ({len(company_names):,} companies)...")
                # build_index_with_location will also save to cache
                self.matcher.build_index_with_location(filepath=filename, data=data)
            else:
                print(f"Building company matching index with {len(company_names):,} companies...")
                # build_index will also save to cache
                self.matcher.build_index(company_names, filepath=filename)
            
            # Store file modification time for change detection
            try:
                self.matcher._last_file_mtime = os.path.getmtime(filename)
            except:
                self.matcher._last_file_mtime = 0
            
            self.company_data_loaded = True
            self.last_data_check = current_time
            
            print(f"SUCCESS: Loaded {len(company_names):,} company name entries")
            if self.matcher.has_location_data:
                print(f"   Location data: Available ({len(self.matcher.company_locations):,} entries)")
            print(f"Webapp is now ready for company matching!")
            self._loading = False
            return True
            
        except Exception as e:
            print(f"Error loading company data: {e}")
            import traceback
            traceback.print_exc()
            self._loading = False
            return False

    def search(self, query, top_k=10, city=None, state=None):
        """Perform search with optional location filtering"""
        if not self.company_data_loaded or self.matcher is None:
            if not self.load_company_data():
                raise Exception("Company data not available")

        # Perform search - use location-aware matching if location data is available
        print(f"Searching for companies matching: {query}")
        if city or state:
            print(f"   Location filter: city='{city}', state='{state}'")
        
        # Use location-aware matching if available and location params provided
        if self.matcher.has_location_data and (city or state):
            matches = self.matcher.match_with_location(query, city=city, state=state, top_k=top_k)
            print(f"Found {len(matches)} matches (location-aware)")
        else:
            matches = self.matcher.match(query, top_k=top_k)
            print(f"Found {len(matches)} matches")
        
        # Format results for display
        results = []
        for i, match in enumerate(matches, 1):
            # Generate match rationale based on the explanation
            explanation = self.matcher.explain_match(query, match['name'])
            rationale = RationaleService.generate_match_rationale(query, match['name'], explanation, match['score'])
            
            result_entry = {
                'rank': i,
                'company_name': match['name'],
                'likeness_percent': round(match['score'] * 100, 1),
                'match_rationale': rationale,
                'raw_score': match['score'],
                'explanation_details': {
                    'query_tokens': list(explanation.get('query_tokens', [])),
                    'match_tokens': list(explanation.get('match_tokens', [])),
                    'overlap_tokens': list(explanation.get('overlap', [])),
                    'overlap_score': explanation.get('overlap_score', 0.0),
                    'string_score': explanation.get('string_score', match.get('string_score', 0.0)),
                    'semantic_score': explanation.get('semantic_score', match.get('semantic_score', 0.0)),
                    'normalized_semantic_score': explanation.get('normalized_semantic_score', match.get('normalized_semantic_score', 0.0)),
                    'acronym_fidelity': explanation.get('acronym_fidelity', match.get('acronym_fidelity', 0.0)),
                    'match_type': match.get('match_type', explanation.get('match_type', 'hybrid'))
                }
            }
            
            # Add top-level fields for convenience
            result_entry['string_score'] = result_entry['explanation_details']['string_score']
            result_entry['semantic_score'] = result_entry['explanation_details']['semantic_score']
            result_entry['normalized_semantic_score'] = result_entry['explanation_details']['normalized_semantic_score']
            result_entry['acronym_fidelity'] = result_entry['explanation_details']['acronym_fidelity']
            result_entry['match_type'] = result_entry['explanation_details']['match_type']
            
            # Add location and count data if available
            if 'city' in match:
                result_entry['city'] = match.get('city', '')
            if 'state' in match:
                result_entry['state'] = match.get('state', '')
            if 'count' in match:
                result_entry['record_count'] = match.get('count', 0)
            if 'location_score' in match:
                result_entry['location_score'] = round(match.get('location_score', 0) * 100, 1)
            if 'name_score' in match:
                result_entry['name_score'] = round(match.get('name_score', 0) * 100, 1)
            
            results.append(result_entry)
        
        return results

    def clear_cache(self):
        """Clear cache and force fresh data loading"""
        if self.matcher is not None:
            # Clear the cache for this matcher
            cache_key = self.matcher.get_cache_key(self.matcher.original_company_names)
            self.matcher.clear_cache(cache_key)
            print(f"Cleared cache: {cache_key}")
        
        # Reset state
        self.matcher = None
        self.company_data_loaded = False
        
        return self.load_company_data(force_reload=True)

    def get_status(self):
        """Get current service status"""
        if self._loading:
            return {
                'status': 'loading',
                'message': 'Building company matching index...',
                'progress': 'indexing'
            }
        
        # Ensure data is loaded
        loaded = self.load_company_data()
        
        if loaded:
            response = {
                'status': 'ready',
                'companies_loaded': len(self.matcher.original_company_names) if self.matcher else 0,
                'last_updated': self.last_data_check,
                'message': f'Ready with {len(self.matcher.original_company_names):,} companies' if self.matcher else 'Ready'
            }
            # Add location data status
            if self.matcher:
                response['has_location_data'] = self.matcher.has_location_data
                if self.matcher.has_location_data:
                    response['location_entries'] = len(self.matcher.company_locations)
                    response['message'] += ' (with location data)'
            return response
        else:
            return {
                'status': 'not_ready',
                'error': 'Company data not available',
                'message': 'Please ensure companies.json exists and is accessible'
            }

    def get_cache_info(self):
        """Get cache debugging info"""
        if self.matcher is None:
            return None
        
        cache_info = self.matcher.get_cache_info()
        cache_key = self.matcher.get_cache_key(self.matcher.original_company_names) if self.matcher.original_company_names else None
        
        return {
            'cache_info': cache_info,
            'current_cache_key': cache_key,
            'companies_loaded': len(self.matcher.original_company_names) if self.matcher.original_company_names else 0,
            'model_name': self.matcher.model_name
        }
