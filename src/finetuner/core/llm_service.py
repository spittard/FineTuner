import ollama
import logging
import json
import hashlib
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LLMService:
    """Service for managing interactions with local LLM (Ollama)"""
    
    DEFAULT_MODEL = "llama3.2:1b"
    CACHE_FILE = "e:/projects/FineTuner/FineTuner/cache/llm_cache.json"
    _cache = None

    @staticmethod
    def _load_cache():
        """Load persistent cache from disk"""
        if LLMService._cache is not None:
            return
        
        try:
            import os
            if os.path.exists(LLMService.CACHE_FILE):
                with open(LLMService.CACHE_FILE, 'r', encoding='utf-8') as f:
                    LLMService._cache = json.load(f)
            else:
                LLMService._cache = {}
        except Exception as e:
            logger.error(f"Failed to load LLM cache: {e}")
            LLMService._cache = {}

    @staticmethod
    def _save_cache():
        """Save persistent cache to disk"""
        if LLMService._cache is None:
            return
        
        try:
            import os
            os.makedirs(os.path.dirname(LLMService.CACHE_FILE), exist_ok=True)
            with open(LLMService.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(LLMService._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save LLM cache: {e}")

    @staticmethod
    def get_cache_key(prompt: str, model: str, system_prompt: Optional[str] = None) -> str:
        """Generate a unique hash key for a prompt/model/system combination"""
        content = f"{model}|{system_prompt or ''}|{prompt}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def is_available() -> bool:
        """Check if Ollama server is running and model is available"""
        try:
            models_resp = ollama.list()
            models = models_resp.models if hasattr(models_resp, 'models') else models_resp.get('models', [])
            model_names = [m.model if hasattr(m, 'model') else m.get('name', '') for m in models]
            return any(LLMService.DEFAULT_MODEL in name for name in model_names)
        except Exception:
            return False
    
    @staticmethod
    def get_completion(prompt: str, model: str = DEFAULT_MODEL, system_prompt: Optional[str] = None) -> Optional[str]:
        """Get a completion from the LLM with strict resource management and caching"""
        LLMService._load_cache()
        cache_key = LLMService.get_cache_key(prompt, model, system_prompt)
        
        if cache_key in LLMService._cache:
            print(f"   [LLM] {model} completion (Cache: HIT)")
            return LLMService._cache[cache_key]
        
        try:
            start_time = time.time()
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            
            messages.append({'role': 'user', 'content': prompt})
            
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': 0.1,
                    'num_predict': 100,
                },
                keep_alive='10m'
            )
            
            duration = time.time() - start_time
            
            result = None
            if hasattr(response, 'message'):
                result = response.message.content.strip()
            else:
                result = response.get('message', {}).get('content', '').strip()
            
            if result:
                LLMService._cache[cache_key] = result
                LLMService._save_cache()
                
            print(f"   [LLM] {model} completion in {duration:.2f}s (Cache: MISS)")
            return result
            
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            return None

    @staticmethod
    def analyze_semantic_context(query: str, company_name: str) -> Optional[str]:
        """Analyze business alignment and semantic context between two names"""
        if not LLMService.is_available():
            return None
        
        system_prompt = (
            "You are a professional business analyst. Provide a one-sentence "
            "description of the relationship between the search query and the company."
        )
        
        prompt = (
            f"Query: {query}\n"
            f"Company: {company_name}\n\n"
            "Instructions:\n"
            "1. Identify if they are the same entity, a branch, or unrelated.\n"
            "2. Mention the shared industry context.\n"
            "3. If entities are clearly unrelated (e.g. Apple Inc vs Apple Orchard), state they are 'Unrelated'.\n"
            "4. Response MUST be a single specific sentence.\n\n"
            "Relationship Analysis:"
        )
        
        return LLMService.get_completion(prompt, system_prompt=system_prompt)

    @staticmethod
    def get_match_narrative(query: str, company_name: str, scores: Dict[str, float]) -> Optional[str]:
        """Generate a rich narrative explanation for a match"""
        
        system_prompt = (
            "You are a matching engine explainer. Create a natural, professional narrative "
            "explaining why a search result was ranked highly or poorly."
        )
        
        score_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
        
        prompt = (
            f"Explain the match for:\n"
            f"Query: \"{query}\"\n"
            f"Result: \"{company_name}\"\n"
            f"Component Scores: {score_str}\n\n"
            "Write a 1-2 sentence narrative that justifies the ranking based on the name similarities and scores provided."
        )
        
        return LLMService.get_completion(prompt, system_prompt=system_prompt)
