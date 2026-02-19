"""
LLM Service stub.

Ollama integration has been removed. All methods return None / False so that
callers that guard with `LLMService.is_available()` continue to work without
any network dependency.
"""
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMService:
    """No-op stub — Ollama dependency removed."""

    DEFAULT_MODEL = "none"

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def get_completion(prompt: str, model: str = DEFAULT_MODEL,
                       system_prompt: Optional[str] = None) -> Optional[str]:
        return None

    @staticmethod
    def analyze_semantic_context(query: str, company_name: str) -> Optional[str]:
        return None

    @staticmethod
    def get_match_narrative(query: str, company_name: str,
                            scores: Dict[str, Any]) -> Optional[str]:
        return None
