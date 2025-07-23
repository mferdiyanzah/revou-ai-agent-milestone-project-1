"""
Services package for TReA application
"""

from .api_client import TReAAPIClient
from .openai_service import OpenAIService

# Optional PostgreSQL Vector DB support
try:
    from .vector_db import VectorDBService
    __all__ = ["TReAAPIClient", "OpenAIService", "VectorDBService"]
except ImportError:
    __all__ = ["TReAAPIClient", "OpenAIService"] 