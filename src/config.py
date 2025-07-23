"""
Configuration settings for TReA application with LangGraph integration
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings with LangGraph and LangSmith support"""
    
    # Core AI Services
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Faster model
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_timeout: int = int(os.getenv("OPENAI_TIMEOUT", "5"))  # Ultra short timeout
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))  # Limit response length
    
    # LangSmith Configuration (for LangGraph monitoring and tracing)
    langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langchain_api_key: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "trea-treasury-ai")
    
    # Database Configuration
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "trea_db")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: Optional[str] = os.getenv("DB_PASSWORD")
    
    # TReA API Configuration
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    api_token: Optional[str] = os.getenv("API_TOKEN")
    api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    
    # Optional Features
    brave_api_key: Optional[str] = os.getenv("BRAVE_API_KEY")
    brave_search_enabled: bool = os.getenv("BRAVE_SEARCH_ENABLED", "false").lower() == "true"
    
    # PostgreSQL Configuration (for vector DB compatibility)
    postgres_host: str = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost"))
    postgres_port: int = int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432")))
    postgres_db: str = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "trea_db"))
    postgres_user: str = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "postgres"))
    postgres_password: Optional[str] = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD"))
    
    # File Upload
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploads")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # UI Configuration
    app_name: str = "TReA - Treasury with Embedded AI"
    app_version: str = "1.0.0"
    page_title: str = "TReA - Treasury AI"
    page_icon: str = "🏦"
    layout: str = "wide"
    
    # Security & Performance
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "10000"))
    response_time_threshold: float = float(os.getenv("RESPONSE_TIME_THRESHOLD", "5.0"))
    error_rate_threshold: float = float(os.getenv("ERROR_RATE_THRESHOLD", "0.1"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # LangGraph Specific Settings
    auto_enable_langgraph: bool = os.getenv("AUTO_ENABLE_LANGGRAPH", "true").lower() == "true"
    agent_timeout: int = int(os.getenv("AGENT_TIMEOUT", "30"))
    workflow_timeout: int = int(os.getenv("WORKFLOW_TIMEOUT", "120"))
    checkpoint_storage: str = os.getenv("CHECKPOINT_STORAGE", "memory")
    
    @property
    def database_url(self) -> Optional[str]:
        """Get PostgreSQL database URL if configured"""
        if self.db_password:
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        return None
    
    @property
    def langsmith_configured(self) -> bool:
        """Check if LangSmith is properly configured"""
        return self.langchain_tracing_v2 and bool(self.langchain_api_key)
    
    def setup_langsmith(self):
        """Setup LangSmith environment variables for tracing"""
        if self.langsmith_configured:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            return True
        return False

# Global settings instance
settings = Settings()

# Setup LangSmith if configured
if settings.setup_langsmith():
    print(f"🔍 LangSmith tracing enabled for project: {settings.langchain_project}")
else:
    print("ℹ️ LangSmith tracing not configured (optional)") 