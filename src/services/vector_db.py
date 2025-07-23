"""
PostgreSQL Vector Database Service
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from pgvector.psycopg2 import register_vector
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None
    RealDictCursor = None
    register_vector = None

from ..config import settings


class VectorDBService:
    """PostgreSQL Vector Database Service with pgvector extension"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        if not POSTGRES_AVAILABLE:
            raise ImportError("PostgreSQL dependencies (psycopg2, pgvector) are not installed. Install with: pip install psycopg2-binary pgvector")
        
        self.host = host or settings.postgres_host
        self.port = port or settings.postgres_port
        self.database = database or settings.postgres_db
        self.user = user or settings.postgres_user
        self.password = password or settings.postgres_password
        
        if not all([self.host, self.database, self.user, self.password]):
            raise ValueError("PostgreSQL connection parameters are required")
        
        self.connection_string = f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"
        self._ensure_tables()
    
    def _get_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(self.connection_string)
            register_vector(conn)
            return conn
        except Exception as e:
            raise Exception(f"Failed to connect to PostgreSQL: {str(e)}")
    
    def _ensure_tables(self):
        """Ensure required tables exist"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Create transactions table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS transaction_embeddings (
                            id SERIAL PRIMARY KEY,
                            transaction_id VARCHAR(255) UNIQUE,
                            transaction_type VARCHAR(255),
                            asset_class VARCHAR(255),
                            description TEXT,
                            embedding vector(%s),
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """, (settings.vector_dimension,))
                    
                    # Create index for vector similarity search
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS transaction_embeddings_vector_idx 
                        ON transaction_embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    
                    # Create journal mappings table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS journal_mappings (
                            id SERIAL PRIMARY KEY,
                            transaction_type VARCHAR(255),
                            asset_class VARCHAR(255),
                            debit_account VARCHAR(255),
                            credit_account VARCHAR(255),
                            description TEXT,
                            confidence_score FLOAT,
                            embedding vector(%s),
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(transaction_type, asset_class)
                        );
                    """, (settings.vector_dimension,))
                    
                    conn.commit()
        except Exception as e:
            raise Exception(f"Failed to create tables: {str(e)}")
    
    def store_transaction_embedding(
        self,
        transaction_id: str,
        transaction_type: str,
        asset_class: str,
        description: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store transaction embedding in database
        
        Args:
            transaction_id: Unique transaction identifier
            transaction_type: Type of transaction
            asset_class: Asset class
            description: Transaction description
            embedding: Vector embedding
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO transaction_embeddings 
                        (transaction_id, transaction_type, asset_class, description, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (transaction_id) 
                        DO UPDATE SET 
                            transaction_type = EXCLUDED.transaction_type,
                            asset_class = EXCLUDED.asset_class,
                            description = EXCLUDED.description,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata;
                    """, (
                        transaction_id,
                        transaction_type,
                        asset_class,
                        description,
                        embedding,
                        json.dumps(metadata or {})
                    ))
                    conn.commit()
            return True
        except Exception as e:
            raise Exception(f"Failed to store transaction embedding: {str(e)}")
    
    def find_similar_transactions(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar transactions using vector similarity
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar transactions
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            transaction_id,
                            transaction_type,
                            asset_class,
                            description,
                            metadata,
                            1 - (embedding <=> %s) as similarity
                        FROM transaction_embeddings
                        WHERE 1 - (embedding <=> %s) > %s
                        ORDER BY embedding <=> %s
                        LIMIT %s;
                    """, (query_embedding, query_embedding, similarity_threshold, query_embedding, limit))
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            raise Exception(f"Failed to find similar transactions: {str(e)}")
    
    def store_journal_mapping(
        self,
        transaction_type: str,
        asset_class: str,
        debit_account: str,
        credit_account: str,
        description: str,
        embedding: List[float],
        confidence_score: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store journal mapping with embedding
        
        Args:
            transaction_type: Type of transaction
            asset_class: Asset class
            debit_account: Debit account code
            credit_account: Credit account code
            description: Mapping description
            embedding: Vector embedding
            confidence_score: Confidence in the mapping
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO journal_mappings 
                        (transaction_type, asset_class, debit_account, credit_account, 
                         description, confidence_score, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (transaction_type, asset_class) 
                        DO UPDATE SET 
                            debit_account = EXCLUDED.debit_account,
                            credit_account = EXCLUDED.credit_account,
                            description = EXCLUDED.description,
                            confidence_score = EXCLUDED.confidence_score,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata;
                    """, (
                        transaction_type,
                        asset_class,
                        debit_account,
                        credit_account,
                        description,
                        confidence_score,
                        embedding,
                        json.dumps(metadata or {})
                    ))
                    conn.commit()
            return True
        except Exception as e:
            raise Exception(f"Failed to store journal mapping: {str(e)}")
    
    def find_journal_mapping(
        self,
        transaction_type: str,
        asset_class: str,
        query_embedding: List[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find journal mapping for transaction
        
        Args:
            transaction_type: Type of transaction
            asset_class: Asset class
            query_embedding: Optional vector for similarity search
            
        Returns:
            Journal mapping if found
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First try exact match
                    cur.execute("""
                        SELECT * FROM journal_mappings
                        WHERE transaction_type = %s AND asset_class = %s;
                    """, (transaction_type, asset_class))
                    
                    result = cur.fetchone()
                    if result:
                        return dict(result)
                    
                    # If no exact match and we have a query embedding, try similarity search
                    if query_embedding:
                        cur.execute("""
                            SELECT 
                                *,
                                1 - (embedding <=> %s) as similarity
                            FROM journal_mappings
                            WHERE 1 - (embedding <=> %s) > 0.8
                            ORDER BY embedding <=> %s
                            LIMIT 1;
                        """, (query_embedding, query_embedding, query_embedding))
                        
                        result = cur.fetchone()
                        if result:
                            return dict(result)
                    
                    return None
        except Exception as e:
            raise Exception(f"Failed to find journal mapping: {str(e)}")
    
    def get_all_mappings(self) -> List[Dict[str, Any]]:
        """Get all journal mappings"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT transaction_type, asset_class, debit_account, 
                               credit_account, description, confidence_score
                        FROM journal_mappings
                        ORDER BY transaction_type, asset_class;
                    """)
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            raise Exception(f"Failed to get mappings: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM transaction_embeddings;")
                    transaction_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM journal_mappings;")
                    mapping_count = cur.fetchone()[0]
                    
                    return {
                        "success": True,
                        "version": version,
                        "transaction_count": transaction_count,
                        "mapping_count": mapping_count,
                        "vector_dimension": settings.vector_dimension
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 