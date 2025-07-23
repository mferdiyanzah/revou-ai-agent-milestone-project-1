"""
Monitoring Service for AI Agent Performance Tracking
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
import threading
from enum import Enum

from ..config import settings


class OperationStatus(Enum):
    """Status of operations"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    GUARDRAIL_BLOCKED = "guardrail_blocked"


class AgentType(Enum):
    """Types of AI agents"""
    JOURNAL_MAPPER = "journal_mapper"
    TRANSACTION_ANALYZER = "transaction_analyzer"
    VALIDATOR = "validator"
    DOCUMENT_PROCESSOR = "document_processor"
    EMBEDDING_GENERATOR = "embedding_generator"


@dataclass
class MetricEvent:
    """Individual metric event"""
    timestamp: float
    agent_type: AgentType
    operation: str
    status: OperationStatus
    response_time: float
    token_count: Optional[int] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentMetrics:
    """Aggregated metrics for an agent"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_response_time: float = 0.0
    total_tokens: int = 0
    avg_tokens_per_operation: float = 0.0
    uptime_percentage: float = 100.0
    last_operation_time: Optional[float] = None


class MonitoringService:
    """Comprehensive monitoring service for AI agents"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # In-memory metrics for real-time dashboard
        self.recent_events = deque(maxlen=1000)
        self.agent_metrics = defaultdict(AgentMetrics)
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        
        # Performance thresholds
        self.response_time_threshold = getattr(settings, 'response_time_threshold', 5.0)
        self.error_rate_threshold = getattr(settings, 'error_rate_threshold', 0.1)
        
        # Initialize database
        self._init_database()
        
        # Load recent metrics on startup
        self._load_recent_metrics()
    
    def _init_database(self):
        """Initialize SQLite database for persistent metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    agent_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    token_count INTEGER,
                    input_size INTEGER,
                    output_size INTEGER,
                    error_message TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON metric_events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_type ON metric_events(agent_type)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    agent_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                    feedback_text TEXT,
                    helpful BOOLEAN,
                    correct_answer TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def record_operation(
        self,
        agent_type: AgentType,
        operation: str,
        status: OperationStatus,
        response_time: float,
        token_count: Optional[int] = None,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record an operation metric event
        
        Returns:
            Event ID for reference
        """
        timestamp = time.time()
        
        event = MetricEvent(
            timestamp=timestamp,
            agent_type=agent_type,
            operation=operation,
            status=status,
            response_time=response_time,
            token_count=token_count,
            input_size=input_size,
            output_size=output_size,
            error_message=error_message,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        # Store in memory for real-time access
        with self.lock:
            self.recent_events.append(event)
            self._update_agent_metrics(event)
            self._update_hourly_stats(event)
        
        # Store in database for persistence
        event_id = self._store_event_to_db(event)
        
        # Check for alerts
        self._check_alerts(event)
        
        return str(event_id)
    
    def record_feedback(
        self,
        agent_type: AgentType,
        operation: str,
        user_id: str,
        rating: int,
        helpful: bool,
        feedback_text: Optional[str] = None,
        correct_answer: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record user feedback for an operation"""
        timestamp = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO feedback_events 
                (timestamp, agent_type, operation, user_id, session_id, rating, 
                 feedback_text, helpful, correct_answer, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, agent_type.value, operation, user_id, session_id,
                rating, feedback_text, helpful, correct_answer,
                json.dumps(metadata) if metadata else None
            ))
            
            return str(cursor.lastrowid)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for dashboard"""
        with self.lock:
            current_time = time.time()
            last_hour = current_time - 3600
            
            # Filter recent events
            recent_events = [
                event for event in self.recent_events
                if event.timestamp >= last_hour
            ]
            
            # Calculate real-time stats
            total_operations = len(recent_events)
            successful_operations = len([e for e in recent_events if e.status == OperationStatus.SUCCESS])
            failed_operations = len([e for e in recent_events if e.status == OperationStatus.FAILURE])
            
            success_rate = successful_operations / total_operations if total_operations > 0 else 0
            
            avg_response_time = sum(e.response_time for e in recent_events) / total_operations if total_operations > 0 else 0
            
            total_tokens = sum(e.token_count for e in recent_events if e.token_count)
            
            # Agent-specific metrics
            agent_stats = {}
            for agent_type in AgentType:
                agent_events = [e for e in recent_events if e.agent_type == agent_type]
                if agent_events:
                    agent_success = len([e for e in agent_events if e.status == OperationStatus.SUCCESS])
                    agent_stats[agent_type.value] = {
                        "total_operations": len(agent_events),
                        "success_rate": agent_success / len(agent_events),
                        "avg_response_time": sum(e.response_time for e in agent_events) / len(agent_events),
                        "total_tokens": sum(e.token_count for e in agent_events if e.token_count)
                    }
            
            return {
                "timestamp": current_time,
                "time_window": "last_hour",
                "overall_metrics": {
                    "total_operations": total_operations,
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                    "total_tokens": total_tokens,
                    "operations_per_minute": total_operations / 60 if total_operations > 0 else 0
                },
                "agent_metrics": agent_stats,
                "health_status": self._get_health_status(),
                "alerts": self._get_active_alerts()
            }
    
    def get_historical_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical metrics from database"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get overall stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_operations,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_operations,
                    SUM(CASE WHEN token_count IS NOT NULL THEN token_count ELSE 0 END) as total_tokens
                FROM metric_events 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_time, end_time))
            
            overall_stats = cursor.fetchone()
            
            # Get hourly breakdown
            cursor = conn.execute("""
                SELECT 
                    datetime(timestamp, 'unixepoch') as hour,
                    agent_type,
                    COUNT(*) as operations,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_operations
                FROM metric_events 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY strftime('%Y-%m-%d %H', datetime(timestamp, 'unixepoch')), agent_type
                ORDER BY hour
            """, (start_time, end_time))
            
            hourly_breakdown = cursor.fetchall()
            
            # Get error breakdown
            cursor = conn.execute("""
                SELECT 
                    error_message,
                    COUNT(*) as count
                FROM metric_events 
                WHERE timestamp BETWEEN ? AND ? AND status != 'success'
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 10
            """, (start_time, end_time))
            
            error_breakdown = cursor.fetchall()
            
            return {
                "time_range": f"last_{hours}_hours",
                "overall_stats": {
                    "total_operations": overall_stats[0] or 0,
                    "avg_response_time": overall_stats[1] or 0,
                    "success_rate": (overall_stats[2] or 0) / (overall_stats[0] or 1),
                    "total_tokens": overall_stats[3] or 0
                },
                "hourly_breakdown": [
                    {
                        "hour": row[0],
                        "agent_type": row[1],
                        "operations": row[2],
                        "avg_response_time": row[3],
                        "success_rate": row[4] / row[2] if row[2] > 0 else 0
                    }
                    for row in hourly_breakdown
                ],
                "error_breakdown": [
                    {"error": row[0], "count": row[1]}
                    for row in error_breakdown
                ]
            }
    
    def get_feedback_analytics(self, hours: int = 168) -> Dict[str, Any]:
        """Get feedback analytics (default: last week)"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall feedback stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(rating) as avg_rating,
                    SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count,
                    COUNT(CASE WHEN correct_answer IS NOT NULL THEN 1 END) as corrections_provided
                FROM feedback_events 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_time, end_time))
            
            overall_feedback = cursor.fetchone()
            
            # Feedback by agent type
            cursor = conn.execute("""
                SELECT 
                    agent_type,
                    COUNT(*) as feedback_count,
                    AVG(rating) as avg_rating,
                    SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count
                FROM feedback_events 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY agent_type
            """, (start_time, end_time))
            
            agent_feedback = cursor.fetchall()
            
            return {
                "time_range": f"last_{hours}_hours",
                "overall_feedback": {
                    "total_feedback": overall_feedback[0] or 0,
                    "avg_rating": overall_feedback[1] or 0,
                    "helpfulness_rate": (overall_feedback[2] or 0) / (overall_feedback[0] or 1),
                    "corrections_provided": overall_feedback[3] or 0
                },
                "agent_feedback": [
                    {
                        "agent_type": row[0],
                        "feedback_count": row[1],
                        "avg_rating": row[2],
                        "helpfulness_rate": row[3] / row[1] if row[1] > 0 else 0
                    }
                    for row in agent_feedback
                ]
            }
    
    def _update_agent_metrics(self, event: MetricEvent):
        """Update in-memory agent metrics"""
        agent_key = event.agent_type.value
        metrics = self.agent_metrics[agent_key]
        
        metrics.total_operations += 1
        if event.status == OperationStatus.SUCCESS:
            metrics.successful_operations += 1
        else:
            metrics.failed_operations += 1
        
        # Update rolling average response time
        if metrics.total_operations == 1:
            metrics.avg_response_time = event.response_time
        else:
            metrics.avg_response_time = (
                (metrics.avg_response_time * (metrics.total_operations - 1) + event.response_time) /
                metrics.total_operations
            )
        
        # Update token metrics
        if event.token_count:
            metrics.total_tokens += event.token_count
            metrics.avg_tokens_per_operation = metrics.total_tokens / metrics.total_operations
        
        metrics.last_operation_time = event.timestamp
    
    def _update_hourly_stats(self, event: MetricEvent):
        """Update hourly statistics"""
        hour_key = int(event.timestamp // 3600)
        self.hourly_stats[hour_key]["total"] += 1
        self.hourly_stats[hour_key][event.status.value] += 1
        self.hourly_stats[hour_key][f"{event.agent_type.value}_operations"] += 1
    
    def _store_event_to_db(self, event: MetricEvent) -> int:
        """Store event to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO metric_events 
                (timestamp, agent_type, operation, status, response_time, token_count,
                 input_size, output_size, error_message, user_id, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp, event.agent_type.value, event.operation, event.status.value,
                event.response_time, event.token_count, event.input_size, event.output_size,
                event.error_message, event.user_id, event.session_id,
                json.dumps(event.metadata) if event.metadata else None
            ))
            
            return cursor.lastrowid
    
    def _load_recent_metrics(self):
        """Load recent metrics from database on startup"""
        current_time = time.time()
        last_hour = current_time - 3600
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM metric_events 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (last_hour,))
            
            for row in cursor.fetchall():
                event = MetricEvent(
                    timestamp=row[1],
                    agent_type=AgentType(row[2]),
                    operation=row[3],
                    status=OperationStatus(row[4]),
                    response_time=row[5],
                    token_count=row[6],
                    input_size=row[7],
                    output_size=row[8],
                    error_message=row[9],
                    user_id=row[10],
                    session_id=row[11],
                    metadata=json.loads(row[12]) if row[12] else None
                )
                
                self.recent_events.append(event)
                self._update_agent_metrics(event)
    
    def _get_health_status(self) -> str:
        """Determine overall system health status"""
        current_time = time.time()
        last_5_minutes = current_time - 300
        
        recent_events = [
            event for event in self.recent_events
            if event.timestamp >= last_5_minutes
        ]
        
        if not recent_events:
            return "unknown"
        
        success_rate = len([e for e in recent_events if e.status == OperationStatus.SUCCESS]) / len(recent_events)
        avg_response_time = sum(e.response_time for e in recent_events) / len(recent_events)
        
        if success_rate >= 0.95 and avg_response_time <= self.response_time_threshold:
            return "healthy"
        elif success_rate >= 0.8:
            return "degraded"
        else:
            return "unhealthy"
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        alerts = []
        current_time = time.time()
        last_10_minutes = current_time - 600
        
        recent_events = [
            event for event in self.recent_events
            if event.timestamp >= last_10_minutes
        ]
        
        if not recent_events:
            return alerts
        
        # Check error rate
        error_rate = len([e for e in recent_events if e.status != OperationStatus.SUCCESS]) / len(recent_events)
        if error_rate > self.error_rate_threshold:
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"Error rate {error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}",
                "timestamp": current_time
            })
        
        # Check response time
        avg_response_time = sum(e.response_time for e in recent_events) / len(recent_events)
        if avg_response_time > self.response_time_threshold:
            alerts.append({
                "type": "slow_response",
                "severity": "warning",
                "message": f"Average response time {avg_response_time:.2f}s exceeds threshold {self.response_time_threshold}s",
                "timestamp": current_time
            })
        
        return alerts
    
    def _check_alerts(self, event: MetricEvent):
        """Check if event triggers any alerts"""
        # Log performance issues
        if event.response_time > self.response_time_threshold * 2:
            self.logger.warning(
                f"Slow operation detected: {event.agent_type.value} {event.operation} "
                f"took {event.response_time:.2f}s"
            )
        
        if event.status != OperationStatus.SUCCESS:
            self.logger.error(
                f"Operation failed: {event.agent_type.value} {event.operation} "
                f"failed with status {event.status.value}: {event.error_message}"
            ) 