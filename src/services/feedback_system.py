"""
Feedback System for Continuous AI Improvement
"""

import json
import sqlite3
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import defaultdict

from .monitoring import MonitoringService, AgentType
from ..config import settings


@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    feedback_id: str
    timestamp: float
    agent_type: AgentType
    operation: str
    user_input: str
    ai_output: str
    user_id: str
    session_id: str
    rating: int  # 1-5 scale
    helpful: bool
    feedback_text: Optional[str] = None
    correct_answer: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class ImprovementInsight:
    """Insights for AI improvement"""
    pattern_type: str
    frequency: int
    description: str
    suggested_action: str
    priority: str  # high, medium, low
    examples: List[str]


class FeedbackSystem:
    """System for collecting and analyzing feedback to improve AI performance"""
    
    def __init__(self, monitoring_service: MonitoringService):
        self.monitoring = monitoring_service
        self.lock = threading.Lock()
        
        # Feedback storage
        self.db_path = monitoring_service.db_path
        self.recent_feedback = defaultdict(list)
        
        # Analysis thresholds
        self.min_feedback_for_analysis = getattr(settings, 'min_feedback_for_analysis', 10)
        self.improvement_threshold = getattr(settings, 'improvement_threshold', 0.7)
        
        # Initialize feedback-specific tables
        self._init_feedback_tables()
    
    def _init_feedback_tables(self):
        """Initialize feedback-specific database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Detailed feedback entries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detailed_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    agent_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    ai_output TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                    helpful BOOLEAN,
                    feedback_text TEXT,
                    correct_answer TEXT,
                    improvement_suggestions TEXT,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Improvement insights
            conn.execute("""
                CREATE TABLE IF NOT EXISTS improvement_insights (
                    insight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    suggested_action TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    examples TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Training examples for fine-tuning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_examples (
                    example_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    correct_output TEXT NOT NULL,
                    feedback_source TEXT NOT NULL,
                    quality_score REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def record_feedback(
        self,
        agent_type: AgentType,
        operation: str,
        user_input: str,
        ai_output: str,
        user_id: str,
        rating: int,
        helpful: bool,
        session_id: str = None,
        feedback_text: str = None,
        correct_answer: str = None,
        improvement_suggestions: str = None,
        tags: List[str] = None
    ) -> str:
        """
        Record detailed feedback for an AI interaction
        
        Returns:
            Feedback ID for reference
        """
        timestamp = time.time()
        feedback_id = f"{agent_type.value}_{operation}_{int(timestamp)}_{user_id}"
        
        feedback_entry = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=timestamp,
            agent_type=agent_type,
            operation=operation,
            user_input=user_input,
            ai_output=ai_output,
            user_id=user_id,
            session_id=session_id or f"session_{int(timestamp)}",
            rating=rating,
            helpful=helpful,
            feedback_text=feedback_text,
            correct_answer=correct_answer,
            improvement_suggestions=improvement_suggestions,
            tags=tags or []
        )
        
        # Store in database
        self._store_feedback_to_db(feedback_entry)
        
        # Store in memory for quick access
        with self.lock:
            self.recent_feedback[agent_type.value].append(feedback_entry)
            
            # Keep only recent feedback in memory
            cutoff_time = timestamp - 86400  # 24 hours
            self.recent_feedback[agent_type.value] = [
                fb for fb in self.recent_feedback[agent_type.value]
                if fb.timestamp >= cutoff_time
            ]
        
        # Record in monitoring system
        self.monitoring.record_feedback(
            agent_type=agent_type,
            operation=operation,
            user_id=user_id,
            rating=rating,
            helpful=helpful,
            feedback_text=feedback_text,
            correct_answer=correct_answer,
            session_id=session_id,
            metadata={"tags": tags, "improvement_suggestions": improvement_suggestions}
        )
        
        # Create training example if correct answer provided
        if correct_answer and correct_answer.strip():
            self._create_training_example(
                agent_type=agent_type,
                operation=operation,
                input_text=user_input,
                correct_output=correct_answer,
                feedback_source=feedback_id,
                quality_score=rating / 5.0
            )
        
        return feedback_id
    
    def analyze_feedback_patterns(self, agent_type: AgentType = None, days: int = 7) -> List[ImprovementInsight]:
        """
        Analyze feedback patterns to identify improvement opportunities
        
        Args:
            agent_type: Specific agent to analyze (None for all)
            days: Number of days to analyze
            
        Returns:
            List of improvement insights
        """
        end_time = time.time()
        start_time = end_time - (days * 86400)
        
        insights = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Base query condition
            where_clause = "WHERE timestamp BETWEEN ? AND ?"
            params = [start_time, end_time]
            
            if agent_type:
                where_clause += " AND agent_type = ?"
                params.append(agent_type.value)
            
            # Analyze low ratings
            cursor = conn.execute(f"""
                SELECT agent_type, operation, COUNT(*) as count,
                       GROUP_CONCAT(feedback_text, ' | ') as feedback_samples
                FROM detailed_feedback 
                {where_clause} AND rating <= 2
                GROUP BY agent_type, operation
                HAVING count >= ?
                ORDER BY count DESC
            """, params + [self.min_feedback_for_analysis])
            
            for row in cursor.fetchall():
                insights.append(ImprovementInsight(
                    pattern_type="low_rating",
                    frequency=row[2],
                    description=f"Low ratings for {row[0]} {row[1]} operations",
                    suggested_action=f"Review and improve {row[1]} logic for {row[0]}",
                    priority="high",
                    examples=row[3].split(' | ')[:3] if row[3] else []
                ))
            
            # Analyze common correction patterns
            cursor = conn.execute(f"""
                SELECT agent_type, operation, COUNT(*) as count,
                       GROUP_CONCAT(correct_answer, ' | ') as correction_samples
                FROM detailed_feedback 
                {where_clause} AND correct_answer IS NOT NULL AND correct_answer != ''
                GROUP BY agent_type, operation
                HAVING count >= ?
                ORDER BY count DESC
            """, params + [3])  # Lower threshold for corrections
            
            for row in cursor.fetchall():
                insights.append(ImprovementInsight(
                    pattern_type="frequent_corrections",
                    frequency=row[2],
                    description=f"Frequent corrections needed for {row[0]} {row[1]}",
                    suggested_action=f"Update training data and prompts for {row[1]}",
                    priority="high",
                    examples=row[3].split(' | ')[:3] if row[3] else []
                ))
            
            # Analyze unhelpful responses
            cursor = conn.execute(f"""
                SELECT agent_type, operation, COUNT(*) as count,
                       AVG(rating) as avg_rating
                FROM detailed_feedback 
                {where_clause} AND helpful = 0
                GROUP BY agent_type, operation
                HAVING count >= ?
                ORDER BY count DESC
            """, params + [self.min_feedback_for_analysis])
            
            for row in cursor.fetchall():
                insights.append(ImprovementInsight(
                    pattern_type="unhelpful_responses",
                    frequency=row[2],
                    description=f"High frequency of unhelpful responses for {row[0]} {row[1]}",
                    suggested_action=f"Improve response relevance and clarity for {row[1]}",
                    priority="medium",
                    examples=[]
                ))
        
        # Store insights in database
        for insight in insights:
            self._store_insight_to_db(insight)
        
        return insights
    
    def get_improvement_suggestions(self, agent_type: AgentType) -> Dict[str, Any]:
        """
        Get specific improvement suggestions for an agent type
        """
        insights = self.analyze_feedback_patterns(agent_type=agent_type, days=14)
        
        # Get training examples
        training_examples = self._get_training_examples(agent_type)
        
        # Get performance trends
        performance_trends = self._get_performance_trends(agent_type)
        
        return {
            "agent_type": agent_type.value,
            "analysis_period": "14_days",
            "insights": [
                {
                    "pattern_type": insight.pattern_type,
                    "frequency": insight.frequency,
                    "description": insight.description,
                    "suggested_action": insight.suggested_action,
                    "priority": insight.priority,
                    "examples": insight.examples
                }
                for insight in insights
            ],
            "training_examples_available": len(training_examples),
            "performance_trends": performance_trends,
            "recommendations": self._generate_recommendations(insights, training_examples, performance_trends)
        }
    
    def get_training_data(self, agent_type: AgentType, limit: int = 100) -> List[Dict[str, str]]:
        """
        Get curated training data for model improvement
        
        Args:
            agent_type: Agent type to get training data for
            limit: Maximum number of examples
            
        Returns:
            List of training examples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT input_text, correct_output, quality_score
                FROM training_examples 
                WHERE agent_type = ?
                ORDER BY quality_score DESC, created_at DESC
                LIMIT ?
            """, (agent_type.value, limit))
            
            return [
                {
                    "input": row[0],
                    "output": row[1],
                    "quality_score": row[2]
                }
                for row in cursor.fetchall()
            ]
    
    def get_feedback_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive feedback data for dashboard"""
        current_time = time.time()
        last_week = current_time - (7 * 86400)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall feedback stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(rating) as avg_rating,
                    SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count,
                    COUNT(CASE WHEN correct_answer IS NOT NULL AND correct_answer != '' THEN 1 END) as corrections_count
                FROM detailed_feedback 
                WHERE timestamp >= ?
            """, (last_week,))
            
            overall_stats = cursor.fetchone()
            
            # Feedback by agent type
            cursor = conn.execute("""
                SELECT 
                    agent_type,
                    COUNT(*) as feedback_count,
                    AVG(rating) as avg_rating,
                    SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count
                FROM detailed_feedback 
                WHERE timestamp >= ?
                GROUP BY agent_type
            """, (last_week,))
            
            agent_feedback = cursor.fetchall()
            
            # Recent insights
            cursor = conn.execute("""
                SELECT pattern_type, description, priority, frequency
                FROM improvement_insights 
                WHERE created_at >= datetime('now', '-7 days')
                ORDER BY frequency DESC, priority
                LIMIT 10
            """)
            
            recent_insights = cursor.fetchall()
            
            # Top improvement areas
            cursor = conn.execute("""
                SELECT operation, agent_type, COUNT(*) as issue_count
                FROM detailed_feedback 
                WHERE timestamp >= ? AND (rating <= 2 OR helpful = 0)
                GROUP BY operation, agent_type
                ORDER BY issue_count DESC
                LIMIT 5
            """, (last_week,))
            
            improvement_areas = cursor.fetchall()
            
            return {
                "time_period": "last_7_days",
                "overall_stats": {
                    "total_feedback": overall_stats[0] or 0,
                    "avg_rating": round(overall_stats[1] or 0, 2),
                    "helpfulness_rate": round((overall_stats[2] or 0) / (overall_stats[0] or 1), 2),
                    "corrections_provided": overall_stats[3] or 0
                },
                "agent_feedback": [
                    {
                        "agent_type": row[0],
                        "feedback_count": row[1],
                        "avg_rating": round(row[2], 2),
                        "helpfulness_rate": round(row[3] / row[1], 2) if row[1] > 0 else 0
                    }
                    for row in agent_feedback
                ],
                "recent_insights": [
                    {
                        "pattern_type": row[0],
                        "description": row[1],
                        "priority": row[2],
                        "frequency": row[3]
                    }
                    for row in recent_insights
                ],
                "top_improvement_areas": [
                    {
                        "operation": row[0],
                        "agent_type": row[1],
                        "issue_count": row[2]
                    }
                    for row in improvement_areas
                ]
            }
    
    def _store_feedback_to_db(self, feedback: FeedbackEntry):
        """Store feedback entry to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO detailed_feedback 
                (feedback_id, timestamp, agent_type, operation, user_input, ai_output,
                 user_id, session_id, rating, helpful, feedback_text, correct_answer,
                 improvement_suggestions, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id, feedback.timestamp, feedback.agent_type.value,
                feedback.operation, feedback.user_input, feedback.ai_output,
                feedback.user_id, feedback.session_id, feedback.rating, feedback.helpful,
                feedback.feedback_text, feedback.correct_answer,
                feedback.improvement_suggestions, json.dumps(feedback.tags)
            ))
    
    def _create_training_example(
        self,
        agent_type: AgentType,
        operation: str,
        input_text: str,
        correct_output: str,
        feedback_source: str,
        quality_score: float
    ):
        """Create a training example from feedback"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO training_examples 
                (agent_type, operation, input_text, correct_output, feedback_source, quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (agent_type.value, operation, input_text, correct_output, feedback_source, quality_score))
    
    def _store_insight_to_db(self, insight: ImprovementInsight):
        """Store improvement insight to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO improvement_insights 
                (pattern_type, frequency, description, suggested_action, priority, examples)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                insight.pattern_type, insight.frequency, insight.description,
                insight.suggested_action, insight.priority, json.dumps(insight.examples)
            ))
    
    def _get_training_examples(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get training examples for an agent type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*), AVG(quality_score)
                FROM training_examples 
                WHERE agent_type = ?
            """, (agent_type.value,))
            
            row = cursor.fetchone()
            return {
                "count": row[0] or 0,
                "avg_quality": row[1] or 0
            }
    
    def _get_performance_trends(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get performance trends for an agent type"""
        current_time = time.time()
        last_month = current_time - (30 * 86400)
        
        with sqlite3.connect(self.db_path) as conn:
            # Weekly performance trend
            cursor = conn.execute("""
                SELECT 
                    strftime('%W', datetime(timestamp, 'unixepoch')) as week,
                    AVG(rating) as avg_rating,
                    COUNT(*) as feedback_count
                FROM detailed_feedback 
                WHERE agent_type = ? AND timestamp >= ?
                GROUP BY week
                ORDER BY week
            """, (agent_type.value, last_month))
            
            weekly_trends = cursor.fetchall()
            
            return {
                "weekly_trends": [
                    {
                        "week": row[0],
                        "avg_rating": round(row[1], 2),
                        "feedback_count": row[2]
                    }
                    for row in weekly_trends
                ]
            }
    
    def _generate_recommendations(
        self,
        insights: List[ImprovementInsight],
        training_examples: Dict[str, Any],
        performance_trends: Dict[str, Any]
    ) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        # High priority insights
        high_priority_insights = [i for i in insights if i.priority == "high"]
        if high_priority_insights:
            recommendations.append(
                f"Address {len(high_priority_insights)} high-priority issues immediately"
            )
        
        # Training data recommendations
        if training_examples["count"] < 50:
            recommendations.append(
                "Collect more training examples to improve model performance"
            )
        elif training_examples["avg_quality"] < 0.7:
            recommendations.append(
                "Focus on improving quality of training examples"
            )
        
        # Performance trend recommendations
        if performance_trends["weekly_trends"]:
            recent_ratings = [t["avg_rating"] for t in performance_trends["weekly_trends"][-2:]]
            if len(recent_ratings) >= 2 and recent_ratings[-1] < recent_ratings[-2]:
                recommendations.append(
                    "Performance is declining - investigate recent changes"
                )
        
        return recommendations 