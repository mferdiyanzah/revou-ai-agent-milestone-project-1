"""
Chain-of-Thought Visualization Service
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class ThoughtType(Enum):
    """Types of thoughts in reasoning chain"""
    INPUT_ANALYSIS = "input_analysis"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    REASONING_STEP = "reasoning_step"
    DECISION_POINT = "decision_point"
    VALIDATION = "validation"
    OUTPUT_GENERATION = "output_generation"
    ERROR_HANDLING = "error_handling"


class StepStatus(Enum):
    """Status of reasoning steps"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReasoningStep:
    """Individual step in reasoning chain"""
    step_id: str
    thought_type: ThoughtType
    title: str
    description: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    alternatives: Optional[List[str]] = None
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningChain:
    """Complete chain of reasoning for a query"""
    chain_id: str
    query: str
    agent_type: str
    operation: str
    steps: List[ReasoningStep]
    final_output: Optional[str] = None
    total_confidence: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class CoTVisualizer:
    """Service for creating and visualizing chain-of-thought reasoning"""
    
    def __init__(self):
        self.active_chains: Dict[str, ReasoningChain] = {}
        self.completed_chains: List[ReasoningChain] = []
        self.max_stored_chains = 100
    
    def start_reasoning_chain(
        self,
        query: str,
        agent_type: str,
        operation: str
    ) -> str:
        """
        Start a new reasoning chain
        
        Returns:
            Chain ID for tracking
        """
        chain_id = str(uuid.uuid4())
        
        chain = ReasoningChain(
            chain_id=chain_id,
            query=query,
            agent_type=agent_type,
            operation=operation,
            steps=[],
            start_time=time.time()
        )
        
        self.active_chains[chain_id] = chain
        return chain_id
    
    def add_reasoning_step(
        self,
        chain_id: str,
        thought_type: ThoughtType,
        title: str,
        description: str,
        input_data: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a reasoning step to an active chain
        
        Returns:
            Step ID
        """
        if chain_id not in self.active_chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        step_id = str(uuid.uuid4())
        
        step = ReasoningStep(
            step_id=step_id,
            thought_type=thought_type,
            title=title,
            description=description,
            input_data=input_data,
            reasoning=reasoning,
            alternatives=alternatives,
            status=StepStatus.IN_PROGRESS,
            start_time=time.time(),
            dependencies=dependencies,
            metadata=metadata
        )
        
        self.active_chains[chain_id].steps.append(step)
        return step_id
    
    def complete_reasoning_step(
        self,
        chain_id: str,
        step_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        status: StepStatus = StepStatus.COMPLETED
    ):
        """Complete a reasoning step"""
        if chain_id not in self.active_chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.active_chains[chain_id]
        
        for step in chain.steps:
            if step.step_id == step_id:
                step.output_data = output_data
                step.confidence = confidence
                step.status = status
                step.end_time = time.time()
                if step.start_time:
                    step.duration = step.end_time - step.start_time
                break
    
    def finish_reasoning_chain(
        self,
        chain_id: str,
        final_output: str,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Finish and store a reasoning chain"""
        if chain_id not in self.active_chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.active_chains[chain_id]
        chain.final_output = final_output
        chain.success = success
        chain.error_message = error_message
        chain.end_time = time.time()
        
        if chain.start_time:
            chain.total_duration = chain.end_time - chain.start_time
        
        # Calculate overall confidence
        completed_steps = [s for s in chain.steps if s.confidence is not None]
        if completed_steps:
            chain.total_confidence = sum(s.confidence for s in completed_steps) / len(completed_steps)
        
        # Move to completed chains
        self.completed_chains.append(chain)
        del self.active_chains[chain_id]
        
        # Keep only recent chains
        if len(self.completed_chains) > self.max_stored_chains:
            self.completed_chains = self.completed_chains[-self.max_stored_chains:]
    
    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Get a reasoning chain by ID"""
        # Check active chains first
        if chain_id in self.active_chains:
            return self.active_chains[chain_id]
        
        # Check completed chains
        for chain in self.completed_chains:
            if chain.chain_id == chain_id:
                return chain
        
        return None
    
    def generate_visualization_data(self, chain_id: str) -> Dict[str, Any]:
        """
        Generate data for visualizing the reasoning chain
        
        Returns:
            Dictionary with visualization data including nodes, edges, and metadata
        """
        chain = self.get_reasoning_chain(chain_id)
        if not chain:
            return {"error": "Chain not found"}
        
        # Create nodes for each step
        nodes = []
        edges = []
        
        for i, step in enumerate(chain.steps):
            node = {
                "id": step.step_id,
                "label": step.title,
                "type": step.thought_type.value,
                "description": step.description,
                "status": step.status.value,
                "confidence": step.confidence,
                "duration": step.duration,
                "input_data": step.input_data,
                "output_data": step.output_data,
                "reasoning": step.reasoning,
                "alternatives": step.alternatives,
                "position": {"x": i * 200, "y": 100},  # Simple linear layout
                "metadata": step.metadata
            }
            nodes.append(node)
            
            # Create edges based on dependencies or sequence
            if step.dependencies:
                for dep_id in step.dependencies:
                    edges.append({
                        "id": f"{dep_id}_{step.step_id}",
                        "source": dep_id,
                        "target": step.step_id,
                        "type": "dependency"
                    })
            elif i > 0:
                # Sequential edge
                prev_step = chain.steps[i-1]
                edges.append({
                    "id": f"{prev_step.step_id}_{step.step_id}",
                    "source": prev_step.step_id,
                    "target": step.step_id,
                    "type": "sequence"
                })
        
        return {
            "chain_id": chain.chain_id,
            "query": chain.query,
            "agent_type": chain.agent_type,
            "operation": chain.operation,
            "success": chain.success,
            "total_confidence": chain.total_confidence,
            "total_duration": chain.total_duration,
            "final_output": chain.final_output,
            "error_message": chain.error_message,
            "nodes": nodes,
            "edges": edges,
            "layout": "hierarchical",  # or "force", "circular"
            "metadata": {
                "step_count": len(chain.steps),
                "completed_steps": len([s for s in chain.steps if s.status == StepStatus.COMPLETED]),
                "failed_steps": len([s for s in chain.steps if s.status == StepStatus.FAILED]),
                "avg_confidence": chain.total_confidence,
                "start_time": chain.start_time,
                "end_time": chain.end_time
            }
        }
    
    def generate_mermaid_diagram(self, chain_id: str) -> str:
        """
        Generate Mermaid diagram syntax for the reasoning chain
        
        Returns:
            Mermaid diagram as string
        """
        chain = self.get_reasoning_chain(chain_id)
        if not chain:
            return "graph TD\n    A[Chain not found]"
        
        mermaid_lines = ["graph TD"]
        
        # Add nodes
        for i, step in enumerate(chain.steps):
            node_id = f"step{i}"
            node_label = step.title.replace('"', "'")
            
            # Choose shape based on thought type
            if step.thought_type == ThoughtType.INPUT_ANALYSIS:
                shape = f'{node_id}["{node_label}"]'
            elif step.thought_type == ThoughtType.DECISION_POINT:
                shape = f'{node_id}{{{node_label}}}'
            elif step.thought_type == ThoughtType.OUTPUT_GENERATION:
                shape = f'{node_id}("{node_label}")'
            else:
                shape = f'{node_id}["{node_label}"]'
            
            # Add status styling
            status_class = ""
            if step.status == StepStatus.COMPLETED:
                status_class = ":::completed"
            elif step.status == StepStatus.FAILED:
                status_class = ":::failed"
            elif step.status == StepStatus.IN_PROGRESS:
                status_class = ":::inprogress"
            
            mermaid_lines.append(f"    {shape}{status_class}")
        
        # Add edges
        for i in range(len(chain.steps) - 1):
            mermaid_lines.append(f"    step{i} --> step{i+1}")
        
        # Add styling
        mermaid_lines.extend([
            "    classDef completed fill:#d4edda,stroke:#28a745",
            "    classDef failed fill:#f8d7da,stroke:#dc3545", 
            "    classDef inprogress fill:#fff3cd,stroke:#ffc107"
        ])
        
        return "\n".join(mermaid_lines)
    
    def get_chain_summary(self, chain_id: str) -> Dict[str, Any]:
        """Get summary statistics for a reasoning chain"""
        chain = self.get_reasoning_chain(chain_id)
        if not chain:
            return {"error": "Chain not found"}
        
        step_types = {}
        total_confidence = 0
        confidence_count = 0
        
        for step in chain.steps:
            step_type = step.thought_type.value
            step_types[step_type] = step_types.get(step_type, 0) + 1
            
            if step.confidence is not None:
                total_confidence += step.confidence
                confidence_count += 1
        
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else None
        
        return {
            "chain_id": chain.chain_id,
            "query": chain.query,
            "agent_type": chain.agent_type,
            "operation": chain.operation,
            "total_steps": len(chain.steps),
            "step_types": step_types,
            "success": chain.success,
            "avg_confidence": avg_confidence,
            "total_duration": chain.total_duration,
            "status_breakdown": {
                "completed": len([s for s in chain.steps if s.status == StepStatus.COMPLETED]),
                "failed": len([s for s in chain.steps if s.status == StepStatus.FAILED]),
                "in_progress": len([s for s in chain.steps if s.status == StepStatus.IN_PROGRESS]),
                "pending": len([s for s in chain.steps if s.status == StepStatus.PENDING])
            }
        }
    
    def get_recent_chains(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reasoning chains with summary info"""
        recent_chains = self.completed_chains[-limit:]
        
        summaries = []
        for chain in recent_chains:
            summary = {
                "chain_id": chain.chain_id,
                "query": chain.query[:100] + "..." if len(chain.query) > 100 else chain.query,
                "agent_type": chain.agent_type,
                "operation": chain.operation,
                "success": chain.success,
                "step_count": len(chain.steps),
                "confidence": chain.total_confidence,
                "duration": chain.total_duration,
                "timestamp": chain.start_time
            }
            summaries.append(summary)
        
        return summaries
    
    def create_treasury_reasoning_template(self, operation_type: str) -> List[Dict[str, Any]]:
        """
        Create reasoning step templates for common treasury operations
        
        Returns:
            List of step templates
        """
        templates = {
            "journal_mapping": [
                {
                    "thought_type": ThoughtType.INPUT_ANALYSIS,
                    "title": "Analyze Transaction Data",
                    "description": "Extract transaction type, asset class, and amount information"
                },
                {
                    "thought_type": ThoughtType.KNOWLEDGE_RETRIEVAL,
                    "title": "Retrieve Journal Rules",
                    "description": "Find matching journal mapping rules from database"
                },
                {
                    "thought_type": ThoughtType.REASONING_STEP,
                    "title": "Apply Business Logic",
                    "description": "Determine if transaction is invoice/payment or debit memo/refund"
                },
                {
                    "thought_type": ThoughtType.DECISION_POINT,
                    "title": "Select Account Mapping",
                    "description": "Choose appropriate debit and credit accounts"
                },
                {
                    "thought_type": ThoughtType.VALIDATION,
                    "title": "Validate Journal Entry",
                    "description": "Ensure debits equal credits and comply with rules"
                },
                {
                    "thought_type": ThoughtType.OUTPUT_GENERATION,
                    "title": "Generate Journal Suggestion",
                    "description": "Format final journal entry recommendation"
                }
            ],
            "transaction_analysis": [
                {
                    "thought_type": ThoughtType.INPUT_ANALYSIS,
                    "title": "Parse Transaction Details",
                    "description": "Extract date, amount, currency, and description"
                },
                {
                    "thought_type": ThoughtType.KNOWLEDGE_RETRIEVAL,
                    "title": "Gather Historical Context",
                    "description": "Retrieve similar transactions and patterns"
                },
                {
                    "thought_type": ThoughtType.REASONING_STEP,
                    "title": "Identify Transaction Category",
                    "description": "Classify transaction type and assess risk"
                },
                {
                    "thought_type": ThoughtType.REASONING_STEP,
                    "title": "Calculate Impact",
                    "description": "Determine financial and operational impact"
                },
                {
                    "thought_type": ThoughtType.OUTPUT_GENERATION,
                    "title": "Generate Analysis Report",
                    "description": "Compile findings and recommendations"
                }
            ]
        }
        
        return templates.get(operation_type, [])


class CoTContextManager:
    """Context manager for automatic chain-of-thought tracking"""
    
    def __init__(self, visualizer: CoTVisualizer, query: str, agent_type: str, operation: str):
        self.visualizer = visualizer
        self.query = query
        self.agent_type = agent_type
        self.operation = operation
        self.chain_id = None
        self.current_step_id = None
    
    def __enter__(self):
        self.chain_id = self.visualizer.start_reasoning_chain(
            query=self.query,
            agent_type=self.agent_type,
            operation=self.operation
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Handle error case
            self.visualizer.finish_reasoning_chain(
                chain_id=self.chain_id,
                final_output="",
                success=False,
                error_message=str(exc_val)
            )
        
        return False  # Don't suppress exceptions
    
    def add_step(self, thought_type: ThoughtType, title: str, description: str, **kwargs) -> str:
        """Add a reasoning step and return step ID"""
        self.current_step_id = self.visualizer.add_reasoning_step(
            chain_id=self.chain_id,
            thought_type=thought_type,
            title=title,
            description=description,
            **kwargs
        )
        return self.current_step_id
    
    def complete_step(self, output_data: Dict[str, Any] = None, confidence: float = None):
        """Complete the current reasoning step"""
        if self.current_step_id:
            self.visualizer.complete_reasoning_step(
                chain_id=self.chain_id,
                step_id=self.current_step_id,
                output_data=output_data,
                confidence=confidence
            )
    
    def finish(self, final_output: str, success: bool = True):
        """Finish the reasoning chain"""
        self.visualizer.finish_reasoning_chain(
            chain_id=self.chain_id,
            final_output=final_output,
            success=success
        ) 