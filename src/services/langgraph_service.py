"""
LangGraph Service for Agentic Treasury Document Processing Workflows

This service implements sophisticated agentic workflows using LangGraph for:
- Document parsing and analysis
- Journal mapping suggestions  
- Multi-agent collaboration for validation
- Complex treasury transaction processing
"""

import json
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..config import settings
from .monitoring import MonitoringService, AgentType, OperationStatus
from .guardrails import AIGuardrails


class WorkflowState(TypedDict):
    """State management for LangGraph workflows"""
    # Core data
    messages: Annotated[Sequence[BaseMessage], "Chat messages"]
    document_content: Optional[str]
    transaction_pairs: List[Dict[str, Any]]
    
    # Processing results
    parsed_data: Optional[Dict[str, Any]]
    journal_suggestions: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    
    # Workflow control
    current_agent: str
    next_action: str
    error_message: Optional[str]
    retry_count: int
    
    # Metadata
    workflow_id: str
    timestamp: str
    user_id: Optional[str]


@dataclass
class AgentConfig:
    """Configuration for specialized agents"""
    name: str
    role: str
    system_prompt: str
    tools: List[str]
    max_iterations: int = 3
    temperature: float = 0.1


class TreasuryDocumentProcessor:
    """Specialized agent for document parsing and initial analysis"""
    
    def __init__(self, llm: ChatOpenAI, monitoring: MonitoringService):
        self.llm = llm
        self.monitoring = monitoring
        self.logger = logging.getLogger(__name__)
        
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Process treasury document and extract transaction data"""
        
        start_time = time.time()
        
        try:
            document_content = state.get("document_content", "")
            
            if not document_content:
                state["error_message"] = "No document content provided"
                state["next_action"] = "error"
                return state
            
            # Create specialized system prompt for treasury document processing
            system_prompt = """
            You are a Treasury Document Processing Agent specialized in extracting structured data from treasury statements and financial documents.
            
            Your tasks:
            1. Parse treasury documents (statements, confirmations, reports)
            2. Identify transaction types and asset classes
            3. Extract relevant financial data (amounts, dates, counterparties)
            4. Structure data for journal entry processing
            
            Focus on:
            - Transaction types: BUY, SELL, DIVIDEND, INTEREST, PLACEMENT, WITHDRAWAL, etc.
            - Asset classes: BOND, STOCK, FUND, ETF, CASH, DERIVATIVE, etc.
            - Key details: amounts, currencies, dates, reference numbers
            
            Return structured JSON with extracted transaction pairs and metadata.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract transaction data from this treasury document:\n\n{document_content}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse the response and extract transaction pairs
            parsed_data = self._parse_document_response(response.content)
            
            state["parsed_data"] = parsed_data
            state["transaction_pairs"] = parsed_data.get("transaction_pairs", [])
            state["current_agent"] = "document_processor"
            state["next_action"] = "journal_mapping" if state["transaction_pairs"] else "validation"
            
            # Record successful operation
            self.monitoring.record_operation(
                agent_type=AgentType.DOCUMENT_PARSER,
                operation="document_processing",
                status=OperationStatus.SUCCESS,
                response_time=time.time() - start_time,
                user_id=state.get("user_id")
            )
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            state["error_message"] = f"Document processing failed: {str(e)}"
            state["next_action"] = "error"
            
            self.monitoring.record_operation(
                agent_type=AgentType.DOCUMENT_PARSER,
                operation="document_processing",
                status=OperationStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e),
                user_id=state.get("user_id")
            )
        
        return state
    
    def _parse_document_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        try:
            # Try to extract JSON from response
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
                return json.loads(json_content)
            
            # Fallback: try to parse entire response as JSON
            return json.loads(response_content)
            
        except json.JSONDecodeError:
            # Fallback: extract transaction pairs manually
            return self._extract_transaction_pairs_fallback(response_content)
    
    def _extract_transaction_pairs_fallback(self, content: str) -> Dict[str, Any]:
        """Fallback method to extract transaction pairs from text response"""
        # Simple pattern matching for common treasury terms
        transaction_types = ["BUY", "SELL", "DIVIDEND", "INTEREST", "PLACEMENT", "WITHDRAWAL", "REDEMPTION", "SUBSCRIPTION"]
        asset_classes = ["BOND", "STOCK", "FUND", "ETF", "CASH", "DERIVATIVE"]
        
        found_pairs = []
        content_upper = content.upper()
        
        for txn_type in transaction_types:
            for asset_class in asset_classes:
                if txn_type in content_upper and asset_class in content_upper:
                    found_pairs.append({
                        "transaction_type": txn_type,
                        "asset_class": asset_class,
                        "description": f"{txn_type} - {asset_class}",
                        "source": "fallback_extraction"
                    })
        
        return {
            "transaction_pairs": found_pairs,
            "extraction_method": "fallback",
            "confidence": 0.7
        }


class JournalMappingAgent:
    """Specialized agent for creating journal entry mappings"""
    
    def __init__(self, llm: ChatOpenAI, monitoring: MonitoringService, guardrails: AIGuardrails):
        self.llm = llm
        self.monitoring = monitoring
        self.guardrails = guardrails
        self.logger = logging.getLogger(__name__)
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Generate journal mapping suggestions for transaction pairs"""
        
        start_time = time.time()
        
        try:
            transaction_pairs = state.get("transaction_pairs", [])
            
            if not transaction_pairs:
                state["journal_suggestions"] = {"success": False, "error": "No transaction pairs to process"}
                state["next_action"] = "validation"
                return state
            
            # Create specialized system prompt for journal mapping
            system_prompt = """
            You are a Treasury Journal Mapping Agent specialized in creating proper double-entry journal mappings for treasury transactions.
            
            CRITICAL TREASURY SYSTEM RULES:
            1. Every transaction creates TWO journal entries (a pair):
               - For OUTGOING transactions: INVOICE + PAYMENT
               - For INCOMING transactions: DEBIT MEMO + REFUND
            
            2. Transaction Classification:
               - OUTGOING (Invoice + Payment): BUY, PURCHASE, SUBSCRIPTION, PLACEMENT, INVESTMENT, ACQUISITION, DEPOSIT, TRANSFER_OUT, PAYMENT
               - INCOMING (Debit Memo + Refund): SELL, DISPOSAL, REDEMPTION, WITHDRAWAL, MATURITY, REPAYMENT, DIVIDEND, INTEREST, COUPON, INCOME, RETURN, REFUND, TRANSFER_IN, RECEIPT
            
            3. Each journal entry requires:
               - Multiple accounts (minimum 1 debit + 1 credit, can have more)
               - Debits must equal credits
               - Clear business purpose
            
            Use standard treasury account names and provide complete transaction flows.
            """
            
            # Create unique pairs for processing
            unique_pairs = list(set(
                (pair["transaction_type"], pair["asset_class"]) 
                for pair in transaction_pairs
            ))
            
            prompt = f"""
            Create proper double-entry journal mappings for these treasury transaction pairs:
            
            Transaction Pairs: {unique_pairs}
            
            For each pair, provide BOTH journal entries with complete account details and business logic explanation.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Generate journal suggestions with guardrails
            response = await self.llm.ainvoke(messages)
            
            # Validate output with guardrails
            guardrail_result = self.guardrails.validate_output(response.content)
            
            journal_suggestions = {
                "success": True,
                "suggestions": response.content,
                "unique_pairs_count": len(unique_pairs),
                "model_used": self.llm.model_name,
                "guardrail_passed": guardrail_result.passed,
                "agent": "journal_mapping_agent"
            }
            
            if not guardrail_result.passed:
                journal_suggestions["guardrail_warning"] = guardrail_result.message
                journal_suggestions["suggestions"] = self.guardrails.sanitize_output(response.content)
                journal_suggestions["sanitized"] = True
            
            state["journal_suggestions"] = journal_suggestions
            state["current_agent"] = "journal_mapping"
            state["next_action"] = "validation"
            
            # Record successful operation
            self.monitoring.record_operation(
                agent_type=AgentType.JOURNAL_MAPPER,
                operation="journal_mapping",
                status=OperationStatus.SUCCESS,
                response_time=time.time() - start_time,
                user_id=state.get("user_id")
            )
            
        except Exception as e:
            self.logger.error(f"Journal mapping failed: {str(e)}")
            state["journal_suggestions"] = {"success": False, "error": str(e)}
            state["next_action"] = "validation"
            
            self.monitoring.record_operation(
                agent_type=AgentType.JOURNAL_MAPPER,
                operation="journal_mapping",
                status=OperationStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e),
                user_id=state.get("user_id")
            )
        
        return state


class ValidationAgent:
    """Specialized agent for validating and fact-checking results"""
    
    def __init__(self, llm: ChatOpenAI, monitoring: MonitoringService):
        self.llm = llm
        self.monitoring = monitoring
        self.logger = logging.getLogger(__name__)
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Validate journal mappings and transaction data for accuracy"""
        
        start_time = time.time()
        
        try:
            journal_suggestions = state.get("journal_suggestions", {})
            transaction_pairs = state.get("transaction_pairs", [])
            
            if not journal_suggestions or not journal_suggestions.get("success"):
                validation_results = {
                    "success": False,
                    "error": "No journal suggestions to validate"
                }
            else:
                # Create validation prompt
                system_prompt = """
                You are a Treasury Validation Agent responsible for fact-checking and validating journal entries.
                
                Your tasks:
                1. Verify double-entry bookkeeping compliance (debits = credits)
                2. Check account name accuracy and standardization
                3. Validate business logic for transaction flows
                4. Identify potential errors or inconsistencies
                5. Suggest improvements or corrections
                
                Provide structured feedback with specific recommendations.
                """
                
                suggestions_text = journal_suggestions.get("suggestions", "")
                
                prompt = f"""
                Validate these journal mapping suggestions:
                
                Transaction Pairs: {transaction_pairs}
                
                Journal Suggestions:
                {suggestions_text}
                
                Check for:
                - Double-entry compliance
                - Account name accuracy
                - Business logic correctness
                - Treasury best practices
                
                Provide validation score (0-100) and specific feedback.
                """
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]
                
                response = await self.llm.ainvoke(messages)
                
                # Parse validation response
                validation_results = self._parse_validation_response(response.content)
            
            state["validation_results"] = validation_results
            state["current_agent"] = "validation"
            state["next_action"] = "complete"
            
            # Record successful operation
            self.monitoring.record_operation(
                agent_type=AgentType.VALIDATOR,
                operation="validation",
                status=OperationStatus.SUCCESS,
                response_time=time.time() - start_time,
                user_id=state.get("user_id")
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            state["validation_results"] = {"success": False, "error": str(e)}
            state["next_action"] = "complete"
            
            self.monitoring.record_operation(
                agent_type=AgentType.VALIDATOR,
                operation="validation",
                status=OperationStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e),
                user_id=state.get("user_id")
            )
        
        return state
    
    def _parse_validation_response(self, response_content: str) -> Dict[str, Any]:
        """Parse validation response and extract structured feedback"""
        try:
            # Try to extract JSON from response
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
                return json.loads(json_content)
            
            # Fallback: create structured response from text
            return {
                "success": True,
                "validation_score": 85,  # Default good score
                "feedback": response_content,
                "recommendations": [],
                "validation_method": "text_analysis"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse validation response: {str(e)}",
                "raw_response": response_content
            }


class LangGraphService:
    """Main LangGraph service for orchestrating agentic workflows"""
    
    def __init__(self, monitoring_service: MonitoringService = None, guardrails: AIGuardrails = None):
        self.monitoring = monitoring_service or MonitoringService()
        self.guardrails = guardrails or AIGuardrails()
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required for LangGraph service")
        
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.1
        )
        
        # Initialize specialized agents
        self.document_processor = TreasuryDocumentProcessor(self.llm, self.monitoring)
        self.journal_agent = JournalMappingAgent(self.llm, self.monitoring, self.guardrails)
        self.validator = ValidationAgent(self.llm, self.monitoring)
        
        # Initialize checkpointer for state management
        self.checkpointer = MemorySaver()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for treasury document processing"""
        
        # Define the workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("document_processor", self.document_processor.process)
        workflow.add_node("journal_mapping", self.journal_agent.process)
        workflow.add_node("validation", self.validator.process)
        workflow.add_node("error_handler", self._handle_error)
        
        # Define the workflow edges
        workflow.add_edge(START, "document_processor")
        
        # Conditional routing based on next_action
        workflow.add_conditional_edges(
            "document_processor",
            self._route_next_action,
            {
                "journal_mapping": "journal_mapping",
                "validation": "validation",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "journal_mapping",
            self._route_next_action,
            {
                "validation": "validation",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "validation",
            self._route_next_action,
            {
                "complete": END,
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _route_next_action(self, state: WorkflowState) -> str:
        """Route to next action based on state"""
        return state.get("next_action", "error")
    
    async def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors and prepare final state"""
        self.logger.error(f"Workflow error: {state.get('error_message', 'Unknown error')}")
        
        state["current_agent"] = "error_handler"
        state["next_action"] = "complete"
        
        return state
    
    async def process_document(
        self, 
        document_content: str, 
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a treasury document using the agentic workflow
        
        Args:
            document_content: The document content to process
            user_id: Optional user identifier
            workflow_id: Optional workflow identifier for tracking
            
        Returns:
            Processing results with agent outputs
        """
        
        if not workflow_id:
            workflow_id = f"workflow_{int(time.time())}_{user_id or 'anonymous'}"
        
        # Initialize workflow state
        initial_state = WorkflowState(
            messages=[],
            document_content=document_content,
            transaction_pairs=[],
            parsed_data=None,
            journal_suggestions=None,
            validation_results=None,
            current_agent="start",
            next_action="document_processing",
            error_message=None,
            retry_count=0,
            workflow_id=workflow_id,
            timestamp=datetime.now().isoformat(),
            user_id=user_id
        )
        
        try:
            # Execute the workflow
            config = {"configurable": {"thread_id": workflow_id}}
            result_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Compile final results
            return {
                "success": True,
                "workflow_id": workflow_id,
                "transaction_pairs": result_state.get("transaction_pairs", []),
                "parsed_data": result_state.get("parsed_data", {}),
                "journal_suggestions": result_state.get("journal_suggestions", {}),
                "validation_results": result_state.get("validation_results", {}),
                "processing_path": self._extract_processing_path(result_state),
                "metadata": {
                    "timestamp": result_state.get("timestamp"),
                    "current_agent": result_state.get("current_agent"),
                    "user_id": user_id
                }
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                }
            }
    
    async def process_manual_input(
        self,
        transaction_types: List[str],
        asset_classes: List[str],
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process manual transaction input using journal mapping agent
        
        Args:
            transaction_types: List of transaction types
            asset_classes: List of asset classes
            user_id: Optional user identifier
            workflow_id: Optional workflow identifier
            
        Returns:
            Processing results with journal suggestions
        """
        
        if not workflow_id:
            workflow_id = f"manual_{int(time.time())}_{user_id or 'anonymous'}"
        
        # Create transaction pairs from manual input
        transaction_pairs = []
        for txn_type in transaction_types:
            for asset_class in asset_classes:
                transaction_pairs.append({
                    "transaction_type": txn_type.strip().upper(),
                    "asset_class": asset_class.strip().upper(),
                    "description": f"{txn_type} - {asset_class}",
                    "source": "manual_input"
                })
        
        # Initialize state for journal mapping only
        initial_state = WorkflowState(
            messages=[],
            document_content=None,
            transaction_pairs=transaction_pairs,
            parsed_data={"transaction_pairs": transaction_pairs, "source": "manual"},
            journal_suggestions=None,
            validation_results=None,
            current_agent="start",
            next_action="journal_mapping",
            error_message=None,
            retry_count=0,
            workflow_id=workflow_id,
            timestamp=datetime.now().isoformat(),
            user_id=user_id
        )
        
        try:
            # Execute journal mapping directly
            result_state = await self.journal_agent.process(initial_state)
            
            # Execute validation
            if result_state.get("next_action") == "validation":
                result_state = await self.validator.process(result_state)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "input_type": "manual",
                "transaction_pairs": transaction_pairs,
                "journal_suggestions": result_state.get("journal_suggestions", {}),
                "validation_results": result_state.get("validation_results", {}),
                "manual_input": {
                    "transaction_types": transaction_types,
                    "asset_classes": asset_classes,
                    "total_combinations": len(transaction_pairs)
                },
                "metadata": {
                    "timestamp": result_state.get("timestamp"),
                    "current_agent": result_state.get("current_agent"),
                    "user_id": user_id
                }
            }
            
        except Exception as e:
            self.logger.error(f"Manual input processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id,
                "input_type": "manual",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                }
            }
    
    def _extract_processing_path(self, state: WorkflowState) -> List[str]:
        """Extract the path of agents that processed this workflow"""
        # This would be enhanced with actual state tracking
        # For now, return a simple path based on what was processed
        path = ["document_processor"]
        
        if state.get("journal_suggestions"):
            path.append("journal_mapping")
        
        if state.get("validation_results"):
            path.append("validation")
        
        if state.get("error_message"):
            path.append("error_handler")
        
        return path
    
    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve workflow state for a given workflow ID"""
        try:
            config = {"configurable": {"thread_id": workflow_id}}
            # This would retrieve state from checkpointer
            # Implementation depends on specific checkpointer features
            return {"workflow_id": workflow_id, "status": "retrievable"}
        except Exception as e:
            self.logger.error(f"Failed to retrieve workflow state: {str(e)}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the status of the LangGraph service"""
        return {
            "service": "LangGraphService",
            "status": "active",
            "llm_model": self.llm.model_name,
            "agents": ["document_processor", "journal_mapping", "validation"],
            "checkpointer": "MemorySaver",
            "timestamp": datetime.now().isoformat()
        } 