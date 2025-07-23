"""
OpenAI Service for Embeddings and LLM with Integrated Monitoring and Guardrails
"""

from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import time
import logging
import json

from ..config import settings
from .monitoring import MonitoringService, AgentType, OperationStatus
from .guardrails import AIGuardrails
from .agent_personas import AgentPersonalizationService
from .cot_visualizer import CoTVisualizer, CoTContextManager, ThoughtType


class OpenAIService:
    """Enhanced OpenAI service with monitoring, guardrails, and personalization"""
    
    def __init__(
        self, 
        api_key: str = None,
        monitoring_service: MonitoringService = None,
        guardrails: AIGuardrails = None,
        personas: AgentPersonalizationService = None,
        cot_visualizer: CoTVisualizer = None
    ):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key, timeout=settings.openai_timeout)
        self.model = settings.openai_model
        self.embedding_model = settings.openai_embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Performance optimizations
        self._cache = {}  # Simple response cache
        self._cache_ttl = 60   # 1 minute cache TTL for faster updates
        self.fast_mode = True  # Enable aggressive optimizations
        
        # Initialize integrated services
        self.monitoring = monitoring_service or MonitoringService()
        self.guardrails = guardrails or AIGuardrails()
        self.personas = personas or AgentPersonalizationService()
        self.cot_visualizer = cot_visualizer or CoTVisualizer()
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Failed to create embedding: {str(e)}")
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding lists
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise Exception(f"Failed to create batch embeddings: {str(e)}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        user_id: str = None,
        agent_type: AgentType = AgentType.JOURNAL_MAPPER,
        operation: str = "chat_completion",
        enable_monitoring: bool = False,  # Disable for speed
        enable_guardrails: bool = False,  # Disable for speed
        persona_mode: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced chat completion with monitoring, guardrails, and personalization
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            user_id: User identifier for personalization and rate limiting
            agent_type: Type of agent for monitoring
            operation: Operation name for monitoring
            enable_monitoring: Whether to enable monitoring
            enable_guardrails: Whether to enable guardrails
            persona_mode: Persona mode to apply
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        # Generate cache key for identical requests
        cache_key = f"{hash(str(messages))}-{temperature}-{max_tokens}-{self.model}"
        
        # Check cache first
        if cache_key in self._cache:
            cached_response, cache_time = self._cache[cache_key]
            if time.time() - cache_time < self._cache_ttl:
                cached_response["cached"] = True
                return cached_response
        
        try:
            # Input validation with guardrails
            if enable_guardrails and messages:
                user_input = messages[-1].get('content', '') if messages else ''
                guardrail_result = self.guardrails.validate_input(user_input, user_id)
                
                if not guardrail_result.passed:
                    if enable_monitoring:
                        self.monitoring.record_operation(
                            agent_type=agent_type,
                            operation=operation,
                            status=OperationStatus.GUARDRAIL_BLOCKED,
                            response_time=time.time() - start_time,
                            error_message=guardrail_result.message,
                            user_id=user_id
                        )
                    
                    return {
                        "success": False,
                        "error": "Input blocked by guardrails",
                        "details": guardrail_result.message,
                        "threat_level": guardrail_result.threat_level.value
                    }
            
            # Apply persona if specified
            if persona_mode and user_id:
                persona_config = self.personas.get_persona_config(user_id, persona_mode)
                system_prompt = self.personas.generate_system_prompt(persona_config, "treasury")
                
                # Update or add system message
                if messages and messages[0].get('role') == 'system':
                    messages[0]['content'] = system_prompt
                else:
                    messages.insert(0, {'role': 'system', 'content': system_prompt})
            
            # Make API call with performance optimizations
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or settings.openai_max_tokens,
                stream=False,  # Ensure no streaming
                top_p=0.9,     # Focus on higher probability tokens
                frequency_penalty=0.2  # Reduce repetition
            )
            
            response_text = response.choices[0].message.content
            token_count = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            # Output validation with guardrails
            if enable_guardrails:
                output_guardrail_result = self.guardrails.validate_output(response_text)
                
                if not output_guardrail_result.passed:
                    # Sanitize output
                    response_text = self.guardrails.sanitize_output(response_text)
                    
                    if enable_monitoring:
                        self.monitoring.record_operation(
                            agent_type=agent_type,
                            operation=operation,
                            status=OperationStatus.SUCCESS,
                            response_time=time.time() - start_time,
                            token_count=token_count,
                            user_id=user_id,
                            metadata={"output_sanitized": True, "guardrail_issue": output_guardrail_result.message}
                        )
                    
                    return {
                        "success": True,
                        "response": response_text,
                        "token_count": token_count,
                        "sanitized": True,
                        "guardrail_warning": output_guardrail_result.message
                    }
            
            # Apply persona formatting
            if persona_mode and user_id:
                persona_config = self.personas.get_persona_config(user_id, persona_mode)
                response_text = self.personas.adapt_response(response_text, persona_config)
            
            # Record successful operation
            if enable_monitoring:
                self.monitoring.record_operation(
                    agent_type=agent_type,
                    operation=operation,
                    status=OperationStatus.SUCCESS,
                    response_time=time.time() - start_time,
                    token_count=token_count,
                    user_id=user_id,
                    input_size=len(str(messages)),
                    output_size=len(response_text)
                )
            
            result = {
                "success": True,
                "response": response_text,
                "token_count": token_count,
                "response_time": time.time() - start_time,
                "sanitized": False,
                "cached": False
            }
            
            # Cache successful response
            self._cache[cache_key] = (result.copy(), time.time())
            
            return result
            
        except Exception as e:
            error_message = str(e)
            
            # Record failed operation
            if enable_monitoring:
                self.monitoring.record_operation(
                    agent_type=agent_type,
                    operation=operation,
                    status=OperationStatus.FAILURE,
                    response_time=time.time() - start_time,
                    error_message=error_message,
                    user_id=user_id
                )
            
            return {
                "success": False,
                "error": f"Failed to generate chat completion: {error_message}",
                "response_time": time.time() - start_time
            }
    
    def analyze_transactions_with_cot(
        self, 
        transaction_data: Dict[str, Any], 
        user_id: str = None,
        enable_cot: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze transaction data using OpenAI with chain-of-thought reasoning
        
        Args:
            transaction_data: Processed transaction data
            user_id: User identifier for personalization
            enable_cot: Whether to enable chain-of-thought tracking
            
        Returns:
            Analysis results with reasoning chain
        """
        if enable_cot:
            with CoTContextManager(
                self.cot_visualizer, 
                f"Analyze {len(transaction_data.get('transaction_pairs', []))} transactions",
                "transaction_analyzer",
                "analyze_transactions"
            ) as cot:
                return self._analyze_transactions_internal(transaction_data, user_id, cot)
        else:
            return self._analyze_transactions_internal(transaction_data, user_id)
    
    def _analyze_transactions_internal(
        self, 
        transaction_data: Dict[str, Any], 
        user_id: str = None,
        cot: CoTContextManager = None
    ) -> Dict[str, Any]:
        """Internal transaction analysis with optional CoT tracking"""
        try:
            # Step 1: Parse and validate input data
            if cot:
                cot.add_step(
                    ThoughtType.INPUT_ANALYSIS,
                    "Parse Transaction Data",
                    "Extract and validate transaction information",
                    input_data={"transaction_count": len(transaction_data.get("transaction_pairs", []))}
                )
            
            transaction_pairs = transaction_data.get("transaction_pairs", [])
            if not transaction_pairs:
                if cot:
                    cot.complete_step({"error": "No transaction pairs found"}, confidence=0.0)
                    cot.finish("No transaction data to analyze", success=False)
                return {"success": False, "error": "No transaction data provided"}
            
            if cot:
                cot.complete_step(
                    {"valid_transactions": len(transaction_pairs)}, 
                    confidence=1.0
                )
            
            # Step 2: Categorize transactions
            if cot:
                cot.add_step(
                    ThoughtType.REASONING_STEP,
                    "Categorize Transactions",
                    "Group transactions by type and asset class",
                    reasoning="Analyzing transaction patterns to identify categories"
                )
            
            transaction_categories = {}
            asset_classes = set()
            
            for pair in transaction_pairs:
                trans_type = pair.get("transaction_type", "unknown")
                asset_class = pair.get("asset_class", "unknown")
                
                key = f"{trans_type}_{asset_class}"
                transaction_categories[key] = transaction_categories.get(key, 0) + 1
                asset_classes.add(asset_class)
            
            if cot:
                cot.complete_step(
                    {
                        "categories": transaction_categories,
                        "unique_asset_classes": list(asset_classes),
                        "category_count": len(transaction_categories)
                    },
                    confidence=0.95
                )
            
            # Step 3: Risk assessment
            if cot:
                cot.add_step(
                    ThoughtType.REASONING_STEP,
                    "Assess Risk Factors",
                    "Evaluate potential risks in transaction patterns",
                    reasoning="Looking for unusual patterns or high-risk transactions"
                )
            
            # Simple risk assessment logic
            high_volume_threshold = 1000000  # $1M
            risk_factors = []
            
            for pair in transaction_pairs:
                # Check for high-value transactions (if amount data available)
                if "amount" in pair and float(pair.get("amount", 0)) > high_volume_threshold:
                    risk_factors.append(f"High-value transaction: {pair['transaction_type']}")
            
            risk_level = "LOW"
            if len(risk_factors) > 5:
                risk_level = "HIGH"
            elif len(risk_factors) > 2:
                risk_level = "MEDIUM"
            
            if cot:
                cot.complete_step(
                    {
                        "risk_level": risk_level,
                        "risk_factors": risk_factors,
                        "risk_count": len(risk_factors)
                    },
                    confidence=0.8
                )
            
            # Step 4: Generate AI analysis
            if cot:
                cot.add_step(
                    ThoughtType.OUTPUT_GENERATION,
                    "Generate AI Analysis",
                    "Create comprehensive analysis using language model",
                    reasoning="Combining categorization and risk assessment for detailed analysis"
                )
            
            prompt = f"""
            Analyze the following treasury transaction data and provide insights:
            
            Transaction Summary:
            - Total transactions: {len(transaction_pairs)}
            - Categories: {transaction_categories}
            - Asset classes: {list(asset_classes)}
            - Risk level: {risk_level}
            - Risk factors: {risk_factors}
            
            Please provide:
            1. Summary of transaction types and patterns
            2. Risk assessment and mitigation recommendations
            3. Journal mapping suggestions
            4. Any anomalies or operational concerns
            5. Compliance considerations
            
            Format your response as a structured analysis with clear headings.
            """
            
            messages = [
                {"role": "system", "content": "You are a treasury operations expert specializing in transaction analysis and risk assessment."},
                {"role": "user", "content": prompt}
            ]
            
            ai_result = self.chat_completion(
                messages,
                user_id=user_id,
                agent_type=AgentType.TRANSACTION_ANALYZER,
                operation="analyze_transactions"
            )
            
            if not ai_result["success"]:
                if cot:
                    cot.complete_step({"error": ai_result["error"]}, confidence=0.0)
                    cot.finish("AI analysis failed", success=False)
                return ai_result
            
            ai_analysis = ai_result["response"]
            
            if cot:
                cot.complete_step(
                    {
                        "analysis_length": len(ai_analysis),
                        "token_count": ai_result.get("token_count"),
                        "response_time": ai_result.get("response_time")
                    },
                    confidence=0.9
                )
            
            # Compile final results
            final_result = {
                "success": True,
                "analysis": ai_analysis,
                "metadata": {
                    "transaction_count": len(transaction_pairs),
                    "categories": transaction_categories,
                    "asset_classes": list(asset_classes),
                    "risk_level": risk_level,
                    "risk_factors": risk_factors,
                    "model_used": self.model,
                    "token_count": ai_result.get("token_count"),
                    "response_time": ai_result.get("response_time")
                }
            }
            
            if cot:
                cot.finish(ai_analysis[:200] + "...", success=True)
            
            return final_result
            
        except Exception as e:
            error_message = str(e)
            
            if cot:
                cot.finish("", success=False, error_message=error_message)
            
            return {
                "success": False,
                "error": error_message
            }
    
    # Backward compatibility
    def analyze_transactions(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        result = self.analyze_transactions_with_cot(transaction_data, enable_cot=False)
        
        # Convert to legacy format
        if result["success"]:
            return {
                "success": True,
                "analysis": result["analysis"],
                "transaction_count": result["metadata"]["transaction_count"],
                "model_used": result["metadata"]["model_used"]
            }
        else:
            return result
    
    def suggest_journal_mappings(self, transaction_pairs: List[Dict[str, str]], enable_guardrails: bool = True) -> Dict[str, Any]:
        """
        Suggest journal mappings for transaction pairs
        
        Args:
            transaction_pairs: List of transaction type and asset class pairs
            enable_guardrails: Whether to enable guardrails (can be disabled for trusted treasury operations)
            
        Returns:
            Suggested mappings
        """
        try:
            # Create prompt for journal mapping suggestions
            unique_pairs = list(set(
                (pair["transaction_type"], pair["asset_class"]) 
                for pair in transaction_pairs
            ))
            
            prompt = f"""
            Based on these unique transaction type and asset class pairs, create proper double-entry journal mappings following treasury management system rules:
            
            Transaction Pairs:
            {unique_pairs}
            
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
            
            For each transaction pair, provide BOTH journal entries:
            
            **[Transaction Type] - [Asset Class]**
            
            **Entry 1: [INVOICE/DEBIT MEMO]**
            - Account 1 (Debit): [Account Name] - [Amount/Description]
            - Account 2 (Credit): [Account Name] - [Amount/Description]
            - Account 3 (if needed): [Account Name] - [Amount/Description]
            - Purpose: [Why this entry is created]
            
            **Entry 2: [PAYMENT/REFUND]**
            - Account 1 (Debit): [Account Name] - [Amount/Description]
            - Account 2 (Credit): [Account Name] - [Amount/Description]
            - Account 3 (if needed): [Account Name] - [Amount/Description]
            - Purpose: [Why this entry is created]
            
            **Business Logic:** [Explain the complete transaction flow]
            
            Use standard treasury account names like:
            - Cash and Cash Equivalents
            - Investment Securities
            - Accounts Payable
            - Accounts Receivable
            - Interest Income
            - Interest Expense
            - Dividend Income
            - Investment Gains/Losses
            - Settlement Clearing
            - Custody Bank Account
            - Operational Bank Account
            """
            
            messages = [
                {"role": "system", "content": "You are a treasury accounting expert specializing in double-entry journal mappings for treasury management systems. You understand that EVERY transaction requires TWO journal entries: 1) Invoice/Debit Memo entry, and 2) Payment/Refund entry. Each entry must have balanced debits and credits. You create complete transaction flows showing how money moves between accounts in treasury operations."},
                {"role": "user", "content": prompt}
            ]
            
            # Call with optional guardrails
            response = self.chat_completion(
                messages, 
                max_tokens=2000,
                enable_guardrails=enable_guardrails,
                agent_type=AgentType.JOURNAL_MAPPER,
                operation="suggest_journal_mappings"
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "suggestions": response["response"],
                    "unique_pairs_count": len(unique_pairs),
                    "model_used": self.model,
                    "sanitized": response.get("sanitized", False),
                    "guardrail_warning": response.get("guardrail_warning")
                }
            else:
                return response
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 

    def batch_journal_mapping(
        self,
        transactions: List[Dict[str, Any]],
        user_id: str = None,
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple transactions in batches for better performance
        
        Args:
            transactions: List of transaction data
            user_id: User identifier
            batch_size: Number of transactions to process in one API call
            
        Returns:
            List of journal mapping results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            
            # Create batch prompt
            batch_prompt = "Process these transactions for journal mapping:\n\n"
            for idx, txn in enumerate(batch):
                batch_prompt += f"Transaction {idx + 1}:\n{json.dumps(txn, indent=2)}\n\n"
            
            batch_prompt += """
For each transaction, provide journal mapping in this exact format:
{
  "transaction_1": {"account_dr": "...", "account_cr": "...", "confidence": 0.95},
  "transaction_2": {"account_dr": "...", "account_cr": "...", "confidence": 0.95},
  ...
}"""

            messages = [
                {"role": "system", "content": "You are a treasury accounting expert. Map transactions to journal entries efficiently."},
                {"role": "user", "content": batch_prompt}
            ]
            
            # Get batch response
            response = self.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
                user_id=user_id,
                agent_type=AgentType.JOURNAL_MAPPER,
                operation="batch_journal_mapping"
            )
            
            if response["success"]:
                try:
                    # Parse batch response
                    batch_results = json.loads(response["response"])
                    for j, txn in enumerate(batch):
                        key = f"transaction_{j + 1}"
                        if key in batch_results:
                            results.append({
                                "transaction": txn,
                                "mapping": batch_results[key],
                                "batch_processed": True
                            })
                        else:
                            results.append({
                                "transaction": txn,
                                "mapping": None,
                                "error": "Failed to parse batch response"
                            })
                except json.JSONDecodeError:
                    # Fallback to individual processing
                    for txn in batch:
                        results.append({
                            "transaction": txn,
                            "mapping": None,
                            "error": "Batch parsing failed, use individual processing"
                        })
            else:
                # Fallback to individual processing
                for txn in batch:
                    results.append({
                        "transaction": txn,
                        "mapping": None,
                        "error": response.get("error", "Batch processing failed")
                    })
        
        return results 

    def fast_journal_mapping(
        self,
        transaction_data: Dict[str, Any],
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Ultra-fast journal mapping with minimal prompt and aggressive optimization
        
        Args:
            transaction_data: Transaction to map
            user_id: User identifier
            
        Returns:
            Quick journal mapping result
        """
        # Ultra-short prompt for speed
        prompt = f"Map to journal entry: {json.dumps(transaction_data, separators=(',', ':'))}"
        
        messages = [
            {"role": "system", "content": "Map transaction to DR/CR accounts. Format: DR:account_name,CR:account_name"},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(
            messages=messages,
            temperature=0,  # No randomness for speed
            max_tokens=100,  # Very short response
            user_id=user_id,
            agent_type=AgentType.JOURNAL_MAPPER,
            operation="fast_journal_mapping"
        ) 