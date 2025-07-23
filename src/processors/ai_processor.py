"""
AI-Enhanced Multimodal Processor for TReA
Combines your existing API with OpenAI and PostgreSQL vector database
Supports PDF, text, JSON, and CSV inputs
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st

from ..services.api_client import TReAAPIClient
from ..services.openai_service import OpenAIService
from ..services.vector_db import VectorDBService
from .brave_processor import BraveSearchProcessor, search_multiple_transactions_sync
from .pdf_processor import PDFProcessor
from .text_processor import TextProcessor, FileValidator
from ..config import settings


class AIEnhancedProcessor:
    """AI-Enhanced multimodal processor supporting PDF, text, JSON, and CSV inputs"""
    
    def __init__(
        self,
        api_client: TReAAPIClient = None,
        openai_service: OpenAIService = None,
        vector_db: VectorDBService = None,
        brave_processor: BraveSearchProcessor = None
    ):
        self.api_client = api_client or TReAAPIClient()
        
        # Initialize processors for different file types
        self.pdf_processor = PDFProcessor(self.api_client)
        self.text_processor = TextProcessor(self.api_client)
        
        # Initialize OpenAI service (optional)
        try:
            self.openai_service = openai_service or OpenAIService()
            self.openai_available = True
        except ValueError:
            self.openai_service = None
            self.openai_available = False
        except Exception as e:
            self.openai_service = None
            self.openai_available = False
        
        # Initialize Vector DB (optional)
        try:
            self.vector_db = vector_db or VectorDBService()
            self.vector_db_available = True
        except (ValueError, Exception):
            self.vector_db = None
            self.vector_db_available = False
        
        # Initialize Brave Search (optional)
        try:
            if settings.brave_search_enabled:
                self.brave_processor = brave_processor or BraveSearchProcessor(settings.brave_api_key)
                self.brave_search_available = True
            else:
                self.brave_processor = None
                self.brave_search_available = False
        except Exception:
            self.brave_processor = None
            self.brave_search_available = False
        
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location"""
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{uploaded_file.name}"
        file_path = self.upload_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension"""
        extension = Path(file_path).suffix.lower()
        if extension == '.pdf':
            return 'pdf'
        elif extension in ['.txt', '.json', '.csv']:
            return 'text'
        else:
            return 'unknown'
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process document with AI enhancement (supports PDF, text, JSON, CSV)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Enhanced processing results
        """
        # Detect file type and route to appropriate processor
        file_type = self.detect_file_type(file_path)
        
        if file_type == 'pdf':
            return self._process_pdf_with_ai(file_path)
        elif file_type == 'text':
            return self._process_text_with_ai(file_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {Path(file_path).suffix}",
                "file_info": self._get_file_info(file_path)
            }
    
    def _process_pdf_with_ai(self, file_path: str) -> Dict[str, Any]:
        """
        Process PDF document with AI enhancement
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Enhanced processing results
        """
        results = {
            "success": False,
            "file_info": {},
            "extraction_data": {},
            "transformed_data": {},
            "mapped_data": {},
            "transaction_pairs": [],
            "ai_analysis": {},
            "journal_suggestions": {},
            "similar_transactions": [],
            "transaction_definitions": {},
            "error": None,
            "input_type": "pdf"  # Identify this as PDF processing
        }
        
        try:
            # Get file information
            results["file_info"] = self._get_file_info(file_path)
            
            # Step 1: Use your existing API for PDF processing
            upload_result = self.api_client.upload_file(file_path)
            if not upload_result["success"]:
                results["error"] = f"Upload failed: {upload_result.get('error', 'Unknown error')}"
                return results
            
            results["extraction_data"] = upload_result["data"]
            
            # Step 2: Extract PDF content using your API
            extract_result = self.api_client.extract_pdf(upload_result["data"])
            if not extract_result["success"]:
                results["error"] = f"Extraction failed: {extract_result.get('error', 'Unknown error')}"
                return results
            
            # Step 3: Transform data using your API
            transform_result = self.api_client.transform_data(extract_result["data"]["data"])
            if not transform_result["success"]:
                results["error"] = f"Transformation failed: {transform_result.get('error', 'Unknown error')}"
                return results
            
            results["transformed_data"] = transform_result["data"]
            
            # Step 4: Map transactions using your API
            map_result = self.api_client.map_transactions(transform_result["data"]["data"])
            if not map_result["success"]:
                results["error"] = f"Mapping failed: {map_result.get('error', 'Unknown error')}"
                return results
            
            results["mapped_data"] = map_result["data"]
            results["transaction_pairs"] = self._extract_transaction_pairs(map_result["data"])

            st.write(results["transaction_pairs"], 'transaction_pairs')
            
            # Step 5: Search for transaction definitions using Brave Search
            if self.brave_search_available and results["transaction_pairs"]:
                try:
                    transaction_definitions = search_multiple_transactions_sync(
                        results["transaction_pairs"], 
                        settings.brave_api_key
                    )
                    results["transaction_definitions"] = transaction_definitions
                    
                    # Display definitions in Streamlit
                    if transaction_definitions.get("success") and transaction_definitions.get("definitions_found") > 0:
                        st.success(f"Found definitions for {transaction_definitions['definitions_found']} transaction types")
                        
                        # Show definitions in expandable sections
                        for key, definition_data in transaction_definitions["transactions"].items():
                            if definition_data.get("success") and definition_data.get("definitions"):
                                transaction_type = definition_data["transaction_type"]
                                asset_class = definition_data["asset_class"]
                                
                                with st.expander(f"📖 Definition: {transaction_type} ({asset_class})"):
                                    st.write(definition_data["summary"])
                                    
                                    if definition_data["definitions"]:
                                        st.subheader("Sources:")
                                        for i, defn in enumerate(definition_data["definitions"][:3]):
                                            st.write(f"**{i+1}. {defn['title']}**")
                                            st.write(defn["description"])
                                            st.write(f"🔗 [Source]({defn['url']})")
                                            st.write("---")
                    else:
                        st.warning("No transaction definitions found in search results")
                        
                except Exception as e:
                    results["transaction_definitions"] = {"success": False, "error": str(e)}
                    st.error(f"Error searching transaction definitions: {str(e)}")
                    
                # Show more detailed error information if search failed
                if not transaction_definitions.get("success"):
                    if "API key is required" in str(transaction_definitions.get("error", "")):
                        st.warning("🔑 Brave Search API key is required. Please add BRAVE_API_KEY to your .env file to enable transaction definition search.")
                        st.info("You can get a free API key from: https://brave.com/search/api/")
                    else:
                        st.error(f"Brave Search Error: {transaction_definitions.get('error', 'Unknown error')}")
                        
                        # Show individual transaction errors if available
                        if transaction_definitions.get("errors"):
                            with st.expander("🔍 Detailed Error Information"):
                                for error in transaction_definitions["errors"][:3]:  # Show first 3 errors
                                    st.write(f"**{error['transaction_type']} ({error['asset_class']})**: {error['error']}")
            
            # Step 6: AI Enhancement (if available)
            if self.openai_available and results["transaction_pairs"]:
                try:
                    # AI analysis of transactions
                    ai_analysis = self.openai_service.analyze_transactions(results["mapped_data"])
                    results["ai_analysis"] = ai_analysis
                    
                    # AI suggestions for journal mappings - DISABLE GUARDRAILS FOR PDF PROCESSING
                    journal_suggestions = self.openai_service.suggest_journal_mappings(
                        results["transaction_pairs"],
                        enable_guardrails=False  # Disable guardrails for PDF processing
                    )
                    results["journal_suggestions"] = journal_suggestions
                    
                except Exception as e:
                    results["ai_analysis"] = {"success": False, "error": str(e)}
                    results["journal_suggestions"] = {"success": False, "error": str(e)}
            
            # Step 6b: Fallback journal suggestions (if OpenAI not available)
            if not self.openai_available and results["transaction_pairs"]:
                try:
                    fallback_suggestions = self._generate_fallback_journal_suggestions(results["transaction_pairs"])
                    results["journal_suggestions"] = fallback_suggestions
                except Exception as e:
                    results["journal_suggestions"] = {"success": False, "error": f"Fallback suggestions failed: {str(e)}"}
            
            # Step 7: Vector database operations (if available)
            if self.vector_db_available and self.openai_available and results["transaction_pairs"]:
                try:
                    similar_transactions = self._find_and_store_similar_transactions(results["transaction_pairs"])
                    results["similar_transactions"] = similar_transactions
                except Exception as e:
                    results["similar_transactions"] = []
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = f"Processing error: {str(e)}"
        
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        return results
    
    def _process_text_with_ai(self, file_path: str) -> Dict[str, Any]:
        """
        Process text-based document (TXT, JSON, CSV) with AI enhancement
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Enhanced processing results
        """
        results = {
            "success": False,
            "file_info": {},
            "extraction_data": {},
            "transformed_data": {},
            "mapped_data": {},
            "transaction_pairs": [],
            "ai_analysis": {},
            "journal_suggestions": {},
            "similar_transactions": [],
            "transaction_definitions": {},
            "error": None,
            "input_type": "text"
        }
        
        try:
            # Get file information
            results["file_info"] = self._get_file_info(file_path)
            
            # Step 1: Process text/JSON/CSV file
            text_result = self.text_processor.process_text_document(file_path)
            if not text_result["success"]:
                results["error"] = f"Text processing failed: {text_result.get('error', 'Unknown error')}"
                return results
            
            # Copy results from text processor
            results["extraction_data"] = text_result["extraction_data"]
            results["transformed_data"] = text_result["transformed_data"]
            results["mapped_data"] = text_result["mapped_data"]
            results["transaction_pairs"] = text_result["transaction_pairs"]
            
            st.write(results["transaction_pairs"], 'transaction_pairs from text')
            
            # Step 2: Search for transaction definitions using Brave Search
            if self.brave_search_available and results["transaction_pairs"]:
                try:
                    transaction_definitions = search_multiple_transactions_sync(
                        results["transaction_pairs"], 
                        settings.brave_api_key
                    )
                    results["transaction_definitions"] = transaction_definitions
                    
                    # Display definitions in Streamlit
                    if transaction_definitions.get("success") and transaction_definitions.get("definitions_found") > 0:
                        st.success(f"Found definitions for {transaction_definitions['definitions_found']} transaction types")
                        
                        # Show definitions in expandable sections
                        for key, definition_data in transaction_definitions["transactions"].items():
                            if definition_data.get("success") and definition_data.get("definitions"):
                                transaction_type = definition_data["transaction_type"]
                                asset_class = definition_data["asset_class"]
                                
                                with st.expander(f"📖 Definition: {transaction_type} ({asset_class})"):
                                    st.write(definition_data["summary"])
                                    
                                    if definition_data["definitions"]:
                                        st.subheader("Sources:")
                                        for i, defn in enumerate(definition_data["definitions"][:3]):
                                            st.write(f"**{i+1}. {defn['title']}**")
                                            st.write(defn["description"])
                                            st.write(f"🔗 [Source]({defn['url']})")
                                            st.write("---")
                    else:
                        st.warning("No transaction definitions found in search results")
                        
                except Exception as e:
                    results["transaction_definitions"] = {"success": False, "error": str(e)}
                    st.error(f"Error searching transaction definitions: {str(e)}")
                    
                # Show more detailed error information if search failed
                if not transaction_definitions.get("success"):
                    if "API key is required" in str(transaction_definitions.get("error", "")):
                        st.warning("🔑 Brave Search API key is required. Please add BRAVE_API_KEY to your .env file to enable transaction definition search.")
                        st.info("You can get a free API key from: https://brave.com/search/api/")
                    else:
                        st.error(f"Brave Search Error: {transaction_definitions.get('error', 'Unknown error')}")
                        
                        # Show individual transaction errors if available
                        if transaction_definitions.get("errors"):
                            with st.expander("🔍 Detailed Error Information"):
                                for error in transaction_definitions["errors"][:3]:  # Show first 3 errors
                                    st.write(f"**{error['transaction_type']} ({error['asset_class']})**: {error['error']}")
            
            # Step 3: AI Enhancement (if available)
            if self.openai_available and results["transaction_pairs"]:
                try:
                    # AI analysis of transactions
                    ai_analysis = self.openai_service.analyze_transactions(results["mapped_data"])
                    results["ai_analysis"] = ai_analysis
                    
                    # AI suggestions for journal mappings - DISABLE GUARDRAILS FOR TEXT FILE PROCESSING
                    journal_suggestions = self.openai_service.suggest_journal_mappings(
                        results["transaction_pairs"],
                        enable_guardrails=False  # Disable guardrails for automatic file processing
                    )
                    results["journal_suggestions"] = journal_suggestions
                    
                except Exception as e:
                    results["ai_analysis"] = {"success": False, "error": str(e)}
                    results["journal_suggestions"] = {"success": False, "error": str(e)}
            
            # Step 3b: Fallback journal suggestions (if OpenAI not available)
            if not self.openai_available and results["transaction_pairs"]:
                try:
                    fallback_suggestions = self._generate_fallback_journal_suggestions(results["transaction_pairs"])
                    results["journal_suggestions"] = fallback_suggestions
                except Exception as e:
                    results["journal_suggestions"] = {"success": False, "error": f"Fallback suggestions failed: {str(e)}"}
            
            # Step 4: Vector database operations (if available)
            if self.vector_db_available and self.openai_available and results["transaction_pairs"]:
                try:
                    similar_transactions = self._find_and_store_similar_transactions(results["transaction_pairs"])
                    results["similar_transactions"] = similar_transactions
                except Exception as e:
                    results["similar_transactions"] = []
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = f"Processing error: {str(e)}"
        
        finally:
            # Clean up uploaded file (handled by text processor)
            pass
        
        return results
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information"""
        path = Path(file_path)
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "extension": path.suffix.lower()
        }
    
    def _extract_transaction_pairs(self, map_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract unique transaction pairs from mapped data"""
        pairs = []
        seen_pairs = set()  # Track unique combinations of (transaction_type, asset_class)
        
        if not map_data or 'data' not in map_data:
            return pairs
        
        data = map_data['data']
        
        # Process cash transactions
        if 'cash_transactions' in data:
            for transaction in data['cash_transactions']:
                transaction_type = transaction.get('TRANSACTION_TYPE', '')
                asset_class = transaction.get('ASSET_CLASS', 'CASH')
                
                # Create unique key for this combination
                pair_key = (transaction_type, asset_class)
                
                # Only add if we haven't seen this combination before
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pair = {
                        'transaction_type': transaction_type,
                        'asset_class': asset_class,
                        'category': 'Cash Transaction',
                        'description': f"{transaction_type} - {asset_class}"
                    }
                    pairs.append(pair)
        
        # Process asset transactions  
        if 'asset_transactions' in data:
            for transaction in data['asset_transactions']:
                transaction_type = transaction.get('TRANSACTION_TYPE', '')
                if not transaction_type:
                    continue
                    
                asset_class = transaction.get('ASSET_CLASS', 'ASSET')

                if asset_class == 'ASSET':
                    asset_name = transaction.get('ASSET_NAME', '')
                    if 'fund' in asset_name.lower():
                        asset_class = 'FUND'
                    elif 'stock' in asset_name.lower():
                        asset_class = 'STOCK'
                    elif 'bond' in asset_name.lower():
                        asset_class = 'BOND'
                    elif 'etf' in asset_name.lower():
                        asset_class = 'ETF'
                    else:
                        asset_class = 'CASH'
                
                # Create unique key for this combination
                pair_key = (transaction_type, asset_class)
                
                # Only add if we haven't seen this combination before
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pair = {
                        'transaction_type': transaction_type,
                        'asset_class': asset_class,
                        'category': 'Asset Transaction',
                        'description': f"{transaction_type} - {asset_class}"
                    }
                    pairs.append(pair)
        
        return pairs
    
    def _find_and_store_similar_transactions(self, transaction_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Find similar transactions and store new ones in vector database"""
        similar_results = []
        
        if not self.vector_db_available or not self.openai_available:
            return similar_results
        
        for pair in transaction_pairs:
            try:
                # Create embedding for the transaction description
                description = pair.get('description', '')
                if not description:
                    continue
                
                embedding = self.openai_service.create_embedding(description)
                
                # Generate unique transaction ID
                transaction_id = f"{pair['transaction_type']}_{pair['asset_class']}_{uuid.uuid4().hex[:8]}"
                
                # Store transaction embedding
                self.vector_db.store_transaction_embedding(
                    transaction_id=transaction_id,
                    transaction_type=pair['transaction_type'],
                    asset_class=pair['asset_class'],
                    description=description,
                    embedding=embedding,
                    metadata={"category": pair['category']}
                )
                
                # Find similar transactions
                similar_transactions = self.vector_db.find_similar_transactions(
                    query_embedding=embedding,
                    limit=5,
                    similarity_threshold=0.8
                )
                
                if similar_transactions:
                    similar_results.append({
                        "transaction": pair,
                        "similar": similar_transactions
                    })
                
            except Exception as e:
                # Continue processing other transactions if one fails
                continue
        
        return similar_results
    
    def get_transaction_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced transaction summary"""
        summary = {
            "total_transactions": 0,
            "cash_transactions": 0,
            "asset_transactions": 0,
            "transaction_types": set(),
            "asset_classes": set(),
            "unique_pairs": 0,
            "ai_analysis_available": False,
            "vector_db_available": False,
            "similar_transactions_found": 0,
            "brave_search_available": False,
            "definitions_found": 0
        }
        
        if not results.get("success") or not results.get("transaction_pairs"):
            return summary
        
        pairs = results["transaction_pairs"]
        summary["total_transactions"] = len(pairs)
        
        for pair in pairs:
            summary["transaction_types"].add(pair["transaction_type"])
            summary["asset_classes"].add(pair["asset_class"])
            
            if pair["category"] == "Cash Transaction":
                summary["cash_transactions"] += 1
            elif pair["category"] == "Asset Transaction":
                summary["asset_transactions"] += 1
        
        # Convert sets to lists for JSON serialization
        summary["transaction_types"] = list(summary["transaction_types"])
        summary["asset_classes"] = list(summary["asset_classes"])
        summary["unique_pairs"] = len(set(
            (pair["transaction_type"], pair["asset_class"]) for pair in pairs
        ))
        
        # AI and vector DB status
        summary["ai_analysis_available"] = results.get("ai_analysis", {}).get("success", False)
        summary["vector_db_available"] = self.vector_db_available
        summary["similar_transactions_found"] = len(results.get("similar_transactions", []))
        
        # Brave Search status
        summary["brave_search_available"] = self.brave_search_available
        transaction_definitions = results.get("transaction_definitions", {})
        summary["definitions_found"] = transaction_definitions.get("definitions_found", 0)
        
        return summary
    
    def create_transactions_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create enhanced pandas DataFrame from transaction pairs"""
        if not results.get("success") or not results.get("transaction_pairs"):
            return pd.DataFrame()
        
        return pd.DataFrame(results["transaction_pairs"])
    
    def get_journal_mappings(self) -> List[Dict[str, Any]]:
        """Get all stored journal mappings from vector database"""
        if not self.vector_db_available:
            return []
        
        try:
            return self.vector_db.get_all_mappings()
        except Exception:
            return []
    
    def search_transaction_definition(
        self, 
        transaction_type: str, 
        asset_class: str = ""
    ) -> Dict[str, Any]:
        """
        Manually search for a specific transaction definition
        
        Args:
            transaction_type: Transaction type to search for
            asset_class: Asset class for context
            
        Returns:
            Search results with definition
        """
        if not self.brave_search_available:
            return {
                "success": False,
                "error": "Brave Search is not available",
                "transaction_type": transaction_type,
                "asset_class": asset_class
            }
        
        try:
            from .brave_processor import search_transaction_definition_sync
            
            result = search_transaction_definition_sync(
                transaction_type=transaction_type,
                asset_class=asset_class,
                api_key=settings.brave_api_key
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Search error: {str(e)}",
                "transaction_type": transaction_type,
                "asset_class": asset_class
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "api_client": self.api_client.check_health(),
            "openai_available": self.openai_available,
            "vector_db_available": self.vector_db_available,
            "brave_search_available": self.brave_search_available,
            "vector_db_status": {},
            "brave_search_status": {}
        }
        
        if self.vector_db_available:
            try:
                status["vector_db_status"] = self.vector_db.health_check()
            except Exception as e:
                status["vector_db_status"] = {"success": False, "error": str(e)}
        
        if self.brave_search_available:
            try:
                status["brave_search_status"] = self.brave_processor.health_check()
            except Exception as e:
                status["brave_search_status"] = {"success": False, "error": str(e)}
        
        return status
    
    def _generate_fallback_journal_suggestions(self, transaction_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate rule-based journal suggestions when OpenAI is not available"""
        
        # Get unique pairs
        unique_pairs = list(set(
            (pair["transaction_type"], pair["asset_class"]) 
            for pair in transaction_pairs
        ))
        
        suggestions_text = "## Journal Entry Suggestions (Rule-Based)\n\n"
        suggestions_text += "*Generated using treasury accounting best practices*\n\n"
        
        for transaction_type, asset_class in unique_pairs:
            suggestions_text += f"### {transaction_type} - {asset_class}\n\n"
            
            # Determine journal type and entries based on transaction classification
            journal_type, debit_account, credit_account, description = self._classify_transaction_journal(transaction_type, asset_class)
            
            suggestions_text += f"**Journal Type:** {journal_type}\n\n"
            suggestions_text += f"**Debit:** {debit_account}\n\n"
            suggestions_text += f"**Credit:** {credit_account}\n\n"
            suggestions_text += f"**Description:** {description}\n\n"
            suggestions_text += f"**Note:** {journal_type} entry for {transaction_type.lower()} transaction\n\n"
            suggestions_text += "---\n\n"
        
        suggestions_text += "\n💡 **Tip:** For more sophisticated suggestions, configure OpenAI API key for AI-powered analysis.\n"
        
        return {
            "success": True,
            "suggestions": suggestions_text,
            "unique_pairs_count": len(unique_pairs),
            "model_used": "Rule-Based System",
            "note": "Basic rule-based suggestions. For AI-powered suggestions, configure OpenAI."
        }
    
    def _classify_transaction_journal(self, transaction_type: str, asset_class: str) -> tuple:
        """Classify transaction and return appropriate journal type and accounts"""
        transaction_upper = transaction_type.upper()
        
        # Determine if this is an Invoice (Payment) or Debit Memo (Refund) transaction
        if self._is_invoice_transaction(transaction_upper):
            journal_type = "PAYMENT"
            debit_account, credit_account, description = self._get_payment_journal_entries(transaction_upper, asset_class)
        elif self._is_debit_memo_transaction(transaction_upper):
            journal_type = "REFUND"
            debit_account, credit_account, description = self._get_refund_journal_entries(transaction_upper, asset_class)
        else:
            # Default to payment for unknown transactions
            journal_type = "PAYMENT"
            debit_account, credit_account, description = self._get_payment_journal_entries(transaction_upper, asset_class)
        
        return journal_type, debit_account, credit_account, description
    
    def _is_invoice_transaction(self, transaction_type: str) -> bool:
        """Check if transaction is an invoice-type (payment journal)"""
        invoice_types = [
            'BUY', 'PURCHASE', 'SUBSCRIPTION', 'PLACEMENT',
            'INVESTMENT', 'ACQUISITION', 'DEPOSIT',
            'TRANSFER_OUT', 'PAYMENT'
        ]
        return transaction_type in invoice_types
    
    def _is_debit_memo_transaction(self, transaction_type: str) -> bool:
        """Check if transaction is a debit memo-type (refund journal)"""
        debit_memo_types = [
            'SELL', 'DISPOSAL', 'REDEMPTION', 'WITHDRAWAL',
            'MATURITY', 'REPAYMENT', 'DIVIDEND', 'INTEREST', 
            'COUPON', 'INCOME', 'RETURN', 'REFUND',
            'TRANSFER_IN', 'RECEIPT'
        ]
        return transaction_type in debit_memo_types
    
    def _get_payment_journal_entries(self, transaction_type: str, asset_class: str) -> tuple:
        """Get journal entries for payment (invoice) transactions"""
        
        if transaction_type in ['BUY', 'PURCHASE', 'SUBSCRIPTION']:
            debit_account = self._get_asset_account(asset_class)
            credit_account = "Cash and Cash Equivalents"
            description = f"Payment for purchase of {asset_class.lower()} securities"
            
        elif transaction_type in ['PLACEMENT', 'INVESTMENT']:
            debit_account = "Investment Account"
            credit_account = "Cash and Cash Equivalents"
            description = f"Payment for {asset_class.lower()} placement/investment"
            
        elif transaction_type in ['TRANSFER_OUT', 'PAYMENT']:
            debit_account = "Accounts Payable"
            credit_account = "Cash and Cash Equivalents"
            description = f"Payment transfer for {asset_class.lower()}"
            
        else:
            # Generic payment
            debit_account = self._get_asset_account(asset_class)
            credit_account = "Cash and Cash Equivalents"
            description = f"Payment for {transaction_type.lower()} of {asset_class.lower()}"
        
        return debit_account, credit_account, description
    
    def _get_refund_journal_entries(self, transaction_type: str, asset_class: str) -> tuple:
        """Get journal entries for refund (debit memo) transactions"""
        
        if transaction_type in ['SELL', 'DISPOSAL', 'REDEMPTION']:
            debit_account = "Cash and Cash Equivalents"
            credit_account = self._get_asset_account(asset_class)
            description = f"Refund from sale/redemption of {asset_class.lower()} securities"
            
        elif transaction_type in ['DIVIDEND', 'INTEREST', 'COUPON', 'INCOME']:
            debit_account = "Cash and Cash Equivalents"
            credit_account = f"{transaction_type.title()} Income"
            description = f"Refund - {transaction_type.title()} received on {asset_class.lower()}"
            
        elif transaction_type in ['MATURITY', 'REPAYMENT']:
            debit_account = "Cash and Cash Equivalents"
            credit_account = self._get_asset_account(asset_class)
            description = f"Refund from maturity/repayment of {asset_class.lower()}"
            
        elif transaction_type in ['WITHDRAWAL', 'TRANSFER_IN', 'RECEIPT']:
            debit_account = "Cash and Cash Equivalents"
            credit_account = "Investment Account"
            description = f"Refund from {transaction_type.lower()} of {asset_class.lower()}"
            
        else:
            # Generic refund
            debit_account = "Cash and Cash Equivalents"
            credit_account = self._get_asset_account(asset_class)
            description = f"Refund from {transaction_type.lower()} of {asset_class.lower()}"
        
        return debit_account, credit_account, description
    
    def _get_asset_account(self, asset_class: str) -> str:
        """Get appropriate asset account based on asset class"""
        asset_class_upper = asset_class.upper()
        
        if asset_class_upper in ['STOCK', 'EQUITY']:
            return "Investment in Equity Securities"
        elif asset_class_upper in ['BOND', 'FIXED_INCOME']:
            return "Investment in Fixed Income Securities"
        elif asset_class_upper in ['FUND', 'MUTUAL_FUND']:
            return "Investment in Mutual Funds"
        elif asset_class_upper in ['ETF']:
            return "Investment in Exchange Traded Funds"
        elif asset_class_upper in ['CASH', 'MONEY_MARKET']:
            return "Cash and Cash Equivalents"
        elif asset_class_upper in ['DERIVATIVE']:
            return "Investment in Derivatives"
        else:
            return "Investment in Financial Instruments" 