"""
Text and JSON Processor for TReA
Handles text-based transaction data input
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import re
from datetime import datetime

from ..services.api_client import TReAAPIClient
from ..config import settings


class TextProcessor:
    """Handles text and JSON document processing for treasury transactions"""
    
    def __init__(self, api_client: TReAAPIClient = None):
        self.api_client = api_client or TReAAPIClient()
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """
        Save uploaded file to temporary location
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Path to saved file
        """
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{uploaded_file.name}"
        file_path = self.upload_dir / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    def process_text_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process text document containing transaction data
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Processing results including transactions and journal mappings
        """
        results = {
            "success": False,
            "file_info": {},
            "extraction_data": {},
            "transformed_data": {},
            "mapped_data": {},
            "transaction_pairs": [],
            "error": None,
            "input_type": "text"
        }
        
        try:
            # Get file information
            results["file_info"] = self._get_file_info(file_path)
            
            # Determine file type and process accordingly
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.json':
                transaction_data = self._process_json_file(file_path)
            elif file_extension == '.csv':
                transaction_data = self._process_csv_file(file_path)
            elif file_extension == '.txt':
                transaction_data = self._process_text_file(file_path)
            else:
                results["error"] = f"Unsupported file type: {file_extension}"
                return results
            
            if not transaction_data["success"]:
                results["error"] = transaction_data["error"]
                return results
            
            # Structure data for TReA API format
            structured_data = self._structure_transaction_data(transaction_data["data"])
            results["extraction_data"] = structured_data
            
            # Transform data using standard format
            transform_result = self._transform_structured_data(structured_data)
            if not transform_result["success"]:
                results["error"] = f"Transformation failed: {transform_result.get('error', 'Unknown error')}"
                return results
            
            results["transformed_data"] = transform_result["data"]
            
            # Create transaction pairs for AI processing
            results["transaction_pairs"] = self._extract_transaction_pairs(transform_result["data"])
            
            # Mock journal mapping (would typically use API)
            results["mapped_data"] = self._create_mock_journal_mapping(results["transaction_pairs"])
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = f"Processing error: {str(e)}"
        
        finally:
            # Clean up uploaded file
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            except:
                pass
        
        return results
    
    def _process_json_file(self, file_path: str) -> Dict[str, Any]:
        """Process JSON file containing transaction data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate JSON structure
            if isinstance(data, dict) and 'transactions' in data:
                transactions = data['transactions']
            elif isinstance(data, list):
                transactions = data
            else:
                return {
                    "success": False,
                    "error": "JSON must contain 'transactions' array or be an array of transactions"
                }
            
            # Validate transaction structure
            validated_transactions = []
            for i, txn in enumerate(transactions):
                if not isinstance(txn, dict):
                    continue
                
                # Ensure required fields exist
                validated_txn = {
                    "transaction_id": txn.get("transaction_id", f"TXN_{i+1}"),
                    "transaction_type": txn.get("transaction_type", "UNKNOWN"),
                    "asset_class": txn.get("asset_class", "UNKNOWN"),
                    "amount": float(txn.get("amount", 0.0)),
                    "currency": txn.get("currency", "USD"),
                    "date": txn.get("date", datetime.now().strftime("%Y-%m-%d")),
                    "description": txn.get("description", ""),
                    "reference": txn.get("reference", ""),
                    "counterparty": txn.get("counterparty", "")
                }
                validated_transactions.append(validated_txn)
            
            return {
                "success": True,
                "data": validated_transactions,
                "format": "json"
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON format: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing JSON file: {str(e)}"
            }
    
    def _process_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Process CSV file containing transaction data"""
        try:
            # Try to read CSV with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return {
                    "success": False,
                    "error": "Could not read CSV file with any encoding"
                }
            
            if df.empty:
                return {
                    "success": False,
                    "error": "CSV file is empty"
                }
            
            # Map common CSV column names to standard format
            column_mapping = {
                'type': 'transaction_type',
                'transaction_type': 'transaction_type',
                'asset': 'asset_class',
                'asset_class': 'asset_class',
                'amount': 'amount',
                'value': 'amount',
                'currency': 'currency',
                'ccy': 'currency',
                'date': 'date',
                'transaction_date': 'date',
                'description': 'description',
                'desc': 'description',
                'reference': 'reference',
                'ref': 'reference',
                'counterparty': 'counterparty'
            }
            
            # Rename columns to standard format
            df_columns_lower = {col: col.lower().replace(' ', '_') for col in df.columns}
            df = df.rename(columns=df_columns_lower)
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            required_columns = ['transaction_type', 'asset_class']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 'UNKNOWN'
            
            # Add missing optional columns
            if 'amount' not in df.columns:
                df['amount'] = 0.0
            if 'currency' not in df.columns:
                df['currency'] = 'USD'
            if 'date' not in df.columns:
                df['date'] = datetime.now().strftime("%Y-%m-%d")
            if 'description' not in df.columns:
                df['description'] = ''
            if 'reference' not in df.columns:
                df['reference'] = ''
            if 'counterparty' not in df.columns:
                df['counterparty'] = ''
            
            # Add transaction IDs
            df['transaction_id'] = [f"TXN_{i+1}" for i in range(len(df))]
            
            # Convert to list of dictionaries
            transactions = df.to_dict('records')
            
            return {
                "success": True,
                "data": transactions,
                "format": "csv"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing CSV file: {str(e)}"
            }
    
    def _process_text_file(self, file_path: str) -> Dict[str, Any]:
        """Process plain text file with transaction data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse structured text data
            transactions = self._parse_text_content(content)
            
            if not transactions:
                return {
                    "success": False,
                    "error": "Could not extract transaction data from text file"
                }
            
            return {
                "success": True,
                "data": transactions,
                "format": "text"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing text file: {str(e)}"
            }
    
    def _parse_text_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse transaction data from text content"""
        transactions = []
        
        # Pattern for structured transaction data
        # Example: "BUY STOCK 1000 USD 2024-01-15 Purchase of equity"
        pattern = r'(\w+)\s+(\w+)\s+([\d.]+)\s+(\w+)\s+(\d{4}-\d{2}-\d{2})\s*(.*)'
        
        lines = content.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                transaction_type, asset_class, amount, currency, date, description = match.groups()
                
                transaction = {
                    "transaction_id": f"TXN_{i+1}",
                    "transaction_type": transaction_type.upper(),
                    "asset_class": asset_class.upper(),
                    "amount": float(amount),
                    "currency": currency.upper(),
                    "date": date,
                    "description": description.strip(),
                    "reference": "",
                    "counterparty": ""
                }
                transactions.append(transaction)
            else:
                # Try to extract basic transaction type and asset class
                words = line.upper().split()
                if len(words) >= 2:
                    transaction = {
                        "transaction_id": f"TXN_{i+1}",
                        "transaction_type": words[0],
                        "asset_class": words[1] if len(words) > 1 else "UNKNOWN",
                        "amount": 0.0,
                        "currency": "USD",
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "description": line,
                        "reference": "",
                        "counterparty": ""
                    }
                    transactions.append(transaction)
        
        return transactions
    
    def _structure_transaction_data(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structure transaction data for TReA API format"""
        return {
            "data": {
                "cash_transactions": [
                    {
                        "TRANSACTION_TYPE": txn["transaction_type"],
                        "ASSET_CLASS": txn["asset_class"],
                        "AMOUNT": txn["amount"],
                        "CURRENCY": txn["currency"],
                        "DATE": txn["date"],
                        "DESCRIPTION": txn["description"],
                        "REFERENCE": txn["reference"],
                        "COUNTERPARTY": txn["counterparty"]
                    }
                    for txn in transactions
                ],
                "asset_transactions": [],
                "metadata": {
                    "total_transactions": len(transactions),
                    "processing_date": datetime.now().isoformat(),
                    "source_type": "text_input"
                }
            }
        }
    
    def _transform_structured_data(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform structured data to standard format"""
        try:
            return {
                "success": True,
                "data": structured_data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_transaction_pairs(self, transform_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract unique transaction pairs from transformed data"""
        pairs = []
        seen_pairs = set()
        
        if not transform_data or 'data' not in transform_data:
            return pairs
        
        data = transform_data['data']
        
        # Process cash transactions
        if 'cash_transactions' in data:
            for transaction in data['cash_transactions']:
                transaction_type = transaction.get('TRANSACTION_TYPE', '')
                asset_class = transaction.get('ASSET_CLASS', 'CASH')
                
                pair_key = (transaction_type, asset_class)
                
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pair = {
                        'transaction_type': transaction_type,
                        'asset_class': asset_class,
                        'category': 'Cash Transaction',
                        'description': f"{transaction_type} - {asset_class}"
                    }
                    pairs.append(pair)
        
        return pairs
    
    def _create_mock_journal_mapping(self, transaction_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create mock journal mapping for demonstration"""
        journal_entries = []
        
        for pair in transaction_pairs:
            transaction_type = pair['transaction_type']
            asset_class = pair['asset_class']
            
            # Simple mapping logic based on transaction type
            if transaction_type in ['BUY', 'PURCHASE', 'SUBSCRIPTION']:
                debit_account = "Investment Account"
                credit_account = "Cash Account"
            elif transaction_type in ['SELL', 'DISPOSAL', 'REDEMPTION']:
                debit_account = "Cash Account"
                credit_account = "Investment Account"
            elif transaction_type in ['DIVIDEND', 'INTEREST', 'INCOME']:
                debit_account = "Cash Account"
                credit_account = "Income Account"
            else:
                debit_account = "General Account"
                credit_account = "General Account"
            
            journal_entry = {
                "transaction_type": transaction_type,
                "asset_class": asset_class,
                "debit_account": debit_account,
                "credit_account": credit_account,
                "journal_type": "Standard",
                "description": f"{transaction_type} - {asset_class}"
            }
            journal_entries.append(journal_entry)
        
        return {
            "data": {
                "journal_mappings": journal_entries,
                "metadata": {
                    "total_mappings": len(journal_entries),
                    "mapping_date": datetime.now().isoformat()
                }
            }
        }
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information"""
        path = Path(file_path)
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "extension": path.suffix.lower()
        }


class FileValidator:
    """Enhanced file validator for multiple file types"""
    
    @staticmethod
    def validate_file(uploaded_file) -> tuple[bool, str]:
        """
        Validate uploaded file for multiple formats
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, message)
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({settings.max_file_size_mb}MB)"
        
        # Check file extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in settings.allowed_file_types:
            return False, f"File type {file_extension} not supported. Allowed types: {', '.join(settings.allowed_file_types)}"
        
        # Additional validation based on file type
        if file_extension == '.json':
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                json.loads(content)
                uploaded_file.seek(0)  # Reset file pointer
            except json.JSONDecodeError:
                return False, "Invalid JSON format"
            except UnicodeDecodeError:
                return False, "Invalid file encoding for JSON"
        
        elif file_extension == '.csv':
            try:
                content = uploaded_file.getvalue()
                # Try common encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        content.decode(encoding)
                        uploaded_file.seek(0)  # Reset file pointer
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return False, "Invalid file encoding for CSV"
            except Exception:
                return False, "Invalid CSV file"
        
        elif file_extension == '.txt':
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                uploaded_file.seek(0)  # Reset file pointer
                if len(content.strip()) == 0:
                    return False, "Text file is empty"
            except UnicodeDecodeError:
                return False, "Invalid file encoding for text file"
        
        return True, "File validation passed"
    
    @staticmethod
    def get_file_type_description(file_extension: str) -> str:
        """Get description for file type"""
        descriptions = {
            '.pdf': 'Treasury statement PDF (DBS Singapore format)',
            '.txt': 'Plain text transaction data',
            '.json': 'JSON formatted transaction data',
            '.csv': 'CSV transaction data'
        }
        return descriptions.get(file_extension, 'Unknown file type') 