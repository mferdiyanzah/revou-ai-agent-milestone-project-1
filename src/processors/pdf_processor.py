"""
PDF Processing Module for TReA
Handles PDF upload, parsing, and transaction extraction
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from io import BytesIO

from ..services.api_client import TReAAPIClient
from ..config import settings


class PDFProcessor:
    """Handles PDF document processing for treasury statements"""
    
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
    
    def process_pdf_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process PDF document through the complete TReA pipeline
        
        Args:
            file_path: Path to the PDF file
            
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
            "error": None
        }
        
        try:
            # Get file information
            results["file_info"] = self._get_file_info(file_path)
            print(results["file_info"], 'file_info')
            
            # Step 1: Upload file and get initial data
            upload_result = self.api_client.upload_file(file_path)
            print(upload_result, 'upload_result')
            if not upload_result["success"]:
                results["error"] = f"Upload failed: {upload_result.get('error', 'Unknown error')}"
                return results
            
            upload_data = upload_result["data"]
            results["extraction_data"] = upload_data
            
            # Step 2: Extract PDF content
            extract_result = self.api_client.extract_pdf(upload_data)
            if not extract_result["success"]:
                results["error"] = f"Extraction failed: {extract_result.get('error', 'Unknown error')}"
                return results
            
            extract_data = extract_result["data"]
            
            # Step 3: Transform extracted data
            transform_result = self.api_client.transform_data(extract_data["data"])
            if not transform_result["success"]:
                results["error"] = f"Transformation failed: {transform_result.get('error', 'Unknown error')}"
                return results
            
            transform_data = transform_result["data"]
            results["transformed_data"] = transform_data
            
            # Step 4: Map transactions to journal entries
            map_result = self.api_client.map_transactions(transform_data["data"])
            if not map_result["success"]:
                results["error"] = f"Mapping failed: {map_result.get('error', 'Unknown error')}"
                return results
            
            map_data = map_result["data"]
            results["mapped_data"] = map_data
            
            # Step 5: Extract transaction pairs
            results["transaction_pairs"] = self._extract_transaction_pairs(map_data)
            
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
        """
        Extract transaction type and asset class pairs from mapped data
        
        Args:
            map_data: Mapped transaction data
            
        Returns:
            List of transaction pairs
        """
        pairs = []
        
        if not map_data or 'data' not in map_data:
            return pairs
        
        data = map_data['data']
        
        # Process cash transactions
        if 'cash_transactions' in data:
            for transaction in data['cash_transactions']:
                pair = {
                    'transaction_type': transaction.get('TRANSACTION_TYPE', ''),
                    'asset_class': transaction.get('ASSET_CLASS', 'CASH'),
                    'category': 'Cash Transaction'
                }
                pairs.append(pair)
        
        # Process asset transactions  
        if 'asset_transactions' in data:
            for transaction in data['asset_transactions']:
                if not transaction.get('TRANSACTION_TYPE'):
                    continue
                pair = {
                    'transaction_type': transaction.get('TRANSACTION_TYPE', ''),
                    'asset_class': transaction.get('ASSET_CLASS', 'ASSET'),
                    'category': 'Asset Transaction'
                }
                pairs.append(pair)
        
        return pairs
    
    def get_transaction_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate transaction summary from processing results
        
        Args:
            results: Processing results
            
        Returns:
            Transaction summary
        """
        summary = {
            "total_transactions": 0,
            "cash_transactions": 0,
            "asset_transactions": 0,
            "transaction_types": set(),
            "asset_classes": set(),
            "unique_pairs": 0
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
        
        return summary
    
    def create_transactions_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create pandas DataFrame from transaction pairs
        
        Args:
            results: Processing results
            
        Returns:
            DataFrame with transaction data
        """
        if not results.get("success") or not results.get("transaction_pairs"):
            return pd.DataFrame()
        
        return pd.DataFrame(results["transaction_pairs"])


class FileValidator:
    """Validates uploaded files"""
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext not in settings.allowed_file_types:
            return False, f"File type {file_ext} not allowed. Allowed types: {settings.allowed_file_types}"
        
        # Check file size
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({settings.max_file_size_mb}MB)"
        
        return True, "File is valid" 