"""
API Client Service for TReA Backend Communication
"""

import os
import mimetypes
from typing import Dict, Any, Optional, Union
import requests
from pathlib import Path

from ..config import settings


class TReAAPIClient:
    """API Client for TReA Backend Services"""
    
    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or settings.api_base_url).rstrip('/')
        self.token = token or settings.api_token
        self.timeout = settings.api_timeout
        
    def _get_headers(self, additional_headers: Dict[str, str] = None) -> Dict[str, str]:
        """Get headers with authentication"""
        headers = {}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        if additional_headers:
            headers.update(additional_headers)
        return headers
    
    def upload_file(self, file_path: str, endpoint: str = "/api/check") -> Dict[str, Any]:
        """
        Upload file to TReA backend
        
        Args:
            file_path: Path to the file to upload
            endpoint: API endpoint for file upload
            
        Returns:
            Response data from the API
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}
        
        url = f"{self.base_url}{endpoint}"
        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/pdf'
        
        headers = self._get_headers()
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, mime_type)}
                response = requests.post(url, headers=headers, files=files, timeout=self.timeout)
                
                if response.status_code == 200:
                    return {"success": True, "data": response.json()}
                else:
                    return {
                        "success": False, 
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from PDF using TReA backend"""
        return self._post("/api/pdf-parser/extract", {"data": data['data']})
    
    def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform extracted data"""
        return self._post("/api/pdf-parser/transform", {"data": data})
    
    def map_transactions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map transactions to journal entries"""
        return self._post("/api/pdf-parser/map", {"data": data})
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal POST method"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers({'Content-Type': 'application/json'})
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_health(self) -> Dict[str, Any]:
        """Check API health"""
        url = f"{self.base_url}/api/check"
        headers = self._get_headers()
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "message": "API is healthy" if response.status_code == 200 else "API unhealthy"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": "API unreachable"} 