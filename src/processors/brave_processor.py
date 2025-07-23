"""
Brave Search Processor for TReA
Searches for transaction type definitions and financial terminology
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

class BraveSearchProcessor:
    """Processor for searching transaction definitions using Brave Search API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Brave Search processor
        
        Args:
            api_key: Brave Search API key (required for API access)
        """
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip"
        }
        if self.api_key:
            self.headers["X-Subscription-Token"] = self.api_key
        else:
            # Without API key, we can't use Brave Search API
            # We'll provide helpful error messages
            pass
    
    async def search_transaction_definition(
        self, 
        transaction_type: str,
        asset_class: str = "",
        context: str = "banking finance treasury"
    ) -> Dict[str, Any]:
        """
        Search for transaction type definition
        
        Args:
            transaction_type: The transaction type to search for
            asset_class: Optional asset class for context
            context: Additional search context
            
        Returns:
            Search results with definitions and explanations
        """
        try:
            # Check if API key is available
            if not self.api_key:
                return {
                    "success": False,
                    "error": "Brave Search API key is required",
                    "transaction_type": transaction_type,
                    "asset_class": asset_class
                }
            
            # Construct search query - simplified
            query_parts = [
                transaction_type,
                "definition",
                "banking"
            ]
            
            if asset_class:
                query_parts.append(asset_class)
            
            query = " ".join(query_parts)
            
            # Search parameters
            params = {
                "q": query,
                "count": 5,  # Number of results
                "search_lang": "en",
                "country": "US"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._process_search_results(data, transaction_type, asset_class)
                    else:
                        return {
                            "success": False,
                            "error": f"Search failed with status {response.status}",
                            "transaction_type": transaction_type,
                            "asset_class": asset_class
                        }
                        
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Search request timed out",
                "transaction_type": transaction_type,
                "asset_class": asset_class
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Search error: {str(e)}",
                "transaction_type": transaction_type,
                "asset_class": asset_class,
                "debug_info": {
                    "query": query,
                    "params": params,
                    "exception_type": type(e).__name__
                }
            }
    
    def _process_search_results(
        self, 
        data: Dict[str, Any], 
        transaction_type: str, 
        asset_class: str
    ) -> Dict[str, Any]:
        """Process and extract relevant information from search results"""
        
        result = {
            "success": True,
            "transaction_type": transaction_type,
            "asset_class": asset_class,
            "definitions": [],
            "sources": [],
            "summary": "",
            "searched_at": datetime.now().isoformat()
        }
        
        try:
            # Extract web results
            web_results = data.get("web", {}).get("results", [])
            
            for item in web_results:
                title = item.get("title", "")
                url = item.get("url", "")
                description = item.get("description", "")
                
                # Look for definition-like content
                if any(keyword in description.lower() for keyword in [
                    "definition", "means", "refers to", "is a", "type of"
                ]):
                    result["definitions"].append({
                        "title": title,
                        "description": description,
                        "url": url,
                        "relevance_score": self._calculate_relevance(
                            description, transaction_type, asset_class
                        )
                    })
                
                result["sources"].append({
                    "title": title,
                    "url": url,
                    "description": description
                })
            
            # Sort definitions by relevance
            result["definitions"].sort(
                key=lambda x: x["relevance_score"], 
                reverse=True
            )
            
            # Create summary from top definitions
            if result["definitions"]:
                top_definitions = result["definitions"][:2]
                result["summary"] = self._create_summary(
                    top_definitions, transaction_type, asset_class
                )
            else:
                result["summary"] = f"No specific definition found for '{transaction_type}' in {asset_class} context."
            
        except Exception as e:
            result["success"] = False
            result["error"] = f"Error processing results: {str(e)}"
        
        return result
    
    def _calculate_relevance(
        self, 
        text: str, 
        transaction_type: str, 
        asset_class: str
    ) -> float:
        """Calculate relevance score for a piece of text"""
        
        text_lower = text.lower()
        transaction_lower = transaction_type.lower()
        asset_lower = asset_class.lower() if asset_class else ""
        
        score = 0.0
        
        # Exact match bonus
        if transaction_lower in text_lower:
            score += 3.0
        
        if asset_lower and asset_lower in text_lower:
            score += 2.0
        
        # Definition keywords
        definition_keywords = [
            "definition", "means", "refers to", "is a", "type of",
            "involves", "process of", "method of", "way of"
        ]
        
        for keyword in definition_keywords:
            if keyword in text_lower:
                score += 1.0
        
        # Financial context keywords
        financial_keywords = [
            "bank", "finance", "treasury", "investment", "trading",
            "settlement", "clearing", "payment", "transaction"
        ]
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                score += 0.5
        
        return score
    
    def _create_summary(
        self, 
        definitions: List[Dict[str, Any]], 
        transaction_type: str, 
        asset_class: str
    ) -> str:
        """Create a summary from the top definitions"""
        
        if not definitions:
            return f"No definition found for {transaction_type}."
        
        summary_parts = []
        
        for i, defn in enumerate(definitions[:2]):
            desc = defn["description"]
            # Clean up the description
            if len(desc) > 200:
                desc = desc[:200] + "..."
            
            summary_parts.append(f"({i+1}) {desc}")
        
        summary = f"Definition of '{transaction_type}'"
        if asset_class:
            summary += f" in {asset_class} context"
        summary += ":\n\n" + "\n\n".join(summary_parts)
        
        return summary
    
    async def search_multiple_transactions(
        self, 
        transaction_pairs: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Search for definitions of multiple transaction types
        
        Args:
            transaction_pairs: List of transaction dictionaries with type and class
            
        Returns:
            Combined search results for all transactions
        """
        
        results = {
            "success": True,
            "total_searched": len(transaction_pairs),
            "definitions_found": 0,
            "transactions": {},
            "errors": []
        }
        
        # Create search tasks for all transactions
        tasks = []
        for pair in transaction_pairs:
            transaction_type = pair.get("transaction_type", "")
            asset_class = pair.get("asset_class", "")
            
            if transaction_type:
                task = self.search_transaction_definition(
                    transaction_type, asset_class
                )
                tasks.append((transaction_type, asset_class, task))
        
        # Execute all searches concurrently
        for transaction_type, asset_class, task in tasks:
            try:
                result = await task
                
                key = f"{transaction_type}_{asset_class}" if asset_class else transaction_type
                results["transactions"][key] = result
                
                if result.get("success") and result.get("definitions"):
                    results["definitions_found"] += 1
                elif not result.get("success"):
                    results["errors"].append({
                        "transaction_type": transaction_type,
                        "asset_class": asset_class,
                        "error": result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                results["errors"].append({
                    "transaction_type": transaction_type,
                    "asset_class": asset_class,
                    "error": str(e)
                })
        
        if results["errors"]:
            results["success"] = len(results["errors"]) < len(transaction_pairs)
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Brave Search API is accessible"""
        return {
            "service": "Brave Search",
            "api_key_configured": bool(self.api_key),
            "base_url": self.base_url,
            "status": "ready"
        }


# Synchronous wrapper for easier integration
def search_transaction_definition_sync(
    transaction_type: str,
    asset_class: str = "",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for searching transaction definition
    
    Args:
        transaction_type: Transaction type to search
        asset_class: Asset class for context
        api_key: Brave Search API key
        
    Returns:
        Search results
    """
    processor = BraveSearchProcessor(api_key)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        processor.search_transaction_definition(transaction_type, asset_class)
    )


def search_multiple_transactions_sync(
    transaction_pairs: List[Dict[str, str]],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for searching multiple transaction definitions
    
    Args:
        transaction_pairs: List of transaction dictionaries
        api_key: Brave Search API key
        
    Returns:
        Combined search results
    """
    processor = BraveSearchProcessor(api_key)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        processor.search_multiple_transactions(transaction_pairs)
    )
