"""
AI Guardrails Service for Input/Output Validation and Safety
"""

import re
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import settings


class ThreatLevel(Enum):
    """Threat levels for security assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailType(Enum):
    """Types of guardrails"""
    INPUT_VALIDATION = "input_validation"
    OUTPUT_SAFETY = "output_safety"
    CONTENT_FILTER = "content_filter"
    RATE_LIMIT = "rate_limit"
    DATA_PRIVACY = "data_privacy"


@dataclass
class GuardrailResult:
    """Result of guardrail check"""
    passed: bool
    threat_level: ThreatLevel
    guardrail_type: GuardrailType
    message: str
    metadata: Dict[str, Any] = None


class AIGuardrails:
    """Comprehensive AI Guardrails for TReA System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Sensitive data patterns
        self.sensitive_patterns = {
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'account_number': r'\b\d{10,}\b',
            # Much more restrictive SWIFT code pattern that excludes common treasury terms
            # Must be exactly 8 or 11 chars, start with bank code, have country code, and not be common words
            'swift_code': r'\b(?!(?:REVENUE|EXPENSE|ACCOUNT|BALANCE|PAYMENT|INVOICE|JOURNAL|BOOKING|MAPPING|REVEU|BOND|STOCK|FUND|CASH|ASSET|TREASURY|CUSTODY|OPERATIONAL|SETTLEMENT|CLEARING|PLACEMENT|WITHDRAWAL|DIVIDEND|INTEREST|COUPON|REDEMPTION|SUBSCRIPTION)\b)[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b'
        }
        
        # Treasury-specific terms that should NOT be flagged as sensitive
        self.treasury_safe_terms = {
            'swift_code', 'swift', 'journal', 'debit', 'credit', 'account', 'balance',
            'transaction', 'payment', 'invoice', 'settlement', 'clearing', 'custody',
            'operational', 'placement', 'withdrawal', 'asset', 'liability', 'equity',
            'revenue', 'expense', 'bank', 'code', 'mapping', 'entry', 'reconciliation',
            'dividend', 'interest', 'coupon', 'redemption', 'subscription', 'purchase'
        }
        
        # Common treasury terms that look like SWIFT codes but aren't
        self.treasury_terms_not_swift = {
            'DIVIDEND', 'INTEREST', 'PURCHASE', 'REDEMPTION', 'SUBSCRIPTION', 
            'PLACEMENT', 'WITHDRAWAL', 'TRANSFER', 'SETTLEMENT', 'CLEARING',
            'CUSTODY', 'OPERATIONAL', 'INVESTMENT', 'SECURITIES', 'TREASURY',
            'REVENUE', 'EXPENSE', 'PAYMENT', 'INVOICE', 'JOURNAL', 'ACCOUNT',
            'BALANCE', 'BOOKING', 'MAPPING', 'REVEU', 'BOND', 'STOCK', 'FUND',
            'CASH', 'ASSET', 'COUPON'
        }
        
        # Malicious content patterns
        self.malicious_patterns = {
            # SQL injection pattern - detects common SQL commands
            'sql_injection': r'(?i)\b(select\s+\*\s+from|drop\s+table|delete\s+from|insert\s+into\s+\w+\s+values|update\s+\w+\s+set|union\s+select)\b',
            'xss': r'<script|javascript:|on\w+\s*=',
            'command_injection': r'(;|\||&|`|\$\()',
            'path_traversal': r'\.\./',
            'prompt_injection': r'(ignore|forget|override).*(instruction|rule|prompt)'
        }
        
        # Rate limiting tracking
        self.rate_limit_tracker = {}
        
        # Prohibited topics for treasury context
        self.prohibited_topics = [
            'personal financial advice',
            'investment recommendations',
            'illegal activities',
            'market manipulation',
            'insider trading',
            'money laundering'
        ]
    
    def validate_input(self, user_input: str, user_id: str = None) -> GuardrailResult:
        """
        Validate user input for security and safety
        
        Args:
            user_input: The input to validate
            user_id: Optional user identifier for rate limiting
            
        Returns:
            GuardrailResult with validation results
        """
        # Check rate limiting
        if user_id:
            rate_limit_result = self._check_rate_limit(user_id)
            if not rate_limit_result.passed:
                return rate_limit_result
        
        # Check for malicious content
        malicious_result = self._check_malicious_content(user_input)
        if not malicious_result.passed:
            return malicious_result
        
        # Check for sensitive data
        sensitive_result = self._check_sensitive_data(user_input)
        if not sensitive_result.passed:
            return sensitive_result
        
        # Check input length and format
        format_result = self._check_input_format(user_input)
        if not format_result.passed:
            return format_result
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.INPUT_VALIDATION,
            message="Input validation passed",
            metadata={"input_length": len(user_input)}
        )
    
    def validate_output(self, ai_output: str, context: Dict[str, Any] = None) -> GuardrailResult:
        """
        Validate AI output for safety and appropriateness
        
        Args:
            ai_output: The AI-generated output to validate
            context: Optional context information
            
        Returns:
            GuardrailResult with validation results
        """
        # Check for sensitive data in output
        sensitive_result = self._check_output_sensitive_data(ai_output)
        if not sensitive_result.passed:
            return sensitive_result
        
        # Check for prohibited content
        prohibited_result = self._check_prohibited_content(ai_output)
        if not prohibited_result.passed:
            return prohibited_result
        
        # Check output quality and coherence
        quality_result = self._check_output_quality(ai_output)
        if not quality_result.passed:
            return quality_result
        
        # Check treasury domain relevance
        relevance_result = self._check_domain_relevance(ai_output, context)
        if not relevance_result.passed:
            return relevance_result
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.OUTPUT_SAFETY,
            message="Output validation passed",
            metadata={"output_length": len(ai_output)}
        )
    
    def _check_rate_limit(self, user_id: str) -> GuardrailResult:
        """Check rate limiting for user"""
        current_time = time.time()
        max_requests_per_minute = getattr(settings, 'rate_limit_per_minute', 30)
        
        if user_id not in self.rate_limit_tracker:
            self.rate_limit_tracker[user_id] = []
        
        # Clean old requests
        self.rate_limit_tracker[user_id] = [
            req_time for req_time in self.rate_limit_tracker[user_id]
            if current_time - req_time < 60
        ]
        
        # Check if limit exceeded
        if len(self.rate_limit_tracker[user_id]) >= max_requests_per_minute:
            return GuardrailResult(
                passed=False,
                threat_level=ThreatLevel.MEDIUM,
                guardrail_type=GuardrailType.RATE_LIMIT,
                message=f"Rate limit exceeded: {max_requests_per_minute} requests per minute",
                metadata={"current_requests": len(self.rate_limit_tracker[user_id])}
            )
        
        # Add current request
        self.rate_limit_tracker[user_id].append(current_time)
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.RATE_LIMIT,
            message="Rate limit check passed",
            metadata={"requests_remaining": max_requests_per_minute - len(self.rate_limit_tracker[user_id])}
        )
    
    def _check_malicious_content(self, text: str) -> GuardrailResult:
        """Check for malicious content patterns with treasury context awareness"""
        text_lower = text.lower()
        
        # Check if this is treasury content first
        is_treasury = self._is_treasury_context(text)
        
        for pattern_name, pattern in self.malicious_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # For treasury content, be more lenient with SQL injection detection
                if pattern_name == 'sql_injection' and is_treasury:
                    # Check if it's actually malicious SQL or just treasury language
                    if not self._is_malicious_sql_in_treasury_context(text, matches):
                        continue
                
                return GuardrailResult(
                    passed=False,
                    threat_level=ThreatLevel.HIGH,
                    guardrail_type=GuardrailType.CONTENT_FILTER,
                    message=f"Malicious content detected: {pattern_name}",
                    metadata={"pattern_matched": pattern_name, "matches": matches[:3]}  # Show first 3 matches
                )
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.CONTENT_FILTER,
            message="No malicious content detected"
        )
    
    def _check_sensitive_data(self, text: str) -> GuardrailResult:
        """Check for sensitive data patterns with treasury context awareness"""
        detected_patterns = []
        
        # First check if this is clearly treasury content
        is_treasury = self._is_treasury_context(text)
        
        for pattern_name, pattern in self.sensitive_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                # Enhanced treasury-aware processing for input validation
                if pattern_name == 'swift_code':
                    # For SWIFT codes, be much more intelligent about treasury terminology
                    real_swift_codes = []
                    for match in matches:
                        # Only flag if it's actually a real SWIFT code and not treasury terminology
                        if self._is_real_swift_code(match) and not self._is_treasury_terminology(match, text):
                            real_swift_codes.append(match)
                    
                    if real_swift_codes:
                        detected_patterns.append(pattern_name)
                elif pattern_name == 'email' and is_treasury:
                    # In treasury context, be more lenient with email-like patterns
                    # Check if they're actually email addresses or just account references
                    real_emails = []
                    for match in matches:
                        if '@' in match and not self._is_account_reference(match):
                            real_emails.append(match)
                    
                    if real_emails:
                        detected_patterns.append(pattern_name)
                elif pattern_name == 'account_number' and is_treasury:
                    # In treasury context, be more lenient with long numbers
                    # Check if they're actually sensitive account numbers or just reference codes
                    real_account_nums = []
                    for match in matches:
                        if not self._is_treasury_reference_code(match, text):
                            real_account_nums.append(match)
                    
                    if real_account_nums:
                        detected_patterns.append(pattern_name)
                else:
                    # For other patterns, only flag if not in treasury context or clearly sensitive
                    if not is_treasury or self._is_clearly_sensitive(pattern_name, matches, text):
                        detected_patterns.append(pattern_name)
        
        if detected_patterns:
            return GuardrailResult(
                passed=False,
                threat_level=ThreatLevel.HIGH,
                guardrail_type=GuardrailType.DATA_PRIVACY,
                message=f"Sensitive data detected: {', '.join(detected_patterns)}",
                metadata={"detected_patterns": detected_patterns}
            )
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.DATA_PRIVACY,
            message="No sensitive data detected"
        )
    
    def _check_input_format(self, text: str) -> GuardrailResult:
        """Check input format and length"""
        max_length = getattr(settings, 'max_input_length', 10000)
        
        if len(text) > max_length:
            return GuardrailResult(
                passed=False,
                threat_level=ThreatLevel.MEDIUM,
                guardrail_type=GuardrailType.INPUT_VALIDATION,
                message=f"Input too long: {len(text)} > {max_length}",
                metadata={"input_length": len(text), "max_length": max_length}
            )
        
        if len(text.strip()) == 0:
            return GuardrailResult(
                passed=False,
                threat_level=ThreatLevel.LOW,
                guardrail_type=GuardrailType.INPUT_VALIDATION,
                message="Empty input not allowed",
                metadata={"input_length": len(text)}
            )
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.INPUT_VALIDATION,
            message="Input format validation passed"
        )
    
    def _check_output_sensitive_data(self, text: str) -> GuardrailResult:
        """Check AI output for sensitive data leakage with enhanced treasury context awareness"""
        detected_patterns = []
        
        # First check if this is clearly treasury content
        is_treasury = self._is_treasury_context(text)
        
        for pattern_name, pattern in self.sensitive_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                # Enhanced treasury-aware processing
                if pattern_name == 'swift_code':
                    # For SWIFT codes, be much more intelligent
                    real_swift_codes = []
                    for match in matches:
                        # Only flag if it's actually a real SWIFT code and not treasury terminology
                        if self._is_real_swift_code(match) and not self._is_treasury_terminology(match, text):
                            real_swift_codes.append(match)
                    
                    if real_swift_codes:
                        detected_patterns.append(pattern_name)
                elif pattern_name == 'email' and is_treasury:
                    # In treasury context, be more lenient with email-like patterns
                    # Check if they're actually email addresses or just account references
                    real_emails = []
                    for match in matches:
                        if '@' in match and not self._is_account_reference(match):
                            real_emails.append(match)
                    
                    if real_emails:
                        detected_patterns.append(pattern_name)
                else:
                    # For other patterns, only flag if not in treasury context or clearly sensitive
                    if not is_treasury or self._is_clearly_sensitive(pattern_name, matches, text):
                        detected_patterns.append(pattern_name)
        
        if detected_patterns:
            return GuardrailResult(
                passed=False,
                threat_level=ThreatLevel.CRITICAL,
                guardrail_type=GuardrailType.OUTPUT_SAFETY,
                message=f"AI output contains sensitive data: {', '.join(detected_patterns)}",
                metadata={"detected_patterns": detected_patterns}
            )
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.OUTPUT_SAFETY,
            message="No sensitive data in output"
        )
    
    def _is_treasury_context(self, text: str) -> bool:
        """Check if the text is in a treasury/banking context"""
        text_lower = text.lower()
        treasury_indicators = [
            'journal', 'transaction', 'debit', 'credit', 'account', 'bank',
            'payment', 'invoice', 'treasury', 'settlement', 'clearing', 
            'reconciliation', 'placement', 'withdrawal', 'asset', 'liability'
        ]
        
        # If we find multiple treasury indicators, it's likely treasury context
        indicator_count = sum(1 for indicator in treasury_indicators if indicator in text_lower)
        return indicator_count >= 2
    
    def _is_real_swift_code(self, code: str) -> bool:
        """Check if a string looks like a real SWIFT code vs generic text"""
        code_upper = code.upper()
        
        # Check against known treasury terms that aren't SWIFT codes
        if code_upper in self.treasury_terms_not_swift:
            return False
        
        # Real SWIFT codes should not be common words or abbreviations
        common_false_positives = [
            'ABCDEF12', 'XXXXXX11', 'TEST123456', 'SAMPLE12', 'DEMO123456',
            'REVENUE', 'EXPENSE', 'ACCOUNT', 'BALANCE', 'PAYMENT',
            'INVOICE', 'JOURNAL', 'BOOKING', 'MAPPING'
        ]
        
        if code_upper in common_false_positives:
            return False
        
        # SWIFT codes should be exactly 8 or 11 characters
        if len(code) not in [8, 11]:
            return False
        
        # First 4 characters should be bank code (letters)
        if not code[:4].isalpha():
            return False
        
        # Characters 5-6 should be country code (letters)
        if not code[4:6].isalpha():
            return False
        
        # If it passes all checks, treat as potentially real SWIFT code
        return True
    
    def _is_malicious_sql_in_treasury_context(self, text: str, matches: List[str]) -> bool:
        """Check if SQL-like terms are actually malicious in treasury context"""
        text_lower = text.lower()
        
        # Common false positives in treasury content
        treasury_false_positives = [
            'select account', 'select appropriate', 'select the', 'select specific',
            'update account', 'update the', 'update balance', 'update status',
            'create account', 'create entry', 'create journal', 'create transaction',
            'insert transaction', 'insert entry', 'insert record'
        ]
        
        # Check if any matches are actually legitimate treasury language
        for match in matches:
            match_clean = match.strip().lower()
            
            # Check for legitimate treasury usage
            for false_positive in treasury_false_positives:
                if false_positive in text_lower and match_clean in false_positive:
                    return False
            
            # Check for accounting/treasury context around the SQL terms
            accounting_context = [
                'account', 'journal', 'entry', 'debit', 'credit', 'transaction',
                'payment', 'invoice', 'balance', 'reconciliation', 'settlement'
            ]
            
            # Find text around the match
            match_pos = text_lower.find(match_clean)
            if match_pos >= 0:
                # Look at 50 characters before and after the match
                start = max(0, match_pos - 50)
                end = min(len(text_lower), match_pos + len(match_clean) + 50)
                context = text_lower[start:end]
                
                # If accounting terms are nearby, likely not malicious
                context_words = set(context.split())
                if any(term in context_words for term in accounting_context):
                    return False
        
        # If we get here, treat as potentially malicious
        return True
    
    def _check_prohibited_content(self, text: str) -> GuardrailResult:
        """Check for prohibited content in output"""
        text_lower = text.lower()
        
        for topic in self.prohibited_topics:
            if topic.lower() in text_lower:
                return GuardrailResult(
                    passed=False,
                    threat_level=ThreatLevel.HIGH,
                    guardrail_type=GuardrailType.CONTENT_FILTER,
                    message=f"Prohibited content detected: {topic}",
                    metadata={"prohibited_topic": topic}
                )
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.CONTENT_FILTER,
            message="No prohibited content detected"
        )
    
    def _check_output_quality(self, text: str) -> GuardrailResult:
        """Check output quality and coherence"""
        # Check for minimum length
        if len(text.strip()) < 10:
            return GuardrailResult(
                passed=False,
                threat_level=ThreatLevel.MEDIUM,
                guardrail_type=GuardrailType.OUTPUT_SAFETY,
                message="Output too short or empty",
                metadata={"output_length": len(text)}
            )
        
        # Check for repetitive content
        words = text.split()
        if len(words) > 10:
            word_ratio = len(set(words)) / len(words)
            if word_ratio < 0.3:  # Too many repeated words
                return GuardrailResult(
                    passed=False,
                    threat_level=ThreatLevel.MEDIUM,
                    guardrail_type=GuardrailType.OUTPUT_SAFETY,
                    message="Output appears repetitive or low quality",
                    metadata={"word_diversity_ratio": word_ratio}
                )
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.OUTPUT_SAFETY,
            message="Output quality check passed"
        )
    
    def _check_domain_relevance(self, text: str, context: Dict[str, Any] = None) -> GuardrailResult:
        """Check if output is relevant to treasury domain"""
        treasury_keywords = [
            'treasury', 'journal', 'transaction', 'asset', 'bank', 'account',
            'payment', 'invoice', 'debit', 'credit', 'balance', 'reconciliation',
            'placement', 'withdrawal', 'currency', 'exchange', 'settlement'
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in treasury_keywords if keyword in text_lower)
        
        # If context suggests treasury operation, require some domain relevance
        if context and context.get('domain') == 'treasury':
            if keyword_matches == 0 and len(text.split()) > 20:
                return GuardrailResult(
                    passed=False,
                    threat_level=ThreatLevel.MEDIUM,
                    guardrail_type=GuardrailType.CONTENT_FILTER,
                    message="Output not relevant to treasury domain",
                    metadata={"keyword_matches": keyword_matches}
                )
        
        return GuardrailResult(
            passed=True,
            threat_level=ThreatLevel.LOW,
            guardrail_type=GuardrailType.CONTENT_FILTER,
            message="Domain relevance check passed",
            metadata={"keyword_matches": keyword_matches}
        )
    
    def sanitize_output(self, text: str) -> str:
        """Sanitize output by removing/masking sensitive information"""
        sanitized = text
        
        # Mask credit card numbers
        sanitized = re.sub(
            r'\b(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})\b',
            r'\1-****-****-\4',
            sanitized
        )
        
        # Mask account numbers (keep first 4 and last 4 digits)
        sanitized = re.sub(
            r'\b(\d{4})\d{4,}(\d{4})\b',
            r'\1****\2',
            sanitized
        )
        
        # Mask email addresses partially
        sanitized = re.sub(
            r'\b([A-Za-z0-9._%+-]{1,3})[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
            r'\1***@\2',
            sanitized
        )
        
        return sanitized
    
    def get_guardrail_stats(self) -> Dict[str, Any]:
        """Get guardrail statistics"""
        return {
            "rate_limit_tracker_users": len(self.rate_limit_tracker),
            "sensitive_patterns_count": len(self.sensitive_patterns),
            "malicious_patterns_count": len(self.malicious_patterns),
            "prohibited_topics_count": len(self.prohibited_topics)
        } 

    def _is_treasury_terminology(self, term: str, text: str) -> bool:
        """Check if a term is legitimate treasury terminology that should not be flagged"""
        term_upper = term.upper()
        text_upper = text.upper()
        
        # Check against treasury safe terms
        if term_upper.lower() in self.treasury_safe_terms:
            return True
            
        # Check against treasury terms that look like SWIFT codes but aren't
        if term_upper in self.treasury_terms_not_swift:
            return True
        
        # Check if it appears in a treasury context within the text
        treasury_context_phrases = [
            f"TRANSACTION {term_upper}", f"{term_upper} TRANSACTION",
            f"ASSET {term_upper}", f"{term_upper} ASSET", 
            f"JOURNAL {term_upper}", f"{term_upper} JOURNAL",
            f"PAYMENT {term_upper}", f"{term_upper} PAYMENT",
            f"INVOICE {term_upper}", f"{term_upper} INVOICE"
        ]
        
        for phrase in treasury_context_phrases:
            if phrase in text_upper:
                return True
        
        # Check if the term is a common treasury abbreviation
        treasury_abbreviations = [
            'REVEU', 'REVENUE', 'EXPNS', 'EXPENSE', 'ACCT', 'ACCOUNT',
            'PYMNT', 'PAYMENT', 'INVST', 'INVEST', 'BOND', 'STOCK',
            'FUND', 'ETF', 'CASH', 'DERIV', 'DERIVATIVE', 'PLACE',
            'PLACEMENT', 'WITHD', 'WITHDRAWAL', 'SETT', 'SETTLEMENT'
        ]
        
        if term_upper in treasury_abbreviations:
            return True
        
        return False
    
    def _is_account_reference(self, text: str) -> bool:
        """Check if an email-like pattern is actually an account reference"""
        # Treasury systems often use patterns like "account@internal" or "ref@system"
        if '@' not in text:
            return False
            
        local_part, domain_part = text.split('@', 1)
        
        # Check for treasury-specific patterns
        treasury_patterns = [
            'account', 'acct', 'ref', 'reference', 'internal', 'system',
            'treasury', 'bank', 'custody', 'operational', 'settlement'
        ]
        
        for pattern in treasury_patterns:
            if pattern in local_part.lower() or pattern in domain_part.lower():
                return True
        
        return False
    
    def _is_treasury_reference_code(self, number: str, text: str) -> bool:
        """Check if a long number is a treasury reference code rather than sensitive account number"""
        text_lower = text.lower()
        
        # Treasury reference codes are often surrounded by specific context
        treasury_ref_contexts = [
            'reference', 'ref', 'code', 'id', 'identifier', 'transaction',
            'journal', 'entry', 'mapping', 'asset', 'account id', 'system id'
        ]
        
        # Check if the number appears near treasury context words
        for context in treasury_ref_contexts:
            if context in text_lower:
                return True
        
        # Check for patterns that indicate it's not a real account number
        if len(number) < 12:  # Real account numbers are usually 12+ digits
            return True
            
        # Sequential numbers (like 123456789012) are likely test data
        if number == ''.join(str(i % 10) for i in range(len(number))):
            return True
            
        return False
    
    def _is_clearly_sensitive(self, pattern_name: str, matches: List[str], text: str) -> bool:
        """Check if detected patterns are clearly sensitive despite treasury context"""
        # Credit cards and SSNs should always be flagged
        if pattern_name in ['credit_card', 'ssn']:
            return True
        
        # Phone numbers in treasury context might be legitimate business numbers
        if pattern_name == 'phone':
            # Check if it's in a business context
            text_lower = text.lower()
            business_contexts = ['contact', 'support', 'help', 'service', 'bank contact']
            if any(context in text_lower for context in business_contexts):
                return False
            return True
        
        # For other patterns, default to not sensitive in treasury context
        return False 