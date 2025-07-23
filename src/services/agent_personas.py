"""
Agent Personalization Service for Different Modes and Personas
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

from ..config import settings


class PersonaMode(Enum):
    """Available persona modes"""
    SUMMARY = "summary"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"


class ExpertiseLevel(Enum):
    """User expertise levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


@dataclass
class PersonaConfig:
    """Configuration for agent persona"""
    mode: PersonaMode
    expertise_level: ExpertiseLevel
    tone: str  # formal, casual, friendly, authoritative
    verbosity: str  # concise, normal, detailed
    include_examples: bool
    include_explanations: bool
    use_technical_terms: bool
    response_format: str  # bullet_points, paragraphs, structured
    max_response_length: int


class AgentPersonalizationService:
    """Service for managing agent personas and personalized responses"""
    
    def __init__(self):
        self.default_personas = self._init_default_personas()
        self.user_preferences = {}  # user_id -> PersonaConfig
    
    def _init_default_personas(self) -> Dict[PersonaMode, PersonaConfig]:
        """Initialize default persona configurations"""
        return {
            PersonaMode.SUMMARY: PersonaConfig(
                mode=PersonaMode.SUMMARY,
                expertise_level=ExpertiseLevel.INTERMEDIATE,
                tone="concise",
                verbosity="concise",
                include_examples=False,
                include_explanations=False,
                use_technical_terms=False,
                response_format="bullet_points",
                max_response_length=500
            ),
            
            PersonaMode.TECHNICAL: PersonaConfig(
                mode=PersonaMode.TECHNICAL,
                expertise_level=ExpertiseLevel.EXPERT,
                tone="authoritative",
                verbosity="detailed",
                include_examples=True,
                include_explanations=True,
                use_technical_terms=True,
                response_format="structured",
                max_response_length=2000
            ),
            
            PersonaMode.CREATIVE: PersonaConfig(
                mode=PersonaMode.CREATIVE,
                expertise_level=ExpertiseLevel.INTERMEDIATE,
                tone="friendly",
                verbosity="normal",
                include_examples=True,
                include_explanations=True,
                use_technical_terms=False,
                response_format="paragraphs",
                max_response_length=1500
            ),
            
            PersonaMode.ANALYTICAL: PersonaConfig(
                mode=PersonaMode.ANALYTICAL,
                expertise_level=ExpertiseLevel.EXPERT,
                tone="formal",
                verbosity="detailed",
                include_examples=True,
                include_explanations=True,
                use_technical_terms=True,
                response_format="structured",
                max_response_length=2500
            ),
            
            PersonaMode.CONVERSATIONAL: PersonaConfig(
                mode=PersonaMode.CONVERSATIONAL,
                expertise_level=ExpertiseLevel.BEGINNER,
                tone="friendly",
                verbosity="normal",
                include_examples=True,
                include_explanations=True,
                use_technical_terms=False,
                response_format="paragraphs",
                max_response_length=1000
            ),
            
            PersonaMode.PROFESSIONAL: PersonaConfig(
                mode=PersonaMode.PROFESSIONAL,
                expertise_level=ExpertiseLevel.EXPERT,
                tone="formal",
                verbosity="normal",
                include_examples=False,
                include_explanations=False,
                use_technical_terms=True,
                response_format="structured",
                max_response_length=1200
            )
        }
    
    def get_persona_config(self, user_id: str, mode: PersonaMode = None) -> PersonaConfig:
        """
        Get persona configuration for user
        
        Args:
            user_id: User identifier
            mode: Specific mode to use (overrides user preference)
            
        Returns:
            PersonaConfig for the user
        """
        if mode:
            # Use specified mode
            return self.default_personas[mode]
        
        if user_id in self.user_preferences:
            # Use user's saved preference
            return self.user_preferences[user_id]
        
        # Default to conversational mode
        return self.default_personas[PersonaMode.CONVERSATIONAL]
    
    def set_user_persona(self, user_id: str, persona_config: PersonaConfig):
        """Set persona configuration for user"""
        self.user_preferences[user_id] = persona_config
    
    def update_user_persona(self, user_id: str, updates: Dict[str, Any]) -> PersonaConfig:
        """
        Update specific aspects of user's persona
        
        Args:
            user_id: User identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated PersonaConfig
        """
        current_config = self.get_persona_config(user_id)
        
        # Create new config with updates
        config_dict = {
            'mode': updates.get('mode', current_config.mode),
            'expertise_level': updates.get('expertise_level', current_config.expertise_level),
            'tone': updates.get('tone', current_config.tone),
            'verbosity': updates.get('verbosity', current_config.verbosity),
            'include_examples': updates.get('include_examples', current_config.include_examples),
            'include_explanations': updates.get('include_explanations', current_config.include_explanations),
            'use_technical_terms': updates.get('use_technical_terms', current_config.use_technical_terms),
            'response_format': updates.get('response_format', current_config.response_format),
            'max_response_length': updates.get('max_response_length', current_config.max_response_length)
        }
        
        updated_config = PersonaConfig(**config_dict)
        self.user_preferences[user_id] = updated_config
        
        return updated_config
    
    def generate_system_prompt(self, persona_config: PersonaConfig, context: str = "treasury") -> str:
        """
        Generate system prompt based on persona configuration
        
        Args:
            persona_config: Persona configuration
            context: Domain context (e.g., "treasury")
            
        Returns:
            Formatted system prompt
        """
        base_role = self._get_base_role(context, persona_config.mode)
        tone_instruction = self._get_tone_instruction(persona_config.tone)
        verbosity_instruction = self._get_verbosity_instruction(persona_config.verbosity)
        format_instruction = self._get_format_instruction(persona_config.response_format)
        expertise_instruction = self._get_expertise_instruction(persona_config.expertise_level)
        
        system_prompt = f"""You are {base_role}.

{tone_instruction}

{verbosity_instruction}

{format_instruction}

{expertise_instruction}

Additional Guidelines:
- {'Include relevant examples to illustrate your points' if persona_config.include_examples else 'Focus on direct answers without examples'}
- {'Provide detailed explanations of concepts and processes' if persona_config.include_explanations else 'Provide concise answers without lengthy explanations'}
- {'Use technical terminology appropriate for expert users' if persona_config.use_technical_terms else 'Use simple, clear language avoiding technical jargon'}
- Keep responses under {persona_config.max_response_length} characters while maintaining completeness

Treasury Domain Expertise:
- You understand journal entries, account mappings, and transaction processing
- You're familiar with regulatory compliance and treasury operations
- You can analyze financial data and provide insights
- You know about asset management, cash flow, and risk assessment
"""
        
        return system_prompt.strip()
    
    def adapt_response(self, response: str, persona_config: PersonaConfig) -> str:
        """
        Adapt existing response to match persona style
        
        Args:
            response: Original response text
            persona_config: Target persona configuration
            
        Returns:
            Adapted response
        """
        # This is a simplified adaptation - in practice, you might use
        # another LLM call to reformat the response
        
        adapted = response
        
        # Apply format adaptations
        if persona_config.response_format == "bullet_points":
            adapted = self._convert_to_bullet_points(adapted)
        elif persona_config.response_format == "structured":
            adapted = self._add_structure_headers(adapted)
        
        # Apply length constraints
        if len(adapted) > persona_config.max_response_length:
            adapted = self._truncate_response(adapted, persona_config.max_response_length)
        
        return adapted
    
    def get_available_personas(self) -> List[Dict[str, Any]]:
        """Get list of available persona modes with descriptions"""
        personas = []
        
        for mode, config in self.default_personas.items():
            personas.append({
                "mode": mode.value,
                "name": mode.value.replace("_", " ").title(),
                "description": self._get_persona_description(mode),
                "characteristics": {
                    "tone": config.tone,
                    "verbosity": config.verbosity,
                    "expertise_level": config.expertise_level.value,
                    "includes_examples": config.include_examples,
                    "includes_explanations": config.include_explanations,
                    "uses_technical_terms": config.use_technical_terms,
                    "response_format": config.response_format,
                    "max_length": config.max_response_length
                }
            })
        
        return personas
    
    def _get_base_role(self, context: str, mode: PersonaMode) -> str:
        """Get base role description based on context and mode"""
        role_map = {
            PersonaMode.SUMMARY: f"a {context} expert who provides concise, high-level summaries",
            PersonaMode.TECHNICAL: f"a senior {context} technical specialist with deep system knowledge",
            PersonaMode.CREATIVE: f"an innovative {context} consultant who thinks outside the box",
            PersonaMode.ANALYTICAL: f"a {context} analyst who provides detailed data-driven insights",
            PersonaMode.CONVERSATIONAL: f"a friendly {context} assistant who explains things clearly",
            PersonaMode.PROFESSIONAL: f"a professional {context} advisor with extensive experience"
        }
        
        return role_map.get(mode, f"a {context} expert")
    
    def _get_tone_instruction(self, tone: str) -> str:
        """Get tone-specific instructions"""
        tone_map = {
            "formal": "Maintain a formal, professional tone in all responses.",
            "casual": "Use a casual, relaxed tone that's easy to understand.",
            "friendly": "Be warm and approachable while remaining professional.",
            "authoritative": "Speak with confidence and authority on the subject matter.",
            "concise": "Be direct and to-the-point in your communication."
        }
        
        return tone_map.get(tone, "Maintain a professional tone.")
    
    def _get_verbosity_instruction(self, verbosity: str) -> str:
        """Get verbosity-specific instructions"""
        verbosity_map = {
            "concise": "Provide brief, focused responses that get straight to the point.",
            "normal": "Provide balanced responses with sufficient detail without being overwhelming.",
            "detailed": "Provide comprehensive, thorough responses with extensive detail and context."
        }
        
        return verbosity_map.get(verbosity, "Provide appropriately detailed responses.")
    
    def _get_format_instruction(self, format_type: str) -> str:
        """Get format-specific instructions"""
        format_map = {
            "bullet_points": "Structure responses using bullet points and lists for clarity.",
            "paragraphs": "Use well-structured paragraphs with clear topic sentences.",
            "structured": "Use clear headings, sections, and logical organization."
        }
        
        return format_map.get(format_type, "Use clear, well-organized formatting.")
    
    def _get_expertise_instruction(self, expertise: ExpertiseLevel) -> str:
        """Get expertise-level specific instructions"""
        expertise_map = {
            ExpertiseLevel.BEGINNER: "Assume the user is new to the topic and needs foundational explanations.",
            ExpertiseLevel.INTERMEDIATE: "Assume the user has basic knowledge but may need clarification on complex topics.",
            ExpertiseLevel.EXPERT: "Assume the user has advanced knowledge and can handle technical details."
        }
        
        return expertise_map.get(expertise, "Adjust complexity based on context.")
    
    def _get_persona_description(self, mode: PersonaMode) -> str:
        """Get description for persona mode"""
        descriptions = {
            PersonaMode.SUMMARY: "Provides quick, high-level overviews and key points",
            PersonaMode.TECHNICAL: "Offers detailed technical explanations with expert-level depth",
            PersonaMode.CREATIVE: "Suggests innovative approaches and alternative solutions",
            PersonaMode.ANALYTICAL: "Delivers data-driven insights with thorough analysis",
            PersonaMode.CONVERSATIONAL: "Explains concepts in an easy-to-understand, friendly manner",
            PersonaMode.PROFESSIONAL: "Provides formal, business-focused advice and recommendations"
        }
        
        return descriptions.get(mode, "Customized response style")
    
    def _convert_to_bullet_points(self, text: str) -> str:
        """Convert text to bullet point format"""
        # Simple conversion - split by sentences and add bullets
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return text
        
        bullet_points = []
        for sentence in sentences:
            if sentence and not sentence.endswith('.'):
                sentence += '.'
            bullet_points.append(f"• {sentence}")
        
        return '\n'.join(bullet_points)
    
    def _add_structure_headers(self, text: str) -> str:
        """Add structure headers to text"""
        # Simple approach - add headers for major sections
        if len(text) < 300:
            return text
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return text
        
        # Add headers
        structured = []
        for i, paragraph in enumerate(paragraphs):
            if i == 0:
                structured.append(f"## Overview\n{paragraph}")
            else:
                structured.append(f"## Section {i}\n{paragraph}")
        
        return '\n\n'.join(structured)
    
    def _truncate_response(self, text: str, max_length: int) -> str:
        """Truncate response to maximum length"""
        if len(text) <= max_length:
            return text
        
        # Find last complete sentence within limit
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.8:  # If we can keep most of the content
            return text[:last_period + 1]
        else:
            return truncated.rstrip() + "..."
    
    def get_user_persona_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's persona usage"""
        if user_id not in self.user_preferences:
            return {"persona_set": False}
        
        config = self.user_preferences[user_id]
        
        return {
            "persona_set": True,
            "current_mode": config.mode.value,
            "expertise_level": config.expertise_level.value,
            "customizations": {
                "tone": config.tone,
                "verbosity": config.verbosity,
                "includes_examples": config.include_examples,
                "includes_explanations": config.include_explanations,
                "uses_technical_terms": config.use_technical_terms,
                "response_format": config.response_format,
                "max_response_length": config.max_response_length
            }
        } 