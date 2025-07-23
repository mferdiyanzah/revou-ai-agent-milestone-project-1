"""
Voice Service for Speech-to-Text and Text-to-Speech
"""

import io
import time
import json
import logging
from typing import Dict, Any, Optional, BinaryIO, List
import asyncio
from dataclasses import dataclass

try:
    import speech_recognition as sr
    import pyttsx3
    import pyaudio
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

from ..config import settings


@dataclass
class VoiceSettings:
    """Voice configuration settings"""
    language: str = "en-US"
    speech_rate: int = 150  # words per minute
    voice_id: Optional[str] = None  # specific voice ID
    volume: float = 0.9
    pitch: int = 0  # -50 to +50
    recognition_timeout: float = 5.0
    phrase_timeout: float = 1.0


class VoiceService:
    """Service for voice input and output capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not VOICE_AVAILABLE:
            self.logger.warning(
                "Voice dependencies not available. Install with: "
                "pip install SpeechRecognition pyttsx3 pyaudio"
            )
            self.enabled = False
            return
        
        self.enabled = True
        self.voice_settings = VoiceSettings()
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self._configure_tts()
        
        # Adjust for ambient noise
        self._calibrate_microphone()
    
    def _configure_tts(self):
        """Configure text-to-speech engine"""
        if not self.enabled:
            return
        
        # Set speech rate
        self.tts_engine.setProperty('rate', self.voice_settings.speech_rate)
        
        # Set volume
        self.tts_engine.setProperty('volume', self.voice_settings.volume)
        
        # Set voice if specified
        if self.voice_settings.voice_id:
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if self.voice_settings.voice_id in voice.id:
                    self.tts_engine.setProperty('voice', voice.id)
                    break
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        if not self.enabled:
            return
        
        try:
            with self.microphone as source:
                self.logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)
                self.logger.info("Microphone calibration complete")
        except Exception as e:
            self.logger.error(f"Failed to calibrate microphone: {str(e)}")
    
    def listen_for_speech(
        self, 
        timeout: Optional[float] = None,
        phrase_timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Listen for speech input and convert to text
        
        Args:
            timeout: Maximum time to wait for speech
            phrase_timeout: Maximum time to wait between words
            
        Returns:
            Dict with transcription results
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Voice service not available",
                "text": ""
            }
        
        timeout = timeout or self.voice_settings.recognition_timeout
        phrase_timeout = phrase_timeout or self.voice_settings.phrase_timeout
        
        try:
            with self.microphone as source:
                self.logger.info("Listening for speech...")
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_timeout
                )
                
                self.logger.info("Processing speech...")
                
                # Recognize speech using Google Web Speech API
                text = self.recognizer.recognize_google(
                    audio, 
                    language=self.voice_settings.language
                )
                
                self.logger.info(f"Speech recognized: {text}")
                
                return {
                    "success": True,
                    "text": text,
                    "confidence": 1.0,  # Google API doesn't provide confidence
                    "language": self.voice_settings.language,
                    "duration": timeout
                }
                
        except sr.WaitTimeoutError:
            return {
                "success": False,
                "error": "No speech detected within timeout period",
                "text": "",
                "timeout": True
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "error": "Could not understand speech",
                "text": "",
                "unclear": True
            }
        except sr.RequestError as e:
            return {
                "success": False,
                "error": f"Speech recognition service error: {str(e)}",
                "text": "",
                "service_error": True
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in speech recognition: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "text": ""
            }
    
    def speak_text(self, text: str, interrupt: bool = False) -> Dict[str, Any]:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
            
        Returns:
            Dict with speech results
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Voice service not available"
            }
        
        if not text or not text.strip():
            return {
                "success": False,
                "error": "No text provided"
            }
        
        try:
            if interrupt:
                self.tts_engine.stop()
            
            # Clean text for better speech
            clean_text = self._clean_text_for_speech(text)
            
            self.logger.info(f"Speaking text: {clean_text[:100]}...")
            
            # Speak the text
            self.tts_engine.say(clean_text)
            self.tts_engine.runAndWait()
            
            return {
                "success": True,
                "text": clean_text,
                "duration_estimate": len(clean_text.split()) / (self.voice_settings.speech_rate / 60)
            }
            
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {str(e)}")
            return {
                "success": False,
                "error": f"Speech synthesis error: {str(e)}"
            }
    
    def speak_text_async(self, text: str) -> None:
        """Speak text asynchronously without blocking"""
        if not self.enabled:
            return
        
        try:
            import threading
            thread = threading.Thread(target=self.speak_text, args=(text,))
            thread.daemon = True
            thread.start()
        except Exception as e:
            self.logger.error(f"Error in async speech: {str(e)}")
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        if not self.enabled:
            return []
        
        try:
            voices = self.tts_engine.getProperty('voices')
            return [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "language": getattr(voice, 'languages', ['unknown'])[0] if hasattr(voice, 'languages') else 'unknown',
                    "gender": getattr(voice, 'gender', 'unknown')
                }
                for voice in voices
            ]
        except Exception as e:
            self.logger.error(f"Error getting voices: {str(e)}")
            return []
    
    def set_voice_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update voice settings
        
        Args:
            settings: Dictionary of voice settings to update
            
        Returns:
            True if settings were updated successfully
        """
        if not self.enabled:
            return False
        
        try:
            if 'language' in settings:
                self.voice_settings.language = settings['language']
            
            if 'speech_rate' in settings:
                self.voice_settings.speech_rate = max(50, min(300, settings['speech_rate']))
                self.tts_engine.setProperty('rate', self.voice_settings.speech_rate)
            
            if 'volume' in settings:
                self.voice_settings.volume = max(0.0, min(1.0, settings['volume']))
                self.tts_engine.setProperty('volume', self.voice_settings.volume)
            
            if 'voice_id' in settings:
                self.voice_settings.voice_id = settings['voice_id']
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if settings['voice_id'] in voice.id:
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            if 'recognition_timeout' in settings:
                self.voice_settings.recognition_timeout = max(1.0, min(30.0, settings['recognition_timeout']))
            
            if 'phrase_timeout' in settings:
                self.voice_settings.phrase_timeout = max(0.5, min(10.0, settings['phrase_timeout']))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating voice settings: {str(e)}")
            return False
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        # Remove or replace problematic characters
        replacements = {
            '&': ' and ',
            '@': ' at ',
            '#': ' hash ',
            '$': ' dollar ',
            '%': ' percent ',
            '^': ' caret ',
            '*': ' asterisk ',
            '|': ' pipe ',
            '\\': ' backslash ',
            '/': ' slash ',
            '~': ' tilde ',
            '`': ' backtick ',
            '_': ' underscore ',
            '=': ' equals ',
            '+': ' plus ',
            '-': ' minus ',
            '[': ' left bracket ',
            ']': ' right bracket ',
            '{': ' left brace ',
            '}': ' right brace ',
            '(': ' left paren ',
            ')': ' right paren ',
            '<': ' less than ',
            '>': ' greater than '
        }
        
        clean_text = text
        for char, replacement in replacements.items():
            clean_text = clean_text.replace(char, replacement)
        
        # Remove multiple spaces
        clean_text = ' '.join(clean_text.split())
        
        # Add pauses for better speech flow
        clean_text = clean_text.replace('. ', '. ... ')
        clean_text = clean_text.replace('! ', '! ... ')
        clean_text = clean_text.replace('? ', '? ... ')
        clean_text = clean_text.replace(': ', ': ... ')
        
        return clean_text
    
    def process_voice_command(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Complete voice command processing: listen -> transcribe -> return
        
        Args:
            timeout: Maximum time to wait for speech
            
        Returns:
            Dict with command processing results
        """
        start_time = time.time()
        
        # Listen for speech
        speech_result = self.listen_for_speech(timeout=timeout)
        
        if not speech_result["success"]:
            return {
                "success": False,
                "error": speech_result["error"],
                "processing_time": time.time() - start_time
            }
        
        command_text = speech_result["text"].strip()
        
        # Basic command preprocessing
        command_text = command_text.lower()
        
        # Check for common voice command patterns
        command_patterns = {
            "help": ["help", "what can you do", "commands", "assistance"],
            "stop": ["stop", "quit", "exit", "cancel"],
            "repeat": ["repeat", "say again", "what"],
            "louder": ["louder", "volume up", "speak up"],
            "quieter": ["quieter", "volume down", "softer"],
            "slower": ["slower", "slow down"],
            "faster": ["faster", "speed up"]
        }
        
        detected_command = None
        for command, patterns in command_patterns.items():
            if any(pattern in command_text for pattern in patterns):
                detected_command = command
                break
        
        return {
            "success": True,
            "text": speech_result["text"],
            "command": detected_command,
            "processed_text": command_text,
            "confidence": speech_result.get("confidence", 1.0),
            "processing_time": time.time() - start_time,
            "language": speech_result.get("language", self.voice_settings.language)
        }
    
    def handle_voice_interaction(
        self, 
        ai_response_text: str, 
        speak_response: bool = True,
        listen_for_followup: bool = False
    ) -> Dict[str, Any]:
        """
        Handle complete voice interaction: speak AI response and optionally listen for follow-up
        
        Args:
            ai_response_text: AI response to speak
            speak_response: Whether to speak the response
            listen_for_followup: Whether to listen for follow-up question
            
        Returns:
            Dict with interaction results
        """
        results = {
            "spoken": False,
            "speech_result": None,
            "followup_command": None,
            "error": None
        }
        
        # Speak the AI response
        if speak_response and ai_response_text:
            speech_result = self.speak_text(ai_response_text)
            results["spoken"] = speech_result["success"]
            results["speech_result"] = speech_result
            
            if not speech_result["success"]:
                results["error"] = speech_result.get("error")
        
        # Listen for follow-up if requested
        if listen_for_followup:
            followup_result = self.process_voice_command(timeout=5.0)
            results["followup_command"] = followup_result
        
        return results
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice service status and settings"""
        return {
            "enabled": self.enabled,
            "available": VOICE_AVAILABLE,
            "settings": {
                "language": self.voice_settings.language,
                "speech_rate": self.voice_settings.speech_rate,
                "volume": self.voice_settings.volume,
                "voice_id": self.voice_settings.voice_id,
                "recognition_timeout": self.voice_settings.recognition_timeout,
                "phrase_timeout": self.voice_settings.phrase_timeout
            } if self.enabled else None,
            "microphone_available": self.microphone is not None if self.enabled else False,
            "tts_available": self.tts_engine is not None if self.enabled else False
        }


class VoiceCommandProcessor:
    """Processor for voice commands in treasury context"""
    
    def __init__(self, voice_service: VoiceService):
        self.voice_service = voice_service
        self.logger = logging.getLogger(__name__)
        
        # Treasury-specific command mappings
        self.treasury_commands = {
            "search": ["search", "find", "look for", "locate"],
            "create": ["create", "add", "new", "make"],
            "analyze": ["analyze", "examine", "review", "check"],
            "journal": ["journal", "entry", "booking", "record"],
            "transaction": ["transaction", "payment", "transfer"],
            "balance": ["balance", "amount", "total"],
            "report": ["report", "summary", "overview"],
            "help": ["help", "assistance", "guide", "instructions"]
        }
    
    def process_treasury_command(self, command_text: str) -> Dict[str, Any]:
        """
        Process voice command in treasury context
        
        Args:
            command_text: Transcribed command text
            
        Returns:
            Dict with processed command information
        """
        command_lower = command_text.lower()
        
        # Detect treasury operation
        detected_operation = None
        for operation, keywords in self.treasury_commands.items():
            if any(keyword in command_lower for keyword in keywords):
                detected_operation = operation
                break
        
        # Extract entities (simplified)
        entities = self._extract_entities(command_text)
        
        # Generate structured command
        structured_command = {
            "original_text": command_text,
            "operation": detected_operation,
            "entities": entities,
            "confidence": 0.8 if detected_operation else 0.3,
            "suggested_actions": self._suggest_actions(detected_operation, entities)
        }
        
        return structured_command
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from command text"""
        entities = {
            "amounts": [],
            "dates": [],
            "currencies": [],
            "account_types": [],
            "transaction_types": []
        }
        
        # Simple pattern matching (can be enhanced with NLP)
        import re
        
        # Extract amounts
        amount_patterns = [
            r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:\.\d+)?)\s*(million|thousand|k)',
            r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*(million|thousand)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["amounts"].extend([match[0] if isinstance(match, tuple) else match for match in matches])
        
        # Extract currencies
        currency_patterns = [
            r'\b(USD|EUR|GBP|JPY|AUD|CAD|CHF|SGD|HKD)\b',
            r'\b(dollar|euro|pound|yen|franc)\b'
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["currencies"].extend(matches)
        
        # Extract account types
        account_patterns = [
            r'\b(checking|savings|investment|custody|operational)\s*account\b',
            r'\b(cash|deposit|placement|withdrawal)\b'
        ]
        
        for pattern in account_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["account_types"].extend(matches)
        
        return entities
    
    def _suggest_actions(self, operation: str, entities: Dict[str, List[str]]) -> List[str]:
        """Suggest possible actions based on operation and entities"""
        suggestions = []
        
        if operation == "search":
            if entities["amounts"]:
                suggestions.append("Search for transactions with specific amounts")
            if entities["currencies"]:
                suggestions.append("Search by currency type")
            if not entities["amounts"] and not entities["currencies"]:
                suggestions.append("Perform general search")
        
        elif operation == "analyze":
            suggestions.append("Analyze transaction patterns")
            if entities["account_types"]:
                suggestions.append("Analyze specific account types")
        
        elif operation == "journal":
            suggestions.append("Create journal entry")
            suggestions.append("Search journal entries")
        
        elif operation == "help":
            suggestions.append("Show available commands")
            suggestions.append("Provide system guidance")
        
        else:
            suggestions.append("Process treasury operation")
            suggestions.append("Get more specific information")
        
        return suggestions 