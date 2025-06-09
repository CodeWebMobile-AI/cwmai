"""
HTTP AI Client Module

Pure HTTP API client for all AI providers - No SDK dependencies.
Eliminates SDK initialization issues and proxies errors in GitHub Actions.
"""

import asyncio
import json
import logging
import os
import requests
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class HTTPAIClient:
    """Pure HTTP API client for all AI providers - No SDK dependencies."""
    
    def __init__(self):
        """Initialize HTTP AI client with API keys from environment variables."""
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        
        # Initialize comprehensive logging
        self.logger = logging.getLogger(f"{__name__}.HTTPAIClient")
        self.debug_mode = False
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Track which providers are available
        self.providers_available = {
            'anthropic': bool(self.anthropic_api_key),
            'openai': bool(self.openai_api_key),
            'gemini': bool(self.gemini_api_key),
            'deepseek': bool(self.deepseek_api_key)
        }
        
        # Log initialization
        self.logger.info(f"HTTPAIClient initialized with {sum(self.providers_available.values())} available providers")
        for provider, available in self.providers_available.items():
            status = "AVAILABLE" if available else "UNAVAILABLE"
            self.logger.debug(f"Provider {provider}: {status}")
    
    async def generate_enhanced_response(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate enhanced response using the best available AI provider.
        
        Args:
            prompt: The prompt to send to the AI
            model: Optional model preference ('claude', 'gpt', 'gemini', 'deepseek')
            
        Returns:
            Dictionary containing the AI response with content and metadata
        """
        request_id = f"req_{self.request_count}"
        self.request_count += 1
        
        start_time = time.time()
        self.logger.info(f"[{request_id}] Starting AI request - Model preference: {model or 'auto'}, Prompt length: {len(prompt)}")
        
        try:
            # Determine which provider to use
            if model == 'claude' or (model is None and self.providers_available['anthropic']):
                self.logger.debug(f"[{request_id}] Routing to Anthropic Claude")
                result = await self._call_anthropic_http(prompt, request_id)
            elif model == 'gpt' or (model is None and self.providers_available['openai']):
                self.logger.debug(f"[{request_id}] Routing to OpenAI GPT")
                result = await self._call_openai_http(prompt, request_id)
            elif model == 'gemini' or (model is None and self.providers_available['gemini']):
                self.logger.debug(f"[{request_id}] Routing to Google Gemini")
                result = await self._call_gemini_http(prompt, request_id)
            elif model == 'deepseek' or (model is None and self.providers_available['deepseek']):
                self.logger.debug(f"[{request_id}] Routing to DeepSeek")
                result = await self._call_deepseek_http(prompt, request_id)
            else:
                # Fallback: return mock response if no providers available
                self.logger.warning(f"[{request_id}] No AI providers available - returning mock response")
                result = {
                    'content': 'Mock AI response - no providers available',
                    'provider': 'mock',
                    'reasoning': 'No AI providers configured',
                    'confidence': 0.1
                }
            
            # Log successful completion
            duration = time.time() - start_time
            self.total_response_time += duration
            self.logger.info(f"[{request_id}] Request completed successfully in {duration:.2f}s - Provider: {result.get('provider', 'unknown')}")
            
            # Add request metadata
            result['request_id'] = request_id
            result['response_time'] = duration
            result['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            return result
                
        except Exception as e:
            # Error logging and fallback
            duration = time.time() - start_time
            self.error_count += 1
            self.logger.error(f"[{request_id}] Request failed after {duration:.2f}s: {str(e)}")
            self.logger.error(f"[{request_id}] Error details: {type(e).__name__}: {str(e)}")
            
            return {
                'content': f'Error generating response: {str(e)}',
                'provider': 'error',
                'error': str(e),
                'confidence': 0.0,
                'request_id': request_id,
                'response_time': duration,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _call_anthropic_http(self, prompt: str, request_id: str = "unknown") -> Dict[str, Any]:
        """Call Anthropic Claude API via HTTP."""
        if not self.anthropic_api_key:
            self.logger.error(f"[{request_id}] Anthropic API key not configured")
            return {
                'content': 'Anthropic API key not configured',
                'provider': 'anthropic',
                'error': 'API key not available',
                'confidence': 0.0
            }
        
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Log request details
            self.logger.debug(f"[{request_id}] Anthropic request URL: {url}")
            self.logger.debug(f"[{request_id}] Anthropic request headers: {self._sanitize_headers(headers)}")
            self.logger.debug(f"[{request_id}] Anthropic request payload size: {len(json.dumps(data))} bytes")
            
            request_start = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=30)
            request_duration = time.time() - request_start
            
            self.logger.debug(f"[{request_id}] Anthropic HTTP response: {response.status_code} in {request_duration:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                
                self.logger.debug(f"[{request_id}] Anthropic successful response: {len(json.dumps(result))} bytes")
                if self.debug_mode:
                    self.logger.debug(f"[{request_id}] Anthropic raw response: {json.dumps(result, indent=2)[:500]}...")
                
                # Extract content from Anthropic response format
                content = ""
                if result.get("content") and len(result["content"]) > 0:
                    content = result["content"][0].get("text", "No content generated")
                
                self.logger.info(f"[{request_id}] Anthropic content extracted: {len(content)} characters")
                
                return {
                    'content': content,
                    'provider': 'anthropic',
                    'model': 'claude-3-7-sonnet',
                    'confidence': 0.9,
                    'usage': result.get('usage', {}),
                    'raw_response_size': len(json.dumps(result))
                }
            else:
                error_text = response.text[:500]  # Limit error text
                self.logger.error(f"[{request_id}] Anthropic API error: HTTP {response.status_code}")
                self.logger.error(f"[{request_id}] Anthropic error response: {error_text}")
                
                return {
                    'content': f'Anthropic API error: HTTP {response.status_code}',
                    'provider': 'anthropic',
                    'error': f'HTTP {response.status_code}: {error_text}',
                    'confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"[{request_id}] Anthropic exception: {type(e).__name__}: {str(e)}")
            if self.debug_mode:
                import traceback
                self.logger.error(f"[{request_id}] Anthropic full traceback: {traceback.format_exc()}")
                
            return {
                'content': f'Anthropic API error: {str(e)}',
                'provider': 'anthropic',
                'error': str(e),
                'confidence': 0.0
            }
    
    async def _call_openai_http(self, prompt: str, request_id: str = "unknown") -> Dict[str, Any]:
        """Call OpenAI API via HTTP."""
        if not self.openai_api_key:
            return {
                'content': 'OpenAI API key not configured',
                'provider': 'openai',
                'error': 'API key not available',
                'confidence': 0.0
            }
        
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.7
            }
            
            self.logger.debug(f"[{request_id}] OpenAI request URL: {url}")
            request_start = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=30)
            request_duration = time.time() - request_start
            
            self.logger.debug(f"[{request_id}] OpenAI HTTP response: {response.status_code} in {request_duration:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                # Extract content from OpenAI response format
                content = ""
                if result.get("choices") and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"].get("content", "No content generated")
                
                self.logger.debug(f"[{request_id}] OpenAI content extracted: {len(content)} characters")
                return {
                    'content': content,
                    'provider': 'openai',
                    'model': 'gpt-4o',
                    'confidence': 0.8,
                    'usage': result.get('usage', {})
                }
            else:
                self.logger.error(f"[{request_id}] OpenAI HTTP error: {response.status_code}")
                return {
                    'content': f'OpenAI API error: HTTP {response.status_code}',
                    'provider': 'openai',
                    'error': f'HTTP {response.status_code}: {response.text}',
                    'confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"[{request_id}] OpenAI API error: {e}")
            return {
                'content': f'OpenAI API error: {str(e)}',
                'provider': 'openai',
                'error': str(e),
                'confidence': 0.0
            }
    
    async def _call_gemini_http(self, prompt: str, request_id: str = "unknown") -> Dict[str, Any]:
        """Call Google Gemini API via HTTP."""
        if not self.gemini_api_key:
            return {
                'content': 'Gemini API key not configured',
                'provider': 'gemini',
                'error': 'API key not available',
                'confidence': 0.0
            }
        
        try:
            model = "gemini-2.0-flash-001"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            
            self.logger.debug(f"[{request_id}] Gemini request URL: {url}")
            request_start = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=30)
            request_duration = time.time() - request_start
            
            self.logger.debug(f"[{request_id}] Gemini HTTP response: {response.status_code} in {request_duration:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                # Extract content from Gemini response format
                content = ""
                if result.get("candidates") and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if candidate.get("content") and candidate["content"].get("parts"):
                        content = candidate["content"]["parts"][0].get("text", "No content generated")
                
                self.logger.debug(f"[{request_id}] Gemini content extracted: {len(content)} characters")
                return {
                    'content': content,
                    'provider': 'gemini',
                    'model': 'gemini-2.0-flash',
                    'confidence': 0.7,
                    'usage': result.get('usageMetadata', {})
                }
            else:
                self.logger.error(f"[{request_id}] Gemini HTTP error: {response.status_code}")
                return {
                    'content': f'Gemini API error: HTTP {response.status_code}',
                    'provider': 'gemini',
                    'error': f'HTTP {response.status_code}: {response.text}',
                    'confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"[{request_id}] Gemini API error: {e}")
            return {
                'content': f'Gemini API error: {str(e)}',
                'provider': 'gemini',
                'error': str(e),
                'confidence': 0.0
            }
    
    async def _call_deepseek_http(self, prompt: str, request_id: str = "unknown", model: str = "deepseek-chat") -> Dict[str, Any]:
        """Call DeepSeek API via HTTP."""
        if not self.deepseek_api_key:
            return {
                'content': 'DeepSeek API key not configured',
                'provider': 'deepseek',
                'error': 'API key not available',
                'confidence': 0.0
            }
        
        try:
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.7
            }
            
            self.logger.debug(f"[{request_id}] DeepSeek request URL: {url}")
            request_start = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=30)
            request_duration = time.time() - request_start
            
            self.logger.debug(f"[{request_id}] DeepSeek HTTP response: {response.status_code} in {request_duration:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                # Extract content from DeepSeek response format (OpenAI compatible)
                content = ""
                if result.get("choices") and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"].get("content", "No content generated")
                
                self.logger.debug(f"[{request_id}] DeepSeek content extracted: {len(content)} characters")
                return {
                    'content': content,
                    'provider': 'deepseek',
                    'model': model,
                    'confidence': 0.8,
                    'usage': result.get('usage', {})
                }
            else:
                self.logger.error(f"[{request_id}] DeepSeek HTTP error: {response.status_code}")
                return {
                    'content': f'DeepSeek API error: HTTP {response.status_code}',
                    'provider': 'deepseek',
                    'error': f'HTTP {response.status_code}: {response.text}',
                    'confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"[{request_id}] DeepSeek API error: {e}")
            return {
                'content': f'DeepSeek API error: {str(e)}',
                'provider': 'deepseek',
                'error': str(e),
                'confidence': 0.0
            }
    
    def generate_enhanced_response_sync(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for generate_enhanced_response.
        
        This method allows synchronous code to call the async generate_enhanced_response method.
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a running loop, we can't use asyncio.run()
            # Create a new task instead
            return loop.run_until_complete(self.generate_enhanced_response(prompt, model))
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self.generate_enhanced_response(prompt, model))
    
    def get_research_ai_status(self) -> Dict[str, bool]:
        """Get status of research AI providers."""
        return {
            "gemini_available": self.providers_available['gemini'],
            "deepseek_available": self.providers_available['deepseek'],
            "anthropic_primary": self.providers_available['anthropic'],
            "openai_secondary": self.providers_available['openai']
        }
    
    def get_research_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive information about AI capabilities."""
        capabilities = {
            "available_providers": {},
            "research_functions": [],
            "analysis_types": [],
            "total_providers": 0,
            "primary_provider": None,
            "research_ready": False
        }
        
        # Check each provider
        for provider, available in self.providers_available.items():
            if available:
                capabilities["available_providers"][provider] = {
                    "status": "available",
                    "role": "primary_ai" if provider == "anthropic" else "research_ai",
                    "api_key_present": True
                }
                capabilities["total_providers"] += 1
                if provider == "anthropic":
                    capabilities["primary_provider"] = "anthropic"
            else:
                capabilities["available_providers"][provider] = {
                    "status": "unavailable",
                    "reason": "API key not configured"
                }
        
        # Define supported analysis types
        capabilities["analysis_types"] = [
            "general", "security", "trends", "technical", "market",
            "strategic", "performance", "competitive"
        ]
        
        # Add research functions if providers are available
        if any(self.providers_available.values()):
            capabilities["research_functions"] = [
                "generate_enhanced_response",
                "analyze_with_research_ai",
                "multi_provider_synthesis"
            ]
        
        capabilities["research_ready"] = capabilities["total_providers"] > 0
        
        return capabilities
    
    def analyze_with_research_ai(self, content: str, analysis_type: str = "general") -> str:
        """Use research AI providers to analyze content (synchronous version)."""
        prompt = f"Analyze this {analysis_type} content and provide insights: {content[:1000]}"
        
        # Try Gemini first, then DeepSeek as research AIs
        if self.providers_available['gemini']:
            result = self.generate_enhanced_response_sync(prompt, 'gemini')
            return result.get('content', '')
        elif self.providers_available['deepseek']:
            result = self.generate_enhanced_response_sync(prompt, 'deepseek')
            return result.get('content', '')
        
        return ""