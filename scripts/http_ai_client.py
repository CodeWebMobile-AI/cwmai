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
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Import AI API logger
from scripts.ai_api_logger import (
    get_ai_api_logger,
    log_ai_request_start,
    log_ai_request_complete,
    log_ai_request_error,
    AIRequestType,
    AIEventType
)

# Import AI response cache - try Redis first, then fallback
try:
    from redis_ai_response_cache import RedisAIResponseCache
    # Initialize global Redis cache
    _redis_cache = RedisAIResponseCache()
    
    async def cache_ai_response(prompt: str, response: str, provider: str = None, 
                              model: str = None, cost_estimate: float = 0.0):
        """Cache AI response using Redis."""
        await _redis_cache.put(prompt, response, provider, model, cost_estimate)
    
    async def get_cached_response(prompt: str, provider: str = None, model: str = None):
        """Get cached response from Redis."""
        return await _redis_cache.get(prompt, provider, model)
    
    CACHE_AVAILABLE = True
    CACHE_TYPE = "redis"
except ImportError:
    try:
        from ai_response_cache import get_global_cache, cache_ai_response, get_cached_response
        CACHE_AVAILABLE = True
        CACHE_TYPE = "in-memory"
    except ImportError:
        CACHE_AVAILABLE = False
        CACHE_TYPE = None


class HTTPAIClient:
    """Pure HTTP API client for all AI providers - No SDK dependencies."""
    
    def __init__(self, enable_round_robin: bool = False):
        """Initialize HTTP AI client with API keys from environment variables.
        
        Args:
            enable_round_robin: Whether to enable round-robin provider selection
        """
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
        
        # Round-robin configuration
        self.enable_round_robin = enable_round_robin
        # Priority order: Gemini, OpenAI, DeepSeek, Anthropic (since Anthropic has no credits)
        provider_priority = ['gemini', 'openai', 'deepseek', 'anthropic']
        self.available_providers = [p for p in provider_priority if self.providers_available.get(p, False)]
        self.last_provider_index = -1
        
        # Log initialization
        self.logger.info(f"HTTPAIClient initialized with {sum(self.providers_available.values())} available providers")
        if self.enable_round_robin:
            self.logger.info(f"✓ Round-robin enabled across providers: {self.available_providers}")
        for provider, available in self.providers_available.items():
            status = "AVAILABLE" if available else "UNAVAILABLE"
            self.logger.debug(f"Provider {provider}: {status}")
        
        # Log cache type
        if CACHE_AVAILABLE:
            self.logger.info(f"✓ AI response cache enabled (type: {CACHE_TYPE})")
        else:
            self.logger.warning("⚠ AI response cache not available")
    
    def _get_next_provider(self) -> Optional[str]:
        """Get the next provider in round-robin order.
        
        Returns:
            Next available provider name or None if no providers available
        """
        if not self.available_providers:
            return None
            
        self.last_provider_index = (self.last_provider_index + 1) % len(self.available_providers)
        return self.available_providers[self.last_provider_index]
    
    def _determine_provider_name(self, model: Optional[str]) -> str:
        """Determine provider name from model preference.
        
        Args:
            model: Model preference or None
            
        Returns:
            Provider name for caching
        """
        if model == 'claude':
            return 'anthropic'
        elif model == 'gpt':
            return 'openai'
        elif model in ['gemini', 'deepseek']:
            return model
        elif self.enable_round_robin:
            # For round-robin, we don't know yet which provider will be selected
            return 'auto'
        else:
            # Default to first available provider
            return self.available_providers[0] if self.available_providers else 'none'
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers for logging by hiding sensitive information."""
        sanitized = {}
        for key, value in headers.items():
            if 'authorization' in key.lower() or 'key' in key.lower():
                sanitized[key] = '***'
            else:
                sanitized[key] = value
        return sanitized
    
    def _make_request_with_retry(self, url: str, headers: Dict[str, str], data: Dict[str, Any], 
                                 request_id: str, provider: str, timeout: int = 120) -> requests.Response:
        """Make HTTP request with retry logic and exponential backoff."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
                return response
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"[{request_id}] {provider} timeout on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                    ai_logger = get_ai_api_logger()
                    ai_logger.log_retry_attempt(request_id, provider, attempt + 1, "Timeout", retry_delay)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"[{request_id}] {provider} failed after {max_retries} attempts")
                    raise
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"[{request_id}] {provider} connection error on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                    ai_logger = get_ai_api_logger()
                    ai_logger.log_retry_attempt(request_id, provider, attempt + 1, "Connection Error", retry_delay)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error(f"[{request_id}] {provider} connection failed after {max_retries} attempts")
                    raise
    
    async def generate_enhanced_response(self, prompt: str, model: Optional[str] = None, prefill: str = None) -> Dict[str, Any]:
        """Generate enhanced response using the best available AI provider with caching.
        
        Args:
            prompt: The prompt to send to the AI
            model: Optional model preference ('claude', 'gpt', 'gemini', 'deepseek')
            prefill: Optional assistant message to prefill (for structured outputs with Claude)
            
        Returns:
            Dictionary containing the AI response with content and metadata
        """
        request_id = f"req_{self.request_count}"
        self.request_count += 1
        
        start_time = time.time()
        self.logger.info(f"[{request_id}] Starting AI request - Model preference: {model or 'auto'}, Prompt length: {len(prompt)}")
        
        # Log to AI API logger
        ai_logger = get_ai_api_logger()
        
        # Check cache first if available
        if CACHE_AVAILABLE:
            try:
                provider_name = self._determine_provider_name(model)
                cached_response = await get_cached_response(prompt, provider_name, model or 'auto')
                if cached_response:
                    duration = time.time() - start_time
                    self.logger.info(f"[{request_id}] Cache HIT - Returning cached response in {duration:.2f}s")
                    
                    # Log cache hit to AI API logger
                    ai_logger.log_cache_event(
                        AIEventType.CACHE_HIT,
                        request_id,
                        prompt,
                        provider_name,
                        model or 'auto',
                        cache_backend=CACHE_TYPE
                    )
                    
                    cached_response_data = {
                        'content': cached_response,
                        'provider': provider_name,
                        'model': model or 'auto',
                        'request_id': request_id,
                        'response_time': duration,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'cached': True,
                        'confidence': 0.9  # High confidence for cached responses
                    }
                    
                    # Log completion
                    log_ai_request_complete(request_id, cached_response_data, duration, cached=True)
                    return cached_response_data
            except Exception as e:
                self.logger.warning(f"[{request_id}] Cache check failed: {e}")
                ai_logger.log_cache_event(
                    AIEventType.CACHE_MISS,
                    request_id,
                    prompt,
                    provider_name,
                    model or 'auto',
                    error=str(e)
                )
        
        try:
            # Determine which provider to use
            if self.enable_round_robin and model is None:
                # Use round-robin selection
                selected_provider = self._get_next_provider()
                if selected_provider == 'anthropic':
                    self.logger.info(f"[{request_id}] Round-robin selected: Anthropic Claude")
                    log_ai_request_start(request_id, prompt, 'anthropic', 'claude-3-7-sonnet',
                                       round_robin=True, prefill=prefill)
                    result = await self._call_anthropic_http(prompt, request_id, prefill)
                elif selected_provider == 'openai':
                    self.logger.info(f"[{request_id}] Round-robin selected: OpenAI GPT")
                    log_ai_request_start(request_id, prompt, 'openai', 'gpt-4o', round_robin=True)
                    result = await self._call_openai_http(prompt, request_id)
                elif selected_provider == 'gemini':
                    self.logger.info(f"[{request_id}] Round-robin selected: Google Gemini")
                    log_ai_request_start(request_id, prompt, 'gemini', 'gemini-2.0-flash', round_robin=True)
                    result = await self._call_gemini_http(prompt, request_id)
                elif selected_provider == 'deepseek':
                    self.logger.info(f"[{request_id}] Round-robin selected: DeepSeek")
                    log_ai_request_start(request_id, prompt, 'deepseek', 'deepseek-chat', round_robin=True)
                    result = await self._call_deepseek_http(prompt, request_id)
                else:
                    # No providers available
                    self.logger.warning(f"[{request_id}] No providers available for round-robin")
                    result = {
                        'content': 'No AI providers available',
                        'provider': 'none',
                        'error': 'No providers configured',
                        'confidence': 0.0
                    }
            elif model is None:
                # Auto mode: try providers in priority order until one succeeds
                last_error = None
                for provider in self.available_providers:
                    if provider == 'anthropic':
                        self.logger.debug(f"[{request_id}] Auto-selecting Anthropic Claude")
                        log_ai_request_start(request_id, prompt, 'anthropic', 'claude-3-7-sonnet', prefill=prefill)
                        result = await self._call_anthropic_http(prompt, request_id, prefill)
                    elif provider == 'openai':
                        self.logger.debug(f"[{request_id}] Auto-selecting OpenAI GPT")
                        log_ai_request_start(request_id, prompt, 'openai', 'gpt-4o')
                        result = await self._call_openai_http(prompt, request_id)
                    elif provider == 'gemini':
                        self.logger.debug(f"[{request_id}] Auto-selecting Google Gemini")
                        log_ai_request_start(request_id, prompt, 'gemini', 'gemini-2.0-flash')
                        result = await self._call_gemini_http(prompt, request_id)
                    elif provider == 'deepseek':
                        self.logger.debug(f"[{request_id}] Auto-selecting DeepSeek")
                        log_ai_request_start(request_id, prompt, 'deepseek', 'deepseek-chat')
                        result = await self._call_deepseek_http(prompt, request_id)
                    else:
                        continue
                    if not result.get('error'):
                        break
                    last_error = result.get('error')
                    self.logger.warning(f"[{request_id}] Provider {provider} failed: {last_error}, trying next provider")
                else:
                    # All providers failed
                    self.logger.error(f"[{request_id}] All AI providers failed: {last_error}")
                    result = {
                        'content': f"Error generating response: {last_error}",
                        'provider': 'error',
                        'error': f"All providers failed: {last_error}",
                        'confidence': 0.0
                    }
            elif model == 'claude':
                self.logger.debug(f"[{request_id}] Routing to Anthropic Claude")
                log_ai_request_start(request_id, prompt, 'anthropic', 'claude-3-7-sonnet', prefill=prefill)
                result = await self._call_anthropic_http(prompt, request_id, prefill)
            elif model == 'gpt':
                self.logger.debug(f"[{request_id}] Routing to OpenAI GPT")
                log_ai_request_start(request_id, prompt, 'openai', 'gpt-4o')
                result = await self._call_openai_http(prompt, request_id)
            elif model == 'gemini':
                self.logger.debug(f"[{request_id}] Routing to Google Gemini")
                log_ai_request_start(request_id, prompt, 'gemini', 'gemini-2.0-flash')
                result = await self._call_gemini_http(prompt, request_id)
            elif model == 'deepseek':
                self.logger.debug(f"[{request_id}] Routing to DeepSeek")
                log_ai_request_start(request_id, prompt, 'deepseek', 'deepseek-chat')
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
            result['cached'] = False
            
            # Log successful completion
            log_ai_request_complete(request_id, result, duration, cached=False)
            
            # Cache the response if caching is available and response is valid
            if CACHE_AVAILABLE and result.get('content') and 'error' not in result:
                try:
                    # Estimate cost based on provider and token usage
                    cost_estimate = self._estimate_request_cost(result, len(prompt))
                    await cache_ai_response(
                        prompt=prompt,
                        response=result['content'],
                        provider=result.get('provider', 'unknown'),
                        model=result.get('model', model or 'auto'),
                        cost_estimate=cost_estimate
                    )
                    self.logger.debug(f"[{request_id}] Response cached successfully")
                    ai_logger.log_cache_event(
                        AIEventType.CACHE_STORE,
                        request_id,
                        prompt,
                        result.get('provider', 'unknown'),
                        result.get('model', model or 'auto'),
                        cost_estimate=cost_estimate
                    )
                except Exception as e:
                    self.logger.warning(f"[{request_id}] Failed to cache response: {e}")
            
            return result
                
        except Exception as e:
            # Error logging and fallback
            duration = time.time() - start_time
            self.error_count += 1
            self.logger.error(f"[{request_id}] Request failed after {duration:.2f}s: {str(e)}")
            self.logger.error(f"[{request_id}] Error details: {type(e).__name__}: {str(e)}")
            
            # Log error to AI API logger
            log_ai_request_error(request_id, e, model or 'auto', model or 'auto')
            
            return {
                'content': f'Error generating response: {str(e)}',
                'provider': 'error',
                'error': str(e),
                'confidence': 0.0,
                'request_id': request_id,
                'response_time': duration,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cached': False
            }
    
    async def _call_anthropic_http(self, prompt: str, request_id: str = "unknown", prefill: str = None) -> Dict[str, Any]:
        """Call Anthropic Claude API via HTTP.
        
        Args:
            prompt: The user prompt
            request_id: Request ID for tracking
            prefill: Optional assistant message to prefill (for structured outputs)
        """
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
            
            # Build messages with optional prefill
            messages = [{"role": "user", "content": prompt}]
            if prefill:
                messages.append({"role": "assistant", "content": prefill})
            
            data = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 4000,
                "messages": messages
            }
            
            # Log request details
            self.logger.debug(f"[{request_id}] Anthropic request URL: {url}")
            self.logger.debug(f"[{request_id}] Anthropic request headers: {self._sanitize_headers(headers)}")
            self.logger.debug(f"[{request_id}] Anthropic request payload size: {len(json.dumps(data))} bytes")
            
            request_start = time.time()
            response = self._make_request_with_retry(url, headers, data, request_id, "Anthropic")
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
                
                # Check if it's a credit balance error
                if response.status_code == 400 and 'credit balance is too low' in error_text.lower():
                    self.logger.warning(f"[{request_id}] Anthropic API has insufficient credits - will fallback to other providers")
                
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
            response = self._make_request_with_retry(url, headers, data, request_id, "OpenAI")
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
            response = self._make_request_with_retry(url, headers, data, request_id, "Gemini")
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
            response = self._make_request_with_retry(url, headers, data, request_id, "DeepSeek")
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
    
    def generate_enhanced_response_sync(self, prompt: str, model: Optional[str] = None, prefill: str = None) -> Dict[str, Any]:
        """Synchronous wrapper for generate_enhanced_response.
        
        This method allows synchronous code to call the async generate_enhanced_response method.
        
        Args:
            prompt: The prompt to send to the AI
            model: Optional specific model to use
            prefill: Optional assistant message to prefill (for structured outputs with Claude)
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a running loop, we can't use asyncio.run()
            # Create a new task instead
            return loop.run_until_complete(self.generate_enhanced_response(prompt, model, prefill))
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self.generate_enhanced_response(prompt, model, prefill))
    
    def get_research_ai_status(self) -> Dict[str, bool]:
        """Get status of research AI providers."""
        return {
            "gemini_available": self.providers_available['gemini'],
            "deepseek_available": self.providers_available['deepseek'],
            "anthropic_primary": self.providers_available['anthropic'],
            "openai_secondary": self.providers_available['openai']
        }
    
    async def conduct_targeted_research(self, query: str, context: Dict[str, Any], 
                                      research_type: str = "performance") -> Dict[str, Any]:
        """Execute targeted research with CWMAI performance context.
        
        Args:
            query: Specific research question
            context: CWMAI performance context and metrics
            research_type: Type of research (performance, claude_interaction, task_generation)
            
        Returns:
            Research results with actionable insights
        """
        request_id = f"research_{self.request_count}"
        self.request_count += 1
        
        # Enhance query with performance context
        enhanced_query = f"""
        CWMAI AI System Research Request:
        
        Research Question: {query}
        
        Current System Context:
        - Claude Success Rate: {context.get('claude_success_rate', 0)}%
        - Task Completion Rate: {context.get('task_completion_rate', 0)}%
        - Active Projects: {len(context.get('projects', []))}
        - System Health: {context.get('system_health', 'unknown')}
        - Recent Errors: {context.get('recent_errors', [])}
        
        Research Focus: {research_type}
        
        Please provide specific, actionable recommendations for improving CWMAI's performance.
        Include:
        1. Root cause analysis
        2. Specific implementation steps
        3. Expected impact on success rates
        4. Testing and validation approach
        
        Format response as structured insights that can be converted to improvement tasks.
        """
        
        # Execute research with enhanced context
        response = await self.generate_enhanced_response(enhanced_query, model='claude')
        
        # Add research metadata
        if response:
            response['research_type'] = research_type
            response['research_context'] = context
            response['query'] = query
            response['research_id'] = request_id
            
        return response
    
    async def extract_actionable_insights(self, research_content: str, 
                                        performance_gap: str) -> List[Dict[str, Any]]:
        """Extract specific actionable insights from research results.
        
        Args:
            research_content: Research results content
            performance_gap: Specific performance issue to address
            
        Returns:
            List of actionable insight dictionaries
        """
        insights_query = f"""
        Extract actionable insights from this research to address: {performance_gap}
        
        Research Content:
        {research_content[:2000]}
        
        Extract specific insights in this JSON format:
        [
          {{
            "insight": "Specific insight description",
            "action_type": "code_change|configuration|prompt_update|process_change",
            "priority": "critical|high|medium|low",
            "implementation_steps": ["step1", "step2", "step3"],
            "expected_impact": "Description of expected improvement",
            "success_metric": "How to measure success",
            "risk_level": "low|medium|high",
            "effort_estimate": "1-5 scale"
          }}
        ]
        
        Focus on insights that directly address the performance gap.
        """
        
        response = await self.generate_enhanced_response(insights_query, prefill="[")
        
        insights = []
        if response and 'content' in response:
            content = response['content']
            if not content.strip().startswith('['):
                content = '[' + content
            
            try:
                insights = json.loads(content)
                if not isinstance(insights, list):
                    insights = []
            except json.JSONDecodeError:
                # Fallback to basic insight extraction
                insights = [{
                    "insight": content[:500],
                    "action_type": "process_change",
                    "priority": "medium",
                    "implementation_steps": ["Review research findings", "Plan implementation"],
                    "expected_impact": "Performance improvement",
                    "success_metric": "Improved success rates",
                    "risk_level": "low",
                    "effort_estimate": "3"
                }]
        
        return insights
    
    async def evaluate_research_quality(self, research: str, expected_outcome: str) -> float:
        """Evaluate the quality of research results for learning.
        
        Args:
            research: Research content to evaluate
            expected_outcome: What the research was supposed to achieve
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        quality_query = f"""
        Evaluate the quality of this research for achieving the expected outcome.
        
        Research Content: {research[:1000]}
        Expected Outcome: {expected_outcome}
        
        Rate on a scale of 0.0 to 1.0 based on:
        - Relevance to expected outcome (0.3 weight)
        - Actionability of recommendations (0.3 weight)
        - Specificity and detail (0.2 weight)
        - Technical accuracy (0.2 weight)
        
        Respond with just the numeric score (e.g., 0.75)
        """
        
        response = await self.generate_enhanced_response(quality_query)
        
        if response and 'content' in response:
            try:
                import re
                score_match = re.search(r'(\d+\.?\d*)', response['content'])
                if score_match:
                    score = float(score_match.group(1))
                    return min(1.0, max(0.0, score))
            except:
                pass
        
        return 0.5  # Default moderate quality
    
    async def generate_performance_research_query(self, gap_type: str, 
                                                current_metrics: Dict[str, Any]) -> str:
        """Generate targeted research query for specific performance gaps.
        
        Args:
            gap_type: Type of performance gap (claude_interaction, task_completion, etc.)
            current_metrics: Current performance metrics
            
        Returns:
            Targeted research query string
        """
        query_templates = {
            "claude_interaction": f"""
                How to improve Claude AI API interaction success rate from {current_metrics.get('claude_success_rate', 0)}% to >80%?
                
                Current issues may include:
                - API authentication failures
                - Prompt formatting problems
                - Rate limiting issues
                - Response parsing errors
                
                Focus on practical fixes for AI task management systems.
            """,
            "task_completion": f"""
                How to improve AI task completion rate from {current_metrics.get('task_completion_rate', 0)}% to >90%?
                
                Current context:
                - Task types: {current_metrics.get('task_types', [])}
                - Failure patterns: {current_metrics.get('failure_patterns', [])}
                - System load: {current_metrics.get('system_load', 'unknown')}
                
                Focus on task orchestration and AI coordination improvements.
            """,
            "multi_agent_coordination": f"""
                How to improve multi-agent AI coordination for better task distribution and completion?
                
                Current performance:
                - Agent success rates: {current_metrics.get('agent_performance', {})}
                - Coordination failures: {current_metrics.get('coordination_issues', [])}
                - Task distribution efficiency: {current_metrics.get('distribution_efficiency', 0)}%
                
                Focus on swarm intelligence and agent communication patterns.
            """,
            "outcome_learning": f"""
                How to improve AI system learning from task outcomes and failures?
                
                Current learning metrics:
                - Learning rate: {current_metrics.get('learning_rate', 0)}%
                - Pattern recognition: {current_metrics.get('pattern_recognition', 0)}%
                - Adaptation speed: {current_metrics.get('adaptation_speed', 'slow')}
                
                Focus on machine learning and adaptive AI system design.
            """
        }
        
        return query_templates.get(gap_type, f"How to improve {gap_type} performance in AI systems?")

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
    
    def extract_json_from_response(self, text: str) -> Any:
        """Extract JSON from AI response, handling various formats.
        
        Args:
            text: The AI response text that may contain JSON
            
        Returns:
            Parsed JSON object or None if extraction fails
        """
        if not text:
            return None
            
        # Try direct parsing first
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Try extracting from markdown code blocks
        # Matches ```json or ``` followed by JSON array or object
        code_block_patterns = [
            r'```(?:json)?\s*(\[[\s\S]*?\])\s*```',  # JSON array in code block
            r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',  # JSON object in code block
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        
        # Try finding first JSON array or object in the text
        json_patterns = [
            r'(\[[\s\S]*?\])',  # Find first complete array
            r'(\{[\s\S]*?\})',  # Find first complete object
        ]
        
        for pattern in json_patterns:
            # Find all potential JSON structures
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    # Try to parse each match
                    potential_json = match.group(1)
                    # Basic bracket balance check
                    if potential_json.count('[') == potential_json.count(']') or \
                       potential_json.count('{') == potential_json.count('}'):
                        return json.loads(potential_json)
                except:
                    continue
        
        return None
    
    # CWMAI-Specific Research Bridge Methods
    # Note: Basic trend/news gathering removed - Research Intelligence System handles research topic selection
    
    
    
    
    # Include helper methods
    from .http_ai_client_helpers import HTTPAIClientHelpers
    
    # Mix in helper methods
    _determine_provider_name = HTTPAIClientHelpers._determine_provider_name
    _estimate_request_cost = HTTPAIClientHelpers._estimate_request_cost
    get_cache_stats = HTTPAIClientHelpers.get_cache_stats
    warm_cache_from_history = HTTPAIClientHelpers.warm_cache_from_history
    clear_cache = HTTPAIClientHelpers.clear_cache