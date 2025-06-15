"""
Enhanced HTTP AI Client Module

Enhanced HTTP AI client with Redis-backed caching, distributed coordination,
and seamless migration from legacy caching systems.
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

# Import enhanced Redis-backed cache
from scripts.redis_integration_adapters import (
    get_enhanced_cache, 
    get_enhanced_state_manager,
    enhanced_cache_ai_response,
    get_enhanced_cached_response
)

# Fallback to legacy cache if Redis unavailable
try:
    from .ai_response_cache import get_global_cache, cache_ai_response, get_cached_response
    LEGACY_CACHE_AVAILABLE = True
except ImportError:
    LEGACY_CACHE_AVAILABLE = False


class EnhancedHTTPAIClient:
    """Enhanced HTTP AI client with Redis-backed caching and distributed coordination."""
    
    def __init__(self, 
                 enable_redis_cache: bool = True,
                 cache_migration_mode: str = "gradual",
                 enable_distributed_coordination: bool = True):
        """Initialize enhanced HTTP AI client.
        
        Args:
            enable_redis_cache: Whether to use Redis-backed cache
            cache_migration_mode: Cache migration strategy (gradual/immediate/readonly)
            enable_distributed_coordination: Enable distributed state coordination
        """
        # API keys from environment
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        
        # Enhanced caching configuration
        self.enable_redis_cache = enable_redis_cache
        self.cache_migration_mode = cache_migration_mode
        self.enable_distributed_coordination = enable_distributed_coordination
        
        # Client state
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0.0
        self.cache_hit_count = 0
        self.redis_cache_hits = 0
        self.legacy_cache_hits = 0
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.EnhancedHTTPAIClient")
        self.debug_mode = False
        
        # Track available providers
        self.providers_available = {
            'anthropic': bool(self.anthropic_api_key),
            'openai': bool(self.openai_api_key),
            'gemini': bool(self.gemini_api_key),
            'deepseek': bool(self.deepseek_api_key)
        }
        
        # Enhanced cache and state management
        self._cache_adapter = None
        self._state_manager = None
        self._initialized = False
        
        # Performance tracking
        self._performance_metrics = {
            'requests_total': 0,
            'requests_cached': 0,
            'requests_redis_cached': 0,
            'requests_legacy_cached': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'redis_hit_rate': 0.0,
            'error_rate': 0.0
        }
        
        self.logger.info(f"Enhanced HTTP AI Client initialized (Redis: {enable_redis_cache}, "
                        f"Coordination: {enable_distributed_coordination})")
    
    async def _ensure_initialized(self):
        """Ensure the client is properly initialized with enhanced features."""
        if self._initialized:
            return
        
        try:
            # Initialize enhanced cache adapter
            if self.enable_redis_cache:
                self._cache_adapter = await get_enhanced_cache(
                    enable_redis=True,
                    migration_mode=self.cache_migration_mode
                )
                self.logger.info("Redis-backed cache adapter initialized")
            
            # Initialize enhanced state manager if coordination enabled
            if self.enable_distributed_coordination:
                self._state_manager = await get_enhanced_state_manager(
                    enable_redis=self.enable_redis_cache,
                    migration_mode=self.cache_migration_mode
                )
                self.logger.info("Distributed state coordination initialized")
            
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced features: {e}")
            # Continue without enhanced features
            self._initialized = True
    
    async def generate_enhanced_response(self, 
                                       prompt: str, 
                                       model: Optional[str] = None, 
                                       prefill: str = None,
                                       cache_ttl: Optional[int] = None,
                                       distributed: bool = False) -> Dict[str, Any]:
        """Generate enhanced response with Redis-backed caching and coordination.
        
        Args:
            prompt: The prompt to send to the AI
            model: Optional model preference ('claude', 'gpt', 'gemini', 'deepseek')
            prefill: Optional assistant message to prefill
            cache_ttl: Custom cache TTL in seconds
            distributed: Whether to coordinate across distributed instances
            
        Returns:
            Dictionary containing the AI response with enhanced metadata
        """
        await self._ensure_initialized()
        
        request_id = f"req_{self.request_count}"
        self.request_count += 1
        self._performance_metrics['requests_total'] += 1
        
        start_time = time.time()
        self.logger.info(f"[{request_id}] Enhanced AI request - Model: {model or 'auto'}, "
                        f"Prompt: {len(prompt)} chars, Distributed: {distributed}")
        
        # Get AI API logger
        ai_logger = get_ai_api_logger()
        
        try:
            # Check enhanced cache first
            cached_response = await self._get_cached_response(prompt, model, request_id)
            if cached_response:
                return cached_response
            
            # Record request attempt in distributed state if enabled
            if distributed and self._state_manager:
                await self._record_request_attempt(request_id, prompt, model)
            
            # Log request start
            provider_name = self._determine_provider_name(model)
            log_ai_request_start(
                request_id, 
                prompt, 
                provider_name, 
                model or 'auto',
                distributed=distributed,
                cache_enabled=self.enable_redis_cache
            )
            
            # Generate new response
            response = await self._generate_new_response(prompt, model, prefill, request_id)
            
            # Cache the response with enhanced features
            await self._cache_response(prompt, response, model, cache_ttl, distributed, request_id)
            
            # Update distributed state if enabled
            if distributed and self._state_manager:
                await self._update_request_completion(request_id, response)
            
            # Update performance metrics
            duration = time.time() - start_time
            self._update_performance_metrics(duration, cached=False)
            
            # Add enhanced metadata
            response.update({
                'request_id': request_id,
                'response_time': duration,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cached': False,
                'cache_backend': 'redis' if self.enable_redis_cache else 'legacy',
                'distributed_coordination': distributed,
                'performance_metrics': self._get_current_performance_metrics()
            })
            
            # Log completion
            log_ai_request_complete(request_id, response, duration, cached=False)
            
            return response
            
        except Exception as e:
            self.error_count += 1
            self._performance_metrics['error_rate'] = self.error_count / self._performance_metrics['requests_total']
            
            self.logger.error(f"[{request_id}] Error generating enhanced response: {e}")
            
            # Log error
            provider_name = self._determine_provider_name(model)
            log_ai_request_error(request_id, e, provider_name, model or 'auto')
            
            raise
    
    async def _get_cached_response(self, prompt: str, model: Optional[str], request_id: str) -> Optional[Dict[str, Any]]:
        """Get cached response using enhanced cache adapter."""
        try:
            provider_name = self._determine_provider_name(model)
            start_time = time.time()
            
            # Try enhanced cache first
            if self._cache_adapter:
                cached_content = await self._cache_adapter.get(prompt, provider_name, model or 'auto')
                if cached_content:
                    duration = time.time() - start_time
                    self.cache_hit_count += 1
                    self.redis_cache_hits += 1
                    self._performance_metrics['requests_cached'] += 1
                    self._performance_metrics['requests_redis_cached'] += 1
                    
                    self.logger.info(f"[{request_id}] Enhanced cache HIT (Redis) - {duration:.3f}s")
                    
                    # Log cache hit
                    ai_logger.log_cache_event(
                        AIEventType.CACHE_HIT,
                        request_id,
                        prompt,
                        provider_name,
                        model or 'auto',
                        cache_backend='redis',
                        distributed=distributed
                    )
                    
                    response_data = {
                        'content': cached_content,
                        'provider': provider_name,
                        'model': model or 'auto',
                        'request_id': request_id,
                        'response_time': duration,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'cached': True,
                        'cache_backend': 'redis',
                        'cache_type': 'enhanced',
                        'confidence': 0.95
                    }
                    
                    # Log completion
                    log_ai_request_complete(request_id, response_data, duration, cached=True)
                    return response_data
            
            # Fallback to legacy cache if Redis cache unavailable
            if LEGACY_CACHE_AVAILABLE:
                cached_content = await get_cached_response(prompt, provider_name, model or 'auto')
                if cached_content:
                    duration = time.time() - start_time
                    self.cache_hit_count += 1
                    self.legacy_cache_hits += 1
                    self._performance_metrics['requests_cached'] += 1
                    self._performance_metrics['requests_legacy_cached'] += 1
                    
                    self.logger.info(f"[{request_id}] Legacy cache HIT - {duration:.3f}s")
                    
                    # Log cache hit
                    ai_logger.log_cache_event(
                        AIEventType.CACHE_HIT,
                        request_id,
                        prompt,
                        provider_name,
                        model or 'auto',
                        cache_backend='legacy'
                    )
                    
                    response_data = {
                        'content': cached_content,
                        'provider': provider_name,
                        'model': model or 'auto',
                        'request_id': request_id,
                        'response_time': duration,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'cached': True,
                        'cache_backend': 'legacy',
                        'cache_type': 'legacy',
                        'confidence': 0.9
                    }
                    
                    # Log completion
                    log_ai_request_complete(request_id, response_data, duration, cached=True)
                    return response_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Error checking cache: {e}")
            ai_logger.log_cache_event(
                AIEventType.CACHE_MISS,
                request_id,
                prompt,
                provider_name,
                model or 'auto',
                error=str(e)
            )
            return None
    
    async def _cache_response(self, prompt: str, response: Dict[str, Any], model: Optional[str], 
                            cache_ttl: Optional[int], distributed: bool, request_id: str):
        """Cache response using enhanced cache adapter."""
        try:
            content = response.get('content', '')
            if not content:
                return
            
            provider = response.get('provider', 'unknown')
            cost_estimate = response.get('cost_estimate', 0.0)
            
            # Cache with enhanced adapter
            if self._cache_adapter:
                await self._cache_adapter.put(
                    prompt, 
                    content, 
                    provider, 
                    model or 'auto',
                    ttl_seconds=cache_ttl,
                    cost_estimate=cost_estimate
                )
                self.logger.debug(f"[{request_id}] Response cached with enhanced adapter")
                
                # Log cache store event
                ai_logger = get_ai_api_logger()
                ai_logger.log_cache_event(
                    AIEventType.CACHE_STORE,
                    request_id,
                    prompt,
                    provider,
                    model or 'auto',
                    cache_backend='redis',
                    cost_estimate=cost_estimate,
                    distributed=distributed
                )
            
            # Fallback to legacy caching
            elif LEGACY_CACHE_AVAILABLE:
                await cache_ai_response(prompt, content, provider, model or 'auto', cost_estimate)
                self.logger.debug(f"[{request_id}] Response cached with legacy adapter")
                
                # Log cache store event
                ai_logger = get_ai_api_logger()
                ai_logger.log_cache_event(
                    AIEventType.CACHE_STORE,
                    request_id,
                    prompt,
                    provider,
                    model or 'auto',
                    cache_backend='legacy',
                    cost_estimate=cost_estimate
                )
            
            # Update distributed cache statistics if enabled
            if distributed and self._state_manager:
                await self._update_cache_statistics(provider, model, cost_estimate)
                
        except Exception as e:
            self.logger.error(f"[{request_id}] Error caching response: {e}")
    
    async def _record_request_attempt(self, request_id: str, prompt: str, model: Optional[str]):
        """Record request attempt in distributed state."""
        try:
            if self._state_manager:
                await self._state_manager.update(
                    f"ai_requests.{request_id}",
                    {
                        'prompt_hash': self._hash_prompt(prompt),
                        'model': model,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'status': 'in_progress'
                    },
                    distributed=True
                )
        except Exception as e:
            self.logger.error(f"Error recording request attempt: {e}")
    
    async def _update_request_completion(self, request_id: str, response: Dict[str, Any]):
        """Update request completion in distributed state."""
        try:
            if self._state_manager:
                await self._state_manager.update(
                    f"ai_requests.{request_id}.status",
                    'completed',
                    distributed=True
                )
                
                await self._state_manager.update(
                    f"ai_requests.{request_id}.response_metadata",
                    {
                        'provider': response.get('provider'),
                        'model': response.get('model'),
                        'response_time': response.get('response_time'),
                        'cost_estimate': response.get('cost_estimate', 0.0)
                    },
                    distributed=True
                )
        except Exception as e:
            self.logger.error(f"Error updating request completion: {e}")
    
    async def _update_cache_statistics(self, provider: str, model: Optional[str], cost_estimate: float):
        """Update distributed cache statistics."""
        try:
            if self._state_manager:
                # Update provider statistics
                await self._state_manager.update(
                    f"ai_statistics.providers.{provider}.requests",
                    1,  # This would need to be an increment operation
                    distributed=True
                )
                
                await self._state_manager.update(
                    f"ai_statistics.providers.{provider}.total_cost",
                    cost_estimate,  # This would need to be an add operation
                    distributed=True
                )
                
                # Update model statistics
                if model:
                    await self._state_manager.update(
                        f"ai_statistics.models.{model}.requests",
                        1,
                        distributed=True
                    )
        except Exception as e:
            self.logger.error(f"Error updating cache statistics: {e}")
    
    async def _generate_new_response(self, prompt: str, model: Optional[str], prefill: str, request_id: str) -> Dict[str, Any]:
        """Generate new AI response (delegate to existing HTTP client logic)."""
        # This would delegate to the existing HTTP AI client implementation
        # For now, return a mock response
        provider_name = self._determine_provider_name(model)
        
        # Simulate API call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            'content': f"Mock response for prompt: {prompt[:50]}...",
            'provider': provider_name,
            'model': model or 'auto',
            'confidence': 0.8,
            'cost_estimate': 0.001,
            'token_usage': {
                'input_tokens': len(prompt.split()),
                'output_tokens': 20,
                'total_tokens': len(prompt.split()) + 20
            }
        }
    
    def _determine_provider_name(self, model: Optional[str]) -> str:
        """Determine provider name from model preference."""
        if model:
            if 'claude' in model.lower():
                return 'anthropic'
            elif 'gpt' in model.lower() or 'openai' in model.lower():
                return 'openai'
            elif 'gemini' in model.lower() or 'google' in model.lower():
                return 'gemini'
            elif 'deepseek' in model.lower():
                return 'deepseek'
        
        # Default to first available provider
        for provider, available in self.providers_available.items():
            if available:
                return provider
        
        return 'unknown'
    
    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash for prompt (for privacy in distributed state)."""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _update_performance_metrics(self, response_time: float, cached: bool):
        """Update internal performance metrics."""
        self.total_response_time += response_time
        
        # Update averages
        total_requests = self._performance_metrics['requests_total']
        self._performance_metrics['average_response_time'] = self.total_response_time / max(total_requests, 1)
        self._performance_metrics['cache_hit_rate'] = self._performance_metrics['requests_cached'] / max(total_requests, 1)
        
        if self.enable_redis_cache:
            self._performance_metrics['redis_hit_rate'] = self._performance_metrics['requests_redis_cached'] / max(total_requests, 1)
    
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics snapshot."""
        return self._performance_metrics.copy()
    
    async def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status and statistics."""
        status = {
            'redis_cache_enabled': self.enable_redis_cache,
            'cache_migration_mode': self.cache_migration_mode,
            'distributed_coordination': self.enable_distributed_coordination,
            'performance_metrics': self._performance_metrics.copy(),
            'cache_hits': {
                'total': self.cache_hit_count,
                'redis': self.redis_cache_hits,
                'legacy': self.legacy_cache_hits
            },
            'providers_available': self.providers_available.copy()
        }
        
        # Get enhanced cache status
        if self._cache_adapter:
            try:
                cache_migration_status = await self._cache_adapter.get_migration_status()
                status['cache_migration'] = cache_migration_status
                
                cache_stats = self._cache_adapter.get_stats()
                status['cache_statistics'] = cache_stats
            except Exception as e:
                status['cache_error'] = str(e)
        
        # Get state manager status
        if self._state_manager:
            try:
                state_migration_status = await self._state_manager.get_migration_status()
                status['state_migration'] = state_migration_status
                
                state_metrics = self._state_manager.get_metrics()
                status['state_metrics'] = state_metrics
            except Exception as e:
                status['state_error'] = str(e)
        
        return status
    
    async def warm_cache(self, historical_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Warm cache with historical AI requests."""
        if not self._cache_adapter:
            return {'error': 'Cache adapter not available'}
        
        try:
            warmed_count = await self._cache_adapter.warm_cache(historical_requests)
            
            self.logger.info(f"Cache warmed with {warmed_count} historical requests")
            
            return {
                'success': True,
                'warmed_entries': warmed_count,
                'total_provided': len(historical_requests)
            }
            
        except Exception as e:
            self.logger.error(f"Error warming cache: {e}")
            return {'error': str(e)}
    
    async def clear_cache(self, confirm: bool = False) -> Dict[str, Any]:
        """Clear all cached responses."""
        if not confirm:
            return {'error': 'Confirmation required to clear cache'}
        
        try:
            if self._cache_adapter:
                await self._cache_adapter.clear()
                return {'success': True, 'message': 'Enhanced cache cleared'}
            else:
                return {'error': 'Cache adapter not available'}
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return {'error': str(e)}
    
    async def get_distributed_statistics(self) -> Dict[str, Any]:
        """Get distributed AI usage statistics."""
        if not self._state_manager:
            return {'error': 'Distributed coordination not enabled'}
        
        try:
            # Get distributed AI statistics
            ai_stats = await self._state_manager.get('ai_statistics', {})
            
            return {
                'success': True,
                'statistics': ai_stats,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting distributed statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown enhanced HTTP AI client."""
        try:
            self.logger.info("Shutting down Enhanced HTTP AI Client")
            
            # Shutdown cache adapter
            if self._cache_adapter:
                await self._cache_adapter.shutdown()
            
            # Shutdown state manager
            if self._state_manager:
                await self._state_manager.shutdown()
            
            self.logger.info("Enhanced HTTP AI Client shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error shutting down Enhanced HTTP AI Client: {e}")


# Global enhanced client instance
_global_enhanced_client: Optional[EnhancedHTTPAIClient] = None


async def get_enhanced_ai_client(enable_redis: bool = True, 
                               migration_mode: str = "gradual",
                               distributed: bool = True) -> EnhancedHTTPAIClient:
    """Get global enhanced HTTP AI client instance."""
    global _global_enhanced_client
    
    if _global_enhanced_client is None:
        _global_enhanced_client = EnhancedHTTPAIClient(
            enable_redis_cache=enable_redis,
            cache_migration_mode=migration_mode,
            enable_distributed_coordination=distributed
        )
        await _global_enhanced_client._ensure_initialized()
    
    return _global_enhanced_client


# Convenience functions that match existing API
async def enhanced_generate_response(prompt: str, 
                                   model: Optional[str] = None,
                                   prefill: str = None,
                                   distributed: bool = False) -> Dict[str, Any]:
    """Enhanced generate AI response with Redis caching."""
    client = await get_enhanced_ai_client()
    return await client.generate_enhanced_response(prompt, model, prefill, distributed=distributed)


async def get_ai_cache_status() -> Dict[str, Any]:
    """Get AI cache status and migration information."""
    client = await get_enhanced_ai_client()
    return await client.get_cache_status()