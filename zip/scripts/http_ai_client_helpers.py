"""
HTTP AI Client Helper Methods

Additional helper methods for the HTTP AI Client with caching integration.
"""

from typing import Dict, Any, Optional


class HTTPAIClientHelpers:
    """Helper methods for HTTP AI Client caching integration."""
    
    def _determine_provider_name(self, model: Optional[str]) -> str:
        """Determine provider name based on model preference."""
        if model == 'claude':
            return 'anthropic'
        elif model == 'gpt':
            return 'openai'
        elif model == 'gemini':
            return 'gemini'
        elif model == 'deepseek':
            return 'deepseek'
        elif model is None:
            # Auto-select based on availability
            if self.providers_available['anthropic']:
                return 'anthropic'
            elif self.providers_available['openai']:
                return 'openai'
            elif self.providers_available['gemini']:
                return 'gemini'
            elif self.providers_available['deepseek']:
                return 'deepseek'
        
        return 'unknown'
    
    def _estimate_request_cost(self, result: Dict[str, Any], prompt_length: int) -> float:
        """Estimate the cost of an AI request."""
        provider = result.get('provider', 'unknown')
        
        # Rough cost estimates per 1K tokens (in USD)
        cost_per_1k_tokens = {
            'anthropic': 0.025,  # Claude-3.5 Sonnet
            'openai': 0.030,     # GPT-4
            'gemini': 0.002,     # Gemini Pro
            'deepseek': 0.001,   # DeepSeek Chat
            'mock': 0.0,
            'error': 0.0
        }
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = (prompt_length + len(result.get('content', ''))) / 4
        
        # Calculate cost
        rate = cost_per_1k_tokens.get(provider, 0.02)  # Default rate
        estimated_cost = (estimated_tokens / 1000) * rate
        
        return round(estimated_cost, 6)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics if caching is available."""
        if not CACHE_AVAILABLE:
            return {'cache_available': False}
        
        try:
            from ai_response_cache import get_global_cache
            cache = get_global_cache()
            return {
                'cache_available': True,
                'cache_stats': cache.get_stats()
            }
        except Exception as e:
            return {
                'cache_available': False,
                'error': str(e)
            }
    
    async def warm_cache_from_history(self, history_data: list) -> Dict[str, Any]:
        """Warm cache with historical AI interactions."""
        if not CACHE_AVAILABLE:
            return {'cache_available': False}
        
        try:
            from ai_response_cache import get_global_cache
            cache = get_global_cache()
            count = await cache.warm_cache(history_data)
            return {
                'cache_available': True,
                'entries_added': count,
                'status': 'success'
            }
        except Exception as e:
            return {
                'cache_available': False,
                'error': str(e),
                'status': 'failed'
            }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear the AI response cache."""
        if not CACHE_AVAILABLE:
            return {'cache_available': False}
        
        try:
            from ai_response_cache import get_global_cache
            cache = get_global_cache()
            await cache.clear()
            return {
                'cache_available': True,
                'status': 'cleared'
            }
        except Exception as e:
            return {
                'cache_available': False,
                'error': str(e),
                'status': 'failed'
            }