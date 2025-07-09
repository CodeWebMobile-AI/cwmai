"""
AI API Communication Logger

Dedicated logging module for monitoring all AI API communications in real-time.
Provides structured JSON logging with detailed request/response information.
"""

import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue
import hashlib


class AIRequestType(Enum):
    """Types of AI API requests."""
    GENERATE = "generate"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    EVALUATION = "evaluation"


class AIEventType(Enum):
    """Types of AI API events."""
    REQUEST_START = "request_start"
    REQUEST_COMPLETE = "request_complete"
    REQUEST_ERROR = "request_error"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_STORE = "cache_store"
    RETRY_ATTEMPT = "retry_attempt"
    RATE_LIMIT = "rate_limit"
    PROVIDER_SWITCH = "provider_switch"


@dataclass
class AIRequestMetadata:
    """Metadata for AI API requests."""
    request_id: str
    timestamp: str
    provider: str
    model: str
    request_type: AIRequestType
    prompt_length: int
    prompt_hash: str  # For privacy
    prefill: Optional[str] = None
    distributed: bool = False
    cache_enabled: bool = True
    round_robin: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data['request_type'] = self.request_type.value
        return data


@dataclass
class AIResponseMetadata:
    """Metadata for AI API responses."""
    response_length: int
    response_time: float
    cached: bool
    cache_backend: Optional[str] = None
    confidence: float = 0.0
    cost_estimate: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class AIAPILogger:
    """Logger for AI API communications with real-time file writing."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global logger instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the AI API logger."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.log_file = "ai_api_communication.log"
        self.logger = logging.getLogger("AIAPILogger")
        
        # Configuration from environment
        self.log_level = os.getenv('AI_API_LOG_LEVEL', 'INFO')
        self.log_sensitive_data = os.getenv('AI_API_LOG_SENSITIVE', 'false').lower() == 'true'
        self.max_content_length = int(os.getenv('AI_API_MAX_LOG_LENGTH', '500'))
        self.enable_file_logging = os.getenv('AI_API_FILE_LOGGING', 'true').lower() == 'true'
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'total_errors': 0,
            'total_cache_hits': 0,
            'total_response_time': 0.0,
            'provider_counts': {},
            'model_counts': {},
            'error_counts': {},
            'hourly_requests': {}
        }
        
        # Async write queue for performance
        self.write_queue = Queue()
        self.writer_thread = None
        
        # Start the writer thread if file logging is enabled
        if self.enable_file_logging:
            self._start_writer_thread()
        
        self.logger.info(f"AI API Logger initialized (file_logging={self.enable_file_logging}, "
                        f"sensitive_data={self.log_sensitive_data})")
    
    def _start_writer_thread(self):
        """Start the background thread for writing logs."""
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        """Background thread that writes logs to file."""
        while True:
            try:
                # Get log entry from queue (blocks until available)
                entry = self.write_queue.get()
                
                if entry is None:  # Shutdown signal
                    break
                
                # Write to file
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()  # Ensure real-time writing
                    
            except Exception as e:
                self.logger.error(f"Error writing AI API log: {e}")
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content for logging if needed."""
        if not self.log_sensitive_data:
            return "[REDACTED]"
        
        if len(content) > self.max_content_length:
            return content[:self.max_content_length] + "...[truncated]"
        
        return content
    
    def _hash_content(self, content: str) -> str:
        """Generate hash of content for tracking without exposing data."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _update_stats(self, event_type: AIEventType, metadata: Dict[str, Any]):
        """Update internal statistics."""
        # Update request count
        if event_type == AIEventType.REQUEST_START:
            self.stats['total_requests'] += 1
            
            # Track hourly requests
            hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:00")
            self.stats['hourly_requests'][hour_key] = self.stats['hourly_requests'].get(hour_key, 0) + 1
            
            # Track provider usage
            provider = metadata.get('provider', 'unknown')
            self.stats['provider_counts'][provider] = self.stats['provider_counts'].get(provider, 0) + 1
            
            # Track model usage
            model = metadata.get('model', 'unknown')
            self.stats['model_counts'][model] = self.stats['model_counts'].get(model, 0) + 1
        
        # Update error count
        elif event_type == AIEventType.REQUEST_ERROR:
            self.stats['total_errors'] += 1
            error_type = metadata.get('error_type', 'unknown')
            self.stats['error_counts'][error_type] = self.stats['error_counts'].get(error_type, 0) + 1
        
        # Update cache hits
        elif event_type == AIEventType.CACHE_HIT:
            self.stats['total_cache_hits'] += 1
        
        # Update response time
        if 'response_time' in metadata:
            self.stats['total_response_time'] += metadata['response_time']
    
    def log_request_start(self, request_id: str, prompt: str, provider: str, model: str,
                         request_type: AIRequestType = AIRequestType.GENERATE,
                         prefill: Optional[str] = None, **kwargs) -> None:
        """Log the start of an AI API request."""
        metadata = AIRequestMetadata(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            request_type=request_type,
            prompt_length=len(prompt),
            prompt_hash=self._hash_content(prompt),
            prefill=prefill,
            distributed=kwargs.get('distributed', False),
            cache_enabled=kwargs.get('cache_enabled', True),
            round_robin=kwargs.get('round_robin', False)
        )
        
        log_entry = {
            'event_type': AIEventType.REQUEST_START.value,
            'request_metadata': metadata.to_dict(),
            'prompt_preview': self._truncate_content(prompt),
            'additional_params': kwargs
        }
        
        self._update_stats(AIEventType.REQUEST_START, metadata.to_dict())
        self._write_log(log_entry)
        
        # Also log to standard logger
        self.logger.info(f"[{request_id}] AI Request START - Provider: {provider}, Model: {model}, "
                        f"Type: {request_type.value}, Length: {len(prompt)}")
    
    def log_request_complete(self, request_id: str, response: Dict[str, Any],
                           response_time: float, cached: bool = False) -> None:
        """Log the completion of an AI API request."""
        content = response.get('content', '')
        
        metadata = AIResponseMetadata(
            response_length=len(content),
            response_time=response_time,
            cached=cached,
            cache_backend=response.get('cache_backend'),
            confidence=response.get('confidence', 0.0),
            cost_estimate=response.get('cost_estimate', 0.0),
            token_usage=response.get('usage') or response.get('token_usage')
        )
        
        log_entry = {
            'event_type': AIEventType.REQUEST_COMPLETE.value,
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'response_metadata': metadata.to_dict(),
            'response_preview': self._truncate_content(content),
            'provider': response.get('provider', 'unknown'),
            'model': response.get('model', 'unknown')
        }
        
        self._update_stats(AIEventType.REQUEST_COMPLETE, metadata.to_dict())
        self._write_log(log_entry)
        
        # Also log to standard logger
        cache_str = f" (CACHED from {response.get('cache_backend', 'unknown')})" if cached else ""
        self.logger.info(f"[{request_id}] AI Request COMPLETE{cache_str} - Time: {response_time:.2f}s, "
                        f"Length: {len(content)}, Cost: ${metadata.cost_estimate:.4f}")
    
    def log_request_error(self, request_id: str, error: Exception, provider: str,
                         model: str, retry_count: int = 0) -> None:
        """Log an error in AI API request."""
        error_type = type(error).__name__
        error_message = str(error)
        
        log_entry = {
            'event_type': AIEventType.REQUEST_ERROR.value,
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'provider': provider,
            'model': model,
            'error_type': error_type,
            'error_message': error_message,
            'retry_count': retry_count,
            'traceback': self._truncate_content(traceback.format_exc()) if self.log_sensitive_data else None
        }
        
        self._update_stats(AIEventType.REQUEST_ERROR, {
            'error_type': error_type,
            'provider': provider
        })
        self._write_log(log_entry)
        
        # Also log to standard logger
        self.logger.error(f"[{request_id}] AI Request ERROR - Provider: {provider}, "
                         f"Error: {error_type}: {error_message}")
    
    def log_cache_event(self, event_type: AIEventType, request_id: str,
                       cache_key: str, provider: str, model: str, **kwargs) -> None:
        """Log cache-related events."""
        log_entry = {
            'event_type': event_type.value,
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cache_key': self._hash_content(cache_key),
            'provider': provider,
            'model': model,
            'cache_backend': kwargs.get('cache_backend', 'unknown'),
            'additional_data': kwargs
        }
        
        self._update_stats(event_type, {'provider': provider})
        self._write_log(log_entry)
        
        # Also log to standard logger
        self.logger.debug(f"[{request_id}] Cache {event_type.value} - Provider: {provider}, Model: {model}")
    
    def log_retry_attempt(self, request_id: str, provider: str, attempt: int,
                         reason: str, delay: float) -> None:
        """Log retry attempts."""
        log_entry = {
            'event_type': AIEventType.RETRY_ATTEMPT.value,
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'provider': provider,
            'attempt': attempt,
            'reason': reason,
            'delay_seconds': delay
        }
        
        self._write_log(log_entry)
        
        # Also log to standard logger
        self.logger.warning(f"[{request_id}] Retry attempt {attempt} - Provider: {provider}, "
                           f"Reason: {reason}, Delay: {delay}s")
    
    def log_provider_switch(self, request_id: str, from_provider: str,
                           to_provider: str, reason: str) -> None:
        """Log provider switching events."""
        log_entry = {
            'event_type': AIEventType.PROVIDER_SWITCH.value,
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'from_provider': from_provider,
            'to_provider': to_provider,
            'reason': reason
        }
        
        self._write_log(log_entry)
        
        # Also log to standard logger
        self.logger.info(f"[{request_id}] Provider switch: {from_provider} -> {to_provider} ({reason})")
    
    def _write_log(self, entry: Dict[str, Any]) -> None:
        """Write log entry to file via queue."""
        if self.enable_file_logging:
            self.write_queue.put(entry)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        total_requests = self.stats['total_requests']
        avg_response_time = (self.stats['total_response_time'] / total_requests) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'total_errors': self.stats['total_errors'],
            'error_rate': (self.stats['total_errors'] / total_requests) if total_requests > 0 else 0,
            'total_cache_hits': self.stats['total_cache_hits'],
            'cache_hit_rate': (self.stats['total_cache_hits'] / total_requests) if total_requests > 0 else 0,
            'average_response_time': avg_response_time,
            'provider_usage': self.stats['provider_counts'],
            'model_usage': self.stats['model_counts'],
            'error_breakdown': self.stats['error_counts'],
            'hourly_requests': self.stats['hourly_requests']
        }
    
    def shutdown(self) -> None:
        """Shutdown the logger gracefully."""
        if self.enable_file_logging and self.writer_thread:
            # Signal shutdown
            self.write_queue.put(None)
            # Wait for writer thread to finish
            self.writer_thread.join(timeout=5.0)
            
        self.logger.info("AI API Logger shutdown complete")


# Global logger instance
_ai_api_logger: Optional[AIAPILogger] = None


def get_ai_api_logger() -> AIAPILogger:
    """Get the global AI API logger instance."""
    global _ai_api_logger
    if _ai_api_logger is None:
        _ai_api_logger = AIAPILogger()
    return _ai_api_logger


# Convenience functions
def log_ai_request_start(request_id: str, prompt: str, provider: str, model: str, **kwargs) -> None:
    """Log the start of an AI API request."""
    logger = get_ai_api_logger()
    logger.log_request_start(request_id, prompt, provider, model, **kwargs)


def log_ai_request_complete(request_id: str, response: Dict[str, Any],
                           response_time: float, cached: bool = False) -> None:
    """Log the completion of an AI API request."""
    logger = get_ai_api_logger()
    logger.log_request_complete(request_id, response, response_time, cached)


def log_ai_request_error(request_id: str, error: Exception, provider: str,
                        model: str, retry_count: int = 0) -> None:
    """Log an error in AI API request."""
    logger = get_ai_api_logger()
    logger.log_request_error(request_id, error, provider, model, retry_count)


def get_ai_api_statistics() -> Dict[str, Any]:
    """Get AI API communication statistics."""
    logger = get_ai_api_logger()
    return logger.get_statistics()
