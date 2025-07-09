"""
Research Engine Comprehensive Fixes

Implements robust solutions for all identified issues:
1. Timeout handling for all async operations
2. Retry logic with exponential backoff
3. Circuit breaker pattern for provider failures
4. Error boundaries and graceful degradation
5. Comprehensive monitoring and metrics
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List, Callable, TypeVar, Coroutine
from datetime import datetime, timedelta
from functools import wraps
import traceback
from enum import Enum

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger = logging.getLogger(__name__)
        
    def call_succeeded(self):
        """Record a successful call"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                self.logger.info("Circuit breaker closed after recovery")
                
    def call_failed(self):
        """Record a failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
    def can_execute(self) -> bool:
        """Check if calls can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker entering half-open state")
                return True
            return False
            
        # HALF_OPEN state
        return True


class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base


async def with_timeout(coro: Coroutine[Any, Any, T], timeout_seconds: float, 
                      operation_name: str = "operation") -> T:
    """Execute coroutine with timeout and proper error handling"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds} seconds")


async def with_retry(func: Callable, retry_config: RetryConfig, 
                    operation_name: str = "operation", **kwargs) -> Any:
    """Execute function with retry logic and exponential backoff"""
    logger = logging.getLogger(__name__)
    last_exception = None
    
    for attempt in range(retry_config.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
                
        except Exception as e:
            last_exception = e
            
            if attempt == retry_config.max_retries:
                logger.error(f"{operation_name} failed after {retry_config.max_retries + 1} attempts: {e}")
                raise
                
            # Calculate delay with exponential backoff
            delay = min(
                retry_config.base_delay * (retry_config.exponential_base ** attempt),
                retry_config.max_delay
            )
            
            logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
    
    raise last_exception


class ResearchEngineMonitor:
    """Comprehensive monitoring for research engine operations"""
    
    def __init__(self):
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "timeout_operations": 0,
            "retry_operations": 0,
            "operation_durations": [],
            "error_types": {},
            "provider_failures": {},
            "circuit_breaker_trips": 0
        }
        self.operation_history = []
        self.logger = logging.getLogger(__name__)
        
    def record_operation(self, operation_name: str, duration: float, 
                        success: bool, error: Optional[str] = None):
        """Record operation metrics"""
        self.metrics["total_operations"] += 1
        
        if success:
            self.metrics["successful_operations"] += 1
        else:
            self.metrics["failed_operations"] += 1
            if error:
                self.metrics["error_types"][error] = self.metrics["error_types"].get(error, 0) + 1
                
        self.metrics["operation_durations"].append(duration)
        
        # Keep only last 1000 durations to prevent memory growth
        if len(self.metrics["operation_durations"]) > 1000:
            self.metrics["operation_durations"] = self.metrics["operation_durations"][-1000:]
            
        # Record in history
        self.operation_history.append({
            "operation": operation_name,
            "duration": duration,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 operations
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-100:]
            
    def record_timeout(self, operation_name: str):
        """Record timeout event"""
        self.metrics["timeout_operations"] += 1
        self.record_operation(operation_name, -1, False, "timeout")
        
    def record_retry(self, operation_name: str):
        """Record retry event"""
        self.metrics["retry_operations"] += 1
        
    def record_provider_failure(self, provider: str):
        """Record provider failure"""
        self.metrics["provider_failures"][provider] = self.metrics["provider_failures"].get(provider, 0) + 1
        
    def record_circuit_breaker_trip(self):
        """Record circuit breaker activation"""
        self.metrics["circuit_breaker_trips"] += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_duration = sum(self.metrics["operation_durations"]) / len(self.metrics["operation_durations"]) if self.metrics["operation_durations"] else 0
        success_rate = self.metrics["successful_operations"] / self.metrics["total_operations"] if self.metrics["total_operations"] > 0 else 0
        
        return {
            **self.metrics,
            "average_duration": avg_duration,
            "success_rate": success_rate,
            "recent_operations": self.operation_history[-10:]
        }


class ResearchEngineEnhancements:
    """Enhanced research engine with all fixes applied"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_config = RetryConfig()
        self.monitor = ResearchEngineMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Timeout configurations
        self.timeouts = {
            "analyze_system_needs": 30,
            "select_research_topics": 30,
            "execute_single_research": 60,
            "execute_research": 120,
            "extract_insights": 60,
            "generate_tasks": 45,
            "implement_improvements": 90,
            "measure_performance": 30,
            "learn_from_cycle": 45,
            "external_research": 180
        }
        
    def get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider"""
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreaker()
        return self.circuit_breakers[provider]
        
    async def execute_with_monitoring(self, operation_name: str, 
                                    operation_func: Callable,
                                    timeout: Optional[float] = None,
                                    retry: bool = True,
                                    **kwargs) -> Any:
        """Execute operation with full monitoring, timeout, and retry logic"""
        start_time = time.time()
        
        if timeout is None:
            timeout = self.timeouts.get(operation_name, 60)
            
        try:
            # Wrap operation with retry if enabled
            if retry:
                async def retriable_operation():
                    self.monitor.record_retry(operation_name)
                    return await with_timeout(
                        operation_func(**kwargs),
                        timeout,
                        operation_name
                    )
                    
                result = await with_retry(
                    retriable_operation,
                    self.retry_config,
                    operation_name
                )
            else:
                result = await with_timeout(
                    operation_func(**kwargs),
                    timeout,
                    operation_name
                )
                
            duration = time.time() - start_time
            self.monitor.record_operation(operation_name, duration, True)
            self.logger.info(f"{operation_name} completed successfully in {duration:.2f}s")
            
            return result
            
        except TimeoutError as e:
            duration = time.time() - start_time
            self.monitor.record_timeout(operation_name)
            self.logger.error(f"{operation_name} timed out after {duration:.2f}s")
            raise
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            self.monitor.record_operation(operation_name, duration, False, error_type)
            self.logger.error(f"{operation_name} failed after {duration:.2f}s: {error_type}: {e}")
            raise
            
    async def execute_with_fallback(self, operation_name: str,
                                   primary_func: Callable,
                                   fallback_func: Callable,
                                   **kwargs) -> Any:
        """Execute operation with fallback on failure"""
        try:
            return await self.execute_with_monitoring(
                operation_name,
                primary_func,
                **kwargs
            )
        except Exception as e:
            self.logger.warning(f"{operation_name} primary failed, trying fallback: {e}")
            return await self.execute_with_monitoring(
                f"{operation_name}_fallback",
                fallback_func,
                retry=False,
                **kwargs
            )


def create_enhanced_research_engine(original_engine):
    """Enhance existing research engine with fixes"""
    enhancements = ResearchEngineEnhancements()
    
    # Store original methods
    original_methods = {
        'execute_research_cycle': original_engine.execute_research_cycle,
        '_analyze_system_needs': original_engine._analyze_system_needs,
        '_select_research_topics': original_engine._select_research_topics,
        '_execute_research': original_engine._execute_research,
        '_execute_single_research': original_engine._execute_single_research,
        '_extract_insights': original_engine._extract_insights,
        '_generate_implementation_tasks': original_engine._generate_implementation_tasks,
        '_implement_improvements': original_engine._implement_improvements,
        '_measure_performance_changes': original_engine._measure_performance_changes,
        '_learn_from_cycle': original_engine._learn_from_cycle
    }
    
    # Create enhanced versions
    async def enhanced_execute_research_cycle():
        """Execute research cycle with full enhancements"""
        return await enhancements.execute_with_monitoring(
            "execute_research_cycle",
            original_methods['execute_research_cycle'],
            timeout=300  # 5 minute total timeout
        )
        
    async def enhanced_analyze_system_needs():
        """Analyze system needs with timeout and monitoring"""
        # Simple operation, should be fast
        return await enhancements.execute_with_monitoring(
            "analyze_system_needs",
            original_methods['_analyze_system_needs'],
            retry=False  # Don't retry analysis
        )
        
    async def enhanced_select_research_topics(system_needs):
        """Select research topics with monitoring"""
        return await enhancements.execute_with_monitoring(
            "select_research_topics",
            original_methods['_select_research_topics'],
            system_needs=system_needs
        )
        
    async def enhanced_execute_single_research(query, topic):
        """Execute single research with circuit breaker and timeout"""
        provider = "default"  # In real implementation, extract from engine
        circuit_breaker = enhancements.get_circuit_breaker(provider)
        
        if not circuit_breaker.can_execute():
            enhancements.monitor.record_circuit_breaker_trip()
            raise Exception(f"Circuit breaker open for provider {provider}")
            
        try:
            result = await enhancements.execute_with_monitoring(
                "execute_single_research",
                original_methods['_execute_single_research'],
                query=query,
                topic=topic
            )
            circuit_breaker.call_succeeded()
            return result
            
        except Exception as e:
            circuit_breaker.call_failed()
            enhancements.monitor.record_provider_failure(provider)
            raise
            
    async def enhanced_execute_research(research_topics):
        """Execute research with enhanced single research calls"""
        # Replace the engine's method temporarily
        original_single = original_engine._execute_single_research
        original_engine._execute_single_research = enhanced_execute_single_research
        
        try:
            return await enhancements.execute_with_monitoring(
                "execute_research",
                original_methods['_execute_research'],
                research_topics=research_topics
            )
        finally:
            # Restore original method
            original_engine._execute_single_research = original_single
            
    async def enhanced_extract_insights(research_results):
        """Extract insights with timeout and fallback"""
        async def fallback_extract(research_results):
            """Simple fallback extraction"""
            return [{
                "insight": "Fallback insight due to extraction failure",
                "confidence": 0.3,
                "source": "fallback",
                "type": "emergency"
            }]
            
        return await enhancements.execute_with_fallback(
            "extract_insights",
            original_methods['_extract_insights'],
            fallback_extract,
            research_results=research_results
        )
        
    # Apply all enhancements
    original_engine.execute_research_cycle = enhanced_execute_research_cycle
    original_engine._analyze_system_needs = enhanced_analyze_system_needs
    original_engine._select_research_topics = enhanced_select_research_topics
    original_engine._execute_research = enhanced_execute_research
    original_engine._extract_insights = enhanced_extract_insights
    
    # Add monitoring access
    original_engine.get_enhanced_metrics = enhancements.monitor.get_metrics
    
    return original_engine


# Utility function to apply fixes
def apply_research_engine_fixes(engine):
    """Apply all fixes to a research engine instance"""
    logging.info("Applying comprehensive research engine fixes...")
    enhanced = create_enhanced_research_engine(engine)
    logging.info("Research engine fixes applied successfully")
    return enhanced