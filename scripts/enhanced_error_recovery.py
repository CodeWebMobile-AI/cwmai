"""
Enhanced Error Recovery System

Provides intelligent error recovery mechanisms with automatic retries,
fallback strategies, and self-healing capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback

from scripts.error_analyzer import ErrorCategory, ErrorSeverity, ErrorInstance
from scripts.work_item_types import WorkItem, TaskPriority


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    ALTERNATE_APPROACH = "alternate_approach"
    TASK_DECOMPOSITION = "task_decomposition"
    PROVIDER_FALLBACK = "provider_fallback"
    RESOURCE_CLEANUP = "resource_cleanup"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP_AND_LOG = "skip_and_log"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    timestamp: datetime
    strategy: RecoveryStrategy
    success: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryContext:
    """Context for error recovery."""
    error: ErrorInstance
    work_item: Optional[WorkItem]
    worker_id: str
    attempt_count: int = 0
    recovery_history: List[RecoveryAttempt] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedErrorRecovery:
    """Intelligent error recovery system with self-healing capabilities."""
    
    def __init__(self, ai_brain=None):
        """Initialize error recovery system.
        
        Args:
            ai_brain: AI brain for intelligent recovery decisions
        """
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Recovery configuration
        self.max_retry_attempts = 3
        self.base_backoff_seconds = 1
        self.max_backoff_seconds = 300
        
        # Strategy mapping
        self.category_strategies = {
            ErrorCategory.NETWORK: [
                RecoveryStrategy.EXPONENTIAL_BACKOFF,
                RecoveryStrategy.PROVIDER_FALLBACK
            ],
            ErrorCategory.API_RATE_LIMIT: [
                RecoveryStrategy.EXPONENTIAL_BACKOFF,
                RecoveryStrategy.TASK_DECOMPOSITION
            ],
            ErrorCategory.AUTHENTICATION: [
                RecoveryStrategy.PROVIDER_FALLBACK,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            ErrorCategory.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.RESOURCE_CLEANUP,
                RecoveryStrategy.TASK_DECOMPOSITION
            ],
            ErrorCategory.TIMEOUT: [
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.TASK_DECOMPOSITION
            ],
            ErrorCategory.DATA_VALIDATION: [
                RecoveryStrategy.ALTERNATE_APPROACH,
                RecoveryStrategy.SKIP_AND_LOG
            ]
        }
        
        # Recovery success rates
        self.strategy_success_rates: Dict[RecoveryStrategy, List[bool]] = {
            strategy: [] for strategy in RecoveryStrategy
        }
        
        # Custom recovery handlers
        self.custom_handlers: Dict[str, Callable] = {}
    
    async def recover_from_error(
        self,
        context: RecoveryContext,
        execute_fn: Callable
    ) -> Tuple[bool, Any]:
        """Attempt to recover from an error.
        
        Args:
            context: Recovery context with error details
            execute_fn: Function to execute after recovery
            
        Returns:
            Tuple of (success, result)
        """
        self.logger.info(f"Starting recovery for error: {context.error.error_type}")
        
        # Determine recovery strategies
        strategies = self._determine_strategies(context)
        
        for strategy in strategies:
            if context.attempt_count >= self.max_retry_attempts:
                self.logger.warning(f"Max retry attempts reached for {context.error.error_id}")
                break
            
            # Execute recovery strategy
            recovery_success = await self._execute_strategy(strategy, context)
            
            if recovery_success:
                # Attempt to re-execute the operation
                try:
                    start_time = time.time()
                    result = await execute_fn()
                    duration = time.time() - start_time
                    
                    # Record successful recovery
                    recovery_attempt = RecoveryAttempt(
                        timestamp=datetime.now(timezone.utc),
                        strategy=strategy,
                        success=True,
                        duration=duration
                    )
                    context.recovery_history.append(recovery_attempt)
                    self._record_strategy_success(strategy, True)
                    
                    self.logger.info(f"Recovery successful using {strategy.value}")
                    return True, result
                    
                except Exception as e:
                    self.logger.warning(f"Recovery attempt failed: {e}")
                    self._record_strategy_success(strategy, False)
                    
                    # Create new error instance
                    context.error = ErrorInstance.from_exception(
                        context.worker_id,
                        context.error.operation_type,
                        e,
                        context.metadata
                    )
            
            context.attempt_count += 1
        
        # All strategies failed
        self.logger.error(f"All recovery strategies failed for {context.error.error_id}")
        return False, None
    
    def _determine_strategies(self, context: RecoveryContext) -> List[RecoveryStrategy]:
        """Determine appropriate recovery strategies based on error context.
        
        Args:
            context: Recovery context
            
        Returns:
            List of recovery strategies to try
        """
        # Get base strategies for error category
        strategies = self.category_strategies.get(
            context.error.category,
            [RecoveryStrategy.IMMEDIATE_RETRY]
        )
        
        # Adjust based on attempt count
        if context.attempt_count > 1:
            # Prefer more aggressive strategies after initial failures
            if RecoveryStrategy.TASK_DECOMPOSITION not in strategies:
                strategies.append(RecoveryStrategy.TASK_DECOMPOSITION)
        
        # Sort by success rate if we have data
        strategies_with_rates = []
        for strategy in strategies:
            success_history = self.strategy_success_rates[strategy]
            if success_history:
                success_rate = sum(success_history[-10:]) / len(success_history[-10:])
                strategies_with_rates.append((strategy, success_rate))
            else:
                strategies_with_rates.append((strategy, 0.5))  # Default 50% assumption
        
        # Sort by success rate (descending)
        strategies_with_rates.sort(key=lambda x: x[1], reverse=True)
        
        return [strategy for strategy, _ in strategies_with_rates]
    
    async def _execute_strategy(
        self,
        strategy: RecoveryStrategy,
        context: RecoveryContext
    ) -> bool:
        """Execute a specific recovery strategy.
        
        Args:
            strategy: Recovery strategy to execute
            context: Recovery context
            
        Returns:
            True if strategy executed successfully
        """
        self.logger.info(f"Executing recovery strategy: {strategy.value}")
        
        try:
            if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                # Just return true to signal immediate retry
                return True
                
            elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                # Calculate backoff time
                backoff = min(
                    self.base_backoff_seconds * (2 ** context.attempt_count),
                    self.max_backoff_seconds
                )
                self.logger.info(f"Backing off for {backoff} seconds")
                await asyncio.sleep(backoff)
                return True
                
            elif strategy == RecoveryStrategy.ALTERNATE_APPROACH:
                # Modify work item for alternate approach
                if context.work_item and self.ai_brain:
                    await self._generate_alternate_approach(context)
                    return True
                return False
                
            elif strategy == RecoveryStrategy.TASK_DECOMPOSITION:
                # Decompose complex task into smaller ones
                if context.work_item and self.ai_brain:
                    await self._decompose_task(context)
                    return True
                return False
                
            elif strategy == RecoveryStrategy.PROVIDER_FALLBACK:
                # Switch to different provider
                context.metadata['force_provider_switch'] = True
                return True
                
            elif strategy == RecoveryStrategy.RESOURCE_CLEANUP:
                # Clean up resources
                await self._cleanup_resources(context)
                return True
                
            elif strategy == RecoveryStrategy.SKIP_AND_LOG:
                # Log the issue and skip
                self.logger.warning(f"Skipping task due to repeated errors: {context.error.error_id}")
                return False
                
            else:
                self.logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing recovery strategy {strategy}: {e}")
            return False
    
    async def _generate_alternate_approach(self, context: RecoveryContext):
        """Generate an alternate approach for the task.
        
        Args:
            context: Recovery context
        """
        if not context.work_item or not self.ai_brain:
            return
        
        prompt = f"""
        A task has failed with the following error:
        Error Type: {context.error.error_type}
        Error Message: {context.error.error_message}
        
        Original Task:
        Type: {context.work_item.task_type}
        Title: {context.work_item.title}
        Description: {context.work_item.description}
        
        Please suggest an alternate approach to accomplish this task that avoids the error.
        Return a JSON object with:
        {{
            "modified_description": "New task description with alternate approach",
            "approach_changes": ["change1", "change2"],
            "risk_mitigation": "How this approach avoids the error"
        }}
        """
        
        try:
            response = await self.ai_brain.think(prompt)
            if response and 'modified_description' in response:
                context.work_item.description = response['modified_description']
                context.metadata['alternate_approach'] = response
                self.logger.info("Generated alternate approach for task")
        except Exception as e:
            self.logger.error(f"Failed to generate alternate approach: {e}")
    
    async def _decompose_task(self, context: RecoveryContext):
        """Decompose a complex task into smaller subtasks.
        
        Args:
            context: Recovery context
        """
        if not context.work_item or not self.ai_brain:
            return
        
        prompt = f"""
        A complex task has failed and needs to be broken down into smaller, more manageable subtasks.
        
        Original Task:
        Type: {context.work_item.task_type}
        Title: {context.work_item.title}
        Description: {context.work_item.description}
        
        Error encountered: {context.error.error_message}
        
        Please decompose this into 2-4 smaller subtasks that can be executed independently.
        Return a JSON object with:
        {{
            "subtasks": [
                {{
                    "title": "Subtask title",
                    "description": "Subtask description",
                    "priority": "high/medium/low"
                }}
            ],
            "execution_order": "sequential/parallel",
            "decomposition_rationale": "Why this decomposition helps"
        }}
        """
        
        try:
            response = await self.ai_brain.think(prompt)
            if response and 'subtasks' in response:
                context.metadata['decomposed_subtasks'] = response['subtasks']
                context.metadata['decomposition_strategy'] = response
                self.logger.info(f"Decomposed task into {len(response['subtasks'])} subtasks")
        except Exception as e:
            self.logger.error(f"Failed to decompose task: {e}")
    
    async def _cleanup_resources(self, context: RecoveryContext):
        """Clean up resources that might be causing issues.
        
        Args:
            context: Recovery context
        """
        self.logger.info("Performing resource cleanup")
        
        # Clear any large data structures in context
        if 'large_data' in context.metadata:
            del context.metadata['large_data']
        
        # Suggest garbage collection
        import gc
        gc.collect()
        
        # Add small delay for system recovery
        await asyncio.sleep(1)
    
    def _record_strategy_success(self, strategy: RecoveryStrategy, success: bool):
        """Record strategy success for future optimization.
        
        Args:
            strategy: Recovery strategy used
            success: Whether it was successful
        """
        history = self.strategy_success_rates[strategy]
        history.append(success)
        
        # Keep only recent history (last 100 attempts)
        if len(history) > 100:
            self.strategy_success_rates[strategy] = history[-100:]
    
    def register_custom_handler(
        self,
        error_pattern: str,
        handler: Callable
    ):
        """Register a custom error handler for specific error patterns.
        
        Args:
            error_pattern: Regex pattern to match error messages
            handler: Custom handler function
        """
        self.custom_handlers[error_pattern] = handler
        self.logger.info(f"Registered custom handler for pattern: {error_pattern}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics.
        
        Returns:
            Dictionary of recovery statistics
        """
        stats = {}
        
        for strategy in RecoveryStrategy:
            history = self.strategy_success_rates[strategy]
            if history:
                success_rate = sum(history) / len(history)
                stats[strategy.value] = {
                    'attempts': len(history),
                    'success_rate': success_rate,
                    'recent_success_rate': sum(history[-10:]) / len(history[-10:]) if len(history) >= 10 else success_rate
                }
            else:
                stats[strategy.value] = {
                    'attempts': 0,
                    'success_rate': 0.0,
                    'recent_success_rate': 0.0
                }
        
        return stats