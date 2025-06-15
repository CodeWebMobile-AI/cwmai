"""
Error Analyzer

Intelligent error pattern detection and recovery suggestions for parallel workers.
Analyzes error patterns across workers, predicts potential failures, and provides
automated recovery strategies and prevention recommendations.
"""

import asyncio
import json
import time
import threading
import traceback
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import re
import statistics
from scripts.worker_logging_config import setup_worker_logger, WorkerOperationContext


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    NETWORK = "network"
    API_RATE_LIMIT = "api_rate_limit"
    AUTHENTICATION = "authentication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    LOGIC_ERROR = "logic_error"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    DATA_VALIDATION = "data_validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorInstance:
    """Individual error occurrence."""
    error_id: str
    timestamp: datetime
    worker_id: str
    operation_type: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    resolved: bool = False
    resolution_method: Optional[str] = None
    
    @classmethod
    def from_exception(cls, worker_id: str, operation_type: str, 
                      exception: Exception, context: Dict[str, Any] = None) -> 'ErrorInstance':
        """Create ErrorInstance from exception."""
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        
        # Generate unique error ID based on error characteristics
        error_signature = f"{type(exception).__name__}:{error_message[:100]}:{worker_id}"
        error_id = hashlib.md5(error_signature.encode()).hexdigest()[:8]
        
        return cls(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            worker_id=worker_id,
            operation_type=operation_type,
            error_type=type(exception).__name__,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context or {}
        )


@dataclass
class ErrorPattern:
    """Identified error pattern across multiple occurrences."""
    pattern_id: str
    pattern_signature: str
    occurrences: List[ErrorInstance] = field(default_factory=list)
    frequency: int = 0
    affected_workers: Set[str] = field(default_factory=set)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    trending: bool = False
    recovery_strategies: List[Dict[str, Any]] = field(default_factory=list)
    prevention_recommendations: List[str] = field(default_factory=list)
    
    def add_occurrence(self, error_instance: ErrorInstance):
        """Add new error occurrence to pattern."""
        self.occurrences.append(error_instance)
        self.frequency += 1
        self.affected_workers.add(error_instance.worker_id)
        
        if self.first_seen is None:
            self.first_seen = error_instance.timestamp
        self.last_seen = error_instance.timestamp
        
        # Update severity if new error is more severe
        if error_instance.severity.value > self.severity.value:
            self.severity = error_instance.severity
    
    def calculate_trend(self, window_hours: int = 1) -> float:
        """Calculate error frequency trend."""
        if len(self.occurrences) < 2:
            return 0.0
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        recent_errors = [e for e in self.occurrences if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return -1.0  # Decreasing trend if no recent errors
        
        # Compare recent frequency with historical average
        hours_since_first = (datetime.now(timezone.utc) - self.first_seen).total_seconds() / 3600
        if hours_since_first == 0:
            return 0.0
        
        historical_rate = len(self.occurrences) / hours_since_first
        recent_rate = len(recent_errors) / window_hours
        
        return (recent_rate - historical_rate) / max(historical_rate, 0.1)


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from specific error patterns."""
    strategy_id: str
    name: str
    description: str
    applicable_patterns: List[str]
    automated: bool
    success_rate: float
    implementation: Optional[Callable] = None
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    
    def can_apply(self, error_pattern: ErrorPattern, context: Dict[str, Any]) -> bool:
        """Check if strategy can be applied to error pattern."""
        if error_pattern.pattern_id not in self.applicable_patterns:
            return False
        
        # Check prerequisites
        for prereq in self.prerequisites:
            if prereq not in context:
                return False
        
        return True
    
    async def execute(self, error_instance: ErrorInstance, 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recovery strategy."""
        if not self.implementation:
            return {'success': False, 'reason': 'No implementation available'}
        
        try:
            if asyncio.iscoroutinefunction(self.implementation):
                result = await self.implementation(error_instance, context)
            else:
                result = self.implementation(error_instance, context)
            
            return {
                'success': True,
                'result': result,
                'strategy_used': self.strategy_id
            }
        except Exception as e:
            return {
                'success': False,
                'reason': f'Strategy execution failed: {str(e)}',
                'strategy_used': self.strategy_id
            }


class ErrorAnalyzer:
    """Intelligent error analysis and recovery system."""
    
    def __init__(self, max_error_history: int = 1000):
        """Initialize error analyzer.
        
        Args:
            max_error_history: Maximum number of errors to keep in history
        """
        self.max_error_history = max_error_history
        self.logger = setup_worker_logger("error_analyzer")
        
        # Error storage
        self.error_history: List[ErrorInstance] = []
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Analysis state
        self.pattern_analysis_enabled = True
        self.auto_recovery_enabled = True
        self.learning_enabled = True
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background analysis
        self._analysis_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Initialize built-in recovery strategies
        self._initialize_recovery_strategies()
        
        # Error categorization rules
        self._categorization_rules = self._build_categorization_rules()
    
    def _build_categorization_rules(self) -> List[Tuple[re.Pattern, ErrorCategory, ErrorSeverity]]:
        """Build rules for automatic error categorization."""
        rules = [
            # Network errors
            (re.compile(r'connection.*timeout|network.*unreachable|dns.*failed', re.IGNORECASE), 
             ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            
            # API Rate limiting
            (re.compile(r'rate.*limit|too.*many.*requests|throttle', re.IGNORECASE),
             ErrorCategory.API_RATE_LIMIT, ErrorSeverity.HIGH),
            
            # Authentication errors
            (re.compile(r'auth.*failed|unauthorized|invalid.*token|forbidden', re.IGNORECASE),
             ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH),
            
            # Resource exhaustion
            (re.compile(r'out.*of.*memory|disk.*full|no.*space|resource.*exhausted', re.IGNORECASE),
             ErrorCategory.RESOURCE_EXHAUSTION, ErrorSeverity.CRITICAL),
            
            # Timeout errors
            (re.compile(r'timeout|timed.*out', re.IGNORECASE),
             ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            
            # External service errors
            (re.compile(r'service.*unavailable|external.*error|downstream.*failed', re.IGNORECASE),
             ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM),
            
            # Configuration errors
            (re.compile(r'config.*error|setting.*invalid|missing.*config', re.IGNORECASE),
             ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH),
            
            # Data validation errors
            (re.compile(r'validation.*failed|invalid.*data|schema.*error', re.IGNORECASE),
             ErrorCategory.DATA_VALIDATION, ErrorSeverity.LOW),
        ]
        
        return rules
    
    def _initialize_recovery_strategies(self):
        """Initialize built-in recovery strategies."""
        
        # Rate limit recovery
        async def rate_limit_recovery(error_instance: ErrorInstance, context: Dict[str, Any]):
            """Handle API rate limit errors."""
            wait_time = min(60, 2 ** context.get('retry_count', 0))  # Exponential backoff
            await asyncio.sleep(wait_time)
            return {'action': 'waited', 'duration': wait_time}
        
        self.register_recovery_strategy(RecoveryStrategy(
            strategy_id="rate_limit_backoff",
            name="Rate Limit Exponential Backoff",
            description="Wait with exponential backoff when rate limited",
            applicable_patterns=[],  # Will be populated when patterns are identified
            automated=True,
            success_rate=0.9,
            implementation=rate_limit_recovery,
            prerequisites=[]
        ))
        
        # Network retry recovery
        async def network_retry_recovery(error_instance: ErrorInstance, context: Dict[str, Any]):
            """Handle network errors with retry."""
            retry_count = context.get('retry_count', 0)
            if retry_count >= 3:
                return {'action': 'max_retries_reached', 'retry_count': retry_count}
            
            wait_time = min(30, 2 ** retry_count)
            await asyncio.sleep(wait_time)
            return {'action': 'retry_scheduled', 'retry_count': retry_count + 1, 'wait_time': wait_time}
        
        self.register_recovery_strategy(RecoveryStrategy(
            strategy_id="network_retry",
            name="Network Error Retry",
            description="Retry operations that failed due to network issues",
            applicable_patterns=[],
            automated=True,
            success_rate=0.7,
            implementation=network_retry_recovery,
            prerequisites=[]
        ))
        
        # Worker restart recovery
        async def worker_restart_recovery(error_instance: ErrorInstance, context: Dict[str, Any]):
            """Handle errors by suggesting worker restart."""
            return {
                'action': 'restart_recommended',
                'worker_id': error_instance.worker_id,
                'reason': 'Persistent errors detected'
            }
        
        self.register_recovery_strategy(RecoveryStrategy(
            strategy_id="worker_restart",
            name="Worker Restart",
            description="Restart worker experiencing persistent errors",
            applicable_patterns=[],
            automated=False,  # Requires manual intervention
            success_rate=0.8,
            implementation=worker_restart_recovery,
            prerequisites=['worker_management']
        ))
    
    async def start(self):
        """Start background error analysis."""
        self.logger.info("Starting error analyzer")
        self._analysis_task = asyncio.create_task(self._analysis_loop())
    
    async def stop(self):
        """Stop error analysis."""
        self.logger.info("Stopping error analyzer")
        self._shutdown = True
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
    
    def register_error(self, worker_id: str, operation_type: str, 
                      exception: Exception, context: Dict[str, Any] = None) -> str:
        """Register a new error occurrence.
        
        Args:
            worker_id: ID of worker that encountered error
            operation_type: Type of operation that failed
            exception: The exception that occurred
            context: Additional context information
            
        Returns:
            Error ID for tracking
        """
        error_instance = ErrorInstance.from_exception(
            worker_id, operation_type, exception, context
        )
        
        # Categorize error
        self._categorize_error(error_instance)
        
        with self.lock:
            # Add to history
            self.error_history.append(error_instance)
            
            # Maintain history size
            if len(self.error_history) > self.max_error_history:
                self.error_history.pop(0)
            
            # Update patterns
            self._update_error_patterns(error_instance)
        
        self.logger.error(f"Registered error {error_instance.error_id} from worker {worker_id}: {exception}")
        
        # Trigger immediate analysis if error is severe
        if error_instance.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            asyncio.create_task(self._immediate_analysis(error_instance))
        
        return error_instance.error_id
    
    def _categorize_error(self, error_instance: ErrorInstance):
        """Automatically categorize error based on message and type."""
        error_text = f"{error_instance.error_type} {error_instance.error_message}"
        
        for pattern, category, severity in self._categorization_rules:
            if pattern.search(error_text):
                error_instance.category = category
                error_instance.severity = severity
                return
        
        # Default categorization based on exception type
        if 'Network' in error_instance.error_type or 'Connection' in error_instance.error_type:
            error_instance.category = ErrorCategory.NETWORK
            error_instance.severity = ErrorSeverity.MEDIUM
        elif 'Timeout' in error_instance.error_type:
            error_instance.category = ErrorCategory.TIMEOUT
            error_instance.severity = ErrorSeverity.MEDIUM
        elif 'Permission' in error_instance.error_type or 'Auth' in error_instance.error_type:
            error_instance.category = ErrorCategory.AUTHENTICATION
            error_instance.severity = ErrorSeverity.HIGH
        else:
            error_instance.category = ErrorCategory.LOGIC_ERROR
            error_instance.severity = ErrorSeverity.LOW
    
    def _update_error_patterns(self, error_instance: ErrorInstance):
        """Update error patterns with new error instance."""
        # Create pattern signature based on error characteristics
        signature_elements = [
            error_instance.error_type,
            error_instance.operation_type,
            self._normalize_error_message(error_instance.error_message)
        ]
        pattern_signature = "|".join(signature_elements)
        pattern_id = hashlib.md5(pattern_signature.encode()).hexdigest()[:12]
        
        if pattern_id not in self.error_patterns:
            # Create new pattern
            self.error_patterns[pattern_id] = ErrorPattern(
                pattern_id=pattern_id,
                pattern_signature=pattern_signature,
                category=error_instance.category,
                severity=error_instance.severity
            )
        
        # Add occurrence to pattern
        self.error_patterns[pattern_id].add_occurrence(error_instance)
        
        # Update pattern analysis
        self._analyze_pattern_trends(self.error_patterns[pattern_id])
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message for pattern matching."""
        # Remove specific values that vary between instances
        normalized = re.sub(r'\d+', 'N', message)  # Replace numbers
        normalized = re.sub(r'[a-f0-9]{8,}', 'HASH', normalized)  # Replace hashes/IDs
        normalized = re.sub(r'https?://[^\s]+', 'URL', normalized)  # Replace URLs
        normalized = re.sub(r'/[^\s]+', 'PATH', normalized)  # Replace file paths
        return normalized.lower()
    
    def _analyze_pattern_trends(self, pattern: ErrorPattern):
        """Analyze trends for a specific error pattern."""
        if len(pattern.occurrences) < 3:
            return
        
        # Check if pattern is trending upward
        trend = pattern.calculate_trend(window_hours=1)
        pattern.trending = trend > 0.5  # Significant upward trend
        
        # Generate recovery strategies for trending patterns
        if pattern.trending and not pattern.recovery_strategies:
            self._generate_recovery_strategies(pattern)
    
    def _generate_recovery_strategies(self, pattern: ErrorPattern):
        """Generate recovery strategies for error pattern."""
        strategies = []
        
        # Match with existing recovery strategies
        for strategy_id, strategy in self.recovery_strategies.items():
            if pattern.category == ErrorCategory.API_RATE_LIMIT and strategy_id == "rate_limit_backoff":
                strategies.append({
                    'strategy_id': strategy_id,
                    'confidence': 0.9,
                    'automated': strategy.automated
                })
            elif pattern.category == ErrorCategory.NETWORK and strategy_id == "network_retry":
                strategies.append({
                    'strategy_id': strategy_id,
                    'confidence': 0.8,
                    'automated': strategy.automated
                })
            elif len(pattern.affected_workers) == 1 and strategy_id == "worker_restart":
                strategies.append({
                    'strategy_id': strategy_id,
                    'confidence': 0.7,
                    'automated': strategy.automated
                })
        
        pattern.recovery_strategies = strategies
        
        # Generate prevention recommendations
        pattern.prevention_recommendations = self._generate_prevention_recommendations(pattern)
    
    def _generate_prevention_recommendations(self, pattern: ErrorPattern) -> List[str]:
        """Generate prevention recommendations for error pattern."""
        recommendations = []
        
        if pattern.category == ErrorCategory.API_RATE_LIMIT:
            recommendations.extend([
                "Implement request queuing with rate limiting",
                "Add retry logic with exponential backoff",
                "Monitor API usage and implement circuit breakers"
            ])
        elif pattern.category == ErrorCategory.NETWORK:
            recommendations.extend([
                "Add network connectivity checks before operations",
                "Implement connection pooling and reuse",
                "Add network timeout configuration"
            ])
        elif pattern.category == ErrorCategory.RESOURCE_EXHAUSTION:
            recommendations.extend([
                "Implement resource monitoring and alerting",
                "Add resource cleanup in finally blocks",
                "Consider implementing resource limits"
            ])
        elif pattern.category == ErrorCategory.AUTHENTICATION:
            recommendations.extend([
                "Implement token refresh mechanism",
                "Add authentication status monitoring",
                "Validate credentials before operations"
            ])
        
        return recommendations
    
    async def _analysis_loop(self):
        """Background analysis loop."""
        while not self._shutdown:
            try:
                with WorkerOperationContext("error_analyzer", "background_analysis"):
                    await self._perform_periodic_analysis()
                    await asyncio.sleep(60)  # Analyze every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}", exc_info=True)
                await asyncio.sleep(120)  # Back off on error
    
    async def _perform_periodic_analysis(self):
        """Perform periodic error analysis."""
        with self.lock:
            # Update pattern trends
            for pattern in self.error_patterns.values():
                self._analyze_pattern_trends(pattern)
            
            # Clean up old errors
            self._cleanup_old_errors()
            
            # Generate global insights
            self._generate_global_insights()
    
    async def _immediate_analysis(self, error_instance: ErrorInstance):
        """Perform immediate analysis for severe errors."""
        try:
            with WorkerOperationContext("error_analyzer", "immediate_analysis"):
                # Attempt automatic recovery if enabled
                if self.auto_recovery_enabled:
                    await self._attempt_auto_recovery(error_instance)
                
                # Check for critical patterns
                await self._check_critical_patterns(error_instance)
        
        except Exception as e:
            self.logger.error(f"Error in immediate analysis: {e}", exc_info=True)
    
    async def _attempt_auto_recovery(self, error_instance: ErrorInstance):
        """Attempt automatic recovery for error."""
        # Find matching pattern
        pattern = None
        for p in self.error_patterns.values():
            if any(occ.error_id == error_instance.error_id for occ in p.occurrences):
                pattern = p
                break
        
        if not pattern or not pattern.recovery_strategies:
            return
        
        # Try automated recovery strategies
        for strategy_info in pattern.recovery_strategies:
            if not strategy_info.get('automated', False):
                continue
            
            strategy_id = strategy_info['strategy_id']
            if strategy_id not in self.recovery_strategies:
                continue
            
            strategy = self.recovery_strategies[strategy_id]
            
            try:
                result = await strategy.execute(error_instance, {
                    'retry_count': 0,
                    'pattern_frequency': pattern.frequency
                })
                
                if result.get('success'):
                    self.logger.info(f"Auto-recovery successful for error {error_instance.error_id} "
                                   f"using strategy {strategy_id}")
                    error_instance.resolved = True
                    error_instance.resolution_method = strategy_id
                    break
                else:
                    self.logger.warning(f"Auto-recovery failed for error {error_instance.error_id} "
                                      f"using strategy {strategy_id}: {result.get('reason')}")
            
            except Exception as e:
                self.logger.error(f"Error executing recovery strategy {strategy_id}: {e}")
    
    async def _check_critical_patterns(self, error_instance: ErrorInstance):
        """Check for critical error patterns that need immediate attention."""
        # Check for error storms (many errors in short time)
        recent_errors = [
            e for e in self.error_history 
            if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        
        if len(recent_errors) > 10:
            self.logger.critical(f"Error storm detected: {len(recent_errors)} errors in 5 minutes")
        
        # Check for worker-specific issues
        worker_errors = [e for e in recent_errors if e.worker_id == error_instance.worker_id]
        if len(worker_errors) > 5:
            self.logger.warning(f"Worker {error_instance.worker_id} experiencing multiple errors: "
                              f"{len(worker_errors)} in 5 minutes")
    
    def _cleanup_old_errors(self):
        """Clean up old error data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        # Clean up old patterns with low frequency
        patterns_to_remove = []
        for pattern_id, pattern in self.error_patterns.items():
            if pattern.last_seen < cutoff_time and pattern.frequency < 3:
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.error_patterns[pattern_id]
        
        if patterns_to_remove:
            self.logger.debug(f"Cleaned up {len(patterns_to_remove)} old error patterns")
    
    def _generate_global_insights(self):
        """Generate global insights from error analysis."""
        if not self.error_patterns:
            return
        
        # Find most common error categories
        category_counts = Counter()
        for pattern in self.error_patterns.values():
            category_counts[pattern.category] += pattern.frequency
        
        # Find most affected workers
        worker_error_counts = Counter()
        for error in self.error_history:
            worker_error_counts[error.worker_id] += 1
        
        # Log insights
        if category_counts:
            top_category = category_counts.most_common(1)[0]
            self.logger.info(f"Most common error category: {top_category[0].value} "
                           f"({top_category[1]} occurrences)")
        
        if worker_error_counts:
            most_affected_worker = worker_error_counts.most_common(1)[0]
            self.logger.info(f"Most affected worker: {most_affected_worker[0]} "
                           f"({most_affected_worker[1]} errors)")
    
    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """Register a new recovery strategy."""
        with self.lock:
            self.recovery_strategies[strategy.strategy_id] = strategy
            self.logger.info(f"Registered recovery strategy: {strategy.name}")
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self.lock:
            recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
            
            # Group by category
            category_stats = defaultdict(lambda: {'count': 0, 'workers': set()})
            for error in recent_errors:
                category_stats[error.category.value]['count'] += 1
                category_stats[error.category.value]['workers'].add(error.worker_id)
            
            # Convert sets to counts
            for category in category_stats:
                category_stats[category]['unique_workers'] = len(category_stats[category]['workers'])
                del category_stats[category]['workers']
            
            # Get trending patterns
            trending_patterns = [
                {
                    'pattern_id': p.pattern_id,
                    'frequency': p.frequency,
                    'affected_workers': len(p.affected_workers),
                    'category': p.category.value,
                    'last_seen': p.last_seen.isoformat() if p.last_seen else None
                }
                for p in self.error_patterns.values() 
                if p.trending
            ]
            
            return {
                'period_hours': hours,
                'total_errors': len(recent_errors),
                'unique_patterns': len(self.error_patterns),
                'category_breakdown': dict(category_stats),
                'trending_patterns': trending_patterns,
                'recovery_strategies_available': len(self.recovery_strategies),
                'auto_recovery_enabled': self.auto_recovery_enabled
            }
    
    def get_worker_error_analysis(self, worker_id: str) -> Dict[str, Any]:
        """Get detailed error analysis for specific worker."""
        with self.lock:
            worker_errors = [e for e in self.error_history if e.worker_id == worker_id]
            
            if not worker_errors:
                return {'worker_id': worker_id, 'error_count': 0}
            
            # Calculate statistics
            error_types = Counter(e.error_type for e in worker_errors)
            categories = Counter(e.category.value for e in worker_errors)
            operations = Counter(e.operation_type for e in worker_errors)
            
            # Time distribution
            recent_errors = [
                e for e in worker_errors 
                if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            # Resolution rate
            resolved_errors = [e for e in worker_errors if e.resolved]
            resolution_rate = len(resolved_errors) / len(worker_errors) if worker_errors else 0
            
            return {
                'worker_id': worker_id,
                'total_errors': len(worker_errors),
                'recent_errors_1h': len(recent_errors),
                'resolution_rate': resolution_rate,
                'most_common_error_type': error_types.most_common(1)[0] if error_types else None,
                'most_common_category': categories.most_common(1)[0] if categories else None,
                'most_problematic_operation': operations.most_common(1)[0] if operations else None,
                'error_distribution': {
                    'by_type': dict(error_types.most_common(5)),
                    'by_category': dict(categories),
                    'by_operation': dict(operations.most_common(5))
                },
                'latest_error': {
                    'timestamp': worker_errors[-1].timestamp.isoformat(),
                    'error_type': worker_errors[-1].error_type,
                    'message': worker_errors[-1].error_message[:100]
                } if worker_errors else None
            }
    
    def get_recovery_recommendations(self, worker_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recovery recommendations for worker or system."""
        recommendations = []
        
        with self.lock:
            # Filter patterns by worker if specified
            relevant_patterns = []
            if worker_id:
                relevant_patterns = [
                    p for p in self.error_patterns.values() 
                    if worker_id in p.affected_workers
                ]
            else:
                relevant_patterns = list(self.error_patterns.values())
            
            # Sort by frequency and trending
            relevant_patterns.sort(
                key=lambda p: (p.trending, p.frequency), 
                reverse=True
            )
            
            for pattern in relevant_patterns[:10]:  # Top 10 patterns
                if pattern.recovery_strategies or pattern.prevention_recommendations:
                    recommendations.append({
                        'pattern_id': pattern.pattern_id,
                        'category': pattern.category.value,
                        'frequency': pattern.frequency,
                        'trending': pattern.trending,
                        'affected_workers': list(pattern.affected_workers),
                        'recovery_strategies': pattern.recovery_strategies,
                        'prevention_recommendations': pattern.prevention_recommendations,
                        'last_seen': pattern.last_seen.isoformat() if pattern.last_seen else None
                    })
        
        return recommendations


# Context manager for automatic error registration
class ErrorCaptureContext:
    """Context manager for automatic error capture and analysis."""
    
    def __init__(self, error_analyzer: ErrorAnalyzer, worker_id: str, 
                 operation_type: str, context: Dict[str, Any] = None):
        """Initialize error capture context.
        
        Args:
            error_analyzer: ErrorAnalyzer instance
            worker_id: Worker performing the operation
            operation_type: Type of operation being performed
            context: Additional context information
        """
        self.error_analyzer = error_analyzer
        self.worker_id = worker_id
        self.operation_type = operation_type
        self.context = context or {}
        self.error_id = None
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and capture any exceptions."""
        if exc_type is not None:
            self.error_id = self.error_analyzer.register_error(
                self.worker_id, self.operation_type, exc_val, self.context
            )
        return False  # Don't suppress exceptions


# Example usage and demonstration
async def demonstrate_error_analyzer():
    """Demonstrate the error analyzer capabilities."""
    analyzer = ErrorAnalyzer()
    await analyzer.start()
    
    try:
        # Simulate various types of errors
        errors_to_simulate = [
            ("worker_1", "api_call", ConnectionError("Connection timeout"), {"retry_count": 0}),
            ("worker_1", "api_call", ConnectionError("Connection timeout"), {"retry_count": 1}),
            ("worker_2", "data_processing", ValueError("Invalid data format"), {"data_size": 1000}),
            ("worker_3", "authentication", PermissionError("Access denied"), {"token_expired": True}),
            ("worker_1", "api_call", Exception("Rate limit exceeded"), {"requests_made": 100}),
            ("worker_2", "file_operation", FileNotFoundError("File not found"), {"file_path": "/tmp/data.json"}),
        ]
        
        # Register errors
        for worker_id, operation, exception, context in errors_to_simulate:
            with ErrorCaptureContext(analyzer, worker_id, operation, context):
                raise exception
        
        # Wait for analysis
        await asyncio.sleep(2)
        
        # Get error summary
        summary = analyzer.get_error_summary(hours=1)
        print("Error Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
        # Get worker-specific analysis
        worker_analysis = analyzer.get_worker_error_analysis("worker_1")
        print(f"\nWorker 1 Analysis:")
        print(json.dumps(worker_analysis, indent=2, default=str))
        
        # Get recovery recommendations
        recommendations = analyzer.get_recovery_recommendations()
        print(f"\nRecovery Recommendations:")
        print(json.dumps(recommendations, indent=2, default=str))
        
    finally:
        await analyzer.stop()


if __name__ == "__main__":
    asyncio.run(demonstrate_error_analyzer())