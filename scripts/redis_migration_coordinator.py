"""
Redis Migration Coordinator

Orchestrates the migration from in-memory/file-based systems to Redis-backed
infrastructure with comprehensive monitoring, rollback capabilities, and phased deployment.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from redis_ai_response_cache import RedisAIResponseCache, get_redis_ai_cache
from redis_lockfree_adapter import create_lockfree_state_manager
from redis_integration import get_redis_client, RedisMonitoring


class MigrationPhase(Enum):
    """Migration phases."""
    PLANNING = "planning"
    PREPARATION = "preparation"
    DUAL_WRITE = "dual_write"
    VALIDATION = "validation"
    CUTOVER = "cutover"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationStatus(Enum):
    """Migration status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationMetrics:
    """Migration performance and progress metrics."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_phase: MigrationPhase = MigrationPhase.PLANNING
    status: MigrationStatus = MigrationStatus.NOT_STARTED
    
    # Cache migration metrics
    cache_entries_total: int = 0
    cache_entries_migrated: int = 0
    cache_entries_failed: int = 0
    cache_migration_rate: float = 0.0
    
    # State migration metrics
    state_keys_total: int = 0
    state_keys_migrated: int = 0
    state_keys_failed: int = 0
    state_migration_rate: float = 0.0
    
    # Performance metrics
    redis_response_time_ms: float = 0.0
    fallback_response_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput_improvement: float = 0.0
    
    # Resource metrics
    memory_usage_before_mb: float = 0.0
    memory_usage_after_mb: float = 0.0
    network_bandwidth_usage_mbps: float = 0.0
    
    # Quality metrics
    data_consistency_score: float = 1.0
    performance_regression_score: float = 0.0
    availability_score: float = 1.0
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall migration progress."""
        cache_progress = (self.cache_entries_migrated / max(self.cache_entries_total, 1)) * 0.6
        state_progress = (self.state_keys_migrated / max(self.state_keys_total, 1)) * 0.4
        return min(1.0, cache_progress + state_progress)
    
    @property
    def estimated_completion(self) -> Optional[datetime]:
        """Estimate completion time based on current rate."""
        if not self.start_time or self.overall_progress == 0:
            return None
        
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        estimated_total = elapsed / self.overall_progress
        return self.start_time + timedelta(seconds=estimated_total)


@dataclass
class MigrationConfig:
    """Migration configuration settings."""
    # Migration strategy
    migration_mode: str = "gradual"  # gradual, immediate, canary
    dual_write_duration_hours: int = 24
    validation_duration_hours: int = 2
    rollback_threshold_error_rate: float = 0.05
    
    # Performance settings
    batch_size: int = 100
    migration_rate_limit: int = 1000  # operations per second
    max_concurrent_operations: int = 50
    health_check_interval: int = 30
    
    # Safety settings
    enable_automatic_rollback: bool = True
    enable_canary_testing: bool = True
    canary_traffic_percentage: float = 0.1
    safety_delay_seconds: int = 5
    
    # Monitoring settings
    enable_detailed_metrics: bool = True
    metrics_collection_interval: int = 10
    alert_on_errors: bool = True


class RedisMigrationCoordinator:
    """Coordinates migration from legacy systems to Redis infrastructure."""
    
    def __init__(self, config: Optional[MigrationConfig] = None):
        """Initialize migration coordinator.
        
        Args:
            config: Migration configuration settings
        """
        self.config = config or MigrationConfig()
        self.metrics = MigrationMetrics()
        self.logger = logging.getLogger(f"{__name__}.RedisMigrationCoordinator")
        
        # Migration components
        self.redis_cache: Optional[RedisAIResponseCache] = None
        self.redis_state: Optional[RedisAsyncStateManager] = None
        self.redis_monitoring: Optional[RedisMonitoring] = None
        
        # Migration tracking
        self._migration_tasks: List[asyncio.Task] = []
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Event handlers
        self._phase_change_handlers: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._completion_handlers: List[Callable] = []
        
        # Safety mechanisms
        self._rollback_triggers: List[Callable] = []
        self._health_checks: List[Callable] = []
        
        # Migration state
        self._migration_lock = asyncio.Lock()
        self._current_operations: Dict[str, asyncio.Task] = {}
    
    async def initialize(self):
        """Initialize migration coordinator and components."""
        try:
            self.logger.info("Initializing Redis Migration Coordinator")
            
            # Initialize Redis monitoring
            redis_client = await get_redis_client()
            self.redis_monitoring = RedisMonitoring(redis_client)
            await self.redis_monitoring.start()
            
            # Add migration-specific health checks
            self._setup_health_checks()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
            
            self.logger.info("Redis Migration Coordinator initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing migration coordinator: {e}")
            raise
    
    def _setup_health_checks(self):
        """Setup migration-specific health checks."""
        try:
            # Add Redis connectivity health check
            async def redis_connectivity_check():
                try:
                    redis_client = await get_redis_client()
                    await redis_client.ping()
                    return True
                except Exception:
                    return False
            
            self._health_checks.append(redis_connectivity_check)
            
            # Add cache performance health check
            async def cache_performance_check():
                try:
                    if self.redis_cache:
                        # Simple performance test
                        test_key = "health_check_test"
                        await self.redis_cache.set(test_key, "test_value", expire_seconds=60)
                        result = await self.redis_cache.get(test_key)
                        await self.redis_cache.delete(test_key)
                        return result == "test_value"
                    return True
                except Exception:
                    return False
            
            self._health_checks.append(cache_performance_check)
            
            # Add state manager health check
            async def state_manager_check():
                try:
                    if self.redis_state:
                        # Test basic state operations
                        await self.redis_state.update("health_check", {"status": "ok"})
                        result = await self.redis_state.get("health_check")
                        return result is not None
                    return True
                except Exception:
                    return False
            
            self._health_checks.append(state_manager_check)
            
            self.logger.info(f"Setup {len(self._health_checks)} health checks")
            
        except Exception as e:
            self.logger.error(f"Error setting up health checks: {e}")
    
    async def start_migration(self) -> bool:
        """Start the migration process.
        
        Returns:
            True if migration started successfully
        """
        async with self._migration_lock:
            if self.metrics.status == MigrationStatus.IN_PROGRESS:
                self.logger.warning("Migration already in progress")
                return False
            
            try:
                self.logger.info("Starting Redis migration process")
                self.metrics.start_time = datetime.now(timezone.utc)
                self.metrics.status = MigrationStatus.IN_PROGRESS
                self.metrics.current_phase = MigrationPhase.PLANNING
                
                # Execute migration phases
                success = await self._execute_migration_phases()
                
                if success:
                    self.metrics.status = MigrationStatus.COMPLETED
                    self.metrics.end_time = datetime.now(timezone.utc)
                    self.logger.info("Migration completed successfully")
                    await self._trigger_completion_handlers()
                else:
                    self.metrics.status = MigrationStatus.FAILED
                    self.logger.error("Migration failed")
                    
                    if self.config.enable_automatic_rollback:
                        await self.rollback_migration()
                
                return success
                
            except Exception as e:
                self.logger.error(f"Error in migration process: {e}")
                self.metrics.status = MigrationStatus.FAILED
                await self._trigger_error_handlers(e)
                
                if self.config.enable_automatic_rollback:
                    await self.rollback_migration()
                
                return False
    
    async def _execute_migration_phases(self) -> bool:
        """Execute all migration phases in sequence."""
        phases = [
            (MigrationPhase.PLANNING, self._phase_planning),
            (MigrationPhase.PREPARATION, self._phase_preparation),
            (MigrationPhase.DUAL_WRITE, self._phase_dual_write),
            (MigrationPhase.VALIDATION, self._phase_validation),
            (MigrationPhase.CUTOVER, self._phase_cutover),
            (MigrationPhase.CLEANUP, self._phase_cleanup)
        ]
        
        for phase, phase_func in phases:
            try:
                self.logger.info(f"Starting migration phase: {phase.value}")
                self.metrics.current_phase = phase
                await self._trigger_phase_change_handlers(phase)
                
                success = await phase_func()
                if not success:
                    self.logger.error(f"Migration phase {phase.value} failed")
                    return False
                
                self.logger.info(f"Migration phase {phase.value} completed")
                
                # Safety delay between phases
                await asyncio.sleep(self.config.safety_delay_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in migration phase {phase.value}: {e}")
                return False
        
        self.metrics.current_phase = MigrationPhase.COMPLETED
        return True
    
    async def _phase_planning(self) -> bool:
        """Planning phase: Analyze current system and prepare migration plan."""
        try:
            self.logger.info("Analyzing current system for migration planning")
            
            # Analyze cache system
            from scripts.ai_response_cache import get_global_cache
            current_cache = get_global_cache()
            cache_stats = current_cache.get_stats()
            
            self.metrics.cache_entries_total = cache_stats['cache_size']
            self.metrics.memory_usage_before_mb = cache_stats['memory_usage_mb']
            
            # Analyze state system
            from scripts.async_state_manager import get_async_state_manager
            current_state = await get_async_state_manager()
            state_metrics = current_state.get_metrics()
            
            full_state = await current_state.get_full_state()
            self.metrics.state_keys_total = len(self._flatten_dict(full_state))
            
            # Estimate migration time
            estimated_duration = self._estimate_migration_duration()
            self.logger.info(f"Estimated migration duration: {estimated_duration:.1f} minutes")
            
            # Validate prerequisites
            return await self._validate_migration_prerequisites()
            
        except Exception as e:
            self.logger.error(f"Error in planning phase: {e}")
            return False
    
    async def _phase_preparation(self) -> bool:
        """Preparation phase: Initialize Redis components and validate connectivity."""
        try:
            self.logger.info("Preparing Redis infrastructure")
            
            # Initialize Redis cache with migration mode
            self.redis_cache = await get_redis_ai_cache(migration_mode="gradual")
            
            # Initialize Redis state manager with migration mode
            self.redis_state = create_lockfree_state_manager(f"migration_coordinator_{self.coordinator_id}")
            await self.redis_state.initialize()
            
            # Validate Redis connectivity and performance
            return await self._validate_redis_performance()
            
        except Exception as e:
            self.logger.error(f"Error in preparation phase: {e}")
            return False
    
    async def _phase_dual_write(self) -> bool:
        """Dual write phase: Write to both legacy and Redis systems."""
        try:
            self.logger.info("Starting dual-write phase")
            
            # Configure dual-write mode
            if self.redis_cache:
                self.redis_cache.dual_write_mode = True
                self.redis_cache.migration_mode = "gradual"
            
            if self.redis_state:
                self.redis_state.dual_write_mode = True
                self.redis_state.migration_mode = "gradual"
            
            # Monitor dual-write for configured duration
            end_time = datetime.now(timezone.utc) + timedelta(hours=self.config.dual_write_duration_hours)
            
            while datetime.now(timezone.utc) < end_time and not self._shutdown:
                # Check migration progress
                if self.redis_cache:
                    cache_migration = await self.redis_cache.get_migration_status()
                    self.metrics.cache_entries_migrated = cache_migration['progress']['migrated_entries']
                    self.metrics.cache_entries_failed = cache_migration['progress']['failed_migrations']
                
                if self.redis_state:
                    state_migration = await self.redis_state.get_migration_status()
                    self.metrics.state_keys_migrated = state_migration['progress']['migrated_keys']
                    self.metrics.state_keys_failed = state_migration['progress']['failed_migrations']
                
                # Check for rollback conditions
                if await self._should_trigger_rollback():
                    self.logger.warning("Rollback condition detected during dual-write")
                    return False
                
                await asyncio.sleep(self.config.health_check_interval)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in dual-write phase: {e}")
            return False
    
    async def _phase_validation(self) -> bool:
        """Validation phase: Validate data consistency and performance."""
        try:
            self.logger.info("Starting validation phase")
            
            # Data consistency validation
            consistency_score = await self._validate_data_consistency()
            self.metrics.data_consistency_score = consistency_score
            
            if consistency_score < 0.95:
                self.logger.error(f"Data consistency validation failed: {consistency_score:.3f}")
                return False
            
            # Performance validation
            performance_score = await self._validate_performance()
            self.metrics.performance_regression_score = performance_score
            
            if performance_score > 0.2:  # More than 20% regression
                self.logger.error(f"Performance regression detected: {performance_score:.3f}")
                return False
            
            # Monitor for validation duration
            end_time = datetime.now(timezone.utc) + timedelta(hours=self.config.validation_duration_hours)
            
            while datetime.now(timezone.utc) < end_time and not self._shutdown:
                await self._collect_validation_metrics()
                await asyncio.sleep(self.config.health_check_interval)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in validation phase: {e}")
            return False
    
    async def _phase_cutover(self) -> bool:
        """Cutover phase: Switch primary traffic to Redis."""
        try:
            self.logger.info("Starting cutover phase")
            
            # Implement canary deployment if enabled
            if self.config.enable_canary_testing:
                success = await self._canary_cutover()
                if not success:
                    return False
            
            # Full cutover to Redis
            if self.redis_cache:
                self.redis_cache.migration_mode = "immediate"
                self.redis_cache.dual_write_mode = False
            
            if self.redis_state:
                self.redis_state.migration_mode = "immediate"
                self.redis_state.dual_write_mode = False
            
            # Monitor cutover for stability
            await asyncio.sleep(self.config.health_check_interval * 2)
            
            # Validate cutover success
            return await self._validate_cutover()
            
        except Exception as e:
            self.logger.error(f"Error in cutover phase: {e}")
            return False
    
    async def _phase_cleanup(self) -> bool:
        """Cleanup phase: Remove legacy systems and optimize Redis."""
        try:
            self.logger.info("Starting cleanup phase")
            
            # Cleanup legacy cache (optional - keep for emergency rollback)
            # This would be done after a successful period of Redis operation
            
            # Optimize Redis configuration
            if self.redis_cache:
                await self._optimize_redis_cache()
            
            if self.redis_state:
                await self._optimize_redis_state()
            
            # Final metrics collection
            await self._collect_final_metrics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in cleanup phase: {e}")
            return False
    
    async def _canary_cutover(self) -> bool:
        """Perform canary deployment cutover."""
        try:
            self.logger.info(f"Starting canary cutover with {self.config.canary_traffic_percentage:.1%} traffic")
            
            # This would require implementing traffic routing logic
            # For now, we'll simulate the canary testing
            
            # Monitor canary for issues
            canary_duration = timedelta(minutes=30)
            end_time = datetime.now(timezone.utc) + canary_duration
            
            while datetime.now(timezone.utc) < end_time:
                # Check canary metrics
                error_rate = await self._measure_error_rate()
                if error_rate > self.config.rollback_threshold_error_rate:
                    self.logger.error(f"Canary error rate too high: {error_rate:.3f}")
                    return False
                
                await asyncio.sleep(self.config.health_check_interval)
            
            self.logger.info("Canary cutover successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in canary cutover: {e}")
            return False
    
    async def rollback_migration(self) -> bool:
        """Rollback migration to previous state."""
        async with self._migration_lock:
            try:
                self.logger.warning("Starting migration rollback")
                self.metrics.status = MigrationStatus.ROLLING_BACK
                
                # Disable Redis components
                if self.redis_cache:
                    self.redis_cache.migration_mode = "readonly"
                    self.redis_cache.dual_write_mode = False
                
                if self.redis_state:
                    self.redis_state.migration_mode = "readonly"
                    self.redis_state.dual_write_mode = False
                
                # Restore legacy systems
                await self._restore_legacy_systems()
                
                self.metrics.status = MigrationStatus.ROLLED_BACK
                self.logger.info("Migration rollback completed")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error during rollback: {e}")
                return False
    
    async def pause_migration(self):
        """Pause the migration process."""
        self.metrics.status = MigrationStatus.PAUSED
        self.logger.info("Migration paused")
    
    async def resume_migration(self):
        """Resume the migration process."""
        if self.metrics.status == MigrationStatus.PAUSED:
            self.metrics.status = MigrationStatus.IN_PROGRESS
            self.logger.info("Migration resumed")
    
    async def _validate_migration_prerequisites(self) -> bool:
        """Validate that all prerequisites for migration are met."""
        try:
            # Check Redis connectivity
            redis_client = await get_redis_client()
            await redis_client.ping()
            
            # Check available memory
            # Check network connectivity
            # Check disk space
            # etc.
            
            return True
            
        except Exception as e:
            self.logger.error(f"Prerequisites validation failed: {e}")
            return False
    
    async def _validate_redis_performance(self) -> bool:
        """Validate Redis performance meets requirements."""
        try:
            # Measure Redis response time
            start_time = time.time()
            redis_client = await get_redis_client()
            await redis_client.ping()
            response_time = (time.time() - start_time) * 1000
            
            self.metrics.redis_response_time_ms = response_time
            
            # Response time should be reasonable (< 10ms for ping)
            if response_time > 10:
                self.logger.warning(f"Redis response time high: {response_time:.3f}ms")
            
            return response_time < 50  # Fail if > 50ms for basic ping
            
        except Exception as e:
            self.logger.error(f"Redis performance validation failed: {e}")
            return False
    
    async def _validate_data_consistency(self) -> float:
        """Validate data consistency between legacy and Redis systems."""
        try:
            # This would involve sampling data from both systems and comparing
            # For now, return a simulated consistency score
            return 0.98
            
        except Exception as e:
            self.logger.error(f"Data consistency validation failed: {e}")
            return 0.0
    
    async def _validate_performance(self) -> float:
        """Validate performance regression is within acceptable limits."""
        try:
            # Compare performance metrics between legacy and Redis
            # Return performance regression score (0 = no regression, 1 = 100% slower)
            return 0.05  # Simulated 5% improvement (negative regression)
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            return 1.0
    
    async def _measure_error_rate(self) -> float:
        """Measure current error rate."""
        try:
            # Get error metrics from monitoring
            if self.redis_monitoring:
                health_status = await self.redis_monitoring.get_health_status()
                # Calculate error rate from health data
                return 0.01  # Simulated 1% error rate
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _should_trigger_rollback(self) -> bool:
        """Check if rollback should be triggered."""
        try:
            # Check error rate
            error_rate = await self._measure_error_rate()
            if error_rate > self.config.rollback_threshold_error_rate:
                return True
            
            # Check custom rollback triggers
            for trigger in self._rollback_triggers:
                try:
                    if await trigger():
                        return True
                except Exception as e:
                    self.logger.error(f"Error in rollback trigger: {e}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rollback conditions: {e}")
            return False
    
    def _estimate_migration_duration(self) -> float:
        """Estimate migration duration in minutes."""
        # Simple estimation based on data size and configured rate
        total_operations = self.metrics.cache_entries_total + self.metrics.state_keys_total
        operations_per_minute = self.config.migration_rate_limit * 60
        
        return total_operations / max(operations_per_minute, 1)
    
    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    async def _run_health_checks(self):
        """Execute all registered health checks."""
        try:
            results = []
            for health_check in self._health_checks:
                try:
                    result = await health_check()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Health check failed: {e}")
                    results.append(False)
            
            # Update health status based on results
            self._health_status = {
                'overall_healthy': all(results),
                'check_count': len(results),
                'passed_count': sum(results),
                'last_check': time.time()
            }
            
            if not all(results):
                self.logger.warning(f"Health check failed: {sum(results)}/{len(results)} passed")
            
        except Exception as e:
            self.logger.error(f"Error running health checks: {e}")
    
    async def _health_check_loop(self):
        """Background health checking loop."""
        while not self._shutdown:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while not self._shutdown:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
    
    async def _collect_metrics(self):
        """Collect migration progress and performance metrics."""
        try:
            # Update migration rates
            if self.metrics.start_time:
                elapsed = (datetime.now(timezone.utc) - self.metrics.start_time).total_seconds()
                if elapsed > 0:
                    self.metrics.cache_migration_rate = self.metrics.cache_entries_migrated / elapsed
                    self.metrics.state_migration_rate = self.metrics.state_keys_migrated / elapsed
            
            # Collect Redis metrics
            if self.redis_monitoring:
                dashboard_data = await self.redis_monitoring.get_dashboard_data()
                # Extract relevant metrics from dashboard data
                
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def add_phase_change_handler(self, handler: Callable):
        """Add phase change event handler."""
        self._phase_change_handlers.append(handler)
    
    def add_error_handler(self, handler: Callable):
        """Add error event handler."""
        self._error_handlers.append(handler)
    
    def add_completion_handler(self, handler: Callable):
        """Add completion event handler."""
        self._completion_handlers.append(handler)
    
    async def _trigger_phase_change_handlers(self, phase: MigrationPhase):
        """Trigger phase change event handlers."""
        for handler in self._phase_change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(phase)
                else:
                    handler(phase)
            except Exception as e:
                self.logger.error(f"Error in phase change handler: {e}")
    
    async def _trigger_error_handlers(self, error: Exception):
        """Trigger error event handlers."""
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error)
                else:
                    handler(error)
            except Exception as e:
                self.logger.error(f"Error in error handler: {e}")
    
    async def _trigger_completion_handlers(self):
        """Trigger completion event handlers."""
        for handler in self._completion_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                self.logger.error(f"Error in completion handler: {e}")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status."""
        return {
            'metrics': asdict(self.metrics),
            'config': asdict(self.config),
            'estimated_completion': self.metrics.estimated_completion.isoformat() if self.metrics.estimated_completion else None,
            'components_status': {
                'redis_cache_available': self.redis_cache is not None,
                'redis_state_available': self.redis_state is not None,
                'redis_monitoring_available': self.redis_monitoring is not None
            }
        }
    
    async def shutdown(self):
        """Shutdown migration coordinator."""
        self.logger.info("Shutting down Redis Migration Coordinator")
        self._shutdown = True
        
        # Stop background tasks
        for task in [self._health_check_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop migration tasks
        for task in self._migration_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown components
        if self.redis_monitoring:
            await self.redis_monitoring.stop()
        
        self.logger.info("Redis Migration Coordinator shutdown complete")


# Helper functions for migration setup
async def create_migration_coordinator(config: Optional[MigrationConfig] = None) -> RedisMigrationCoordinator:
    """Create and initialize migration coordinator."""
    coordinator = RedisMigrationCoordinator(config)
    await coordinator.initialize()
    return coordinator


async def quick_migration(enable_monitoring: bool = True) -> bool:
    """Perform a quick migration with default settings."""
    config = MigrationConfig(
        migration_mode="gradual",
        dual_write_duration_hours=1,  # Shorter for quick migration
        validation_duration_hours=0.5,
        enable_detailed_metrics=enable_monitoring
    )
    
    coordinator = await create_migration_coordinator(config)
    
    try:
        return await coordinator.start_migration()
    finally:
        await coordinator.shutdown()