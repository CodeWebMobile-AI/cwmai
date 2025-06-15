"""
Continuous 24/7 AI Orchestrator

Replaces interval-based scheduling with intelligent continuous operation.
No artificial delays - always working on the highest priority task available.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
import re
import os

from scripts.state_manager import StateManager
from scripts.ai_brain import IntelligentAIBrain
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.github_issue_creator import GitHubIssueCreator
from scripts.task_persistence import TaskPersistence
from scripts.research_evolution_engine import ResearchEvolutionEngine
from scripts.ai_brain_factory import AIBrainFactory
from scripts.alternative_task_generator import AlternativeTaskGenerator
from scripts.enhanced_work_generator import EnhancedWorkGenerator
from scripts.github_issue_queue import GitHubIssueQueue
from scripts.mcp_redis_integration import MCPRedisIntegration


class WorkerStatus(Enum):
    """Status of parallel workers."""
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class WorkerState:
    """State of a parallel worker."""
    id: str
    status: WorkerStatus
    current_work: Optional[WorkItem] = None
    total_completed: int = 0
    total_errors: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    specialization: Optional[str] = None  # repository or task type specialization


class ContinuousOrchestrator:
    """24/7 Continuous AI Orchestrator with parallel processing."""
    
    def __init__(self, max_workers: int = 10, enable_parallel: bool = True, enable_research: bool = True,
                 enable_round_robin: bool = False):
        """Initialize the continuous orchestrator.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_parallel: Whether to enable parallel processing
            enable_research: Whether to enable research evolution engine
            enable_round_robin: Whether to enable round-robin AI provider selection
        """
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        self.enable_round_robin = enable_round_robin
        self.logger = logging.getLogger(__name__)
        
        # Core components - try Redis lock-free state manager first
        self.redis_state_manager = None
        try:
            from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager
            self.redis_state_manager = RedisLockFreeStateManager()
            self.logger.info("✓ Using Redis lock-free state manager")
        except Exception as e:
            self.logger.warning(f"Redis lock-free state manager not available: {e}")
        
        # Use regular state manager for file-based state
        try:
            from scripts.redis_state_adapter import RedisEnabledStateManager
            self.state_manager = RedisEnabledStateManager(use_redis=True)
            self.logger.info("✓ Using Redis-enabled state manager")
        except Exception as e:
            self.logger.warning(f"Redis state manager not available: {e}, using file-based")
            self.state_manager = StateManager()
        
        # Load state with repository discovery
        self.system_state = self.state_manager.load_state_with_repository_discovery()
        self.ai_brain = IntelligentAIBrain(self.system_state, {}, enable_round_robin=enable_round_robin)
        
        # Work management - try Redis work queue first
        self.redis_work_queue = None
        self.use_redis_queue = True
        try:
            from scripts.redis_work_queue import RedisWorkQueue
            self.redis_work_queue = RedisWorkQueue()
            self.logger.info("✓ Using Redis-based work queue")
        except Exception as e:
            self.logger.warning(f"Redis work queue not available: {e}, using in-memory queue")
            self.use_redis_queue = False
        
        # Fallback in-memory queue
        self.work_queue: List[WorkItem] = []
        self.work_queue_lock = asyncio.Lock()
        self.completed_work: List[WorkItem] = []
        
        # Worker management
        self.workers: Dict[str, WorkerState] = {}
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        
        # System state
        self.running = False
        self.shutdown_requested = False
        self.start_time: Optional[datetime] = None
        
        # Performance tracking
        self.metrics = {
            'total_work_completed': 0,
            'total_work_created': 0,
            'total_errors': 0,
            'average_completion_time': 0.0,
            'worker_utilization': 0.0,
            'work_per_hour': 0.0
        }
        
        # Failed task tracking with cooldown
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}  # task_key -> {count, last_failure, cooldown_until}
        self.failed_task_cooldown_base = 60  # Base cooldown in seconds
        self.failed_task_cooldown_multiplier = 2  # Exponential backoff multiplier
        
        # Repository cleanup settings
        self.cleanup_check_interval = 3600  # Check every hour
        self.last_cleanup_check = 0
        self.cleanup_manager = None
        
        # Work discovery components (will be initialized later)
        self.work_finder = None
        self.resource_manager = None
        self.task_coordinator = None
        
        # Real work execution components
        self.github_creator = GitHubIssueCreator()
        
        # GitHub issue queue for async processing
        self.github_issue_queue = GitHubIssueQueue(
            redis_client=None,  # Will use default from get_redis_client
            logger=self.logger
        )
        
        # Alternative task generator for handling duplicates
        self.alternative_task_generator = AlternativeTaskGenerator(
            ai_brain=self.ai_brain,
            logger=self.logger
        )
        
        # Enhanced work generator for continuous work generation
        self.enhanced_work_generator = EnhancedWorkGenerator(
            ai_brain=self.ai_brain,
            system_state=self.system_state,
            logger=self.logger
        )
        
        # Initialize task persistence (try Redis first)
        try:
            from redis_task_persistence import RedisTaskPersistence
            self.task_persistence = RedisTaskPersistence()
            self.logger.info("✓ Using Redis-based task persistence")
        except Exception as e:
            self.logger.warning(f"Redis task persistence not available: {e}, falling back to file-based")
            self.task_persistence = TaskPersistence()
        
        # Redis coordination components
        self.worker_coordinator = None
        self.redis_locks = None
        self.event_analytics = None
        self.event_processor = None
        self.redis_analytics = None
        self.workflow_executor = None
        
        # Initialize Research Evolution Engine
        self.research_engine = None
        self.research_task = None
        self.research_enabled = enable_research
        self.research_mode = "proactive"  # proactive or reactive
        
        # Track last state update time for each worker
        self._last_state_update: Dict[str, float] = {}
        
        # MCP-Redis integration
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
        
        if self.research_enabled:
            self._init_research_engine()
        else:
            self.logger.info("🔬 Research Evolution Engine DISABLED by configuration")
        
    async def start(self):
        """Start the continuous orchestrator."""
        self.logger.info("Starting 24/7 Continuous AI Orchestrator")
        self.start_time = datetime.now(timezone.utc)
        self.running = True
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start workers
            await self._start_workers()
            
            # Start main orchestration loop
            await self._run_main_loop()
            
        except Exception as e:
            self.logger.error(f"Error in orchestrator: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the orchestrator gracefully."""
        self.logger.info("Stopping continuous orchestrator...")
        self.shutdown_requested = True
        
        # Stop all workers
        for worker_id in list(self.workers.keys()):
            await self._stop_worker(worker_id)
        
        # Stop GitHub issue queue processor
        await self.github_issue_queue.stop_processor()
        self.logger.info("GitHub issue queue processor stopped")
        
        # Stop research engine if running
        if self.research_engine and self.research_task:
            self.logger.info("Stopping research engine...")
            self.research_engine.stop_continuous_research()
            if not self.research_task.done():
                self.research_task.cancel()
                try:
                    await self.research_task
                except asyncio.CancelledError:
                    pass
            self.logger.info("Research engine stopped")
        
        # Save final state
        await self._save_state()
        
        # Clean up Redis state
        if self.redis_state_manager:
            await self.redis_state_manager.remove_from_set("active_orchestrators", "main_orchestrator")
            await self.redis_state_manager.update_worker_field(
                "orchestrator", "status", "stopped"
            )
        
        # Clean up lock-free state manager
        if self.redis_state_manager:
            await self.redis_state_manager.close()
        
        self.running = False
        self.logger.info("Orchestrator stopped")
    
    async def _init_redis_state(self):
        """Initialize Redis state manager."""
        try:
            # Initialize lock-free state manager if available
            if self.redis_state_manager:
                await self.redis_state_manager.initialize()
                self.logger.info("✓ Redis lock-free state manager initialized")
            
            # Also initialize regular state manager's Redis if available
            if hasattr(self.state_manager, 'initialize_redis'):
                await self.state_manager.initialize_redis()
                self.logger.info("✓ Redis state manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis state: {e}")
    
    async def _init_redis_components(self):
        """Initialize Redis coordination components."""
        try:
            # Initialize Redis client
            from redis_integration.redis_client import RedisClient
            redis_client = RedisClient()
            await redis_client.connect()
            
            # 1. Initialize Worker Coordinator
            try:
                from redis_worker_coordinator import RedisWorkerCoordinator, WorkerEvent
                self.worker_coordinator = RedisWorkerCoordinator(
                    redis_client=redis_client, 
                    worker_id="orchestrator"
                )
                await self.worker_coordinator.initialize()
                
                # Store WorkerEvent for later use
                self.WorkerEvent = WorkerEvent
                
                # Register event handlers
                self.worker_coordinator.register_handler(
                    WorkerEvent.EMERGENCY_STOP, 
                    self._handle_emergency_stop
                )
                self.worker_coordinator.register_handler(
                    WorkerEvent.TASK_REASSIGN,
                    self._handle_task_reassignment
                )
                
                self.logger.info("✓ Redis worker coordination enabled")
            except Exception as e:
                self.logger.warning(f"Worker coordination not available: {e}")
                self.WorkerEvent = None
            
            # 2. Initialize Distributed Locks
            try:
                from redis_integration.redis_locks_manager import RedisLocksManager
                self.redis_locks = RedisLocksManager(redis_client)
                await self.redis_locks.start()
                self.logger.info("✓ Redis distributed locks enabled")
            except Exception as e:
                self.logger.warning(f"Distributed locks not available: {e}")
            
            # 3. Initialize Event Analytics
            try:
                from redis_event_analytics import RedisEventAnalytics
                self.event_analytics = RedisEventAnalytics()
                await self.event_analytics.initialize()
                self.logger.info("✓ Redis event analytics enabled")
            except Exception as e:
                self.logger.warning(f"Event analytics not available: {e}")
            
            # 4. Initialize Event Stream Processor
            try:
                from redis_event_stream_processor import RedisEventStreamProcessor
                self.event_processor = RedisEventStreamProcessor(redis_client)
                await self.event_processor.initialize()
                self.logger.info("✓ Redis event stream processing enabled")
            except Exception as e:
                self.logger.warning(f"Event stream processor not available: {e}")
            
            # 5. Initialize Performance Analytics
            try:
                from redis_integration.redis_analytics import RedisAnalytics
                self.redis_analytics = RedisAnalytics(redis_client)
                self.logger.info("✓ Redis performance analytics enabled")
            except Exception as e:
                self.logger.warning(f"Performance analytics not available: {e}")
            
            # 6. Initialize Workflow Executor
            try:
                from redis_distributed_workflows import RedisWorkflowEngine
                self.workflow_executor = RedisWorkflowEngine()
                await self.workflow_executor.initialize()
                self.logger.info("✓ Redis workflow orchestration enabled")
            except Exception as e:
                self.logger.warning(f"Workflow orchestration not available: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis components: {e}")
    
    async def _initialize_components(self):
        """Initialize all orchestrator components."""
        self.logger.info("Initializing orchestrator components...")
        
        # Run repository cleanup check first
        try:
            from repository_cleanup_manager import RepositoryCleanupManager
            cleanup_manager = RepositoryCleanupManager()
            cleanup_result = cleanup_manager.perform_cleanup()
            if cleanup_result['items_removed'] > 0:
                self.logger.warning(f"Cleaned up {cleanup_result['items_removed']} references to deleted repositories: {', '.join(cleanup_result['deleted_repos'])}")
                # Reload system state after cleanup
                self.system_state = self.state_manager.load_state_with_repository_discovery()
        except Exception as e:
            self.logger.warning(f"Repository cleanup check failed: {e}")
        
        # Initialize Redis components first
        await self._init_redis_state()
        await self._init_redis_components()
        
        # Initialize MCP-Redis if enabled
        if self._use_mcp:
            try:
                self.mcp_redis = MCPRedisIntegration()
                await self.mcp_redis.initialize()
                self.logger.info("✓ MCP-Redis integration enabled for orchestrator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                self._use_mcp = False
        
        # Initialize GitHub issue queue
        await self.github_issue_queue.initialize()
        await self.github_issue_queue.start_processor()
        self.logger.info("✓ GitHub issue queue processor started")
        
        # Import and initialize work finder
        try:
            from intelligent_work_finder import IntelligentWorkFinder
            self.work_finder = IntelligentWorkFinder(self.ai_brain, self.system_state)
            self.logger.info("✓ Work finder initialized")
        except ImportError as e:
            self.logger.warning(f"Work finder not available: {e}")
        
        # Import and initialize resource manager
        try:
            from resource_manager import ResourceManager
            self.resource_manager = ResourceManager()
            # Connect state manager to resource manager for metric updates
            self.resource_manager.set_state_manager(self.state_manager)
            # Start monitoring
            await self.resource_manager.start_monitoring()
            self.logger.info("✓ Resource manager initialized and monitoring started")
        except ImportError as e:
            self.logger.warning(f"Resource manager not available: {e}")
        
        # Import and initialize task coordinator
        try:
            from task_coordinator import TaskCoordinator
            self.task_coordinator = TaskCoordinator(self.ai_brain)
            self.logger.info("✓ Task coordinator initialized")
        except ImportError as e:
            self.logger.warning(f"Task coordinator not available: {e}")
        
        # Load any existing work queue
        await self._load_work_queue()
        
        self.logger.info("All components initialized")
        
        # Start research engine if enabled
        if self.research_enabled and self.research_engine:
            await self._start_research_engine()
    
    async def _start_workers(self):
        """Start parallel workers."""
        if not self.enable_parallel:
            self.max_workers = 1
            self.logger.info("Parallel processing disabled - using single worker")
        
        self.logger.info(f"Starting {self.max_workers} workers...")
        
        for i in range(self.max_workers):
            worker_id = f"worker_{i+1}"
            
            # Assign specialization to workers
            specialization = self._assign_worker_specialization(i)
            
            worker = WorkerState(
                id=worker_id,
                status=WorkerStatus.IDLE,
                specialization=specialization
            )
            
            self.workers[worker_id] = worker
            
            # Start worker task
            worker_task = asyncio.create_task(self._worker_loop(worker))
            self.worker_tasks[worker_id] = worker_task
            
            self.logger.info(f"✓ Started {worker_id} (specialization: {specialization})")
            
            # Track worker start event
            if self.event_processor:
                asyncio.create_task(self.event_processor.track_event({
                    'event_type': 'worker_started',
                    'worker_id': worker_id,
                    'specialization': specialization,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }))
        
        self.logger.info(f"All {self.max_workers} workers started")
    
    def _assign_worker_specialization(self, worker_index: int) -> str:
        """Assign specialization to workers for optimal load distribution.
        
        Better distribution strategy:
        - 30% for system/general tasks
        - 70% distributed among projects
        - At least 1 flex worker that can adapt
        """
        from repository_exclusion import RepositoryExclusion
        
        # Get all projects and filter out excluded ones
        all_projects = list(self.system_state.get('projects', {}).keys())
        projects = RepositoryExclusion.filter_excluded_repos(all_projects)
        
        # For single worker, make it general to handle everything
        if self.max_workers == 1:
            return "general"
        
        # Calculate optimal distribution
        system_workers = max(1, int(self.max_workers * 0.3))  # 30% for system tasks
        flex_workers = max(1, int(self.max_workers * 0.1))    # 10% flexible workers
        project_workers = self.max_workers - system_workers - flex_workers
        
        # Assign based on calculated distribution
        if worker_index < system_workers:
            # First 30% are system/general workers
            return "system_tasks" if worker_index == 0 else "general"
        elif worker_index < system_workers + project_workers and projects:
            # Next 60% are distributed among projects
            project_index = (worker_index - system_workers) % len(projects)
            return projects[project_index]
        else:
            # Last 10% are flexible workers
            return "general"
    
    async def _worker_loop(self, worker: WorkerState):
        """Main loop for a parallel worker."""
        self.logger.info(f"Worker {worker.id} started (specialization: {worker.specialization})")
        
        # Register worker with Redis if available
        if self.redis_state_manager:
            try:
                await self.redis_state_manager.add_to_set("active_workers", worker.id)
                # Increment active worker count
                await self.redis_state_manager.increment_counter("active_worker_count", 1)
                # Store worker state
                worker_data = {
                    'status': worker.status.value,
                    'specialization': worker.specialization,
                    'started_at': worker.started_at.isoformat(),
                    'last_activity': worker.last_activity.isoformat()
                }
                await self.redis_state_manager.update_worker_state(
                    worker.id, 
                    worker_data
                )
                self.logger.debug(f"Worker {worker.id} registered with Redis")
            except Exception as e:
                self.logger.error(f"Failed to register worker {worker.id} with Redis: {e}")
        
        while not self.shutdown_requested:
            try:
                # Update worker activity
                worker.last_activity = datetime.now(timezone.utc)
                
                # Update worker state in Redis periodically
                if self.redis_state_manager:
                    if time.time() - self._last_state_update.get(worker.id, 0) > 30:  # Update every 30 seconds
                        try:
                            worker_data = {
                                'status': worker.status.value,
                                'specialization': worker.specialization,
                                'last_activity': worker.last_activity.isoformat(),
                                'total_completed': worker.total_completed,
                                'total_errors': worker.total_errors
                            }
                            if worker.current_work:
                                worker_data['current_task'] = {
                                    'id': worker.current_work.id,
                                    'title': worker.current_work.title,
                                    'task_type': worker.current_work.task_type,
                                    'repository': worker.current_work.repository,
                                    'started_at': worker.current_work.started_at.isoformat() if worker.current_work.started_at else None
                                }
                            await self.redis_state_manager.update_worker_state(
                                worker.id, 
                                worker_data
                            )
                            self._last_state_update[worker.id] = time.time()
                        except Exception as e:
                            self.logger.error(f"Failed to update worker {worker.id} state in Redis: {e}")
                
                # Find work for this worker
                work_item = await self._find_work_for_worker(worker)
                
                if work_item:
                    # Execute work
                    await self._execute_work(worker, work_item)
                else:
                    # No work available, brief pause
                    worker.status = WorkerStatus.IDLE
                    await asyncio.sleep(1)  # Brief pause to prevent busy waiting
                
            except Exception as e:
                worker.status = WorkerStatus.ERROR
                worker.total_errors += 1
                self.metrics['total_errors'] += 1
                
                self.logger.error(f"Error in worker {worker.id}: {e}")
                self.logger.error(traceback.format_exc())
                
                # Brief pause before retrying
                await asyncio.sleep(5)
                worker.status = WorkerStatus.IDLE
        
        worker.status = WorkerStatus.SHUTTING_DOWN
        self.logger.info(f"Worker {worker.id} shutting down")
        
        # Deregister worker from Redis if available
        if self.redis_state_manager:
            try:
                await self.redis_state_manager.remove_from_set("active_workers", worker.id)
                # Decrement active worker count
                await self.redis_state_manager.increment_counter("active_worker_count", -1)
                self.logger.debug(f"Worker {worker.id} deregistered from Redis")
            except Exception as e:
                self.logger.error(f"Failed to deregister worker {worker.id} from Redis: {e}")
    
    async def _find_work_for_worker(self, worker: WorkerState) -> Optional[WorkItem]:
        """Find appropriate work for a specific worker.
        
        Strategy:
        1. First try to find work matching specialization
        2. If no specialized work, look for general tasks
        3. If still no work and worker is idle too long, allow cross-specialization
        """
        work_found = None
        
        if self.use_redis_queue and self.redis_work_queue:
            # Get work from Redis queue
            self.logger.debug(f"Looking for work for {worker.id} (specialization: {worker.specialization})")
            
            # First try specialized work
            work_items = await self.redis_work_queue.get_work_for_worker(
                worker.id,
                worker.specialization,
                count=1
            )
            
            if work_items:
                self.logger.debug(f"Found specialized work for {worker.id}")
                return work_items[0]
            
            # If no specialized work and worker has been idle, try general tasks
            idle_time = (datetime.now(timezone.utc) - worker.last_activity).total_seconds()
            if idle_time > 30:  # Idle for more than 30 seconds
                self.logger.debug(f"Worker {worker.id} idle for {idle_time}s, looking for any work")
                work_items = await self.redis_work_queue.get_work_for_worker(
                    worker.id,
                    "general",  # Look for general tasks
                    count=1
                )
                if work_items:
                    self.logger.info(f"Assigned general work to specialized worker {worker.id}")
                    return work_items[0]
        else:
            # Use in-memory queue
            async with self.work_queue_lock:
                # First pass: Look for specialized work
                for i, work_item in enumerate(self.work_queue):
                    if self._can_worker_handle_work(worker, work_item):
                        assigned_work = self.work_queue.pop(i)
                        assigned_work.assigned_worker = worker.id
                        assigned_work.started_at = datetime.now(timezone.utc)
                        return assigned_work
                
                # Second pass: If worker is idle and specialized, allow general work
                idle_time = (datetime.now(timezone.utc) - worker.last_activity).total_seconds()
                if worker.specialization != "general" and idle_time > 30:
                    for i, work_item in enumerate(self.work_queue):
                        # Allow taking system tasks or unassigned repository tasks
                        if work_item.repository is None or work_item.task_type in ["DOCUMENTATION", "TESTING"]:
                            self.logger.info(f"Idle worker {worker.id} taking general task: {work_item.title}")
                            assigned_work = self.work_queue.pop(i)
                            assigned_work.assigned_worker = worker.id
                            assigned_work.started_at = datetime.now(timezone.utc)
                            return assigned_work
        
        return None
    
    def _can_worker_handle_work(self, worker: WorkerState, work_item: WorkItem) -> bool:
        """Check if a worker can handle specific work."""
        # Check if task is in cooldown
        task_key = f"{work_item.repository}:{work_item.title}"
        if task_key in self.failed_tasks:
            failure_info = self.failed_tasks[task_key]
            current_time = time.time()
            if current_time < failure_info['cooldown_until']:
                remaining_cooldown = failure_info['cooldown_until'] - current_time
                self.logger.debug(f"Task '{work_item.title}' is in cooldown for {remaining_cooldown:.1f}s more")
                return False
            else:
                # Cooldown expired, remove from failed tasks
                del self.failed_tasks[task_key]
        
        # General workers can handle anything
        if worker.specialization == "general":
            return True
            
        # Check if worker is specialized for this work
        if worker.specialization == "system_tasks":
            # System worker handles system-wide tasks
            return work_item.repository is None
        elif worker.specialization and work_item.repository:
            # Repository-specialized worker
            return worker.specialization == work_item.repository
        else:
            # Worker has specialization but work doesn't match
            return False
    
    async def _execute_work(self, worker: WorkerState, work_item: WorkItem):
        """Execute a work item."""
        worker.status = WorkerStatus.WORKING
        worker.current_work = work_item
        
        self.logger.info(f"Worker {worker.id} executing: {work_item.title}")
        
        # Update worker status in Redis immediately
        if self.redis_state_manager:
            try:
                worker_data = {
                    'status': worker.status.value,
                    'specialization': worker.specialization,
                    'last_activity': datetime.now(timezone.utc).isoformat(),
                    'total_completed': worker.total_completed,
                    'total_errors': worker.total_errors,
                    'current_task': {
                        'id': work_item.id,
                        'title': work_item.title,
                        'task_type': work_item.task_type,
                        'repository': work_item.repository,
                        'started_at': work_item.started_at.isoformat() if work_item.started_at else None
                    }
                }
                await self.redis_state_manager.update_worker_state(worker.id, worker_data)
                self._last_state_update[worker.id] = time.time()
            except Exception as e:
                self.logger.error(f"Failed to update worker {worker.id} status in Redis: {e}")
        
        # Broadcast task start event
        if self.worker_coordinator and hasattr(self, 'WorkerEvent') and self.WorkerEvent:
            await self.worker_coordinator.broadcast_event(
                self.WorkerEvent.TASK_STARTED,
                {
                    'worker_id': worker.id,
                    'task_id': work_item.id,
                    'task_title': work_item.title,
                    'task_type': work_item.task_type
                }
            )
        
        start_time = time.time()
        
        try:
            # Execute the work item
            result = await self._perform_work(work_item)
            
            # Mark as completed
            work_item.completed_at = datetime.now(timezone.utc)
            self.completed_work.append(work_item)
            
            # Update metrics
            worker.total_completed += 1
            self.metrics['total_work_completed'] += 1
            
            completion_time = time.time() - start_time
            self._update_completion_metrics(completion_time)
            
            # Record in resource manager for efficiency calculation
            if self.resource_manager:
                self.resource_manager.record_task_completion(completion_time, success=True)
            
            # Track event for analytics
            if self.event_analytics:
                await self.event_analytics.track_event({
                    'event_type': 'task_completed',
                    'worker_id': worker.id,
                    'task_id': work_item.id,
                    'task_type': work_item.task_type,
                    'repository': work_item.repository,
                    'completion_time': completion_time,
                    'success': True
                })
            
            self.logger.info(f"Worker {worker.id} completed: {work_item.title} in {completion_time:.2f}s")
            
            # Update worker status in Redis after completion
            worker.status = WorkerStatus.IDLE
            worker.current_work = None
            if self.redis_state_manager:
                try:
                    worker_data = {
                        'status': worker.status.value,
                        'specialization': worker.specialization,
                        'last_activity': datetime.now(timezone.utc).isoformat(),
                        'total_completed': worker.total_completed,
                        'total_errors': worker.total_errors,
                        'current_task': None
                    }
                    await self.redis_state_manager.update_worker_state(worker.id, worker_data)
                    self._last_state_update[worker.id] = time.time()
                except Exception as e:
                    self.logger.error(f"Failed to update worker {worker.id} status after completion: {e}")
            
            # Broadcast task completion
            if self.worker_coordinator and hasattr(self, 'WorkerEvent') and self.WorkerEvent:
                await self.worker_coordinator.broadcast_event(
                    self.WorkerEvent.TASK_COMPLETED,
                    {
                        'worker_id': worker.id,
                        'task_id': work_item.id,
                        'task_title': work_item.title,
                        'completion_time': completion_time,
                        'success': True
                    }
                )
            
            # Look for follow-up work
            await self._generate_followup_work(work_item, result)
            
        except Exception as e:
            worker.total_errors += 1
            self.metrics['total_errors'] += 1
            
            # Categorize the error
            error_category = self._categorize_error(e)
            self.logger.error(f"Worker {worker.id} failed on {work_item.title}: [{error_category}] {e}")
            
            # Track failed task for cooldown based on error category
            task_key = f"{work_item.repository}:{work_item.title}"
            current_time = time.time()
            
            # Different cooldown strategies based on error type
            if error_category == "rate_limit":
                # Longer cooldown for rate limits
                base_cooldown = 300  # 5 minutes base
            elif error_category == "duplicate":
                # Permanent cooldown for duplicates (handled elsewhere but safety check)
                base_cooldown = 86400  # 24 hours
            elif error_category == "network":
                # Medium cooldown for network issues
                base_cooldown = 120  # 2 minutes
            else:
                # Default cooldown
                base_cooldown = self.failed_task_cooldown_base
            
            if task_key in self.failed_tasks:
                # Increment failure count and apply exponential backoff
                failure_info = self.failed_tasks[task_key]
                failure_info['count'] += 1
                cooldown_duration = base_cooldown * (self.failed_task_cooldown_multiplier ** (failure_info['count'] - 1))
                failure_info['cooldown_until'] = current_time + cooldown_duration
                failure_info['last_failure'] = current_time
                failure_info['error_category'] = error_category
                self.logger.warning(f"Task '{work_item.title}' failed {failure_info['count']} times ({error_category}). Cooldown for {cooldown_duration}s")
            else:
                # First failure
                cooldown_duration = base_cooldown
                self.failed_tasks[task_key] = {
                    'count': 1,
                    'last_failure': current_time,
                    'cooldown_until': current_time + cooldown_duration,
                    'error_category': error_category
                }
                self.logger.warning(f"Task '{work_item.title}' failed for the first time ({error_category}). Cooldown for {cooldown_duration}s")
            
            # Trigger research on repeated failures
            failure_rate = worker.total_errors / max(worker.total_completed + worker.total_errors, 1)
            if failure_rate > 0.3 and self.research_engine:  # 30% failure rate
                await self.trigger_research(
                    reason=f"High failure rate detected: {failure_rate:.1%}",
                    context={
                        'worker_id': worker.id,
                        'failure_type': type(e).__name__,
                        'failed_task': work_item.to_dict(),
                        'error_message': str(e)
                    }
                )
            
            # Record failure in resource manager
            if self.resource_manager:
                completion_time = time.time() - start_time
                self.resource_manager.record_task_completion(completion_time, success=False)
            
            # Track failure event
            if self.event_processor:
                await self.event_processor.track_event({
                    'event_type': 'task_failed',
                    'worker_id': worker.id,
                    'task_id': work_item.id,
                    'task_type': work_item.task_type,
                    'error': str(e),
                    'completion_time': completion_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        finally:
            worker.current_work = None
            worker.status = WorkerStatus.IDLE
    
    async def _perform_work(self, work_item: WorkItem) -> Dict[str, Any]:
        """Perform the actual work for a work item."""
        self.logger.info(f"🔍 _perform_work called for: {work_item.title} (type: {work_item.task_type})")
        
        # Check if this is a duplicate task
        is_duplicate = False
        if hasattr(self.task_persistence, 'is_duplicate_task'):
            # Check if it's async (Redis) or sync (file-based)
            if asyncio.iscoroutinefunction(self.task_persistence.is_duplicate_task):
                is_duplicate = await self.task_persistence.is_duplicate_task(work_item)
            else:
                is_duplicate = self.task_persistence.is_duplicate_task(work_item)
        
        if is_duplicate:
            self.logger.info(f"⏭️  Duplicate task detected: {work_item.title}")
            
            # Record skipped task
            if hasattr(self.task_persistence, 'record_skipped_task'):
                if asyncio.iscoroutinefunction(self.task_persistence.record_skipped_task):
                    await self.task_persistence.record_skipped_task(work_item.title, reason="duplicate")
                else:
                    self.task_persistence.record_skipped_task(work_item.title, reason="duplicate")
            
            # Generate alternative task with max attempts to prevent infinite loops
            self.logger.info(f"🔄 Generating alternative task for duplicate: {work_item.title}")
            
            # Track attempted alternatives to prevent infinite loops
            max_alternative_attempts = 3
            attempted_alternatives = set()
            attempted_alternatives.add(work_item.title.lower())
            
            alternative_task = None
            for attempt in range(max_alternative_attempts):
                self.logger.debug(f"Alternative generation attempt {attempt + 1}/{max_alternative_attempts}")
                
                candidate_task = await self.alternative_task_generator.generate_alternative_task(
                    work_item,
                    context={
                        'repository': work_item.repository,
                        'original_type': work_item.task_type,
                        'worker_id': work_item.assigned_worker,
                        'attempted_alternatives': list(attempted_alternatives),
                        'attempt_number': attempt + 1
                    }
                )
                
                if not candidate_task:
                    self.logger.warning(f"No alternative generated on attempt {attempt + 1}")
                    break
                
                # Check if the alternative is also a duplicate
                is_alternative_duplicate = False
                if hasattr(self.task_persistence, 'is_duplicate_task'):
                    if asyncio.iscoroutinefunction(self.task_persistence.is_duplicate_task):
                        is_alternative_duplicate = await self.task_persistence.is_duplicate_task(candidate_task)
                    else:
                        is_alternative_duplicate = self.task_persistence.is_duplicate_task(candidate_task)
                
                # Also check if we've already tried this alternative
                if candidate_task.title.lower() in attempted_alternatives:
                    self.logger.debug(f"Alternative '{candidate_task.title}' already attempted")
                    is_alternative_duplicate = True
                
                if not is_alternative_duplicate:
                    # Found a unique alternative!
                    alternative_task = candidate_task
                    self.logger.info(f"✨ Found unique alternative: {alternative_task.title}")
                    break
                else:
                    # This alternative is also a duplicate, try again
                    self.logger.debug(f"Alternative '{candidate_task.title}' is also a duplicate")
                    attempted_alternatives.add(candidate_task.title.lower())
                    
                    # Record this failed alternative
                    if hasattr(self.task_persistence, 'record_skipped_task'):
                        if asyncio.iscoroutinefunction(self.task_persistence.record_skipped_task):
                            await self.task_persistence.record_skipped_task(
                                candidate_task.title, 
                                reason="duplicate_alternative"
                            )
                        else:
                            self.task_persistence.record_skipped_task(
                                candidate_task.title, 
                                reason="duplicate_alternative"
                            )
            
            if alternative_task:
                self.logger.info(f"✨ Successfully generated alternative task: {alternative_task.title}")
                
                # Add the alternative task to the work queue
                if self.use_redis_queue and self.redis_work_queue:
                    await self.redis_work_queue.add_work(alternative_task)
                else:
                    async with self.work_queue_lock:
                        self.work_queue.insert(0, alternative_task)  # Add to front of queue
                
                # Return success with info about alternative
                return {
                    'success': True,
                    'alternative_generated': True,
                    'original_task': work_item.title,
                    'alternative_task': alternative_task.title,
                    'reason': 'duplicate_with_alternative',
                    'value_created': 0.3,  # Some value for generating alternative
                    'attempts_made': len(attempted_alternatives) - 1  # Excluding original
                }
            else:
                # Exhausted attempts or failed to generate alternatives
                self.logger.warning(
                    f"Failed to generate unique alternative for duplicate: {work_item.title} "
                    f"after {len(attempted_alternatives) - 1} attempts"
                )
                
                # Record this as a problematic task
                if hasattr(self.task_persistence, 'record_problematic_task'):
                    if asyncio.iscoroutinefunction(self.task_persistence.record_problematic_task):
                        await self.task_persistence.record_problematic_task(
                            work_item.title,
                            work_item.task_type,
                            work_item.repository
                        )
                    else:
                        self.task_persistence.record_problematic_task(
                            work_item.title,
                            work_item.task_type,
                            work_item.repository
                        )
                
                return {
                    'success': False,
                    'skipped': True,
                    'reason': 'duplicate_task_no_alternatives',
                    'value_created': 0,
                    'attempts_made': len(attempted_alternatives) - 1
                }
        
        # Check if there are any projects in the system
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        from scripts.repository_exclusion import is_excluded_repo
        active_projects = {
            name: data for name, data in all_projects.items()
            if not is_excluded_repo(name)
        }
        
        # If no active projects exist, only allow NEW_PROJECT tasks
        if not active_projects:
            self.logger.warning(f"⚠️ No active projects in system. Only NEW_PROJECT tasks allowed.")
            if work_item.task_type != 'NEW_PROJECT':
                self.logger.info(f"🔄 Converting {work_item.task_type} task to NEW_PROJECT since no projects exist")
                # Skip this task and generate a NEW_PROJECT task instead
                return {
                    'success': False,
                    'skipped': True,
                    'reason': 'no_projects_exist',
                    'message': f'Cannot execute {work_item.task_type} tasks when no projects exist',
                    'value_created': 0
                }
        
        # Route based on task type
        self.logger.info(f"📋 Routing task type: {work_item.task_type}")
        
        # Map task types to execution methods
        task_handlers = {
            'SYSTEM_IMPROVEMENT': self._execute_system_improvement_task,
            'NEW_PROJECT': self._execute_new_project_task,
            'FEATURE': self._execute_feature_task,
            'BUG_FIX': self._execute_bug_fix_task,
            'DOCUMENTATION': self._execute_documentation_task,
            'RESEARCH': self._execute_research_task,
            'TESTING': self._execute_testing_task,
            'MAINTENANCE': self._execute_maintenance_task,
            'MONITORING': self._execute_monitoring_task,
            'INTEGRATION': self._execute_integration_task,
            'REFACTORING': self._execute_refactoring_task,
            'OPTIMIZATION': self._execute_optimization_task,
        }
        
        # Get handler for task type
        handler = task_handlers.get(work_item.task_type)
        self.logger.info(f"📑 Available task handlers: {list(task_handlers.keys())}")
        self.logger.info(f"🔎 Looking for handler for task type: '{work_item.task_type}'")
        self.logger.info(f"✅ Handler found: {handler.__name__ if handler else 'None'}")
        
        if handler:
            self.logger.info(f"🚀 Using specific handler: {handler.__name__} for task type: {work_item.task_type}")
            return await handler(work_item)
        else:
            # Default behavior - try GitHub issues if available
            self.logger.warning(f"⚠️  No specific handler for task type: '{work_item.task_type}', falling back to GitHub integration")
            
            # Execute real work by creating GitHub issues
            can_create = self.github_creator.can_create_issues()
            self.logger.info(f"🔐 GitHub integration check: can_create_issues = {can_create}")
            self.logger.info(f"🔍 GitHub creator instance: {self.github_creator}")
            self.logger.info(f"🔍 GitHub creator class: {type(self.github_creator)}")
            
            if not can_create:
                self.logger.error("❌ GitHub integration NOT available!")
                self.logger.error("📊 Debugging why GitHub integration failed:")
                self.logger.error(f"   - GitHub Token exists: {bool(os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT'))}")
                self.logger.error(f"   - GitHub Repo: {os.getenv('GITHUB_REPOSITORY')}")
                self.logger.error(f"   - Current working directory: {os.getcwd()}")
                raise RuntimeError(
                    f"No handler found for task type '{work_item.task_type}' and GitHub integration is not available. "
                    "Cannot execute task."
                )
            
            self.logger.info("✅ GitHub integration IS available - will create real issue")
            # No distributed lock needed - using optimistic concurrency
            result = await self.github_creator.execute_work_item(work_item)
            
            # Record completion if successful
            if result.get('success'):
                if asyncio.iscoroutinefunction(self.task_persistence.record_completed_task):
                    await self.task_persistence.record_completed_task(work_item, result)
                else:
                    self.task_persistence.record_completed_task(work_item, result)
            
            return result
    
    async def _execute_feature_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a feature development task."""
        # Integrate with existing task manager
        from task_manager import TaskManager, TaskType, TaskPriority as LegacyPriority
        
        # Don't create task manager with default repository
        task_manager = TaskManager(repository=work_item.repository if hasattr(work_item, 'repository') else None)
        
        # Convert our work item to legacy task format
        priority_map = {
            TaskPriority.CRITICAL: LegacyPriority.CRITICAL,
            TaskPriority.HIGH: LegacyPriority.HIGH,
            TaskPriority.MEDIUM: LegacyPriority.MEDIUM,
            TaskPriority.LOW: LegacyPriority.LOW,
            TaskPriority.BACKGROUND: LegacyPriority.LOW
        }
        
        task = task_manager.create_task(
            task_type=TaskType.FEATURE,
            title=work_item.title,
            description=work_item.description,
            priority=priority_map[work_item.priority],
            repository=work_item.repository
        )
        
        # Create GitHub issue for the task
        issue_number = task_manager.create_github_issue(task)
        
        return {
            'success': True,
            'task_id': task['id'],
            'issue_number': issue_number,
            'value_created': 1.0
        }
    
    async def _execute_bug_fix_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a bug fix task."""
        # Similar to feature task but with BUG_FIX type
        from task_manager import TaskManager, TaskType, TaskPriority as LegacyPriority
        
        # Don't create task manager with default repository
        task_manager = TaskManager(repository=work_item.repository if hasattr(work_item, 'repository') else None)
        priority_map = {
            TaskPriority.CRITICAL: LegacyPriority.CRITICAL,
            TaskPriority.HIGH: LegacyPriority.HIGH,
            TaskPriority.MEDIUM: LegacyPriority.MEDIUM,
            TaskPriority.LOW: LegacyPriority.LOW,
            TaskPriority.BACKGROUND: LegacyPriority.LOW
        }
        
        task = task_manager.create_task(
            task_type=TaskType.BUG_FIX,
            title=work_item.title,
            description=work_item.description,
            priority=priority_map[work_item.priority],
            repository=work_item.repository
        )
        
        issue_number = task_manager.create_github_issue(task)
        
        return {
            'success': True,
            'task_id': task['id'],
            'issue_number': issue_number,
            'value_created': 0.8
        }
    
    async def _execute_documentation_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a documentation task."""
        from task_manager import TaskManager, TaskType, TaskPriority as LegacyPriority
        
        # Don't create task manager with default repository
        task_manager = TaskManager(repository=work_item.repository if hasattr(work_item, 'repository') else None)
        priority_map = {
            TaskPriority.CRITICAL: LegacyPriority.CRITICAL,
            TaskPriority.HIGH: LegacyPriority.HIGH,
            TaskPriority.MEDIUM: LegacyPriority.MEDIUM,
            TaskPriority.LOW: LegacyPriority.LOW,
            TaskPriority.BACKGROUND: LegacyPriority.LOW
        }
        
        task = task_manager.create_task(
            task_type=TaskType.DOCUMENTATION,
            title=work_item.title,
            description=work_item.description,
            priority=priority_map[work_item.priority],
            repository=work_item.repository
        )
        
        issue_number = task_manager.create_github_issue(task)
        
        return {
            'success': True,
            'task_id': task['id'],
            'issue_number': issue_number,
            'value_created': 0.6
        }
    
    async def _execute_research_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a research task."""
        # For now, research tasks are logged but not executed
        # The research evolution engine handles actual research
        self.logger.info(f"🔬 Research task logged: {work_item.title}")
        
        # If we have research engine, trigger it
        if self.research_engine and self.research_enabled:
            await self.trigger_research(
                reason=f"Research task: {work_item.title}",
                context={
                    'task_type': 'RESEARCH',
                    'research_area': work_item.metadata.get('research_area', 'general'),
                    'description': work_item.description,
                    'priority': work_item.priority.name
                }
            )
        
        return {
            'success': True,
            'message': f"Research task logged: {work_item.title}",
            'triggered_research': self.research_enabled,
            'value_created': 0.3
        }
    
    async def _execute_system_improvement_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a system improvement task using intelligent improvements."""
        try:
            # Get configuration from environment
            import os
            
            # Check if we're in development mode
            is_development = os.getenv('NODE_ENV', 'production').lower() == 'development'
            if is_development:
                self.logger.info("⚠️ Skipping SYSTEM_IMPROVEMENT task in development mode")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'development_mode',
                    'message': 'System improvement tasks are disabled in development mode',
                    'value_created': 0
                }
            
            use_intelligent = os.getenv('INTELLIGENT_IMPROVEMENT_ENABLED', 'true').lower() == 'true'
            staging_enabled = os.getenv('SELF_IMPROVEMENT_STAGING_ENABLED', 'true').lower() == 'true'
            auto_validate = os.getenv('SELF_IMPROVEMENT_AUTO_VALIDATE', 'true').lower() == 'true'
            auto_apply = os.getenv('SELF_IMPROVEMENT_AUTO_APPLY_VALIDATED', 'false').lower() == 'true'
            max_daily = int(os.getenv('SELF_IMPROVEMENT_MAX_DAILY', '3'))
            
            # Check if we have AI brain and should use intelligent improver
            if use_intelligent and hasattr(self, 'ai_brain') and self.ai_brain:
                # Use intelligent self-improver
                from intelligent_self_improver import IntelligentSelfImprover
                
                improver = IntelligentSelfImprover(
                    ai_brain=self.ai_brain,
                    repo_path="/workspaces/cwmai",
                    staging_enabled=staging_enabled
                )
                
                # Run improvement cycle
                result = await improver.run_improvement_cycle(
                    max_improvements=max_daily,
                    auto_apply=auto_apply
                )
                
                return {
                    'success': result['success'],
                    'staged': result.get('staged', 0),
                    'validated': result.get('validated', 0),
                    'applied': result.get('applied', 0),
                    'total_opportunities': result.get('opportunities', 0),
                    'confidence_score': result.get('confidence_score', 0),
                    'value_created': result.get('staged', 0) * 0.3 + result.get('validated', 0) * 0.5 + result.get('applied', 0) * 1.0,
                    'learning_insights': result.get('learning_insights', {}),
                    'mode': 'intelligent'
                }
            
            # Otherwise use staged improvement system
            from staged_self_improver import StagedSelfImprover
            from staged_improvement_monitor import StagedImprovementMonitor
            from progressive_confidence import ProgressiveConfidence, RiskLevel
            
            if not staging_enabled:
                # Fallback to old behavior
                from safe_self_improver import SafeSelfImprover
                improver = SafeSelfImprover()
                opportunities = improver.analyze_improvement_opportunities()
                
                return {
                    'success': True,
                    'improvement_opportunities': opportunities,
                    'value_created': 0.5,
                    'mode': 'analysis_only'
                }
            
            # Initialize components
            improver = StagedSelfImprover(
                repo_path="/workspaces/cwmai",
                max_changes_per_day=max_daily
            )
            monitor = StagedImprovementMonitor("/workspaces/cwmai")
            confidence = ProgressiveConfidence("/workspaces/cwmai")
            
            # Step 1: Find improvement opportunities
            self.logger.info("Finding improvement opportunities (regex-based)...")
            opportunities = improver.analyze_improvement_opportunities()
            
            if not opportunities:
                self.logger.info("No improvement opportunities found")
                return {
                    'success': True,
                    'message': 'No improvements needed',
                    'value_created': 0.1,
                    'mode': 'staged_regex'
                }
            
            self.logger.info(f"Found {len(opportunities)} improvement opportunities")
            
            # Step 2: Stage improvements (limited by daily max)
            batch_size = min(3, len(opportunities))  # Max 3 per cycle
            staged_ids = await improver.stage_batch_improvements(
                opportunities, 
                max_batch=batch_size
            )
            
            if not staged_ids:
                return {
                    'success': True,
                    'message': 'No improvements could be staged',
                    'value_created': 0.2
                }
            
            # Step 3: Validate if auto-validate is enabled
            validated_count = 0
            if auto_validate:
                self.logger.info(f"Validating {len(staged_ids)} staged improvements...")
                validation_results = await improver.validate_batch(staged_ids)
                
                validated_count = sum(
                    1 for r in validation_results.values() 
                    if r.get('ready_to_apply', False)
                )
                
                self.logger.info(f"{validated_count}/{len(staged_ids)} improvements validated successfully")
            
            # Step 4: Auto-apply if enabled and confidence allows
            applied_count = 0
            if auto_apply and validated_count > 0:
                validated_improvements = improver.get_staged_improvements('validated')
                
                for imp in validated_improvements:
                    # Assess risk level
                    modification_details = {
                        'lines_changed': len(imp.modification.changes),
                        'target_file': imp.modification.target_file,
                        'complexity_change': imp.metadata.get('complexity_change', 0)
                    }
                    risk_level = confidence.assess_risk_level(
                        imp.modification.type,
                        modification_details
                    )
                    
                    # Check if we should auto-apply
                    should_apply, reason = confidence.should_auto_apply(
                        imp.modification.type,
                        risk_level
                    )
                    
                    if should_apply:
                        # Start monitoring
                        metrics = await monitor.start_monitoring(
                            imp.metadata['staging_id'],
                            imp.modification.target_file
                        )
                        
                        # Apply improvement
                        success = await improver.apply_staged_improvement(
                            imp.metadata['staging_id']
                        )
                        
                        # Stop monitoring
                        final_metrics = await monitor.stop_monitoring(
                            imp.metadata['staging_id'],
                            improvement_applied=success
                        )
                        
                        # Record outcome
                        confidence.record_outcome(
                            staging_id=imp.metadata['staging_id'],
                            improvement_type=imp.modification.type,
                            risk_level=risk_level,
                            success=success,
                            performance_impact=final_metrics.verdict == 'improved' and 0.5 or -0.1
                        )
                        
                        if success:
                            applied_count += 1
                            self.logger.info(f"✅ Auto-applied improvement: {imp.modification.description}")
                    else:
                        self.logger.info(f"⏸️  Manual review required: {reason}")
            
            # Generate report
            report = improver.generate_staging_report()
            
            return {
                'success': True,
                'staged': len(staged_ids),
                'validated': validated_count,
                'applied': applied_count,
                'total_opportunities': len(opportunities),
                'confidence_score': confidence.metrics.confidence_score,
                'value_created': 0.3 * len(staged_ids) + 0.5 * validated_count + 1.0 * applied_count,
                'report': report
            }
            
        except Exception as e:
            self.logger.error(f"Error in staged improvement system: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Fallback to simple analysis
            return {
                'success': True,
                'message': f"System improvement logged: {work_item.title}",
                'error': str(e),
                'value_created': 0.1
            }
    
    async def _execute_new_project_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a new project creation task."""
        self.logger.info(f"📂 Executing new project task: {work_item.title}")
        
        try:
            # Initialize project creator with GitHub token
            from scripts.project_creator import ProjectCreator
            import os
            
            github_token = os.getenv('CLAUDE_PAT') or os.getenv('GITHUB_TOKEN')
            if not github_token:
                self.logger.error("No GitHub token available for project creation")
                return {
                    'success': False,
                    'status': 'failed',
                    'error': 'No GitHub token available',
                    'message': 'Cannot create project without GitHub credentials'
                }
            
            project_creator = ProjectCreator(github_token, self.ai_brain)
            
            # Convert WorkItem to task dict format expected by ProjectCreator
            # IMPORTANT: Include metadata so project gets full customization
            task_dict = {
                'title': work_item.title,
                'description': work_item.description,
                'requirements': getattr(work_item, 'requirements', []),
                'type': 'NEW_PROJECT',
                'metadata': getattr(work_item, 'metadata', {})  # Pass metadata to ensure customization
            }
            
            # Log metadata availability
            metadata = task_dict.get('metadata', {})
            self.logger.info(f"📊 Task metadata for project creation:")
            self.logger.info(f"  - Has selected_project: {bool(metadata.get('selected_project'))}")
            self.logger.info(f"  - Has architecture: {bool(metadata.get('architecture'))}")
            
            # Create the new project using the Laravel React starter kit
            self.logger.info(f"🚀 Creating new project from Laravel React starter kit")
            result = await project_creator.create_project(task_dict)
            
            if result.get('success'):
                project_name = result.get('project_name')
                repo_name = result.get('repo_name')
                repo_full_name = f"{ProjectCreator.ORGANIZATION}/{repo_name}"
                
                self.logger.info(f"✅ Successfully created new project: {repo_full_name}")
                
                # Add the new repository to system state with full details
                if self.state_manager:
                    state = self.state_manager.load_state()
                    if 'projects' not in state:
                        state['projects'] = {}
                    
                    # Extract venture analyst research and architecture from work item metadata
                    metadata = getattr(work_item, 'metadata', {})
                    selected_project = metadata.get('selected_project', {})
                    architecture = metadata.get('architecture', {})
                    
                    # Store comprehensive project information
                    state['projects'][repo_name] = {
                        'status': 'active',
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'last_activity': datetime.now(timezone.utc).isoformat(),
                        'project_type': 'laravel-react',
                        'auto_created': True,
                        'full_name': repo_full_name,
                        'description': result.get('description', ''),
                        'venture_analysis': {
                            'problem_solved': selected_project.get('problem_solved', ''),
                            'target_audience': selected_project.get('target_audience', ''),
                            'market_opportunity': selected_project.get('market_opportunity', ''),
                            'monetization_strategy': selected_project.get('monetization_strategy', ''),
                            'competitive_advantage': selected_project.get('competitive_advantage', ''),
                            'key_features': selected_project.get('key_features', []),
                            'project_goal': selected_project.get('project_goal', '')
                        },
                        'architecture': {
                            'design_system': architecture.get('foundational_architecture', {}).get('design_system', {}),
                            'database_schema': architecture.get('foundational_architecture', {}).get('database_schema', {}),
                            'api_architecture': architecture.get('foundational_architecture', {}).get('api_architecture', {}),
                            'authentication': architecture.get('foundational_architecture', {}).get('authentication', {}),
                            'deployment': architecture.get('foundational_architecture', {}).get('deployment', {}),
                            'performance': architecture.get('foundational_architecture', {}).get('performance', {}),
                            'testing_strategy': architecture.get('foundational_architecture', {}).get('testing_strategy', {}),
                            'core_entities': selected_project.get('core_entities', [])
                        },
                        'customizations': result.get('customizations', {}),
                        'initial_issues': result.get('initial_issues', [])
                    }
                    self.state_manager.update_state({'projects': state['projects']})
                
                # Create initial setup tasks for the new project
                if repo_full_name:
                    # Generate follow-up work for Laravel+React project setup
                    setup_tasks = [
                        WorkItem(
                            id=f"setup_{repo_name}_env",
                            title=f"Configure environment variables for {repo_name}",
                            description="Set up .env file with database credentials, API keys, and other configuration",
                            task_type="FEATURE",
                            priority=TaskPriority.CRITICAL,
                            repository=repo_full_name
                        ),
                        WorkItem(
                            id=f"setup_{repo_name}_database",
                            title=f"Set up database migrations for {repo_name}",
                            description="Create initial database schema and run Laravel migrations",
                            task_type="FEATURE",
                            priority=TaskPriority.HIGH,
                            repository=repo_full_name
                        ),
                        WorkItem(
                            id=f"setup_{repo_name}_ci",
                            title=f"Set up CI/CD pipeline for {repo_name}",
                            description="Configure GitHub Actions for Laravel tests, React builds, and deployment",
                            task_type="INTEGRATION",
                            priority=TaskPriority.HIGH,
                            repository=repo_full_name
                        ),
                        WorkItem(
                            id=f"setup_{repo_name}_api",
                            title=f"Create initial API endpoints for {repo_name}",
                            description="Set up RESTful API structure with authentication and basic CRUD operations",
                            task_type="FEATURE",
                            priority=TaskPriority.HIGH,
                            repository=repo_full_name
                        ),
                        WorkItem(
                            id=f"setup_{repo_name}_components",
                            title=f"Create React component structure for {repo_name}",
                            description="Set up component hierarchy, routing, and state management",
                            task_type="FEATURE",
                            priority=TaskPriority.MEDIUM,
                            repository=repo_full_name
                        ),
                        WorkItem(
                            id=f"setup_{repo_name}_docs",
                            title=f"Create comprehensive documentation for {repo_name}",
                            description="Document API endpoints, component structure, and development workflow",
                            task_type="DOCUMENTATION",
                            priority=TaskPriority.MEDIUM,
                            repository=repo_full_name
                        )
                    ]
                    
                    # Queue the setup tasks
                    async with self.work_queue_lock:
                        self.work_queue.extend(setup_tasks)
                        self.metrics['total_work_created'] += len(setup_tasks)
                    
                    self.logger.info(f"📋 Queued {len(setup_tasks)} Laravel+React setup tasks for {repo_name}")
                
                return {
                    'success': True,
                    'status': 'completed',
                    'project_name': project_name,
                    'repository': repo_full_name,
                    'url': result.get('repo_url'),
                    'message': f"Successfully created Laravel+React project: {repo_full_name}",
                    'value_created': 10.0,  # High value for new project creation
                    'initial_issues': result.get('initial_issues', []),
                    'customizations': result.get('customizations', {})
                }
            else:
                self.logger.error(f"❌ Failed to create project: {result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error'),
                    'message': f"Failed to create project from Laravel React starter kit"
                }
                
        except Exception as e:
            self.logger.error(f"Error creating new project: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'status': 'failed',
                'error': str(e),
                'message': f"Error creating project: {str(e)}"
            }
    
    async def _execute_testing_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a testing task."""
        self.logger.info(f"🧪 Executing testing task: {work_item.title}")
        
        # Create GitHub issue for testing task
        return await self._create_github_issue_for_task(work_item)
    
    async def _execute_maintenance_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a maintenance task."""
        self.logger.info(f"🔧 Executing maintenance task: {work_item.title}")
        
        # Check if this is a repository customization fix
        if work_item.metadata.get('fix_type') == 'repository_customization':
            return await self._execute_repository_fix(work_item)
        
        # Create GitHub issue for other maintenance tasks
        return await self._create_github_issue_for_task(work_item)
    
    async def _execute_repository_fix(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute repository customization fixes."""
        try:
            self.logger.info(f"🔧 Executing repository fix for: {work_item.repository}")
            
            # Import the fixer
            from fix_repository_customizations import RepositoryCustomizationFixer
            
            # Initialize the fixer with AI brain
            fixer = RepositoryCustomizationFixer(
                github_token=os.environ.get('GITHUB_TOKEN'),
                organization='CodeWebMobile-AI',
                ai_brain=self.ai_brain
            )
            
            # Fix the specific repository
            results = await fixer.fix_repository(work_item.repository)
            
            if results and results.get('status') == 'fixed':
                fixes_applied = results.get('fixes_applied', [])
                success_count = sum(1 for fix in fixes_applied if fix.get('result') == 'success')
                
                self.logger.info(f"✅ Repository fix completed for {work_item.repository}: {success_count} fixes applied")
                
                return {
                    'status': 'success',
                    'fixes_applied': fixes_applied,
                    'message': f"Applied {success_count} customization fixes to {work_item.repository}"
                }
            else:
                message = results.get('message', f"Failed to apply fixes to {work_item.repository}") if results else f"Failed to apply fixes to {work_item.repository}"
                return {
                    'status': 'failed',
                    'message': message
                }
                
        except Exception as e:
            self.logger.error(f"Error executing repository fix: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': f"Error fixing repository {work_item.repository}: {e}"
            }
    
    async def _execute_monitoring_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a monitoring task."""
        self.logger.info(f"📊 Executing monitoring task: {work_item.title}")
        
        # Create GitHub issue for monitoring task
        return await self._create_github_issue_for_task(work_item)
    
    async def _execute_integration_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute an integration task."""
        self.logger.info(f"🔗 Executing integration task: {work_item.title}")
        
        # Create GitHub issue for integration task
        return await self._create_github_issue_for_task(work_item)
    
    async def _execute_refactoring_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a refactoring task."""
        self.logger.info(f"♻️ Executing refactoring task: {work_item.title}")
        
        # Create GitHub issue for refactoring task
        return await self._create_github_issue_for_task(work_item)
    
    async def _execute_optimization_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute an optimization task."""
        self.logger.info(f"⚡ Executing optimization task: {work_item.title}")
        
        # Create GitHub issue for optimization task
        return await self._create_github_issue_for_task(work_item)
    
    async def _create_github_issue_for_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Create GitHub issue for any task type."""
        self.logger.info(f"📋 Queueing GitHub issue for task: {work_item.title}")
        
        # Generate a task ID for tracking
        task_id = f"task_{int(time.time())}_{work_item.id}"
        
        # Queue the GitHub issue creation
        queue_id = await self.github_issue_queue.add_issue(work_item, task_id)
        
        # Record task completion immediately
        result = {
            'success': True,
            'status': 'completed',
            'github_issue': 'queued',
            'github_queue_id': queue_id,
            'task_id': task_id,
            'message': f"Task completed and queued for GitHub issue creation",
            'value_created': 1.0  # Full value since task is complete
        }
        
        # Record in persistence
        if asyncio.iscoroutinefunction(self.task_persistence.record_completed_task):
            await self.task_persistence.record_completed_task(work_item, result)
        else:
            self.task_persistence.record_completed_task(work_item, result)
        
        self.logger.info(f"✅ Task completed and GitHub issue queued: {work_item.title}")
        return result
        
        # OLD CODE - Keeping for reference but not used
        # Check if GitHub integration is available
        if not self.github_creator.can_create_issues():
            self.logger.error("❌ GitHub integration NOT available!")
            self.logger.error("📊 GitHub configuration check:")
            self.logger.error(f"   - GITHUB_TOKEN exists: {bool(os.getenv('GITHUB_TOKEN'))}")
            self.logger.error(f"   - CLAUDE_PAT exists: {bool(os.getenv('CLAUDE_PAT'))}")
            self.logger.error(f"   - GITHUB_REPOSITORY: {os.getenv('GITHUB_REPOSITORY')}")
            raise RuntimeError(
                "GitHub integration is required but not available. "
                "Please ensure GITHUB_TOKEN/CLAUDE_PAT and GITHUB_REPOSITORY environment variables are set."
            )
        
        self.logger.info("✅ GitHub integration available - creating issue")
        result = await self.github_creator.execute_work_item(work_item)
        
        # Record completion if successful
        if result.get('success'):
            if asyncio.iscoroutinefunction(self.task_persistence.record_completed_task):
                await self.task_persistence.record_completed_task(work_item, result)
            else:
                self.task_persistence.record_completed_task(work_item, result)
        
        return result
    
    async def _log_task_only(self, work_item: WorkItem) -> Dict[str, Any]:
        """Log task without creating GitHub issue (fallback mode)."""
        self.logger.info(f"📝 Logging task (no GitHub integration): {work_item.title}")
        
        # Record as completed even without GitHub issue
        result = {
            'success': True,
            'message': f"Task logged: {work_item.title}",
            'value_created': 0.1,  # Minimal value for logged-only tasks
            'mode': 'logged_only'
        }
        
        # Still record in persistence to prevent duplicates
        if asyncio.iscoroutinefunction(self.task_persistence.record_completed_task):
            await self.task_persistence.record_completed_task(work_item, result)
        else:
            self.task_persistence.record_completed_task(work_item, result)
        
        return result
    
    async def _generate_followup_work(self, completed_work: WorkItem, result: Dict[str, Any]):
        """Generate follow-up work based on completed task."""
        if not result.get('success', False):
            return
        
        # Use workflow orchestration for complex feature development
        if self.workflow_executor and completed_work.task_type == "FEATURE":
            await self._create_feature_workflow(completed_work, result)
            return
        
        followup_items = []
        
        # Generate follow-up based on task type
        if completed_work.task_type == "FEATURE":
            # Add testing and documentation follow-ups
            followup_items.extend([
                WorkItem(
                    id=f"test_{completed_work.id}",
                    task_type="TESTING",
                    title=f"Test {completed_work.title}",
                    description=f"Create comprehensive tests for {completed_work.title}",
                    priority=TaskPriority.HIGH,
                    repository=completed_work.repository,
                    estimated_cycles=2,
                    dependencies=[completed_work.id]
                ),
                WorkItem(
                    id=f"doc_{completed_work.id}",
                    task_type="DOCUMENTATION",
                    title=f"Document {completed_work.title}",
                    description=f"Update documentation for {completed_work.title}",
                    priority=TaskPriority.MEDIUM,
                    repository=completed_work.repository,
                    estimated_cycles=1,
                    dependencies=[completed_work.id]
                )
            ])
        
        elif completed_work.task_type == "BUG_FIX":
            # Add testing follow-up for bug fixes
            followup_items.append(
                WorkItem(
                    id=f"test_{completed_work.id}",
                    task_type="TESTING",
                    title=f"Regression test for {completed_work.title}",
                    description=f"Add regression tests to prevent {completed_work.title} from recurring",
                    priority=TaskPriority.HIGH,
                    repository=completed_work.repository,
                    estimated_cycles=1,
                    dependencies=[completed_work.id]
                )
            )
        
        # Add follow-up work to queue
        if followup_items:
            async with self.work_queue_lock:
                self.work_queue.extend(followup_items)
                self.metrics['total_work_created'] += len(followup_items)
            
            self.logger.info(f"Generated {len(followup_items)} follow-up tasks for {completed_work.title}")
    
    async def _create_feature_workflow(self, feature_work: WorkItem, result: Dict[str, Any]):
        """Create a comprehensive workflow for feature development using Redis workflow orchestration."""
        try:
            from redis_distributed_workflows import WorkflowDefinition, WorkflowTask, WorkflowTrigger
            
            # Create workflow definition for feature development
            workflow_def = WorkflowDefinition(
                workflow_id=f"feature_dev_{feature_work.id}",
                name=f"Feature Development: {feature_work.title}",
                description=f"Complete development workflow for {feature_work.title}",
                trigger=WorkflowTrigger.MANUAL,  # Triggered manually after feature completion
                tasks=[
                    # Phase 1: Initial implementation is done (the feature_work)
                    
                    # Phase 2: Code review and refinement
                    WorkflowTask(
                        task_id=f"review_{feature_work.id}",
                        name="Code Review",
                        task_type="CODE_REVIEW",
                        parameters={
                            'feature_id': feature_work.id,
                            'feature_title': feature_work.title,
                            'repository': feature_work.repository,
                            'pr_number': result.get('pr_number')
                        },
                        dependencies=[],
                        timeout_seconds=7200  # 2 hours
                    ),
                    
                    # Phase 3: Unit testing
                    WorkflowTask(
                        task_id=f"unit_test_{feature_work.id}",
                        name="Unit Testing",
                        task_type="TESTING",
                        parameters={
                            'test_type': 'unit',
                            'feature_id': feature_work.id,
                            'feature_title': feature_work.title,
                            'repository': feature_work.repository
                        },
                        dependencies=[f"review_{feature_work.id}"],
                        timeout_seconds=3600
                    ),
                    
                    # Phase 4: Integration testing
                    WorkflowTask(
                        task_id=f"integration_test_{feature_work.id}",
                        name="Integration Testing",
                        task_type="TESTING",
                        parameters={
                            'test_type': 'integration',
                            'feature_id': feature_work.id,
                            'feature_title': feature_work.title,
                            'repository': feature_work.repository
                        },
                        dependencies=[f"unit_test_{feature_work.id}"],
                        timeout_seconds=5400  # 1.5 hours
                    ),
                    
                    # Phase 5: Documentation
                    WorkflowTask(
                        task_id=f"doc_{feature_work.id}",
                        name="Documentation",
                        task_type="DOCUMENTATION",
                        parameters={
                            'feature_id': feature_work.id,
                            'feature_title': feature_work.title,
                            'repository': feature_work.repository,
                            'doc_types': ['api', 'user_guide', 'changelog']
                        },
                        dependencies=[f"review_{feature_work.id}"],
                        timeout_seconds=3600
                    ),
                    
                    # Phase 6: Performance testing
                    WorkflowTask(
                        task_id=f"perf_test_{feature_work.id}",
                        name="Performance Testing",
                        task_type="PERFORMANCE_TEST",
                        parameters={
                            'feature_id': feature_work.id,
                            'feature_title': feature_work.title,
                            'repository': feature_work.repository,
                            'baseline_metrics': result.get('performance_baseline', {})
                        },
                        dependencies=[f"integration_test_{feature_work.id}"],
                        timeout_seconds=7200
                    ),
                    
                    # Phase 7: Deployment preparation
                    WorkflowTask(
                        task_id=f"deploy_prep_{feature_work.id}",
                        name="Deployment Preparation",
                        task_type="DEPLOYMENT_PREP",
                        parameters={
                            'feature_id': feature_work.id,
                            'feature_title': feature_work.title,
                            'repository': feature_work.repository,
                            'deployment_checklist': [
                                'migration_scripts',
                                'feature_flags',
                                'rollback_plan',
                                'monitoring_alerts'
                            ]
                        },
                        dependencies=[
                            f"perf_test_{feature_work.id}",
                            f"doc_{feature_work.id}"
                        ],
                        timeout_seconds=3600
                    )
                ],
                metadata={
                    'feature_work_item': feature_work.to_dict(),
                    'initial_result': result,
                    'estimated_total_hours': 12,
                    'auto_assign': True
                }
            )
            
            # Register and start the workflow
            await self.workflow_executor.register_workflow(workflow_def)
            workflow_id = await self.workflow_executor.start_workflow(
                workflow_id=workflow_def.workflow_id,
                initial_context={'feature_work': feature_work.to_dict()}
            )
            
            self.logger.info(f"Created feature development workflow {workflow_id} for {feature_work.title}")
            
            # Track workflow creation event
            if self.event_processor:
                await self.event_processor.track_event({
                    'event_type': 'workflow_created',
                    'workflow_id': workflow_id,
                    'workflow_type': 'feature_development',
                    'feature_id': feature_work.id,
                    'feature_title': feature_work.title,
                    'task_count': len(workflow_def.tasks),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
        except Exception as e:
            self.logger.error(f"Error creating feature workflow: {e}")
            # Fall back to traditional follow-up work generation
            await self._generate_followup_work_traditional(feature_work, result)
    
    async def _generate_followup_work_traditional(self, completed_work: WorkItem, result: Dict[str, Any]):
        """Traditional follow-up work generation (fallback for when workflows aren't available)."""
        followup_items = []
        
        if completed_work.task_type == "FEATURE":
            followup_items.extend([
                WorkItem(
                    id=f"test_{completed_work.id}",
                    task_type="TESTING",
                    title=f"Test {completed_work.title}",
                    description=f"Create comprehensive tests for {completed_work.title}",
                    priority=TaskPriority.HIGH,
                    repository=completed_work.repository,
                    estimated_cycles=2,
                    dependencies=[completed_work.id]
                ),
                WorkItem(
                    id=f"doc_{completed_work.id}",
                    task_type="DOCUMENTATION",
                    title=f"Document {completed_work.title}",
                    description=f"Update documentation for {completed_work.title}",
                    priority=TaskPriority.MEDIUM,
                    repository=completed_work.repository,
                    estimated_cycles=1,
                    dependencies=[completed_work.id]
                )
            ])
        
        if followup_items:
            if self.use_redis_queue and self.redis_work_queue:
                await self.redis_work_queue.add_work_batch(followup_items)
                self.metrics['total_work_created'] += len(followup_items)
            else:
                async with self.work_queue_lock:
                    self.work_queue.extend(followup_items)
                    self.metrics['total_work_created'] += len(followup_items)
            
            self.logger.info(f"Generated {len(followup_items)} traditional follow-up tasks for {completed_work.title}")
    
    async def _check_repository_cleanup(self):
        """Periodically check for and cleanup deleted repositories."""
        current_time = time.time()
        
        # Only check periodically based on interval
        if current_time - self.last_cleanup_check < self.cleanup_check_interval:
            return
        
        self.last_cleanup_check = current_time
        
        try:
            # Initialize cleanup manager if needed
            if not self.cleanup_manager:
                from repository_cleanup_manager import RepositoryCleanupManager
                self.cleanup_manager = RepositoryCleanupManager()
            
            # Perform cleanup check
            cleanup_result = self.cleanup_manager.perform_cleanup()
            
            if cleanup_result['items_removed'] > 0:
                self.logger.warning(
                    f"Automatic cleanup: removed {cleanup_result['items_removed']} references to "
                    f"deleted repositories: {', '.join(cleanup_result['deleted_repos'])}"
                )
                
                # Reload system state after cleanup
                self.system_state = self.state_manager.load_state_with_repository_discovery()
                
                # Clear any in-memory work items for deleted repos
                if not self.use_redis_queue:
                    deleted_repos = set(cleanup_result['deleted_repos'])
                    original_count = len(self.work_queue)
                    self.work_queue = [
                        item for item in self.work_queue
                        if item.repository not in deleted_repos
                    ]
                    removed_count = original_count - len(self.work_queue)
                    if removed_count > 0:
                        self.logger.info(f"Removed {removed_count} in-memory work items for deleted repositories")
                
        except Exception as e:
            self.logger.error(f"Repository cleanup check failed: {e}")
    
    async def _check_repositories_needing_fixes(self) -> List[WorkItem]:
        """Check for repositories that need customization fixes."""
        try:
            # Import needed for checking repositories
            from fix_repository_customizations import RepositoryCustomizationFixer
            
            # Only check every 30 minutes to avoid too frequent checks
            current_time = datetime.now(timezone.utc)
            if hasattr(self, '_last_repo_fix_check'):
                if (current_time - self._last_repo_fix_check).total_seconds() < 1800:
                    return []
            
            self._last_repo_fix_check = current_time
            
            self.logger.info("Checking repositories for customization issues")
            
            # Initialize the fixer
            fixer = RepositoryCustomizationFixer(
                github_token=os.environ.get('GITHUB_TOKEN'),
                organization='CodeWebMobile-AI'
            )
            
            # Check repositories
            issues = await fixer.check_repositories()
            
            work_items = []
            
            # Create work items for repositories with issues
            for repo_name, repo_issues in issues.items():
                if repo_issues:
                    # Skip excluded repositories
                    if repo_name in ['cwmai', '.github']:
                        continue
                    
                    # Create a maintenance task to fix the repository
                    work_item = WorkItem(
                        id=f"repo-fix-{repo_name}-{int(current_time.timestamp())}",
                        task_type="maintenance",
                        title=f"Fix customization issues in {repo_name}",
                        description=f"Repository {repo_name} has the following issues that need fixing: {', '.join(issue['type'] for issue in repo_issues)}",
                        priority=TaskPriority.MEDIUM,
                        repository=repo_name,
                        estimated_cycles=1,
                        metadata={
                            'auto_generated': True,
                            'fix_type': 'repository_customization',
                            'issues': repo_issues
                        }
                    )
                    work_items.append(work_item)
            
            if work_items:
                self.logger.info(f"Found {len(work_items)} repositories needing customization fixes")
            
            return work_items
            
        except Exception as e:
            self.logger.error(f"Error checking repositories for fixes: {e}")
            return []
    
    async def _run_main_loop(self):
        """Main orchestration loop - continuously discovers and queues work."""
        self.logger.info("Starting main orchestration loop")
        
        last_maintenance_check = datetime.now(timezone.utc)
        
        while not self.shutdown_requested:
            try:
                # Check for deleted repositories periodically
                await self._check_repository_cleanup()
                
                # Discover new work
                await self._discover_work()
                
                # Generate maintenance work periodically (every 5 minutes)
                current_time = datetime.now(timezone.utc)
                if (current_time - last_maintenance_check).total_seconds() > 300:
                    await self._generate_periodic_maintenance_work()
                    last_maintenance_check = current_time
                
                # Update metrics
                await self._update_metrics()
                
                # Save state periodically
                await self._save_state()
                
                # Brief pause to prevent overwhelming the system
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(5)  # Longer pause on error
    
    async def _discover_work(self):
        """Discover new work opportunities."""
        if not self.work_finder:
            return
        
        try:
            # Check if we need more work
            if self.use_redis_queue and self.redis_work_queue:
                # Get queue stats from Redis
                queue_stats = await self.redis_work_queue.get_queue_stats()
                current_queue_size = queue_stats.get('total_queued', 0)
            else:
                current_queue_size = len(self.work_queue)
            
            active_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.WORKING)
            
            # Maintain a larger buffer of work to prevent starvation
            # Formula: max(10, workers * 3) to ensure at least 10 items queued
            target_queue_size = max(10, self.max_workers * 3)  # 3x buffer with minimum
            
            if current_queue_size < target_queue_size:
                # Discover new work opportunities
                new_work = await self.work_finder.discover_work(
                    max_items=target_queue_size - current_queue_size,
                    current_workload=active_workers
                )
                
                # If no work discovered, use enhanced generator
                if not new_work or len(new_work) < (target_queue_size - current_queue_size) / 2:
                    self.logger.warning(f"Insufficient work discovered ({len(new_work) if new_work else 0}), using enhanced generator")
                    
                    # Generate additional work
                    generated_count = target_queue_size - current_queue_size - (len(new_work) if new_work else 0)
                    
                    # Determine generation strategy based on queue state
                    if current_queue_size == 0:
                        # Emergency generation - queue is empty!
                        emergency_work = await self.enhanced_work_generator.generate_emergency_work(
                            count=max(5, generated_count)
                        )
                        new_work = (new_work or []) + emergency_work
                        self.logger.warning(f"⚠️ Generated {len(emergency_work)} emergency work items")
                    else:
                        # Normal generation
                        additional_work = await self.enhanced_work_generator.generate_work_batch(
                            target_count=generated_count,
                            min_priority=TaskPriority.MEDIUM
                        )
                        new_work = (new_work or []) + additional_work
                        self.logger.info(f"Generated {len(additional_work)} additional work items")
                
                if new_work:
                    if self.use_redis_queue and self.redis_work_queue:
                        # Add to Redis queue
                        await self.redis_work_queue.add_work_batch(new_work)
                        # Force immediate flush
                        await self.redis_work_queue._flush_buffer()
                        self.metrics['total_work_created'] += len(new_work)
                    else:
                        # Add to in-memory queue
                        async with self.work_queue_lock:
                            self.work_queue.extend(new_work)
                            self.metrics['total_work_created'] += len(new_work)
                        
                        # Sort queue by priority
                        self.work_queue.sort(key=lambda x: x.priority.value)
                    
                    self.logger.info(f"Total work items added to queue: {len(new_work)}")
                    
                    # Detect patterns in discovered work
                    work_types = {}
                    for work in new_work:
                        work_type = work.task_type
                        work_types[work_type] = work_types.get(work_type, 0) + 1
                    
                    # Trigger research if unusual patterns detected
                    if len(work_types) > 5 and self.research_engine:  # Many different types
                        await self.trigger_research(
                            reason="Diverse work patterns detected",
                            context={
                                'work_types': work_types,
                                'total_items': len(new_work),
                                'trigger': 'pattern_diversity'
                            }
                        )
                    
                    # Track work discovery event
                    if self.event_processor:
                        await self.event_processor.track_event({
                            'event_type': 'work_discovered',
                            'work_count': len(new_work),
                            'work_types': work_types,
                            'queue_size': current_queue_size,
                            'active_workers': active_workers,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
        
        except Exception as e:
            self.logger.error(f"Error discovering work: {e}")
    
    async def _generate_periodic_maintenance_work(self):
        """Generate periodic maintenance and research work."""
        try:
            self.logger.info("Generating periodic maintenance work")
            
            # Generate a mix of maintenance and research work
            maintenance_work = await self.enhanced_work_generator.generate_maintenance_work(count=2)
            research_work = await self.enhanced_work_generator.generate_research_work(count=1)
            
            # Check for repositories needing customization
            repo_fix_work = await self._check_repositories_needing_fixes()
            
            all_work = maintenance_work + research_work + repo_fix_work
            
            if all_work:
                if self.use_redis_queue and self.redis_work_queue:
                    await self.redis_work_queue.add_work_batch(all_work)
                    await self.redis_work_queue._flush_buffer()
                else:
                    async with self.work_queue_lock:
                        self.work_queue.extend(all_work)
                        self.work_queue.sort(key=lambda x: x.priority.value)
                
                self.metrics['total_work_created'] += len(all_work)
                self.logger.info(f"Added {len(all_work)} periodic maintenance/research tasks")
        
        except Exception as e:
            self.logger.error(f"Error generating periodic maintenance work: {e}")
    
    async def _update_metrics(self):
        """Update performance metrics."""
        if not self.start_time:
            return
        
        runtime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        
        if runtime_hours > 0:
            self.metrics['work_per_hour'] = self.metrics['total_work_completed'] / runtime_hours
        
        # Calculate worker utilization
        working_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.WORKING)
        self.metrics['worker_utilization'] = working_workers / max(len(self.workers), 1)
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize an error for better handling.
        
        Returns:
            Error category: 'rate_limit', 'duplicate', 'network', 'auth', or 'unknown'
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Rate limit errors
        if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429']):
            return 'rate_limit'
        
        # Duplicate errors
        if any(keyword in error_str for keyword in ['duplicate', 'already exists']):
            return 'duplicate'
        
        # Redis errors (check before network to catch "redis connection")
        if 'redis' in error_str or 'RedisError' in error_type:
            return 'redis'
        
        # Network errors
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'network', 'ssl']):
            return 'network'
        
        # Authentication errors
        if any(keyword in error_str for keyword in ['unauthorized', 'forbidden', '401', '403', 'authentication']):
            return 'auth'
        
        # Redis errors
        if 'redis' in error_str or 'RedisError' in error_type:
            return 'redis'
        
        # GitHub API errors
        if 'github' in error_str or 'GithubException' in error_type:
            return 'github_api'
        
        return 'unknown'
    
    def _update_completion_metrics(self, completion_time: float):
        """Update completion time metrics."""
        total_completed = self.metrics['total_work_completed']
        current_avg = self.metrics['average_completion_time']
        
        # Rolling average
        self.metrics['average_completion_time'] = (
            (current_avg * (total_completed - 1) + completion_time) / total_completed
        )
    
    async def _save_state(self):
        """Save orchestrator state."""
        state = {
            'metrics': self.metrics,
            'work_queue': [item.to_dict() for item in self.work_queue],
            'completed_work': [item.to_dict() for item in self.completed_work[-100:]],  # Keep last 100
            'workers': {
                worker_id: {
                    'id': worker.id,
                    'status': worker.status.value,
                    'total_completed': worker.total_completed,
                    'total_errors': worker.total_errors,
                    'specialization': worker.specialization
                }
                for worker_id, worker in self.workers.items()
            },
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        with open('continuous_orchestrator_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    async def _load_work_queue(self):
        """Load existing work queue from state."""
        try:
            with open('continuous_orchestrator_state.json', 'r') as f:
                state = json.load(f)
            
            # Get valid repositories from system state
            valid_repositories = set(self.system_state.get('projects', {}).keys())
            valid_repositories.update(self.system_state.get('repositories', {}).keys())
            self.logger.info(f"Valid repositories: {valid_repositories}")
            
            # Restore work queue
            skipped_count = 0
            for item_data in state.get('work_queue', []):
                try:
                    # Check if repository is valid (skip if it's for a deleted repo)
                    repository = item_data.get('repository')
                    if repository and repository not in valid_repositories:
                        self.logger.warning(f"Skipping work item for non-existent repository: {repository} - {item_data.get('title', 'Unknown')}")
                        skipped_count += 1
                        continue
                    
                    # Handle priority parsing safely
                    priority_value = item_data['priority']
                    if isinstance(priority_value, int):
                        priority = TaskPriority(priority_value)
                    else:
                        # Handle string enum names
                        priority = TaskPriority[priority_value] if isinstance(priority_value, str) else TaskPriority.MEDIUM
                    
                    work_item = WorkItem(
                        id=item_data['id'],
                        task_type=item_data['task_type'],
                        title=item_data['title'],
                        description=item_data['description'],
                        priority=priority,
                        repository=item_data.get('repository'),
                        estimated_cycles=item_data.get('estimated_cycles', 1),
                        dependencies=item_data.get('dependencies', []),
                        metadata=item_data.get('metadata', {}),
                        created_at=datetime.fromisoformat(item_data['created_at'])
                    )
                    self.work_queue.append(work_item)
                except Exception as e:
                    self.logger.warning(f"Error loading work item: {e}")
                    continue
            
            self.logger.info(f"Restored {len(self.work_queue)} work items from state (skipped {skipped_count} for deleted repos)")
            
        except FileNotFoundError:
            self.logger.info("No previous state found - starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
    
    async def _stop_worker(self, worker_id: str):
        """Stop a specific worker."""
        if worker_id in self.worker_tasks:
            task = self.worker_tasks[worker_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # Only delete if still exists (may be deleted by another thread)
            if worker_id in self.worker_tasks:
                del self.worker_tasks[worker_id]
        
        if worker_id in self.workers:
            self.workers[worker_id].status = WorkerStatus.SHUTTING_DOWN
    
    async def _handle_emergency_stop(self, message: Dict[str, Any]):
        """Handle emergency stop event."""
        self.logger.warning(f"⚠️ EMERGENCY STOP: {message.get('data', {}).get('reason', 'Unknown')}")
        self.shutdown_requested = True
    
    async def _handle_task_reassignment(self, message: Dict[str, Any]):
        """Handle task reassignment request."""
        data = message.get('data', {})
        task_id = data.get('task_id')
        reason = data.get('reason', 'Unknown')
        from_worker = data.get('from_worker')
        
        self.logger.info(f"Task reassignment requested for {task_id}: {reason}")
        
        # Find the task in workers using lock-free state
        if self.redis_state_manager and from_worker:
            # Clear task from original worker
            await self.redis_state_manager.update_worker_field(
                from_worker, "current_task_id", None
            )
            await self.redis_state_manager.update_worker_field(
                from_worker, "status", WorkerStatus.IDLE.value
            )
            
            # Mark task as available for reassignment
            await self.redis_state_manager.update_task_state(task_id, {
                "status": "pending_reassignment",
                "previous_worker": from_worker,
                "reassignment_reason": reason
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        runtime = 0
        if self.start_time:
            runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        # Get metrics from Redis if available
        if self.redis_state_manager:
            try:
                # Create a coroutine and run it
                loop = asyncio.new_event_loop()
                metrics_coro = self._get_redis_metrics()
                redis_metrics = loop.run_until_complete(metrics_coro)
                loop.close()
                
                # Update local metrics with Redis values
                self.metrics.update(redis_metrics)
            except Exception as e:
                self.logger.debug(f"Could not fetch Redis metrics: {e}")
        
        # Get task completion statistics
        completion_stats = {}
        if hasattr(self.task_persistence, 'get_completion_stats'):
            if asyncio.iscoroutinefunction(self.task_persistence.get_completion_stats):
                # Create a new event loop if needed for sync context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, can't use run_until_complete
                        completion_stats = {'note': 'Stats available in async context only'}
                    else:
                        completion_stats = loop.run_until_complete(self.task_persistence.get_completion_stats())
                except:
                    completion_stats = {'error': 'Could not fetch stats'}
            else:
                completion_stats = self.task_persistence.get_completion_stats()
        
        # Get Redis components status
        redis_components = {
            'worker_coordination': self.worker_coordinator is not None,
            'distributed_locks': self.redis_locks is not None,
            'event_analytics': self.event_analytics is not None,
            'event_processor': self.event_processor is not None,
            'performance_analytics': self.redis_analytics is not None,
            'workflow_orchestration': self.workflow_executor is not None
        }
        
        return {
            'running': self.running,
            'runtime_seconds': runtime,
            'metrics': self.metrics.copy(),
            'workers': {
                worker_id: {
                    'status': worker.status.value,
                    'current_work': worker.current_work.title if worker.current_work else None,
                    'total_completed': worker.total_completed,
                    'total_errors': worker.total_errors,
                    'specialization': worker.specialization
                }
                for worker_id, worker in self.workers.items()
            },
            'work_queue_size': self._get_queue_size_sync(),
            'completed_work_count': len(self.completed_work),
            'queue_type': 'redis' if self.use_redis_queue else 'in-memory',
            'github_integration': self.github_creator.can_create_issues(),
            'task_completion_stats': completion_stats,
            'rate_limit_status': self.github_creator.get_rate_limit_status() if self.github_creator.can_create_issues() else None,
            'redis_components': redis_components,
            'redis_integration_score': sum(redis_components.values()) / len(redis_components) * 100
        }
    
    async def _get_queue_size(self) -> int:
        """Get current queue size from Redis or in-memory."""
        if self.use_redis_queue and self.redis_work_queue:
            stats = await self.redis_work_queue.get_queue_stats()
            return stats.get('total_queued', 0)
        else:
            return len(self.work_queue)
    
    def _get_queue_size_sync(self) -> int:
        """Get current queue size synchronously."""
        if self.use_redis_queue and self.redis_work_queue:
            # Use the synchronous method from Redis work queue
            try:
                return self.redis_work_queue.get_queue_size_sync()
            except Exception as e:
                self.logger.debug(f"Error getting Redis queue size: {e}")
                return -1  # Fallback to -1 on error
        else:
            return len(self.work_queue)
    
    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive analytics dashboard data."""
        dashboard = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'orchestrator_id': 'continuous_orchestrator',
            'system_health': {},
            'performance_metrics': {},
            'pattern_insights': {},
            'resource_utilization': {},
            'workflow_status': {}
        }
        
        try:
            # Get system health from event processor
            if self.event_processor:
                analytics_summary = await self.event_processor.get_analytics_summary()
                dashboard['system_health'] = {
                    'task_success_rate': self._calculate_success_rate(analytics_summary),
                    'worker_performance': analytics_summary.get('worker_metrics', {}),
                    'recent_patterns': analytics_summary.get('patterns_detected', [])
                }
            
            # Get performance analytics
            if self.redis_analytics:
                # Note: redis_analytics methods may need to be called differently
                dashboard['performance_metrics'] = {
                    'throughput': self.metrics.get('work_per_hour', 0),
                    'average_completion_time': self.metrics.get('average_completion_time', 0),
                    'worker_utilization': self.metrics.get('worker_utilization', 0)
                }
            
            # Get resource utilization
            if self.resource_manager:
                resources = self.resource_manager.get_system_metrics()
                dashboard['resource_utilization'] = {
                    'cpu_usage': resources.get('cpu_usage', 0),
                    'memory_usage': resources.get('memory_usage', 0),
                    'efficiency_score': resources.get('efficiency', 0)
                }
            
            # Get workflow status
            if self.workflow_executor:
                # Get active workflows count
                dashboard['workflow_status'] = {
                    'active_workflows': 0,  # Would need workflow executor method
                    'workflow_types': ['feature_development'],
                    'workflow_enabled': True
                }
            
            # Calculate overall health score
            dashboard['overall_health_score'] = self._calculate_health_score(dashboard)
            
        except Exception as e:
            self.logger.error(f"Error generating analytics dashboard: {e}")
            dashboard['error'] = str(e)
        
        return dashboard
    
    def _calculate_success_rate(self, analytics_summary: Dict[str, Any]) -> float:
        """Calculate task success rate from analytics data."""
        task_metrics = analytics_summary.get('task_metrics', {})
        total_completed = sum(m.get('success', 0) + m.get('failed', 0) for m in task_metrics.values())
        total_success = sum(m.get('success', 0) for m in task_metrics.values())
        
        if total_completed > 0:
            return total_success / total_completed
        return 1.0
    
    def _calculate_health_score(self, dashboard: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        scores = []
        
        # Task success rate contributes 30%
        success_rate = dashboard.get('system_health', {}).get('task_success_rate', 1.0)
        scores.append(success_rate * 0.3)
        
        # Worker utilization contributes 20%
        utilization = dashboard.get('performance_metrics', {}).get('worker_utilization', 0.5)
        scores.append(utilization * 0.2)
        
        # Resource efficiency contributes 20%
        efficiency = dashboard.get('resource_utilization', {}).get('efficiency_score', 0.5)
        scores.append(efficiency * 0.2)
        
        # Redis integration contributes 15%
        redis_score = self.get_status().get('redis_integration_score', 0) / 100
        scores.append(redis_score * 0.15)
        
        # Workflow usage contributes 15%
        workflow_enabled = dashboard.get('workflow_status', {}).get('workflow_enabled', False)
        scores.append((1.0 if workflow_enabled else 0.0) * 0.15)
        
        return sum(scores)
    
    def _init_research_engine(self):
        """Initialize the research evolution engine."""
        try:
            # Create AI brain for research using factory
            research_ai_brain = AIBrainFactory.create_for_research()
            
            # Initialize research engine with all necessary components
            self.research_engine = ResearchEvolutionEngine(
                state_manager=self.state_manager,
                ai_brain=research_ai_brain,
                task_generator=None,  # Will be set when available
                self_improver=None,   # Will be set when available
                outcome_learning=None # Will be set when available
            )
            
            # Configure for development/production mode
            import os
            execution_mode = os.getenv('EXECUTION_MODE', 'development').lower()
            
            if execution_mode == 'development':
                # More aggressive research in development
                self.research_engine.config.update({
                    "cycle_interval_seconds": 2 * 60,  # 2 minutes in dev - very aggressive
                    "emergency_cycle_interval": 1 * 60,  # 1 minute for emergencies
                    "max_research_per_cycle": 10,
                    "min_insight_confidence": 0.4,  # Lower threshold to trigger more research
                    "auto_implement_threshold": 0.6,  # More willing to implement
                    "enable_proactive_research": True,  # Always look for improvements
                    "proactive_research_priority": "high"  # Prioritize proactive research
                })
                self.logger.info("Research engine configured for AGGRESSIVE DEVELOPMENT mode")
            else:
                # Production settings (default)
                self.logger.info("Research engine configured for PRODUCTION mode")
            
            # Enable proactive research
            self.research_engine.config["enable_proactive_research"] = True
            
            self.logger.info("✓ Research Evolution Engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize research engine: {e}")
            self.research_engine = None
    
    async def _start_research_engine(self):
        """Start the research engine in the background."""
        if not self.research_engine:
            return
        
        try:
            self.logger.info("Starting Research Evolution Engine...")
            self.research_task = asyncio.create_task(self.research_engine.start_continuous_research())
            self.logger.info("✓ Research engine started")
            
            # Add research metrics to tracking
            self.metrics['research_cycles_completed'] = 0
            self.metrics['research_insights_generated'] = 0
            self.metrics['research_improvements_applied'] = 0
            
        except Exception as e:
            self.logger.error(f"Failed to start research engine: {e}")
            self.research_task = None
    
    async def trigger_research(self, reason: str, context: Dict[str, Any] = None):
        """Manually trigger research for a specific reason."""
        if not self.research_engine:
            self.logger.warning("Research engine not available")
            return
        
        try:
            # Execute emergency research
            result = await self.research_engine.execute_emergency_research({
                'reason': reason,
                'context': context or {},
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'triggered_by': 'orchestrator'
            })
            
            self.logger.info(f"Emergency research completed: {result.get('insights', [])}")
            
            # Update metrics
            self.metrics['research_cycles_completed'] += 1
            self.metrics['research_insights_generated'] += len(result.get('insights', []))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to trigger research: {e}")
            return None
    
    async def _get_redis_metrics(self) -> Dict[str, Any]:
        """Get metrics from Redis atomic counters."""
        if not self.redis_state_manager:
            return {}
        
        try:
            metrics = {
                'total_work_completed': await self.redis_state_manager.get_counter("total_work_completed"),
                'total_work_created': await self.redis_state_manager.get_counter("total_work_created"),
                'total_errors': await self.redis_state_manager.get_counter("total_errors")
            }
            
            # Get per-worker metrics
            active_workers = await self.redis_state_manager.get_set_members("active_workers")
            for worker_id in active_workers:
                completed = await self.redis_state_manager.get_counter(f"worker:{worker_id}:completed")
                errors = await self.redis_state_manager.get_counter(f"worker:{worker_id}:errors")
                
                # Update worker state if exists
                if worker_id in self.workers:
                    self.workers[worker_id].total_completed = completed
                    self.workers[worker_id].total_errors = errors
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting Redis metrics: {e}")
            return {}
    
    # MCP-Redis Enhanced Methods
    async def analyze_orchestration_patterns(self) -> Dict[str, Any]:
        """Analyze orchestration patterns using MCP-Redis natural language processing."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            # Gather orchestration metrics
            status = self.get_status()
            
            analysis = await self.mcp_redis.execute(f"""
                Analyze the orchestration patterns:
                
                Workers: {len(self.workers)}
                Active Workers: {sum(1 for w in self.workers.values() if w.status == WorkerStatus.WORKING)}
                Queue Size: {status['queue_size']}
                Total Completed: {self.metrics['total_work_completed']}
                Total Errors: {self.metrics['total_errors']}
                Success Rate: {self.metrics.get('success_rate', 0):.2%}
                
                Analyze:
                - Worker utilization patterns
                - Task distribution efficiency
                - Error patterns and causes
                - Bottlenecks in processing
                - Optimal worker count recommendation
                - Task type distribution
                - Performance trends
                
                Provide actionable insights to improve orchestration.
            """)
            
            return analysis if isinstance(analysis, dict) else {"analysis": analysis}
            
        except Exception as e:
            self.logger.error(f"Error analyzing orchestration patterns: {e}")
            return {"error": str(e)}
    
    async def predict_resource_needs(self, time_window: str = "next 24 hours") -> Dict[str, Any]:
        """Predict resource needs using MCP-Redis AI capabilities."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            # Get current and historical metrics
            current_status = self.get_status()
            work_per_hour = self.metrics.get('work_per_hour', 0)
            
            prediction = await self.mcp_redis.execute(f"""
                Predict resource needs for: {time_window}
                
                Current state:
                - Workers: {len(self.workers)} (max: {self.max_workers})
                - Queue size: {current_status['queue_size']}
                - Work completion rate: {work_per_hour:.2f} items/hour
                - Average completion time: {self.metrics['average_completion_time']:.2f}s
                - Error rate: {self.metrics['total_errors'] / max(self.metrics['total_work_completed'], 1):.2%}
                
                Predict:
                - Expected workload changes
                - Optimal worker count
                - Memory usage trends
                - CPU requirements
                - Potential bottlenecks
                - Scaling recommendations
                
                Consider patterns, time of day, and historical trends.
            """)
            
            return prediction if isinstance(prediction, dict) else {"prediction": prediction}
            
        except Exception as e:
            self.logger.error(f"Error predicting resource needs: {e}")
            return {"error": str(e)}
    
    async def optimize_worker_distribution(self) -> Dict[str, Any]:
        """Optimize worker distribution using MCP-Redis intelligent analysis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            # Gather worker specializations and performance
            worker_data = []
            for worker_id, worker in self.workers.items():
                worker_data.append({
                    'id': worker_id,
                    'status': worker.status.value,
                    'specialization': worker.specialization,
                    'completed': worker.total_completed,
                    'errors': worker.total_errors,
                    'success_rate': worker.total_completed / max(worker.total_completed + worker.total_errors, 1)
                })
            
            optimization = await self.mcp_redis.execute(f"""
                Optimize worker distribution and specialization:
                
                Current workers:
                {json.dumps(worker_data, indent=2)}
                
                Queue characteristics:
                - Size: {await self._get_queue_size()}
                - Task types in queue: analyze based on patterns
                
                Optimize:
                - Worker specialization assignments
                - Load balancing strategy
                - Task routing rules
                - Performance-based worker adjustments
                - Specialization effectiveness
                
                Provide specific reassignment recommendations.
            """)
            
            return optimization if isinstance(optimization, dict) else {"optimization": optimization}
            
        except Exception as e:
            self.logger.error(f"Error optimizing worker distribution: {e}")
            return {"error": str(e)}
    
    async def diagnose_performance_issues(self) -> Dict[str, Any]:
        """Diagnose performance issues using MCP-Redis deep analysis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            # Gather comprehensive metrics
            status = self.get_status()
            analytics = await self.get_analytics_dashboard()
            
            diagnosis = await self.mcp_redis.execute(f"""
                Diagnose performance issues in the orchestration system:
                
                System metrics:
                - Uptime: {status['uptime']}
                - Queue size: {status['queue_size']}
                - Worker utilization: {status['worker_utilization']:.2%}
                - Average completion time: {self.metrics['average_completion_time']:.2f}s
                - Error rate: {analytics.get('error_rate', 0):.2%}
                - Work per hour: {self.metrics['work_per_hour']:.2f}
                
                Error patterns:
                {json.dumps(analytics.get('error_breakdown', {}), indent=2)}
                
                Diagnose:
                - Root causes of performance degradation
                - Specific bottlenecks
                - Error pattern analysis
                - Resource constraints
                - Inefficiencies in task processing
                - Queue backup reasons
                
                Provide specific fixes and improvements.
            """)
            
            return diagnosis if isinstance(diagnosis, dict) else {"diagnosis": diagnosis}
            
        except Exception as e:
            self.logger.error(f"Error diagnosing performance: {e}")
            return {"error": str(e)}
    
    async def generate_orchestration_insights(self) -> Dict[str, Any]:
        """Generate comprehensive orchestration insights using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            # Collect all relevant data
            status = self.get_status()
            analytics = await self.get_analytics_dashboard()
            
            insights = await self.mcp_redis.execute(f"""
                Generate orchestration insights report:
                
                System overview:
                - Total tasks completed: {self.metrics['total_work_completed']}
                - Success rate: {analytics.get('success_rate', 0):.2%}
                - Health score: {analytics.get('health_score', 0):.2f}
                - Active since: {status['start_time']}
                
                Worker performance:
                - Active workers: {status['active_workers']}
                - Utilization: {status['worker_utilization']:.2%}
                - Specializations: {[w.specialization for w in self.workers.values()]}
                
                Task processing:
                - Queue depth: {status['queue_size']}
                - Processing rate: {self.metrics['work_per_hour']:.2f}/hour
                - Average time: {self.metrics['average_completion_time']:.2f}s
                
                Generate insights on:
                - Overall system efficiency
                - Worker performance rankings
                - Task type success patterns
                - Optimization opportunities
                - Scaling recommendations
                - Future performance predictions
                
                Format as executive summary with key metrics and actions.
            """)
            
            return insights if isinstance(insights, dict) else {"insights": insights}
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}
    
    async def intelligent_task_routing(self, task: WorkItem) -> Optional[str]:
        """Route task to optimal worker using MCP-Redis intelligence."""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to round-robin
            return self._basic_task_routing(task)
        
        try:
            # Gather worker states and capabilities
            worker_info = []
            for worker_id, worker in self.workers.items():
                if worker.status == WorkerStatus.IDLE:
                    worker_info.append({
                        'id': worker_id,
                        'specialization': worker.specialization,
                        'completed': worker.total_completed,
                        'error_rate': worker.total_errors / max(worker.total_completed + worker.total_errors, 1),
                        'last_activity': worker.last_activity.isoformat()
                    })
            
            if not worker_info:
                return None
            
            routing = await self.mcp_redis.execute(f"""
                Route task to optimal worker:
                
                Task details:
                - Type: {task.type}
                - Repository: {task.repository}
                - Priority: {task.priority.value if hasattr(task.priority, 'value') else task.priority}
                - Description: {task.title}
                
                Available workers:
                {json.dumps(worker_info, indent=2)}
                
                Select the best worker based on:
                - Specialization match
                - Historical performance
                - Error rates
                - Load balancing
                
                Return: worker_id of the best match
            """)
            
            # Extract worker ID from response
            if isinstance(routing, dict) and 'worker_id' in routing:
                return routing['worker_id']
            elif isinstance(routing, str) and routing.startswith('worker_'):
                return routing
            else:
                return self._basic_task_routing(task)
                
        except Exception as e:
            self.logger.error(f"Error in intelligent routing: {e}")
            return self._basic_task_routing(task)
    
    def _basic_task_routing(self, task: WorkItem) -> Optional[str]:
        """Basic round-robin task routing fallback."""
        idle_workers = [w_id for w_id, w in self.workers.items() 
                       if w.status == WorkerStatus.IDLE]
        return idle_workers[0] if idle_workers else None