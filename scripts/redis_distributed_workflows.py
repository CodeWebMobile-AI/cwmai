"""
Redis Distributed Event Processing Workflows

High-performance distributed workflows using Redis Streams for coordinating
complex multi-worker intelligence tasks with fault tolerance and scalability.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum

from scripts.redis_integration.redis_client import get_redis_client
from scripts.redis_integration.redis_streams_manager import RedisStreamsManager
from scripts.redis_integration.redis_pubsub_manager import RedisPubSubManager
from scripts.redis_integration.redis_locks_manager import RedisLocksManager

from scripts.redis_intelligence_hub import IntelligenceEvent, EventType, EventPriority, RedisIntelligenceHub, get_intelligence_hub
from scripts.redis_event_sourcing import RedisEventStore, get_event_store
from scripts.redis_lockfree_adapter import create_lockfree_state_manager


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskStatus(Enum):
    """Individual task status within workflow."""
    WAITING = "waiting"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowTrigger(Enum):
    """Workflow trigger types."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    CHAIN_TRIGGER = "chain_trigger"
    CONDITION_BASED = "condition_based"


@dataclass
class WorkflowTask:
    """Individual task within a workflow."""
    task_id: str
    name: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    assigned_worker: Optional[str] = None
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 3600
    priority: EventPriority = EventPriority.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'task_type': self.task_type,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'assigned_worker': self.assigned_worker,
            'status': self.status.value if isinstance(self.status, TaskStatus) else self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'priority': self.priority.value if hasattr(self.priority, 'value') else self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTask':
        """Create task from dictionary."""
        return cls(
            task_id=data['task_id'],
            name=data['name'],
            task_type=data['task_type'],
            parameters=data['parameters'],
            dependencies=data['dependencies'],
            assigned_worker=data.get('assigned_worker'),
            status=TaskStatus(data['status']),
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            result=data.get('result'),
            error=data.get('error'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            timeout_seconds=data.get('timeout_seconds', 3600),
            priority=EventPriority(data.get('priority', EventPriority.NORMAL.value))
        )


@dataclass
class WorkflowDefinition:
    """Workflow definition and configuration."""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    trigger: WorkflowTrigger
    schedule: Optional[str] = None  # Cron expression for scheduled workflows
    trigger_conditions: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 7200  # 2 hours default
    max_retries: int = 2
    enable_parallel_execution: bool = True
    failure_strategy: str = "stop"  # stop, continue, retry_failed
    notification_channels: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowExecution:
    """Workflow execution instance."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    triggered_by: str = "manual"
    trigger_data: Dict[str, Any] = None
    current_tasks: List[str] = None
    completed_tasks: List[str] = None
    failed_tasks: List[str] = None
    execution_context: Dict[str, Any] = None
    retry_count: int = 0
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.trigger_data is None:
            self.trigger_data = {}
        if self.current_tasks is None:
            self.current_tasks = []
        if self.completed_tasks is None:
            self.completed_tasks = []
        if self.failed_tasks is None:
            self.failed_tasks = []
        if self.execution_context is None:
            self.execution_context = {}


class RedisWorkflowEngine:
    """Redis-powered distributed workflow engine."""
    
    def __init__(self,
                 engine_id: str = None,
                 max_concurrent_workflows: int = 100,
                 max_concurrent_tasks: int = 500,
                 enable_fault_tolerance: bool = True,
                 enable_load_balancing: bool = True):
        """Initialize Redis Workflow Engine.
        
        Args:
            engine_id: Unique engine identifier
            max_concurrent_workflows: Maximum concurrent workflows
            max_concurrent_tasks: Maximum concurrent tasks
            enable_fault_tolerance: Enable fault tolerance mechanisms
            enable_load_balancing: Enable task load balancing
        """
        self.engine_id = engine_id or f"workflow_engine_{uuid.uuid4().hex[:8]}"
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_fault_tolerance = enable_fault_tolerance
        self.enable_load_balancing = enable_load_balancing
        
        self.logger = logging.getLogger(f"{__name__}.RedisWorkflowEngine")
        
        # Redis components
        self.redis_client = None
        self.streams_manager: Optional[RedisStreamsManager] = None
        self.pubsub_manager: Optional[RedisPubSubManager] = None
        self.locks_manager: Optional[RedisLocksManager] = None
        self.intelligence_hub: Optional[RedisIntelligenceHub] = None
        self.event_store: Optional[RedisEventStore] = None
        self.state_manager = None
        
        # Stream keys
        self.workflow_stream = f"workflows:{self.engine_id}"
        self.task_stream = f"tasks:{self.engine_id}"
        self.coordination_stream = f"coordination:{self.engine_id}"
        
        # Engine state
        self._workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._worker_registry: Dict[str, Dict[str, Any]] = {}
        self._task_handlers: Dict[str, Callable] = {}
        
        # Processing tasks
        self._workflow_processor_task: Optional[asyncio.Task] = None
        self._task_processor_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Performance metrics
        self._metrics = {
            'workflows_executed': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'tasks_executed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_workflow_duration': 0.0,
            'average_task_duration': 0.0,
            'current_load': 0.0
        }
    
    async def initialize(self):
        """Initialize workflow engine components."""
        try:
            self.logger.info(f"Initializing Redis Workflow Engine: {self.engine_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            self.streams_manager = RedisStreamsManager(self.redis_client)
            self.pubsub_manager = RedisPubSubManager(self.redis_client)
            self.locks_manager = RedisLocksManager(self.redis_client)
            self.state_manager = create_lockfree_state_manager(f"workflow_engine_{self.engine_id}")
            await self.state_manager.initialize()
            
            # Initialize intelligence hub integration
            self.intelligence_hub = await get_intelligence_hub()
            
            # Initialize event store integration
            self.event_store = await get_event_store()
            
            # Initialize streams
            await self._initialize_streams()
            
            # Register event processors
            await self._register_event_processors()
            
            # Start processing tasks
            await self._start_processors()
            
            # Register engine
            await self._register_engine()
            
            self.logger.info(f"Workflow Engine {self.engine_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Workflow Engine: {e}")
            raise
    
    async def _initialize_streams(self):
        """Initialize Redis streams for workflow processing."""
        try:
            streams = [self.workflow_stream, self.task_stream, self.coordination_stream]
            
            for stream in streams:
                # Create stream
                try:
                    await self.redis_client.xadd(stream, {'init': 'true'}, maxlen=10000)
                except Exception:
                    pass
                
                # Create consumer group
                try:
                    await self.redis_client.xgroup_create(
                        stream, f"processors_{self.engine_id}", id='0', mkstream=True
                    )
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.error(f"Error initializing streams: {e}")
            raise
    
    async def _register_event_processors(self):
        """Register event processors with intelligence hub."""
        try:
            # Register workflow event processor
            self.intelligence_hub.register_event_processor(
                EventType.TASK_ASSIGNMENT,
                self._handle_task_assignment_event
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.TASK_COMPLETION,
                self._handle_task_completion_event
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.TASK_FAILURE,
                self._handle_task_failure_event
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.WORKER_REGISTRATION,
                self._handle_worker_registration_event
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.WORKER_SHUTDOWN,
                self._handle_worker_shutdown_event
            )
            
        except Exception as e:
            self.logger.error(f"Error registering event processors: {e}")
            raise
    
    async def _start_processors(self):
        """Start workflow and task processors."""
        try:
            # Start workflow processor
            self._workflow_processor_task = asyncio.create_task(self._workflow_processor())
            
            # Start task processor
            self._task_processor_task = asyncio.create_task(self._task_processor())
            
            # Start monitor
            self._monitor_task = asyncio.create_task(self._execution_monitor())
            
            # Start scheduler
            self._scheduler_task = asyncio.create_task(self._workflow_scheduler())
            
            self.logger.info("Workflow processors started")
            
        except Exception as e:
            self.logger.error(f"Error starting processors: {e}")
            raise
    
    async def register_workflow(self, workflow: WorkflowDefinition):
        """Register workflow definition.
        
        Args:
            workflow: Workflow definition to register
        """
        try:
            # Validate workflow
            await self._validate_workflow(workflow)
            
            # Store workflow definition
            self._workflow_definitions[workflow.workflow_id] = workflow
            
            # Convert workflow to dict with proper enum handling
            workflow_dict = {
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'tasks': [task.to_dict() for task in workflow.tasks],
                'trigger': workflow.trigger.value if hasattr(workflow.trigger, 'value') else workflow.trigger,
                'schedule': workflow.schedule,
                'trigger_conditions': workflow.trigger_conditions,
                'timeout_seconds': workflow.timeout_seconds,
                'max_retries': workflow.max_retries,
                'enable_parallel_execution': workflow.enable_parallel_execution,
                'failure_strategy': workflow.failure_strategy,
                'notification_channels': workflow.notification_channels,
                'metadata': workflow.metadata
            }
            
            # Store in Redis
            await self.state_manager.update(
                f"workflows.definitions.{workflow.workflow_id}",
                workflow_dict,
                distributed=True
            )
            
            self.logger.info(f"Registered workflow: {workflow.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Error registering workflow {workflow.workflow_id}: {e}")
            raise
    
    async def _validate_workflow(self, workflow: WorkflowDefinition):
        """Validate workflow definition."""
        # Check for circular dependencies
        task_ids = {task.task_id for task in workflow.tasks}
        
        for task in workflow.tasks:
            # Check dependencies exist
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.task_id} depends on non-existent task {dep}")
            
            # Check for circular dependencies (simplified check)
            if task.task_id in task.dependencies:
                raise ValueError(f"Task {task.task_id} has circular dependency on itself")
    
    async def start_workflow(self, 
                           workflow_id: str, 
                           trigger_data: Dict[str, Any] = None,
                           triggered_by: str = "manual") -> str:
        """Start workflow execution.
        
        Args:
            workflow_id: Workflow to execute
            trigger_data: Data passed to workflow
            triggered_by: What triggered the workflow
            
        Returns:
            Execution ID
        """
        try:
            if workflow_id not in self._workflow_definitions:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create execution
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                start_time=datetime.now(timezone.utc),
                triggered_by=triggered_by,
                trigger_data=trigger_data or {}
            )
            
            # Store execution
            self._active_executions[execution_id] = execution
            
            # Store in Redis
            await self.state_manager.update(
                f"workflows.executions.{execution_id}",
                asdict(execution),
                distributed=True
            )
            
            # Publish workflow start event
            await self._publish_workflow_event({
                'event_type': 'workflow_started',
                'execution_id': execution_id,
                'workflow_id': workflow_id,
                'triggered_by': triggered_by
            })
            
            # Log to event store
            workflow_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.COORDINATION_EVENT,
                worker_id=self.engine_id,
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={
                    'event_subtype': 'workflow_started',
                    'execution_id': execution_id,
                    'workflow_id': workflow_id,
                    'triggered_by': triggered_by
                }
            )
            await self.event_store.append_event(workflow_event)
            
            self._metrics['workflows_executed'] += 1
            
            self.logger.info(f"Started workflow {workflow_id} with execution {execution_id}")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Error starting workflow {workflow_id}: {e}")
            raise
    
    async def _workflow_processor(self):
        """Process workflow execution lifecycle."""
        while not self._shutdown:
            try:
                # Process pending workflows
                for execution_id, execution in list(self._active_executions.items()):
                    if execution.status == WorkflowStatus.PENDING:
                        await self._start_workflow_execution(execution)
                    
                    elif execution.status == WorkflowStatus.RUNNING:
                        await self._process_running_workflow(execution)
                
                await asyncio.sleep(1)  # Process every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in workflow processor: {e}")
                await asyncio.sleep(5)
    
    async def _start_workflow_execution(self, execution: WorkflowExecution):
        """Start workflow execution by scheduling initial tasks."""
        try:
            workflow = self._workflow_definitions[execution.workflow_id]
            
            # Find tasks with no dependencies (can start immediately)
            ready_tasks = []
            for task in workflow.tasks:
                if not task.dependencies:
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # No tasks can start - check for configuration error
                self.logger.error(f"No initial tasks found for workflow {execution.workflow_id}")
                execution.status = WorkflowStatus.FAILED
                execution.error = "No initial tasks available"
                return
            
            # Schedule ready tasks
            for task in ready_tasks:
                await self._schedule_task(execution, task)
            
            # Update execution status
            execution.status = WorkflowStatus.RUNNING
            execution.current_tasks = [task.task_id for task in ready_tasks]
            
            await self._update_execution_state(execution)
            
            self.logger.info(f"Started execution {execution.execution_id} with {len(ready_tasks)} initial tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting workflow execution {execution.execution_id}: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            await self._update_execution_state(execution)
    
    async def _process_running_workflow(self, execution: WorkflowExecution):
        """Process running workflow to check for completion or next steps."""
        try:
            workflow = self._workflow_definitions[execution.workflow_id]
            
            # Check if workflow is complete
            all_task_ids = {task.task_id for task in workflow.tasks}
            completed_task_ids = set(execution.completed_tasks)
            failed_task_ids = set(execution.failed_tasks)
            
            if completed_task_ids == all_task_ids:
                # All tasks completed successfully
                execution.status = WorkflowStatus.COMPLETED
                execution.end_time = datetime.now(timezone.utc)
                self._metrics['workflows_completed'] += 1
                
                await self._complete_workflow_execution(execution)
                return
            
            if failed_task_ids and workflow.failure_strategy == "stop":
                # Workflow should stop on failure
                execution.status = WorkflowStatus.FAILED
                execution.end_time = datetime.now(timezone.utc)
                execution.error = f"Workflow stopped due to failed tasks: {failed_task_ids}"
                self._metrics['workflows_failed'] += 1
                
                await self._fail_workflow_execution(execution)
                return
            
            # Check for tasks that can now be scheduled
            ready_tasks = []
            for task in workflow.tasks:
                if (task.task_id not in completed_task_ids and 
                    task.task_id not in failed_task_ids and
                    task.task_id not in execution.current_tasks):
                    
                    # Check if all dependencies are completed
                    dependencies_met = all(
                        dep in completed_task_ids for dep in task.dependencies
                    )
                    
                    if dependencies_met:
                        ready_tasks.append(task)
            
            # Schedule ready tasks
            for task in ready_tasks:
                await self._schedule_task(execution, task)
                execution.current_tasks.append(task.task_id)
            
            if ready_tasks:
                await self._update_execution_state(execution)
                self.logger.debug(f"Scheduled {len(ready_tasks)} new tasks for execution {execution.execution_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing running workflow {execution.execution_id}: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            await self._update_execution_state(execution)
    
    async def _schedule_task(self, execution: WorkflowExecution, task: WorkflowTask):
        """Schedule task for execution."""
        try:
            # Find best worker for task
            assigned_worker = await self._find_best_worker(task)
            
            if not assigned_worker:
                self.logger.warning(f"No suitable worker found for task {task.task_id}")
                return
            
            # Update task status
            task.assigned_worker = assigned_worker
            task.status = TaskStatus.ASSIGNED
            task.start_time = datetime.now(timezone.utc)
            
            # Publish task assignment event
            assignment_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.TASK_ASSIGNMENT,
                worker_id=assigned_worker,
                timestamp=datetime.now(timezone.utc),
                priority=task.priority,
                data={
                    'task_id': task.task_id,
                    'execution_id': execution.execution_id,
                    'workflow_id': execution.workflow_id,
                    'task_type': task.task_type,
                    'parameters': task.parameters,
                    'timeout_seconds': task.timeout_seconds
                }
            )
            
            await self.intelligence_hub.publish_event(assignment_event)
            
            self._metrics['tasks_executed'] += 1
            
            self.logger.info(f"Scheduled task {task.task_id} to worker {assigned_worker}")
            
        except Exception as e:
            self.logger.error(f"Error scheduling task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
    
    async def _find_best_worker(self, task: WorkflowTask) -> Optional[str]:
        """Find best worker for task based on load balancing and capabilities."""
        try:
            # Get available workers
            worker_registry = await self.intelligence_hub.get_worker_registry()
            
            # Filter workers by capabilities
            suitable_workers = []
            for worker_id, worker_data in worker_registry.items():
                if (worker_data.get('status') == 'active' and
                    task.task_type in worker_data.get('capabilities', [])):
                    suitable_workers.append(worker_id)
            
            if not suitable_workers:
                return None
            
            if not self.enable_load_balancing:
                return suitable_workers[0]
            
            # Simple load balancing - find worker with least active tasks
            worker_loads = {}
            for worker_id in suitable_workers:
                active_tasks = 0
                for execution in self._active_executions.values():
                    if execution.status == WorkflowStatus.RUNNING:
                        for task_id in execution.current_tasks:
                            # This is simplified - would need to track which worker has which task
                            pass
                
                worker_loads[worker_id] = active_tasks
            
            # Return worker with minimum load
            return min(worker_loads.keys(), key=lambda w: worker_loads[w])
            
        except Exception as e:
            self.logger.error(f"Error finding best worker for task {task.task_id}: {e}")
            return None
    
    async def _handle_task_completion_event(self, event: IntelligenceEvent, stream_name: str):
        """Handle task completion event."""
        try:
            task_id = event.data.get('task_id')
            execution_id = event.data.get('execution_id')
            result = event.data.get('result', {})
            
            if not execution_id or execution_id not in self._active_executions:
                return
            
            execution = self._active_executions[execution_id]
            workflow = self._workflow_definitions[execution.workflow_id]
            
            # Find and update task
            task = None
            for t in workflow.tasks:
                if t.task_id == task_id:
                    task = t
                    break
            
            if task:
                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now(timezone.utc)
                task.result = result
                
                # Update execution
                if task_id in execution.current_tasks:
                    execution.current_tasks.remove(task_id)
                execution.completed_tasks.append(task_id)
                
                await self._update_execution_state(execution)
                
                self._metrics['tasks_completed'] += 1
                
                self.logger.info(f"Task {task_id} completed in execution {execution_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling task completion event: {e}")
    
    async def _handle_task_failure_event(self, event: IntelligenceEvent, stream_name: str):
        """Handle task failure event."""
        try:
            task_id = event.data.get('task_id')
            execution_id = event.data.get('execution_id')
            error = event.data.get('error', 'Unknown error')
            
            if not execution_id or execution_id not in self._active_executions:
                return
            
            execution = self._active_executions[execution_id]
            workflow = self._workflow_definitions[execution.workflow_id]
            
            # Find and update task
            task = None
            for t in workflow.tasks:
                if t.task_id == task_id:
                    task = t
                    break
            
            if task:
                task.retry_count += 1
                
                if task.retry_count <= task.max_retries:
                    # Retry task
                    task.status = TaskStatus.RETRYING
                    task.error = error
                    await self._schedule_task(execution, task)
                    
                    self.logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                else:
                    # Task failed permanently
                    task.status = TaskStatus.FAILED
                    task.end_time = datetime.now(timezone.utc)
                    task.error = error
                    
                    # Update execution
                    if task_id in execution.current_tasks:
                        execution.current_tasks.remove(task_id)
                    execution.failed_tasks.append(task_id)
                    
                    self._metrics['tasks_failed'] += 1
                    
                    self.logger.error(f"Task {task_id} failed permanently: {error}")
                
                await self._update_execution_state(execution)
            
        except Exception as e:
            self.logger.error(f"Error handling task failure event: {e}")
    
    async def _handle_task_assignment_event(self, event: IntelligenceEvent, stream_name: str):
        """Handle task assignment event."""
        try:
            task_id = event.data.get('task_id')
            worker_id = event.worker_id
            execution_id = event.data.get('execution_id')
            
            self.logger.info(f"Task {task_id} assigned to worker {worker_id}")
            
            # Update worker registry with assigned task
            if worker_id in self._worker_registry:
                if 'assigned_tasks' not in self._worker_registry[worker_id]:
                    self._worker_registry[worker_id]['assigned_tasks'] = []
                self._worker_registry[worker_id]['assigned_tasks'].append(task_id)
                
                # Update last activity time
                self._worker_registry[worker_id]['last_activity'] = datetime.now(timezone.utc).isoformat()
            
            # Emit task assignment notification if needed
            await self._publish_workflow_event({
                'event_type': 'task_assigned',
                'task_id': task_id,
                'worker_id': worker_id,
                'execution_id': execution_id,
                'timestamp': event.timestamp.isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error handling task assignment event: {e}")
    
    async def _handle_worker_registration_event(self, event: IntelligenceEvent, stream_name: str):
        """Handle worker registration event."""
        try:
            worker_id = event.worker_id
            worker_data = event.data
            
            # Update worker registry
            self._worker_registry[worker_id] = {
                'worker_id': worker_id,
                'capabilities': worker_data.get('capabilities', []),
                'status': 'active',
                'registered_at': event.timestamp.isoformat(),
                'last_activity': event.timestamp.isoformat(),
                'metadata': worker_data.get('metadata', {}),
                'assigned_tasks': []
            }
            
            self.logger.info(f"Registered worker {worker_id} with capabilities: {worker_data.get('capabilities', [])}")
            
            # Publish worker registration notification
            await self._publish_workflow_event({
                'event_type': 'worker_registered',
                'worker_id': worker_id,
                'capabilities': worker_data.get('capabilities', []),
                'timestamp': event.timestamp.isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error handling worker registration event: {e}")
    
    async def _handle_worker_shutdown_event(self, event: IntelligenceEvent, stream_name: str):
        """Handle worker shutdown event."""
        try:
            worker_id = event.worker_id
            
            if worker_id in self._worker_registry:
                # Mark worker as inactive
                self._worker_registry[worker_id]['status'] = 'inactive'
                self._worker_registry[worker_id]['shutdown_at'] = event.timestamp.isoformat()
                
                # Reassign any active tasks from this worker
                assigned_tasks = self._worker_registry[worker_id].get('assigned_tasks', [])
                if assigned_tasks:
                    self.logger.warning(f"Worker {worker_id} shutdown with {len(assigned_tasks)} active tasks")
                    
                    # TODO: Implement task reassignment logic
                    for task_id in assigned_tasks:
                        await self._reassign_task(task_id, worker_id)
                
                self.logger.info(f"Worker {worker_id} shutdown gracefully")
            else:
                self.logger.warning(f"Received shutdown event for unknown worker {worker_id}")
            
            # Publish worker shutdown notification
            await self._publish_workflow_event({
                'event_type': 'worker_shutdown',
                'worker_id': worker_id,
                'timestamp': event.timestamp.isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error handling worker shutdown event: {e}")
    
    async def _reassign_task(self, task_id: str, failed_worker_id: str):
        """Reassign task from failed worker to another worker."""
        try:
            # Find the task in active executions
            for execution in self._active_executions.values():
                workflow = self._workflow_definitions[execution.workflow_id]
                
                for task in workflow.tasks:
                    if task.task_id == task_id and task.assigned_worker == failed_worker_id:
                        # Reset task for reassignment
                        task.assigned_worker = None
                        task.status = TaskStatus.WAITING
                        
                        # Schedule task again
                        await self._schedule_task(execution, task)
                        
                        self.logger.info(f"Reassigned task {task_id} from failed worker {failed_worker_id}")
                        return
            
            self.logger.warning(f"Could not find task {task_id} to reassign from worker {failed_worker_id}")
            
        except Exception as e:
            self.logger.error(f"Error reassigning task {task_id}: {e}")
    
    async def _publish_workflow_event(self, event_data: Dict[str, Any]):
        """Publish workflow-related event to coordination stream."""
        try:
            # Add timestamp if not present
            if 'timestamp' not in event_data:
                event_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Add engine ID
            event_data['engine_id'] = self.engine_id
            
            # Publish to coordination stream
            await self.streams_manager.publish(
                self.coordination_stream,
                event_data
            )
            
            # Also publish to pubsub for real-time notifications
            channel = f"workflow_events:{self.engine_id}"
            await self.pubsub_manager.publish(channel, event_data)
            
        except Exception as e:
            self.logger.error(f"Error publishing workflow event: {e}")
    
    async def _update_execution_state(self, execution: WorkflowExecution):
        """Update workflow execution state in Redis."""
        try:
            # Convert execution to dict
            execution_data = {
                'execution_id': execution.execution_id,
                'workflow_id': execution.workflow_id,
                'status': execution.status.value,
                'start_time': execution.start_time.isoformat(),
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'triggered_by': execution.triggered_by,
                'trigger_data': execution.trigger_data,
                'current_tasks': execution.current_tasks,
                'completed_tasks': execution.completed_tasks,
                'failed_tasks': execution.failed_tasks,
                'execution_context': execution.execution_context,
                'retry_count': execution.retry_count,
                'error': execution.error
            }
            
            # Update in Redis
            await self.state_manager.update(
                f"workflows.executions.{execution.execution_id}",
                execution_data,
                distributed=True
            )
            
            # Publish state change event
            await self._publish_workflow_event({
                'event_type': 'execution_state_changed',
                'execution_id': execution.execution_id,
                'workflow_id': execution.workflow_id,
                'status': execution.status.value,
                'previous_status': execution_data.get('previous_status', 'unknown')
            })
            
        except Exception as e:
            self.logger.error(f"Error updating execution state: {e}")
    
    async def _complete_workflow_execution(self, execution: WorkflowExecution):
        """Complete workflow execution successfully."""
        try:
            # Calculate execution duration
            duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Update metrics
            self._metrics['workflows_completed'] += 1
            # Update average duration (simple moving average)
            current_avg = self._metrics['average_workflow_duration']
            total_workflows = self._metrics['workflows_executed']
            self._metrics['average_workflow_duration'] = (
                (current_avg * (total_workflows - 1) + duration) / total_workflows
            )
            
            # Store completion event
            completion_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.COORDINATION_EVENT,
                worker_id=self.engine_id,
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={
                    'event_subtype': 'workflow_completed',
                    'execution_id': execution.execution_id,
                    'workflow_id': execution.workflow_id,
                    'duration_seconds': duration,
                    'total_tasks': len(execution.completed_tasks)
                }
            )
            await self.event_store.append_event(completion_event)
            
            # Publish completion event
            await self._publish_workflow_event({
                'event_type': 'workflow_completed',
                'execution_id': execution.execution_id,
                'workflow_id': execution.workflow_id,
                'duration_seconds': duration,
                'completed_tasks': len(execution.completed_tasks)
            })
            
            # Clean up from active executions
            del self._active_executions[execution.execution_id]
            
            self.logger.info(f"Workflow {execution.workflow_id} completed successfully "
                           f"(execution: {execution.execution_id}, duration: {duration:.2f}s)")
            
        except Exception as e:
            self.logger.error(f"Error completing workflow execution: {e}")
    
    async def _fail_workflow_execution(self, execution: WorkflowExecution):
        """Fail workflow execution."""
        try:
            # Calculate execution duration
            duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Store failure event
            failure_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.COORDINATION_EVENT,
                worker_id=self.engine_id,
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.HIGH,
                data={
                    'event_subtype': 'workflow_failed',
                    'execution_id': execution.execution_id,
                    'workflow_id': execution.workflow_id,
                    'duration_seconds': duration,
                    'error': execution.error,
                    'failed_tasks': execution.failed_tasks
                }
            )
            await self.event_store.append_event(failure_event)
            
            # Publish failure event
            await self._publish_workflow_event({
                'event_type': 'workflow_failed',
                'execution_id': execution.execution_id,
                'workflow_id': execution.workflow_id,
                'duration_seconds': duration,
                'error': execution.error,
                'failed_tasks': execution.failed_tasks
            })
            
            # Clean up from active executions
            del self._active_executions[execution.execution_id]
            
            self.logger.error(f"Workflow {execution.workflow_id} failed "
                            f"(execution: {execution.execution_id}, error: {execution.error})")
            
        except Exception as e:
            self.logger.error(f"Error failing workflow execution: {e}")
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register task handler function.
        
        Args:
            task_type: Type of task to handle
            handler: Handler function (async or sync)
        """
        self._task_handlers[task_type] = handler
        self.logger.info(f"Registered task handler for type: {task_type}")
    
    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status.
        
        Args:
            execution_id: Execution ID to check
            
        Returns:
            Execution status dictionary
        """
        if execution_id not in self._active_executions:
            return None
        
        execution = self._active_executions[execution_id]
        workflow = self._workflow_definitions[execution.workflow_id]
        
        # Calculate progress
        total_tasks = len(workflow.tasks)
        completed_tasks = len(execution.completed_tasks)
        failed_tasks = len(execution.failed_tasks)
        progress = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        return {
            'execution_id': execution_id,
            'workflow_id': execution.workflow_id,
            'status': execution.status.value,
            'progress': progress,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'current_tasks': len(execution.current_tasks),
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'duration_seconds': (
                (execution.end_time or datetime.now(timezone.utc)) - execution.start_time
            ).total_seconds(),
            'error': execution.error
        }
    
    async def _register_engine(self):
        """Register engine in distributed registry."""
        engine_data = {
            'engine_id': self.engine_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'max_concurrent_workflows': self.max_concurrent_workflows,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'active_workflows': len(self._active_executions),
            'registered_workflows': len(self._workflow_definitions)
        }
        
        await self.state_manager.update(
            f"workflow_engines.{self.engine_id}",
            engine_data,
            distributed=True
        )
    
    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get comprehensive engine metrics."""
        active_workflows = len([e for e in self._active_executions.values() 
                               if e.status == WorkflowStatus.RUNNING])
        
        return {
            'engine_id': self.engine_id,
            'active_workflows': active_workflows,
            'total_executions': len(self._active_executions),
            'registered_workflows': len(self._workflow_definitions),
            'registered_task_handlers': len(self._task_handlers),
            'performance': self._metrics.copy(),
            'fault_tolerance_enabled': self.enable_fault_tolerance,
            'load_balancing_enabled': self.enable_load_balancing
        }
    
    async def _task_processor(self):
        """Process tasks from task stream."""
        while not self._shutdown:
            try:
                # Read from task stream
                messages = await self.streams_manager.read_messages(
                    self.task_stream,
                    f"processors_{self.engine_id}",
                    count=10,
                    block=1000  # Block for 1 second
                )
                
                for stream_name, message_id, data in messages:
                    # Process task execution results
                    if 'task_id' in data and 'status' in data:
                        await self._process_task_result(data)
                    
                    # Acknowledge message
                    await self.redis_client.xack(stream_name, f"processors_{self.engine_id}", message_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_task_result(self, task_data: Dict[str, Any]):
        """Process task execution result."""
        try:
            task_id = task_data.get('task_id')
            status = task_data.get('status')
            execution_id = task_data.get('execution_id')
            
            if status == 'completed':
                # Create completion event
                event = IntelligenceEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.TASK_COMPLETION,
                    worker_id=task_data.get('worker_id', 'unknown'),
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.NORMAL,
                    data=task_data
                )
                await self._handle_task_completion_event(event, self.task_stream)
                
            elif status == 'failed':
                # Create failure event
                event = IntelligenceEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.TASK_FAILURE,
                    worker_id=task_data.get('worker_id', 'unknown'),
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.HIGH,
                    data=task_data
                )
                await self._handle_task_failure_event(event, self.task_stream)
                
        except Exception as e:
            self.logger.error(f"Error processing task result: {e}")
    
    async def _execution_monitor(self):
        """Monitor workflow executions for timeouts and health."""
        while not self._shutdown:
            try:
                current_time = datetime.now(timezone.utc)
                
                for execution_id, execution in list(self._active_executions.items()):
                    if execution.status != WorkflowStatus.RUNNING:
                        continue
                    
                    workflow = self._workflow_definitions.get(execution.workflow_id)
                    if not workflow:
                        continue
                    
                    # Check for workflow timeout
                    execution_duration = (current_time - execution.start_time).total_seconds()
                    if execution_duration > workflow.timeout_seconds:
                        self.logger.warning(f"Workflow {execution_id} timed out after {execution_duration}s")
                        execution.status = WorkflowStatus.FAILED
                        execution.end_time = current_time
                        execution.error = f"Workflow timed out after {execution_duration}s"
                        await self._fail_workflow_execution(execution)
                        continue
                    
                    # Check for task timeouts
                    for task in workflow.tasks:
                        if (task.status == TaskStatus.RUNNING and 
                            task.start_time is not None):
                            task_duration = (current_time - task.start_time).total_seconds()
                            if task_duration > task.timeout_seconds:
                                self.logger.warning(f"Task {task.task_id} timed out after {task_duration}s")
                                # Trigger task failure
                                event = IntelligenceEvent(
                                    event_id=str(uuid.uuid4()),
                                    event_type=EventType.TASK_FAILURE,
                                    worker_id=task.assigned_worker or 'unknown',
                                    timestamp=current_time,
                                    priority=EventPriority.HIGH,
                                    data={
                                        'task_id': task.task_id,
                                        'execution_id': execution_id,
                                        'error': f'Task timed out after {task_duration}s'
                                    }
                                )
                                await self._handle_task_failure_event(event, self.task_stream)
                
                # Update engine metrics
                await self._update_engine_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in execution monitor: {e}")
                await asyncio.sleep(10)
    
    async def _workflow_scheduler(self):
        """Handle scheduled workflow executions."""
        while not self._shutdown:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check for scheduled workflows
                for workflow_id, workflow in self._workflow_definitions.items():
                    if workflow.trigger == WorkflowTrigger.SCHEDULED and workflow.schedule:
                        # This is a simplified scheduler - in production, use a proper cron library
                        # For now, just check if we should run based on simple interval
                        # TODO: Implement proper cron expression parsing
                        pass
                
                # Check for condition-based workflows
                for workflow_id, workflow in self._workflow_definitions.items():
                    if workflow.trigger == WorkflowTrigger.CONDITION_BASED and workflow.trigger_conditions:
                        # Evaluate trigger conditions
                        # TODO: Implement condition evaluation logic
                        pass
                
                # Sleep for scheduler interval
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in workflow scheduler: {e}")
                await asyncio.sleep(60)
    
    async def _update_engine_metrics(self):
        """Update engine metrics in Redis."""
        try:
            # Calculate current load
            active_tasks = sum(
                len(e.current_tasks) for e in self._active_executions.values()
                if e.status == WorkflowStatus.RUNNING
            )
            self._metrics['current_load'] = (active_tasks / self.max_concurrent_tasks) * 100
            
            # Update task duration average
            total_task_duration = 0
            completed_task_count = 0
            
            for execution in self._active_executions.values():
                workflow = self._workflow_definitions.get(execution.workflow_id)
                if workflow:
                    for task in workflow.tasks:
                        if task.status == TaskStatus.COMPLETED and task.start_time and task.end_time:
                            duration = (task.end_time - task.start_time).total_seconds()
                            total_task_duration += duration
                            completed_task_count += 1
            
            if completed_task_count > 0:
                self._metrics['average_task_duration'] = total_task_duration / completed_task_count
            
            # Store metrics in Redis
            await self.state_manager.update(
                f"workflow_engines.{self.engine_id}.metrics",
                self._metrics,
                distributed=True
            )
            
        except Exception as e:
            self.logger.error(f"Error updating engine metrics: {e}")
    
    async def shutdown(self):
        """Shutdown workflow engine."""
        self.logger.info(f"Shutting down Workflow Engine: {self.engine_id}")
        self._shutdown = True
        
        # Stop processing tasks
        tasks = [
            self._workflow_processor_task,
            self._task_processor_task,
            self._monitor_task,
            self._scheduler_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Update engine status
        await self.state_manager.update(
            f"workflow_engines.{self.engine_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Workflow Engine {self.engine_id} shutdown complete")


# Global workflow engine instance
_global_workflow_engine: Optional[RedisWorkflowEngine] = None


async def get_workflow_engine(**kwargs) -> RedisWorkflowEngine:
    """Get global workflow engine instance."""
    global _global_workflow_engine
    
    if _global_workflow_engine is None:
        _global_workflow_engine = RedisWorkflowEngine(**kwargs)
        await _global_workflow_engine.initialize()
    
    return _global_workflow_engine


async def create_workflow_engine(**kwargs) -> RedisWorkflowEngine:
    """Create new workflow engine instance."""
    engine = RedisWorkflowEngine(**kwargs)
    await engine.initialize()
    return engine


# Alias for backward compatibility
DistributedWorkflowExecutor = RedisWorkflowEngine