"""
Work Item Tracker

Complete lifecycle tracking and audit trails for work items across parallel workers.
Provides comprehensive visibility into work item progression, dependencies,
bottlenecks, and performance analytics.
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import uuid
from scripts.worker_logging_config import setup_worker_logger, WorkerOperationContext


class WorkItemStatus(Enum):
    """Work item lifecycle status."""
    CREATED = "created"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class WorkItemPriority(Enum):
    """Work item priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class WorkItemEvent:
    """Individual event in work item lifecycle."""
    event_id: str
    timestamp: datetime
    status: WorkItemStatus
    worker_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, status: WorkItemStatus, worker_id: str = None, 
               details: Dict[str, Any] = None) -> 'WorkItemEvent':
        """Create new work item event."""
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            status=status,
            worker_id=worker_id,
            details=details or {},
            metadata={}
        )


@dataclass
class DependencyRelation:
    """Dependency relationship between work items."""
    dependent_item_id: str
    dependency_item_id: str
    dependency_type: str  # "blocks", "requires", "follows"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class WorkItemSnapshot:
    """Point-in-time snapshot of work item state."""
    item_id: str
    timestamp: datetime
    status: WorkItemStatus
    assigned_worker: Optional[str]
    progress_percentage: float
    estimated_completion: Optional[datetime]
    current_operation: Optional[str]
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrackedWorkItem:
    """Complete work item with tracking information."""
    item_id: str
    title: str
    description: str
    work_type: str
    priority: WorkItemPriority
    created_at: datetime
    created_by: str
    
    # Current state
    current_status: WorkItemStatus = WorkItemStatus.CREATED
    assigned_worker: Optional[str] = None
    estimated_duration: Optional[float] = None  # seconds
    actual_duration: Optional[float] = None
    progress_percentage: float = 0.0
    
    # Lifecycle tracking
    events: List[WorkItemEvent] = field(default_factory=list)
    status_history: List[Tuple[WorkItemStatus, datetime]] = field(default_factory=list)
    snapshots: List[WorkItemSnapshot] = field(default_factory=list)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # item_ids this depends on
    dependents: List[str] = field(default_factory=list)    # item_ids that depend on this
    
    # Performance data
    time_in_status: Dict[WorkItemStatus, float] = field(default_factory=dict)
    worker_assignments: List[Tuple[str, datetime, Optional[datetime]]] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    completion_result: Optional[Dict[str, Any]] = None
    
    def add_event(self, status: WorkItemStatus, worker_id: str = None, 
                  details: Dict[str, Any] = None):
        """Add new event to work item lifecycle."""
        event = WorkItemEvent.create(status, worker_id, details)
        
        # Calculate duration from previous event
        if self.events:
            last_event = self.events[-1]
            event.duration_seconds = (event.timestamp - last_event.timestamp).total_seconds()
            
            # Update time in status
            last_status = last_event.status
            if last_status not in self.time_in_status:
                self.time_in_status[last_status] = 0
            self.time_in_status[last_status] += event.duration_seconds
        
        self.events.append(event)
        self.status_history.append((status, event.timestamp))
        self.current_status = status
        
        # Update worker assignment tracking
        if worker_id and status == WorkItemStatus.ASSIGNED:
            self.assigned_worker = worker_id
            self.worker_assignments.append((worker_id, event.timestamp, None))
        elif status in [WorkItemStatus.COMPLETED, WorkItemStatus.FAILED, WorkItemStatus.CANCELLED]:
            # Close current worker assignment
            if self.worker_assignments and self.worker_assignments[-1][2] is None:
                worker, start_time, _ = self.worker_assignments[-1]
                self.worker_assignments[-1] = (worker, start_time, event.timestamp)
            self.assigned_worker = None
    
    def calculate_total_duration(self) -> Optional[float]:
        """Calculate total time from creation to completion."""
        if self.current_status not in [WorkItemStatus.COMPLETED, WorkItemStatus.FAILED, WorkItemStatus.CANCELLED]:
            return None
        
        if not self.events:
            return None
        
        start_time = self.created_at
        end_time = self.events[-1].timestamp
        return (end_time - start_time).total_seconds()
    
    def calculate_active_duration(self) -> float:
        """Calculate time spent actively working (excluding queued/blocked time)."""
        active_statuses = {
            WorkItemStatus.ASSIGNED,
            WorkItemStatus.IN_PROGRESS
        }
        
        total_active = 0.0
        for status, duration in self.time_in_status.items():
            if status in active_statuses:
                total_active += duration
        
        return total_active
    
    def get_current_bottleneck(self) -> Optional[Dict[str, Any]]:
        """Identify current bottleneck if item is blocked or delayed."""
        if self.current_status == WorkItemStatus.BLOCKED:
            return {
                'type': 'blocked',
                'reason': 'Work item is explicitly blocked',
                'since': self.events[-1].timestamp.isoformat()
            }
        
        # Check if waiting for dependencies
        unresolved_deps = [dep for dep in self.dependencies if not self._is_dependency_resolved(dep)]
        if unresolved_deps:
            return {
                'type': 'dependency',
                'reason': f'Waiting for {len(unresolved_deps)} dependencies',
                'blocking_items': unresolved_deps
            }
        
        # Check if item has been in same status too long
        if self.events:
            last_event = self.events[-1]
            time_since_change = (datetime.now(timezone.utc) - last_event.timestamp).total_seconds()
            
            # Define "too long" thresholds for each status
            thresholds = {
                WorkItemStatus.QUEUED: 3600,      # 1 hour
                WorkItemStatus.ASSIGNED: 1800,    # 30 minutes
                WorkItemStatus.IN_PROGRESS: 7200  # 2 hours
            }
            
            threshold = thresholds.get(self.current_status)
            if threshold and time_since_change > threshold:
                return {
                    'type': 'timeout',
                    'reason': f'Item stuck in {self.current_status.value} for {time_since_change/3600:.1f} hours',
                    'threshold_hours': threshold/3600
                }
        
        return None
    
    def _is_dependency_resolved(self, dependency_id: str) -> bool:
        """Check if dependency is resolved (placeholder - would integrate with tracker)."""
        # This would be implemented by the tracker to check dependency status
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'item_id': self.item_id,
            'title': self.title,
            'description': self.description,
            'work_type': self.work_type,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'current_status': self.current_status.value,
            'assigned_worker': self.assigned_worker,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'progress_percentage': self.progress_percentage,
            'events': [asdict(event) for event in self.events],
            'dependencies': self.dependencies,
            'dependents': self.dependents,
            'time_in_status': {k.value: v for k, v in self.time_in_status.items()},
            'tags': list(self.tags),
            'context': self.context,
            'completion_result': self.completion_result
        }


class WorkItemTracker:
    """Comprehensive work item lifecycle tracker."""
    
    def __init__(self, max_completed_items: int = 5000):
        """Initialize work item tracker.
        
        Args:
            max_completed_items: Maximum completed items to keep in memory
        """
        self.max_completed_items = max_completed_items
        self.logger = setup_worker_logger("work_item_tracker")
        
        # Work item storage
        self.active_items: Dict[str, TrackedWorkItem] = {}
        self.completed_items: deque = deque(maxlen=max_completed_items)
        self.dependencies: Dict[str, List[DependencyRelation]] = defaultdict(list)
        
        # Performance tracking
        self.performance_cache: Dict[str, Any] = {}
        self.bottleneck_analysis: Dict[str, Any] = {}
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Analytics data
        self.completion_stats: Dict[str, Any] = defaultdict(int)
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    async def start_monitoring(self):
        """Start background monitoring and analysis."""
        self.logger.info("Starting work item tracker monitoring")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self.logger.info("Stopping work item tracker monitoring")
        self._shutdown = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    def create_work_item(self, title: str, description: str, work_type: str,
                        priority: WorkItemPriority = WorkItemPriority.MEDIUM,
                        created_by: str = "system",
                        estimated_duration: float = None,
                        dependencies: List[str] = None,
                        tags: Set[str] = None,
                        context: Dict[str, Any] = None) -> str:
        """Create new work item for tracking.
        
        Args:
            title: Work item title
            description: Detailed description
            work_type: Type of work (feature, bug_fix, research, etc.)
            priority: Work item priority
            created_by: Creator identifier
            estimated_duration: Estimated duration in seconds
            dependencies: List of item IDs this depends on
            tags: Set of tags for categorization
            context: Additional context data
            
        Returns:
            Work item ID
        """
        item_id = str(uuid.uuid4())
        
        work_item = TrackedWorkItem(
            item_id=item_id,
            title=title,
            description=description,
            work_type=work_type,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            estimated_duration=estimated_duration,
            dependencies=dependencies or [],
            tags=tags or set(),
            context=context or {}
        )
        
        # Add creation event
        work_item.add_event(WorkItemStatus.CREATED, details={
            'created_by': created_by,
            'work_type': work_type,
            'estimated_duration': estimated_duration
        })
        
        with self.lock:
            self.active_items[item_id] = work_item
            
            # Register dependencies
            if dependencies:
                for dep_id in dependencies:
                    self.dependencies[dep_id].append(DependencyRelation(
                        dependent_item_id=item_id,
                        dependency_item_id=dep_id,
                        dependency_type="requires"
                    ))
                    
                    # Update dependent's dependents list if it exists
                    if dep_id in self.active_items:
                        self.active_items[dep_id].dependents.append(item_id)
        
        self.logger.info(f"Created work item {item_id}: {title}")
        return item_id
    
    def update_status(self, item_id: str, new_status: WorkItemStatus,
                     worker_id: str = None, details: Dict[str, Any] = None):
        """Update work item status.
        
        Args:
            item_id: Work item ID
            new_status: New status to set
            worker_id: Worker making the update
            details: Additional details about the status change
        """
        with self.lock:
            if item_id not in self.active_items:
                self.logger.warning(f"Cannot update status for unknown work item: {item_id}")
                return
            
            work_item = self.active_items[item_id]
            old_status = work_item.current_status
            
            # Add status change event
            work_item.add_event(new_status, worker_id, details)
            
            # Handle completion
            if new_status in [WorkItemStatus.COMPLETED, WorkItemStatus.FAILED, WorkItemStatus.CANCELLED]:
                work_item.actual_duration = work_item.calculate_total_duration()
                work_item.progress_percentage = 100.0 if new_status == WorkItemStatus.COMPLETED else work_item.progress_percentage
                
                # Move to completed items
                self.completed_items.append(work_item)
                del self.active_items[item_id]
                
                # Resolve dependencies for completed items
                if new_status == WorkItemStatus.COMPLETED:
                    self._resolve_dependencies(item_id)
                
                # Update completion stats
                self.completion_stats[f"{work_item.work_type}_{new_status.value}"] += 1
                
                # Record performance trends
                if work_item.actual_duration:
                    self.performance_trends[work_item.work_type].append(work_item.actual_duration)
        
        self.logger.info(f"Work item {item_id} status: {old_status.value} -> {new_status.value}")
    
    def update_progress(self, item_id: str, progress_percentage: float,
                       current_operation: str = None, worker_id: str = None):
        """Update work item progress.
        
        Args:
            item_id: Work item ID
            progress_percentage: Progress percentage (0-100)
            current_operation: Description of current operation
            worker_id: Worker reporting progress
        """
        with self.lock:
            if item_id not in self.active_items:
                return
            
            work_item = self.active_items[item_id]
            work_item.progress_percentage = max(0, min(100, progress_percentage))
            
            # Create snapshot
            snapshot = WorkItemSnapshot(
                item_id=item_id,
                timestamp=datetime.now(timezone.utc),
                status=work_item.current_status,
                assigned_worker=work_item.assigned_worker,
                progress_percentage=progress_percentage,
                estimated_completion=self._estimate_completion(work_item),
                current_operation=current_operation
            )
            
            work_item.snapshots.append(snapshot)
            
            # Keep only recent snapshots
            if len(work_item.snapshots) > 50:
                work_item.snapshots.pop(0)
    
    def assign_to_worker(self, item_id: str, worker_id: str):
        """Assign work item to worker."""
        self.update_status(item_id, WorkItemStatus.ASSIGNED, worker_id, {
            'assigned_to': worker_id,
            'assignment_time': datetime.now(timezone.utc).isoformat()
        })
    
    def start_work(self, item_id: str, worker_id: str):
        """Mark work item as in progress."""
        self.update_status(item_id, WorkItemStatus.IN_PROGRESS, worker_id, {
            'started_by': worker_id,
            'start_time': datetime.now(timezone.utc).isoformat()
        })
    
    def complete_work(self, item_id: str, worker_id: str, 
                     result: Dict[str, Any] = None):
        """Mark work item as completed."""
        with self.lock:
            if item_id in self.active_items:
                self.active_items[item_id].completion_result = result
        
        self.update_status(item_id, WorkItemStatus.COMPLETED, worker_id, {
            'completed_by': worker_id,
            'completion_time': datetime.now(timezone.utc).isoformat(),
            'result': result
        })
    
    def fail_work(self, item_id: str, worker_id: str, 
                 error: str, error_details: Dict[str, Any] = None):
        """Mark work item as failed."""
        self.update_status(item_id, WorkItemStatus.FAILED, worker_id, {
            'failed_by': worker_id,
            'failure_time': datetime.now(timezone.utc).isoformat(),
            'error': error,
            'error_details': error_details or {}
        })
    
    def block_work(self, item_id: str, reason: str, blocked_by: str = None):
        """Block work item with reason."""
        self.update_status(item_id, WorkItemStatus.BLOCKED, blocked_by, {
            'blocked_by': blocked_by,
            'block_reason': reason,
            'blocked_time': datetime.now(timezone.utc).isoformat()
        })
    
    def add_dependency(self, item_id: str, dependency_id: str, 
                      dependency_type: str = "requires"):
        """Add dependency relationship."""
        with self.lock:
            if item_id not in self.active_items:
                return
            
            # Add to dependency tracking
            relation = DependencyRelation(
                dependent_item_id=item_id,
                dependency_item_id=dependency_id,
                dependency_type=dependency_type
            )
            
            self.dependencies[dependency_id].append(relation)
            
            # Update work items
            self.active_items[item_id].dependencies.append(dependency_id)
            if dependency_id in self.active_items:
                self.active_items[dependency_id].dependents.append(item_id)
        
        self.logger.info(f"Added dependency: {item_id} {dependency_type} {dependency_id}")
    
    def _resolve_dependencies(self, completed_item_id: str):
        """Resolve dependencies when item completes."""
        if completed_item_id not in self.dependencies:
            return
        
        for relation in self.dependencies[completed_item_id]:
            relation.resolved = True
            relation.resolved_at = datetime.now(timezone.utc)
            
            # Check if dependent item can now proceed
            dependent_id = relation.dependent_item_id
            if dependent_id in self.active_items:
                dependent_item = self.active_items[dependent_id]
                
                # Check if all dependencies are resolved
                all_resolved = all(
                    self._is_dependency_resolved(dep_id) 
                    for dep_id in dependent_item.dependencies
                )
                
                if all_resolved and dependent_item.current_status == WorkItemStatus.BLOCKED:
                    self.update_status(dependent_id, WorkItemStatus.QUEUED, details={
                        'unblocked_reason': f'All dependencies resolved (last: {completed_item_id})'
                    })
    
    def _is_dependency_resolved(self, dependency_id: str) -> bool:
        """Check if dependency is resolved."""
        if dependency_id not in self.dependencies:
            return True  # No dependency relations = resolved
        
        return any(
            relation.resolved 
            for relation in self.dependencies[dependency_id]
        )
    
    def _estimate_completion(self, work_item: TrackedWorkItem) -> Optional[datetime]:
        """Estimate completion time based on progress and historical data."""
        if work_item.progress_percentage <= 0:
            return None
        
        # Use estimated duration if available
        if work_item.estimated_duration:
            total_estimated = work_item.estimated_duration
        else:
            # Use historical average for work type
            if work_item.work_type in self.performance_trends:
                recent_durations = list(self.performance_trends[work_item.work_type])
                if recent_durations:
                    total_estimated = sum(recent_durations) / len(recent_durations)
                else:
                    return None
            else:
                return None
        
        # Calculate based on current progress
        time_elapsed = work_item.calculate_active_duration()
        estimated_total = time_elapsed / (work_item.progress_percentage / 100.0)
        remaining_time = estimated_total - time_elapsed
        
        return datetime.now(timezone.utc) + timedelta(seconds=remaining_time)
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._shutdown:
            try:
                with WorkerOperationContext("work_item_tracker", "monitoring"):
                    await self._perform_monitoring()
                    await asyncio.sleep(30)  # Monitor every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _perform_monitoring(self):
        """Perform monitoring checks."""
        with self.lock:
            # Check for bottlenecks
            self._analyze_bottlenecks()
            
            # Update performance cache
            self._update_performance_cache()
            
            # Check for stale items
            self._check_stale_items()
    
    def _analyze_bottlenecks(self):
        """Analyze current bottlenecks in work items."""
        bottlenecks = {}
        
        for item_id, work_item in self.active_items.items():
            current_bottleneck = work_item.get_current_bottleneck()
            if current_bottleneck:
                bottlenecks[item_id] = current_bottleneck
        
        # Group by bottleneck type
        bottleneck_summary = defaultdict(list)
        for item_id, bottleneck in bottlenecks.items():
            bottleneck_summary[bottleneck['type']].append(item_id)
        
        self.bottleneck_analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_bottlenecks': len(bottlenecks),
            'by_type': dict(bottleneck_summary),
            'details': bottlenecks
        }
        
        if bottlenecks:
            self.logger.warning(f"Found {len(bottlenecks)} work item bottlenecks")
    
    def _update_performance_cache(self):
        """Update performance metrics cache."""
        with self.lock:
            active_count = len(self.active_items)
            completed_count = len(self.completed_items)
            
            # Calculate average durations by work type
            avg_durations = {}
            for work_type, durations in self.performance_trends.items():
                if durations:
                    avg_durations[work_type] = sum(durations) / len(durations)
            
            # Calculate current throughput (items per hour)
            recent_completions = [
                item for item in self.completed_items
                if (datetime.now(timezone.utc) - item.events[-1].timestamp).total_seconds() < 3600
            ]
            throughput = len(recent_completions)
            
            self.performance_cache = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'active_items': active_count,
                'completed_items': completed_count,
                'average_durations': avg_durations,
                'current_throughput': throughput,
                'completion_stats': dict(self.completion_stats)
            }
    
    def _check_stale_items(self):
        """Check for stale work items that need attention."""
        stale_threshold = timedelta(hours=4)
        current_time = datetime.now(timezone.utc)
        
        stale_items = []
        for item_id, work_item in self.active_items.items():
            if work_item.events:
                last_update = work_item.events[-1].timestamp
                if current_time - last_update > stale_threshold:
                    stale_items.append(item_id)
        
        if stale_items:
            self.logger.warning(f"Found {len(stale_items)} stale work items: {stale_items}")
    
    def get_work_item(self, item_id: str) -> Optional[TrackedWorkItem]:
        """Get work item by ID."""
        with self.lock:
            if item_id in self.active_items:
                return self.active_items[item_id]
            
            # Search in completed items
            for item in self.completed_items:
                if item.item_id == item_id:
                    return item
            
            return None
    
    def get_active_items_by_status(self, status: WorkItemStatus) -> List[TrackedWorkItem]:
        """Get active items with specific status."""
        with self.lock:
            return [
                item for item in self.active_items.values()
                if item.current_status == status
            ]
    
    def get_items_by_worker(self, worker_id: str) -> List[TrackedWorkItem]:
        """Get items assigned to specific worker."""
        with self.lock:
            return [
                item for item in self.active_items.values()
                if item.assigned_worker == worker_id
            ]
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        with self.lock:
            # Status distribution
            status_distribution = defaultdict(int)
            for item in self.active_items.values():
                status_distribution[item.current_status.value] += 1
            
            # Priority distribution
            priority_distribution = defaultdict(int)
            for item in self.active_items.values():
                priority_distribution[item.priority.value] += 1
            
            # Worker load distribution
            worker_loads = defaultdict(int)
            for item in self.active_items.values():
                if item.assigned_worker:
                    worker_loads[item.assigned_worker] += 1
            
            # Dependency analysis
            total_dependencies = sum(len(item.dependencies) for item in self.active_items.values())
            blocked_by_deps = len([
                item for item in self.active_items.values()
                if item.current_status == WorkItemStatus.BLOCKED and item.dependencies
            ])
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'active_items': len(self.active_items),
                    'completed_items': len(self.completed_items),
                    'total_dependencies': total_dependencies,
                    'blocked_by_dependencies': blocked_by_deps
                },
                'distributions': {
                    'status': dict(status_distribution),
                    'priority': dict(priority_distribution),
                    'worker_load': dict(worker_loads)
                },
                'performance': self.performance_cache,
                'bottlenecks': self.bottleneck_analysis
            }
    
    def export_audit_trail(self, item_id: str) -> Dict[str, Any]:
        """Export complete audit trail for work item."""
        work_item = self.get_work_item(item_id)
        if not work_item:
            return {}
        
        return {
            'item_id': item_id,
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'work_item': work_item.to_dict(),
            'audit_trail': {
                'total_events': len(work_item.events),
                'status_changes': work_item.status_history,
                'worker_assignments': work_item.worker_assignments,
                'time_breakdown': work_item.time_in_status,
                'performance_snapshots': [asdict(s) for s in work_item.snapshots]
            },
            'dependencies': {
                'depends_on': work_item.dependencies,
                'dependents': work_item.dependents
            }
        }


# Context manager for automatic work item tracking
class WorkItemContext:
    """Context manager for automatic work item lifecycle tracking."""
    
    def __init__(self, tracker: WorkItemTracker, item_id: str, worker_id: str):
        """Initialize work item context.
        
        Args:
            tracker: WorkItemTracker instance
            item_id: Work item ID
            worker_id: Worker performing the work
        """
        self.tracker = tracker
        self.item_id = item_id
        self.worker_id = worker_id
        self.success = False
    
    def __enter__(self):
        """Start work item execution."""
        self.tracker.start_work(self.item_id, self.worker_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete work item execution."""
        if exc_type is None:
            self.tracker.complete_work(self.item_id, self.worker_id, {
                'success': True,
                'completed_at': datetime.now(timezone.utc).isoformat()
            })
        else:
            self.tracker.fail_work(self.item_id, self.worker_id, 
                                 str(exc_val), {'exception_type': exc_type.__name__})
    
    def update_progress(self, percentage: float, operation: str = None):
        """Update progress during execution."""
        self.tracker.update_progress(self.item_id, percentage, operation, self.worker_id)


# Example usage and demonstration
async def demonstrate_work_item_tracker():
    """Demonstrate work item tracker capabilities."""
    tracker = WorkItemTracker()
    await tracker.start_monitoring()
    
    try:
        # Create some work items
        item1_id = tracker.create_work_item(
            title="Implement user authentication",
            description="Add JWT-based authentication system",
            work_type="feature",
            priority=WorkItemPriority.HIGH,
            estimated_duration=3600,  # 1 hour
            tags={"authentication", "security"}
        )
        
        item2_id = tracker.create_work_item(
            title="Write authentication tests",
            description="Create comprehensive test suite for auth system",
            work_type="testing",
            priority=WorkItemPriority.MEDIUM,
            dependencies=[item1_id],  # Depends on auth implementation
            estimated_duration=1800  # 30 minutes
        )
        
        # Simulate work execution
        worker_id = "worker_1"
        
        # Process first item
        tracker.assign_to_worker(item1_id, worker_id)
        
        with WorkItemContext(tracker, item1_id, worker_id) as ctx:
            # Simulate progressive work
            await asyncio.sleep(0.1)
            ctx.update_progress(25, "Setting up JWT library")
            
            await asyncio.sleep(0.1)
            ctx.update_progress(50, "Implementing login endpoint")
            
            await asyncio.sleep(0.1)
            ctx.update_progress(75, "Adding token validation")
            
            await asyncio.sleep(0.1)
            ctx.update_progress(100, "Authentication system complete")
        
        # Process dependent item
        tracker.assign_to_worker(item2_id, worker_id)
        
        with WorkItemContext(tracker, item2_id, worker_id) as ctx:
            await asyncio.sleep(0.1)
            ctx.update_progress(50, "Writing unit tests")
            
            await asyncio.sleep(0.1)
            ctx.update_progress(100, "Tests complete")
        
        # Wait for monitoring
        await asyncio.sleep(1)
        
        # Get analytics
        analytics = tracker.get_analytics_summary()
        print("Analytics Summary:")
        print(json.dumps(analytics, indent=2, default=str))
        
        # Get audit trail
        audit_trail = tracker.export_audit_trail(item1_id)
        print(f"\nAudit Trail for {item1_id}:")
        print(json.dumps(audit_trail, indent=2, default=str))
        
    finally:
        await tracker.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(demonstrate_work_item_tracker())