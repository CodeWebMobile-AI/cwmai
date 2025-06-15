"""
Task Coordinator for Continuous 24/7 AI System

Manages task dependencies, prevents conflicts, and coordinates
work distribution across parallel workers.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

from scripts.continuous_orchestrator import WorkItem, TaskPriority
from scripts.ai_brain import AIBrain


class ConflictType(Enum):
    """Types of task conflicts."""
    RESOURCE_CONFLICT = "resource_conflict"      # Tasks competing for same resource
    DEPENDENCY_CONFLICT = "dependency_conflict"  # Circular or invalid dependencies
    REPOSITORY_CONFLICT = "repository_conflict"  # Multiple tasks modifying same repo
    TIMING_CONFLICT = "timing_conflict"          # Tasks that can't run simultaneously
    SEMANTIC_CONFLICT = "semantic_conflict"      # Tasks with conflicting goals


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""
    dependent_task_id: str
    prerequisite_task_id: str
    dependency_type: str = "completion"  # completion, resource, timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TaskConflict:
    """Represents a conflict between tasks."""
    conflict_id: str
    task_ids: List[str]
    conflict_type: ConflictType
    severity: str  # low, medium, high, critical
    description: str
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TaskLock:
    """Represents a resource lock held by a task."""
    resource_id: str
    task_id: str
    worker_id: str
    lock_type: str  # exclusive, shared
    acquired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


class TaskCoordinator:
    """Coordinates task execution to prevent conflicts and manage dependencies."""
    
    def __init__(self, ai_brain: AIBrain):
        """Initialize the task coordinator.
        
        Args:
            ai_brain: AI brain for intelligent conflict resolution
        """
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Task management
        self.active_tasks: Dict[str, WorkItem] = {}
        self.completed_tasks: Dict[str, WorkItem] = {}
        self.failed_tasks: Dict[str, WorkItem] = {}
        
        # Dependency management
        self.dependencies: Dict[str, List[TaskDependency]] = {}
        self.reverse_dependencies: Dict[str, List[str]] = {}  # What depends on this task
        
        # Conflict management
        self.conflicts: Dict[str, TaskConflict] = {}
        self.conflict_resolution_strategies: Dict[ConflictType, callable] = {
            ConflictType.RESOURCE_CONFLICT: self._resolve_resource_conflict,
            ConflictType.DEPENDENCY_CONFLICT: self._resolve_dependency_conflict,
            ConflictType.REPOSITORY_CONFLICT: self._resolve_repository_conflict,
            ConflictType.TIMING_CONFLICT: self._resolve_timing_conflict,
            ConflictType.SEMANTIC_CONFLICT: self._resolve_semantic_conflict
        }
        
        # Resource locking
        self.resource_locks: Dict[str, TaskLock] = {}
        self.lock_timeout = 3600  # 1 hour default lock timeout
        
        # Coordination metrics
        self.metrics = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'dependencies_managed': 0,
            'resource_locks_acquired': 0,
            'coordination_decisions': 0,
            'tasks_coordinated': 0
        }
        
        # Background task for cleanup and monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start_coordination(self):
        """Start the task coordination system."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._coordination_loop())
        
        self.logger.info("Task coordinator started")
    
    async def stop_coordination(self):
        """Stop the task coordination system."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Task coordinator stopped")
    
    async def _coordination_loop(self):
        """Main coordination monitoring loop."""
        while self.monitoring_active:
            try:
                # Clean up expired locks
                await self._cleanup_expired_locks()
                
                # Check for new conflicts
                await self._detect_conflicts()
                
                # Resolve pending conflicts
                await self._resolve_pending_conflicts()
                
                # Update dependency status
                await self._update_dependency_status()
                
                # Clean up completed tasks
                await self._cleanup_old_tasks()
                
                # Wait before next check
                await asyncio.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(30)  # Longer pause on error
    
    async def coordinate_task_execution(self, work_item: WorkItem, 
                                      worker_id: str) -> Tuple[bool, Optional[str]]:
        """Coordinate the execution of a task.
        
        Args:
            work_item: The work item to coordinate
            worker_id: ID of the worker requesting execution
            
        Returns:
            Tuple of (can_execute, reason_if_not)
        """
        self.metrics['coordination_decisions'] += 1
        
        # Check dependencies
        can_execute, dependency_reason = await self._check_dependencies(work_item)
        if not can_execute:
            return False, f"Dependency not met: {dependency_reason}"
        
        # Check for conflicts
        conflicts = await self._check_conflicts(work_item, worker_id)
        if conflicts:
            conflict_descriptions = [c.description for c in conflicts]
            return False, f"Conflicts detected: {'; '.join(conflict_descriptions)}"
        
        # Acquire necessary resource locks
        locks_acquired, lock_reason = await self._acquire_resource_locks(work_item, worker_id)
        if not locks_acquired:
            return False, f"Resource lock failed: {lock_reason}"
        
        # Register the task as active
        self.active_tasks[work_item.id] = work_item
        self.metrics['tasks_coordinated'] += 1
        
        self.logger.info(f"Task {work_item.id} cleared for execution by worker {worker_id}")
        return True, None
    
    async def report_task_completion(self, work_item: WorkItem, 
                                   success: bool, result: Dict[str, Any] = None):
        """Report completion of a task.
        
        Args:
            work_item: The completed work item
            success: Whether the task completed successfully
            result: Task execution result
        """
        # Move from active to completed/failed
        if work_item.id in self.active_tasks:
            del self.active_tasks[work_item.id]
        
        if success:
            self.completed_tasks[work_item.id] = work_item
            self.logger.info(f"Task {work_item.id} completed successfully")
        else:
            self.failed_tasks[work_item.id] = work_item
            self.logger.warning(f"Task {work_item.id} failed")
        
        # Release resource locks
        await self._release_task_locks(work_item.id)
        
        # Update dependents
        await self._notify_dependents(work_item.id, success)
        
        # Generate follow-up tasks if needed
        if success and result:
            await self._generate_followup_coordination(work_item, result)
    
    async def add_task_dependency(self, dependent_task_id: str, 
                                prerequisite_task_id: str,
                                dependency_type: str = "completion"):
        """Add a dependency between tasks.
        
        Args:
            dependent_task_id: Task that depends on another
            prerequisite_task_id: Task that must complete first
            dependency_type: Type of dependency
        """
        dependency = TaskDependency(
            dependent_task_id=dependent_task_id,
            prerequisite_task_id=prerequisite_task_id,
            dependency_type=dependency_type
        )
        
        if dependent_task_id not in self.dependencies:
            self.dependencies[dependent_task_id] = []
        self.dependencies[dependent_task_id].append(dependency)
        
        if prerequisite_task_id not in self.reverse_dependencies:
            self.reverse_dependencies[prerequisite_task_id] = []
        self.reverse_dependencies[prerequisite_task_id].append(dependent_task_id)
        
        self.metrics['dependencies_managed'] += 1
        
        # Check for circular dependencies
        if await self._has_circular_dependency(dependent_task_id, prerequisite_task_id):
            self.logger.warning(f"Circular dependency detected: {dependent_task_id} <-> {prerequisite_task_id}")
            await self._create_conflict(
                [dependent_task_id, prerequisite_task_id],
                ConflictType.DEPENDENCY_CONFLICT,
                "critical",
                f"Circular dependency between {dependent_task_id} and {prerequisite_task_id}"
            )
    
    async def _check_dependencies(self, work_item: WorkItem) -> Tuple[bool, Optional[str]]:
        """Check if all dependencies for a task are satisfied.
        
        Args:
            work_item: Work item to check
            
        Returns:
            Tuple of (dependencies_met, reason_if_not)
        """
        if work_item.id not in self.dependencies:
            return True, None
        
        for dependency in self.dependencies[work_item.id]:
            prerequisite_id = dependency.prerequisite_task_id
            
            if dependency.dependency_type == "completion":
                if prerequisite_id not in self.completed_tasks:
                    return False, f"Prerequisite task {prerequisite_id} not completed"
            elif dependency.dependency_type == "resource":
                # Check if prerequisite has released necessary resources
                if prerequisite_id in self.active_tasks:
                    return False, f"Prerequisite task {prerequisite_id} still holds resources"
            elif dependency.dependency_type == "timing":
                # Check timing constraints
                if prerequisite_id in self.active_tasks:
                    return False, f"Timing dependency: {prerequisite_id} must complete first"
        
        return True, None
    
    async def _check_conflicts(self, work_item: WorkItem, worker_id: str) -> List[TaskConflict]:
        """Check for conflicts with other active tasks.
        
        Args:
            work_item: Work item to check
            worker_id: Worker requesting execution
            
        Returns:
            List of conflicts detected
        """
        conflicts = []
        
        for active_task_id, active_task in self.active_tasks.items():
            # Repository conflict
            if (work_item.repository and active_task.repository and 
                work_item.repository == active_task.repository):
                
                conflict = await self._create_conflict(
                    [work_item.id, active_task_id],
                    ConflictType.REPOSITORY_CONFLICT,
                    "medium",
                    f"Both tasks operate on repository {work_item.repository}"
                )
                conflicts.append(conflict)
            
            # Semantic conflict detection using AI
            if await self._detect_semantic_conflict(work_item, active_task):
                conflict = await self._create_conflict(
                    [work_item.id, active_task_id],
                    ConflictType.SEMANTIC_CONFLICT,
                    "high",
                    "Tasks have conflicting objectives or approaches"
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_semantic_conflict(self, task1: WorkItem, task2: WorkItem) -> bool:
        """Use AI to detect semantic conflicts between tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            True if tasks have semantic conflicts
        """
        try:
            # Create a prompt for conflict detection
            prompt = f"""
            Analyze these two tasks for potential conflicts:
            
            Task 1: {task1.title}
            Description: {task1.description}
            Type: {task1.task_type}
            Repository: {task1.repository}
            
            Task 2: {task2.title}
            Description: {task2.description}
            Type: {task2.task_type}
            Repository: {task2.repository}
            
            Do these tasks conflict in goals, approach, or implementation?
            Consider: overlapping features, contradictory changes, resource competition.
            
            Respond with: CONFLICT or NO_CONFLICT and brief reason.
            """
            
            # Use AI brain to analyze
            response = await self.ai_brain.process_prompt(
                prompt,
                context={"analysis_type": "conflict_detection"}
            )
            
            return "CONFLICT" in response.get('content', '').upper()
            
        except Exception as e:
            self.logger.error(f"Error in semantic conflict detection: {e}")
            return False  # Default to no conflict if analysis fails
    
    async def _acquire_resource_locks(self, work_item: WorkItem, 
                                    worker_id: str) -> Tuple[bool, Optional[str]]:
        """Acquire necessary resource locks for a task.
        
        Args:
            work_item: Work item requiring resources
            worker_id: Worker requesting locks
            
        Returns:
            Tuple of (locks_acquired, reason_if_not)
        """
        required_locks = self._identify_required_locks(work_item)
        acquired_locks = []
        
        try:
            for resource_id, lock_type in required_locks:
                # Check if resource is already locked
                if resource_id in self.resource_locks:
                    existing_lock = self.resource_locks[resource_id]
                    
                    # Check if lock is exclusive or incompatible
                    if (existing_lock.lock_type == "exclusive" or 
                        lock_type == "exclusive" or
                        existing_lock.task_id != work_item.id):
                        
                        # Release any locks we've acquired
                        for lock_id in acquired_locks:
                            del self.resource_locks[lock_id]
                        
                        return False, f"Resource {resource_id} locked by task {existing_lock.task_id}"
                
                # Acquire the lock
                lock = TaskLock(
                    resource_id=resource_id,
                    task_id=work_item.id,
                    worker_id=worker_id,
                    lock_type=lock_type,
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.lock_timeout)
                )
                
                self.resource_locks[resource_id] = lock
                acquired_locks.append(resource_id)
                self.metrics['resource_locks_acquired'] += 1
            
            return True, None
            
        except Exception as e:
            # Release any locks we've acquired
            for lock_id in acquired_locks:
                if lock_id in self.resource_locks:
                    del self.resource_locks[lock_id]
            
            return False, f"Error acquiring locks: {e}"
    
    def _identify_required_locks(self, work_item: WorkItem) -> List[Tuple[str, str]]:
        """Identify what resource locks a task needs.
        
        Args:
            work_item: Work item to analyze
            
        Returns:
            List of (resource_id, lock_type) tuples
        """
        locks = []
        
        # Repository lock
        if work_item.repository:
            if work_item.task_type in ["FEATURE", "BUG_FIX", "REFACTORING"]:
                locks.append((f"repo:{work_item.repository}", "exclusive"))
            else:
                locks.append((f"repo:{work_item.repository}", "shared"))
        
        # System-wide locks for certain operations
        if work_item.task_type == "SYSTEM_IMPROVEMENT":
            locks.append(("system:core", "exclusive"))
        elif work_item.task_type == "NEW_PROJECT":
            locks.append(("system:project_creation", "exclusive"))
        
        return locks
    
    async def _release_task_locks(self, task_id: str):
        """Release all locks held by a task.
        
        Args:
            task_id: Task whose locks should be released
        """
        locks_to_release = [
            resource_id for resource_id, lock in self.resource_locks.items()
            if lock.task_id == task_id
        ]
        
        for resource_id in locks_to_release:
            del self.resource_locks[resource_id]
            self.logger.debug(f"Released lock on {resource_id} for task {task_id}")
    
    async def _cleanup_expired_locks(self):
        """Clean up expired resource locks."""
        now = datetime.now(timezone.utc)
        expired_locks = [
            resource_id for resource_id, lock in self.resource_locks.items()
            if lock.expires_at and lock.expires_at < now
        ]
        
        for resource_id in expired_locks:
            lock = self.resource_locks[resource_id]
            del self.resource_locks[resource_id]
            self.logger.warning(f"Released expired lock on {resource_id} (task: {lock.task_id})")
    
    async def _create_conflict(self, task_ids: List[str], conflict_type: ConflictType,
                             severity: str, description: str) -> TaskConflict:
        """Create a new task conflict.
        
        Args:
            task_ids: IDs of conflicting tasks
            conflict_type: Type of conflict
            severity: Severity level
            description: Conflict description
            
        Returns:
            Created conflict object
        """
        conflict_id = f"conflict_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        conflict = TaskConflict(
            conflict_id=conflict_id,
            task_ids=task_ids,
            conflict_type=conflict_type,
            severity=severity,
            description=description
        )
        
        self.conflicts[conflict_id] = conflict
        self.metrics['conflicts_detected'] += 1
        
        self.logger.warning(f"Conflict detected: {description} (tasks: {', '.join(task_ids)})")
        
        return conflict
    
    async def _detect_conflicts(self):
        """Detect new conflicts among active tasks."""
        active_task_list = list(self.active_tasks.values())
        
        for i in range(len(active_task_list)):
            for j in range(i + 1, len(active_task_list)):
                task1 = active_task_list[i]
                task2 = active_task_list[j]
                
                # Check for repository conflicts
                if (task1.repository and task2.repository and 
                    task1.repository == task2.repository and
                    task1.task_type in ["FEATURE", "BUG_FIX"] and
                    task2.task_type in ["FEATURE", "BUG_FIX"]):
                    
                    # Check if conflict already exists
                    existing_conflict = any(
                        set([task1.id, task2.id]).issubset(c.task_ids)
                        for c in self.conflicts.values()
                        if not c.resolved
                    )
                    
                    if not existing_conflict:
                        await self._create_conflict(
                            [task1.id, task2.id],
                            ConflictType.REPOSITORY_CONFLICT,
                            "high",
                            f"Both tasks modifying repository {task1.repository}"
                        )
    
    async def _resolve_pending_conflicts(self):
        """Resolve pending conflicts using appropriate strategies."""
        unresolved_conflicts = [c for c in self.conflicts.values() if not c.resolved]
        
        for conflict in unresolved_conflicts:
            try:
                strategy = self.conflict_resolution_strategies.get(conflict.conflict_type)
                if strategy:
                    resolved = await strategy(conflict)
                    if resolved:
                        conflict.resolved = True
                        conflict.resolution_strategy = strategy.__name__
                        self.metrics['conflicts_resolved'] += 1
                        self.logger.info(f"Resolved conflict {conflict.conflict_id}")
            except Exception as e:
                self.logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
    
    async def _resolve_resource_conflict(self, conflict: TaskConflict) -> bool:
        """Resolve a resource conflict."""
        # Strategy: Priority-based resolution
        task_priorities = {}
        
        for task_id in conflict.task_ids:
            if task_id in self.active_tasks:
                task_priorities[task_id] = self.active_tasks[task_id].priority.value
            else:
                task_priorities[task_id] = TaskPriority.LOW.value
        
        # Keep highest priority task, defer others
        highest_priority_task = min(task_priorities.items(), key=lambda x: x[1])[0]
        
        for task_id in conflict.task_ids:
            if task_id != highest_priority_task and task_id in self.active_tasks:
                # Move task back to queue (implementation would depend on orchestrator)
                self.logger.info(f"Deferring task {task_id} due to resource conflict")
        
        return True
    
    async def _resolve_dependency_conflict(self, conflict: TaskConflict) -> bool:
        """Resolve a dependency conflict."""
        # Strategy: Break circular dependencies by removing lower priority dependency
        return True  # Simplified implementation
    
    async def _resolve_repository_conflict(self, conflict: TaskConflict) -> bool:
        """Resolve a repository conflict."""
        # Strategy: Serialize repository operations by priority
        return True  # Simplified implementation
    
    async def _resolve_timing_conflict(self, conflict: TaskConflict) -> bool:
        """Resolve a timing conflict."""
        # Strategy: Reschedule lower priority task
        return True  # Simplified implementation
    
    async def _resolve_semantic_conflict(self, conflict: TaskConflict) -> bool:
        """Resolve a semantic conflict."""
        # Strategy: Use AI to suggest resolution
        return True  # Simplified implementation
    
    async def _has_circular_dependency(self, task1_id: str, task2_id: str) -> bool:
        """Check if adding a dependency would create a circular dependency."""
        # Use DFS to detect cycles
        visited = set()
        
        def dfs(task_id: str, target: str) -> bool:
            if task_id == target:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            
            for dependent in self.reverse_dependencies.get(task_id, []):
                if dfs(dependent, target):
                    return True
            
            return False
        
        return dfs(task1_id, task2_id)
    
    async def _notify_dependents(self, completed_task_id: str, success: bool):
        """Notify dependent tasks that a prerequisite has completed."""
        dependents = self.reverse_dependencies.get(completed_task_id, [])
        
        for dependent_id in dependents:
            self.logger.debug(f"Notifying dependent task {dependent_id} of {completed_task_id} completion")
    
    async def _generate_followup_coordination(self, completed_task: WorkItem, 
                                            result: Dict[str, Any]):
        """Generate coordination for follow-up tasks."""
        # This would integrate with the work finder to ensure proper coordination
        pass
    
    async def _update_dependency_status(self):
        """Update the status of dependencies."""
        # Check if any waiting tasks can now proceed
        pass
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed and failed tasks."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        # Clean up old completed tasks
        old_completed = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in old_completed:
            del self.completed_tasks[task_id]
        
        # Clean up old failed tasks
        old_failed = [
            task_id for task_id, task in self.failed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in old_failed:
            del self.failed_tasks[task_id]
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'pending_conflicts': len([c for c in self.conflicts.values() if not c.resolved]),
            'resolved_conflicts': len([c for c in self.conflicts.values() if c.resolved]),
            'active_locks': len(self.resource_locks),
            'dependencies_tracked': sum(len(deps) for deps in self.dependencies.values()),
            'metrics': self.metrics.copy(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }