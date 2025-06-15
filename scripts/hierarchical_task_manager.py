"""
Hierarchical Task Manager

Manages parent-child task relationships, progress tracking across task hierarchies,
and intelligent task orchestration with dependency management.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

from task_decomposition_engine import SubTask, DecompositionResult, TaskComplexity


class TaskHierarchyStatus(Enum):
    """Status levels for hierarchical tasks."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIALLY_COMPLETE = "partially_complete"
    AWAITING_DEPENDENCIES = "awaiting_dependencies"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class ProgressLevel(Enum):
    """Progress tracking granularity levels."""
    TASK = "task"           # Individual task level
    SUBTASK = "subtask"     # Sub-task level
    MILESTONE = "milestone" # Milestone level
    PROJECT = "project"     # Entire project level


@dataclass
class TaskNode:
    """Represents a node in the task hierarchy."""
    id: str
    parent_id: Optional[str]
    title: str
    description: str
    task_type: str
    status: str
    priority: str
    estimated_hours: float
    actual_hours: float = 0.0
    progress_percentage: float = 0.0
    children: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    github_issue_number: Optional[int] = None
    repository: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a point in time."""
    timestamp: datetime
    task_id: str
    progress_percentage: float
    status: str
    hours_worked: float
    milestones_completed: List[str]
    blockers: List[str]
    next_actions: List[str]


@dataclass
class HierarchyAnalytics:
    """Analytics for task hierarchy."""
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    blocked_tasks: int
    average_completion_time: float
    critical_path_tasks: List[str]
    bottleneck_tasks: List[str]
    efficiency_score: float
    estimated_completion_date: Optional[datetime]


class HierarchicalTaskManager:
    """Manages hierarchical task structures with intelligent orchestration."""
    
    def __init__(self, state_file: str = "hierarchical_tasks.json"):
        """Initialize the hierarchical task manager.
        
        Args:
            state_file: File to store hierarchical task state
        """
        self.state_file = state_file
        self.logger = logging.getLogger(__name__)
        
        # Task hierarchy storage
        self.task_nodes: Dict[str, TaskNode] = {}
        self.hierarchy_graph: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)   # task -> dependencies
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # task -> dependents
        
        # Progress tracking
        self.progress_history: Dict[str, List[ProgressSnapshot]] = defaultdict(list)
        self.milestones: Dict[str, List[str]] = defaultdict(list)
        
        # Load existing state
        self._load_state()
        
    def add_task_hierarchy(self, decomposition_result: DecompositionResult, 
                          parent_task: Dict[str, Any]) -> str:
        """Add a task hierarchy from decomposition result.
        
        Args:
            decomposition_result: Result from task decomposition
            parent_task: Original parent task
            
        Returns:
            Parent task ID in hierarchy
        """
        self.logger.info(f"Adding task hierarchy for: {parent_task.get('title', 'Unknown')}")
        
        # Create parent task node
        parent_node = TaskNode(
            id=parent_task.get('id', f"task_{datetime.now().timestamp()}"),
            parent_id=None,
            title=parent_task.get('title', 'Unknown Task'),
            description=parent_task.get('description', ''),
            task_type=parent_task.get('type', ''),
            status=parent_task.get('status', 'pending'),
            priority=parent_task.get('priority', 'medium'),
            estimated_hours=parent_task.get('estimated_hours', 0.0),
            github_issue_number=parent_task.get('github_issue_number'),
            repository=parent_task.get('repository'),
            metadata=parent_task.copy()
        )
        
        self.task_nodes[parent_node.id] = parent_node
        
        # Add sub-tasks as children
        for sub_task in decomposition_result.sub_tasks:
            child_node = TaskNode(
                id=sub_task.id,
                parent_id=parent_node.id,
                title=sub_task.title,
                description=sub_task.description,
                task_type=sub_task.type,
                status='pending',
                priority=sub_task.priority,
                estimated_hours=sub_task.estimated_hours,
                repository=parent_task.get('repository'),
                tags=['subtask', f'sequence_{sub_task.sequence_order}'],
                metadata={
                    'deliverables': sub_task.deliverables,
                    'acceptance_criteria': sub_task.acceptance_criteria,
                    'technical_requirements': sub_task.technical_requirements,
                    'sequence_order': sub_task.sequence_order,
                    'can_parallelize': sub_task.can_parallelize
                }
            )
            
            self.task_nodes[child_node.id] = child_node
            parent_node.children.append(child_node.id)
            self.hierarchy_graph[parent_node.id].append(child_node.id)
            
            # Add dependencies within sub-tasks
            for dep in sub_task.dependencies:
                if dep in self.task_nodes:
                    child_node.dependencies.append(dep)
                    self.dependency_graph[child_node.id].add(dep)
                    self.reverse_dependency_graph[dep].add(child_node.id)
        
        # Create critical path and milestones
        self._create_milestones(parent_node.id, decomposition_result)
        
        # Save state
        self._save_state()
        
        self.logger.info(f"Added hierarchy with {len(decomposition_result.sub_tasks)} sub-tasks")
        return parent_node.id
    
    def update_task_progress(self, task_id: str, progress_percentage: float, 
                           hours_worked: float = 0.0, status: str = None) -> bool:
        """Update progress for a task and propagate to hierarchy.
        
        Args:
            task_id: Task ID to update
            progress_percentage: Progress percentage (0-100)
            hours_worked: Hours worked on this update
            status: New status (optional)
            
        Returns:
            Success status
        """
        if task_id not in self.task_nodes:
            self.logger.warning(f"Task {task_id} not found in hierarchy")
            return False
        
        task_node = self.task_nodes[task_id]
        
        # Update task progress
        old_progress = task_node.progress_percentage
        task_node.progress_percentage = min(100.0, max(0.0, progress_percentage))
        task_node.actual_hours += hours_worked
        task_node.updated_at = datetime.now(timezone.utc)
        
        if status:
            task_node.status = status
            
        if progress_percentage >= 100.0:
            task_node.status = 'completed'
            task_node.completed_at = datetime.now(timezone.utc)
        
        # Create progress snapshot
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            progress_percentage=progress_percentage,
            status=task_node.status,
            hours_worked=hours_worked,
            milestones_completed=self._get_completed_milestones(task_id),
            blockers=self._get_current_blockers(task_id),
            next_actions=self._get_next_actions(task_id)
        )
        
        self.progress_history[task_id].append(snapshot)
        
        # Propagate progress to parent
        if task_node.parent_id:
            self._update_parent_progress(task_node.parent_id)
        
        # Check if dependencies are unblocked
        self._check_dependency_unblocking(task_id)
        
        # Save state
        self._save_state()
        
        self.logger.info(f"Updated {task_id} progress: {old_progress}% -> {progress_percentage}%")
        return True
    
    def get_task_hierarchy(self, root_task_id: str) -> Dict[str, Any]:
        """Get complete task hierarchy starting from root.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            Hierarchical task structure
        """
        if root_task_id not in self.task_nodes:
            return {}
        
        def build_hierarchy(task_id: str) -> Dict[str, Any]:
            node = self.task_nodes[task_id]
            
            hierarchy = {
                'id': node.id,
                'title': node.title,
                'description': node.description,
                'type': node.task_type,
                'status': node.status,
                'priority': node.priority,
                'progress_percentage': node.progress_percentage,
                'estimated_hours': node.estimated_hours,
                'actual_hours': node.actual_hours,
                'created_at': node.created_at.isoformat(),
                'updated_at': node.updated_at.isoformat(),
                'completed_at': node.completed_at.isoformat() if node.completed_at else None,
                'github_issue_number': node.github_issue_number,
                'repository': node.repository,
                'tags': node.tags,
                'dependencies': node.dependencies,
                'children': []
            }
            
            # Add children recursively
            for child_id in node.children:
                if child_id in self.task_nodes:
                    hierarchy['children'].append(build_hierarchy(child_id))
            
            return hierarchy
        
        return build_hierarchy(root_task_id)
    
    def get_ready_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get tasks that are ready to be worked on (no blockers).
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of ready tasks
        """
        ready_tasks = []
        
        for task_id, node in self.task_nodes.items():
            if (node.status in ['pending', 'in_progress'] and 
                node.progress_percentage < 100.0 and
                self._are_dependencies_satisfied(task_id)):
                
                ready_tasks.append({
                    'id': node.id,
                    'title': node.title,
                    'description': node.description,
                    'type': node.task_type,
                    'priority': node.priority,
                    'estimated_hours': node.estimated_hours,
                    'progress_percentage': node.progress_percentage,
                    'repository': node.repository,
                    'github_issue_number': node.github_issue_number,
                    'next_actions': self._get_next_actions(task_id),
                    'parent_id': node.parent_id
                })
        
        # Sort by priority and progress
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        ready_tasks.sort(key=lambda x: (
            priority_order.get(x['priority'], 0),
            -x['progress_percentage']  # Lower progress first
        ), reverse=True)
        
        return ready_tasks[:limit]
    
    def get_critical_path(self, root_task_id: str) -> List[str]:
        """Calculate critical path through task hierarchy.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            List of task IDs in critical path
        """
        if root_task_id not in self.task_nodes:
            return []
        
        # Simple critical path calculation based on dependencies and duration
        critical_path = []
        visited = set()
        
        def find_longest_path(task_id: str, current_path: List[str], current_duration: float) -> Tuple[List[str], float]:
            if task_id in visited:
                return current_path, current_duration
            
            visited.add(task_id)
            node = self.task_nodes[task_id]
            path_with_current = current_path + [task_id]
            duration_with_current = current_duration + node.estimated_hours
            
            longest_path = path_with_current
            longest_duration = duration_with_current
            
            # Check children
            for child_id in node.children:
                if child_id in self.task_nodes:
                    child_path, child_duration = find_longest_path(
                        child_id, path_with_current, duration_with_current
                    )
                    if child_duration > longest_duration:
                        longest_path = child_path
                        longest_duration = child_duration
            
            visited.remove(task_id)
            return longest_path, longest_duration
        
        critical_path, _ = find_longest_path(root_task_id, [], 0.0)
        return critical_path
    
    def get_hierarchy_analytics(self, root_task_id: str) -> HierarchyAnalytics:
        """Get analytics for task hierarchy.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            Hierarchy analytics
        """
        if root_task_id not in self.task_nodes:
            return HierarchyAnalytics(0, 0, 0, 0, 0.0, [], [], 0.0, None)
        
        # Collect all tasks in hierarchy
        all_tasks = self._get_all_descendants(root_task_id)
        all_tasks.append(root_task_id)
        
        # Calculate statistics
        total_tasks = len(all_tasks)
        completed_tasks = sum(1 for tid in all_tasks 
                             if self.task_nodes[tid].status == 'completed')
        in_progress_tasks = sum(1 for tid in all_tasks 
                               if self.task_nodes[tid].status == 'in_progress')
        blocked_tasks = sum(1 for tid in all_tasks 
                           if self.task_nodes[tid].status == 'blocked')
        
        # Calculate average completion time
        completed_task_nodes = [self.task_nodes[tid] for tid in all_tasks 
                               if self.task_nodes[tid].completed_at]
        avg_completion_time = 0.0
        if completed_task_nodes:
            completion_times = [
                (node.completed_at - node.created_at).total_seconds() / 3600
                for node in completed_task_nodes
            ]
            avg_completion_time = sum(completion_times) / len(completion_times)
        
        # Get critical path
        critical_path = self.get_critical_path(root_task_id)
        
        # Find bottlenecks (tasks with most dependents)
        bottlenecks = sorted(
            all_tasks,
            key=lambda tid: len(self.reverse_dependency_graph.get(tid, set())),
            reverse=True
        )[:3]
        
        # Calculate efficiency score
        total_estimated = sum(self.task_nodes[tid].estimated_hours for tid in all_tasks)
        total_actual = sum(self.task_nodes[tid].actual_hours for tid in all_tasks)
        efficiency_score = min(1.0, total_estimated / max(total_actual, 1.0))
        
        # Estimate completion date
        remaining_hours = sum(
            max(0, self.task_nodes[tid].estimated_hours - self.task_nodes[tid].actual_hours)
            for tid in all_tasks
            if self.task_nodes[tid].status != 'completed'
        )
        estimated_completion = None
        if remaining_hours > 0:
            # Assume 8 hours per day work capacity
            days_remaining = remaining_hours / 8
            estimated_completion = datetime.now(timezone.utc) + timedelta(days=days_remaining)
        
        return HierarchyAnalytics(
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            in_progress_tasks=in_progress_tasks,
            blocked_tasks=blocked_tasks,
            average_completion_time=avg_completion_time,
            critical_path_tasks=critical_path,
            bottleneck_tasks=bottlenecks,
            efficiency_score=efficiency_score,
            estimated_completion_date=estimated_completion
        )
    
    def suggest_next_tasks(self, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Suggest next tasks to work on based on hierarchy and priorities.
        
        Args:
            context: Additional context for suggestions
            
        Returns:
            List of suggested tasks with rationale
        """
        suggestions = []
        
        # Get ready tasks
        ready_tasks = self.get_ready_tasks(20)
        
        # Prioritize suggestions
        for task in ready_tasks[:5]:
            task_id = task['id']
            node = self.task_nodes[task_id]
            
            # Calculate suggestion score
            priority_score = {'critical': 10, 'high': 7, 'medium': 4, 'low': 1}.get(task['priority'], 1)
            progress_score = (100 - task['progress_percentage']) / 10  # Higher for less complete
            dependency_score = len(self.reverse_dependency_graph.get(task_id, set())) * 2
            
            total_score = priority_score + progress_score + dependency_score
            
            # Generate rationale
            rationale_parts = []
            if task['priority'] in ['critical', 'high']:
                rationale_parts.append(f"High priority ({task['priority']})")
            if len(self.reverse_dependency_graph.get(task_id, set())) > 0:
                rationale_parts.append("Unblocks other tasks")
            if task['progress_percentage'] > 0:
                rationale_parts.append(f"In progress ({task['progress_percentage']:.0f}%)")
            
            rationale = "; ".join(rationale_parts) if rationale_parts else "Available to start"
            
            suggestions.append({
                'task': task,
                'score': total_score,
                'rationale': rationale,
                'blocking_count': len(self.reverse_dependency_graph.get(task_id, set())),
                'estimated_impact': self._calculate_task_impact(task_id)
            })
        
        # Sort by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        return suggestions
    
    def _update_parent_progress(self, parent_id: str) -> None:
        """Update parent task progress based on children.
        
        Args:
            parent_id: Parent task ID
        """
        if parent_id not in self.task_nodes:
            return
        
        parent_node = self.task_nodes[parent_id]
        children = [self.task_nodes[child_id] for child_id in parent_node.children 
                   if child_id in self.task_nodes]
        
        if not children:
            return
        
        # Calculate weighted average progress
        total_weight = sum(child.estimated_hours for child in children)
        if total_weight > 0:
            weighted_progress = sum(
                child.progress_percentage * child.estimated_hours 
                for child in children
            ) / total_weight
            
            parent_node.progress_percentage = weighted_progress
            parent_node.updated_at = datetime.now(timezone.utc)
            
            # Update status based on children
            if all(child.status == 'completed' for child in children):
                parent_node.status = 'completed'
                parent_node.completed_at = datetime.now(timezone.utc)
            elif any(child.status == 'in_progress' for child in children):
                parent_node.status = 'in_progress'
            elif any(child.status == 'blocked' for child in children):
                parent_node.status = 'blocked'
        
        # Recursively update grandparent
        if parent_node.parent_id:
            self._update_parent_progress(parent_node.parent_id)
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if dependencies are satisfied
        """
        if task_id not in self.task_nodes:
            return False
        
        node = self.task_nodes[task_id]
        
        for dep_id in node.dependencies:
            if dep_id not in self.task_nodes:
                continue
            
            dep_node = self.task_nodes[dep_id]
            if dep_node.status != 'completed':
                return False
        
        return True
    
    def _check_dependency_unblocking(self, completed_task_id: str) -> None:
        """Check if completing a task unblocks others.
        
        Args:
            completed_task_id: ID of completed task
        """
        if completed_task_id not in self.task_nodes:
            return
        
        completed_node = self.task_nodes[completed_task_id]
        if completed_node.status != 'completed':
            return
        
        # Check all dependents
        for dependent_id in self.reverse_dependency_graph.get(completed_task_id, set()):
            if dependent_id in self.task_nodes:
                dependent_node = self.task_nodes[dependent_id]
                if (dependent_node.status == 'blocked' and 
                    self._are_dependencies_satisfied(dependent_id)):
                    dependent_node.status = 'pending'
                    self.logger.info(f"Unblocked task {dependent_id}")
    
    def _get_completed_milestones(self, task_id: str) -> List[str]:
        """Get completed milestones for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of completed milestones
        """
        return self.milestones.get(task_id, [])
    
    def _get_current_blockers(self, task_id: str) -> List[str]:
        """Get current blockers for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of blocking task IDs
        """
        if task_id not in self.task_nodes:
            return []
        
        node = self.task_nodes[task_id]
        blockers = []
        
        for dep_id in node.dependencies:
            if dep_id in self.task_nodes:
                dep_node = self.task_nodes[dep_id]
                if dep_node.status != 'completed':
                    blockers.append(dep_id)
        
        return blockers
    
    def _get_next_actions(self, task_id: str) -> List[str]:
        """Get next actions for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of next actions
        """
        if task_id not in self.task_nodes:
            return []
        
        node = self.task_nodes[task_id]
        next_actions = []
        
        if node.status == 'pending':
            if self._are_dependencies_satisfied(task_id):
                next_actions.append(f"Start work on: {node.title}")
            else:
                blockers = self._get_current_blockers(task_id)
                next_actions.append(f"Wait for dependencies: {', '.join(blockers)}")
        elif node.status == 'in_progress':
            next_actions.append(f"Continue work on: {node.title}")
            if node.progress_percentage > 75:
                next_actions.append("Prepare for completion review")
        elif node.status == 'completed':
            dependents = list(self.reverse_dependency_graph.get(task_id, set()))
            if dependents:
                next_actions.append(f"Check unblocked tasks: {', '.join(dependents[:3])}")
        
        return next_actions
    
    def _get_all_descendants(self, task_id: str) -> List[str]:
        """Get all descendant task IDs.
        
        Args:
            task_id: Root task ID
            
        Returns:
            List of all descendant task IDs
        """
        descendants = []
        
        def collect_descendants(tid: str):
            if tid in self.task_nodes:
                for child_id in self.task_nodes[tid].children:
                    descendants.append(child_id)
                    collect_descendants(child_id)
        
        collect_descendants(task_id)
        return descendants
    
    def _calculate_task_impact(self, task_id: str) -> float:
        """Calculate impact score for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Impact score
        """
        if task_id not in self.task_nodes:
            return 0.0
        
        node = self.task_nodes[task_id]
        
        # Base impact from estimated hours
        base_impact = node.estimated_hours
        
        # Multiplier for blocking other tasks
        blocking_multiplier = 1 + (len(self.reverse_dependency_graph.get(task_id, set())) * 0.5)
        
        # Priority multiplier
        priority_multiplier = {'critical': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}.get(node.priority, 1.0)
        
        return base_impact * blocking_multiplier * priority_multiplier
    
    def _create_milestones(self, parent_id: str, decomposition_result: DecompositionResult) -> None:
        """Create milestones for task hierarchy.
        
        Args:
            parent_id: Parent task ID
            decomposition_result: Decomposition result
        """
        milestones = []
        
        # Create milestones based on sequence groups
        sequence_groups = defaultdict(list)
        for sub_task in decomposition_result.sub_tasks:
            sequence_groups[sub_task.sequence_order].append(sub_task.id)
        
        for order in sorted(sequence_groups.keys()):
            milestone_name = f"Milestone {order}: Complete sequence {order} tasks"
            milestones.append(milestone_name)
        
        self.milestones[parent_id] = milestones
    
    def _load_state(self) -> None:
        """Load hierarchical task state from file."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Load task nodes
            for task_data in state.get('task_nodes', []):
                node = TaskNode(
                    id=task_data['id'],
                    parent_id=task_data.get('parent_id'),
                    title=task_data['title'],
                    description=task_data['description'],
                    task_type=task_data['task_type'],
                    status=task_data['status'],
                    priority=task_data['priority'],
                    estimated_hours=task_data['estimated_hours'],
                    actual_hours=task_data.get('actual_hours', 0.0),
                    progress_percentage=task_data.get('progress_percentage', 0.0),
                    children=task_data.get('children', []),
                    dependencies=task_data.get('dependencies', []),
                    dependents=task_data.get('dependents', []),
                    created_at=datetime.fromisoformat(task_data['created_at']),
                    updated_at=datetime.fromisoformat(task_data['updated_at']),
                    completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None,
                    github_issue_number=task_data.get('github_issue_number'),
                    repository=task_data.get('repository'),
                    tags=task_data.get('tags', []),
                    metadata=task_data.get('metadata', {})
                )
                self.task_nodes[node.id] = node
            
            # Rebuild graphs
            self._rebuild_graphs()
            
            self.logger.info(f"Loaded {len(self.task_nodes)} tasks from {self.state_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading hierarchical task state: {e}")
    
    def _save_state(self) -> None:
        """Save hierarchical task state to file."""
        try:
            state = {
                'task_nodes': [
                    {
                        'id': node.id,
                        'parent_id': node.parent_id,
                        'title': node.title,
                        'description': node.description,
                        'task_type': node.task_type,
                        'status': node.status,
                        'priority': node.priority,
                        'estimated_hours': node.estimated_hours,
                        'actual_hours': node.actual_hours,
                        'progress_percentage': node.progress_percentage,
                        'children': node.children,
                        'dependencies': node.dependencies,
                        'dependents': node.dependents,
                        'created_at': node.created_at.isoformat(),
                        'updated_at': node.updated_at.isoformat(),
                        'completed_at': node.completed_at.isoformat() if node.completed_at else None,
                        'github_issue_number': node.github_issue_number,
                        'repository': node.repository,
                        'tags': node.tags,
                        'metadata': node.metadata
                    }
                    for node in self.task_nodes.values()
                ],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving hierarchical task state: {e}")
    
    def _rebuild_graphs(self) -> None:
        """Rebuild hierarchy and dependency graphs from loaded nodes."""
        self.hierarchy_graph.clear()
        self.dependency_graph.clear()
        self.reverse_dependency_graph.clear()
        
        for node in self.task_nodes.values():
            # Rebuild hierarchy graph
            if node.parent_id:
                self.hierarchy_graph[node.parent_id].append(node.id)
            
            # Rebuild dependency graphs
            for dep in node.dependencies:
                self.dependency_graph[node.id].add(dep)
                self.reverse_dependency_graph[dep].add(node.id)