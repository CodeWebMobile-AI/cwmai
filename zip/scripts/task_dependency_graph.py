"""
DAG-based Task Dependency Modeling with Advanced Analysis

This module implements a Directed Acyclic Graph for task dependencies,
including Critical Path Method (CPM) and PERT analysis capabilities.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import networkx as nx
from enum import Enum
import json

from scripts.task_types import SmartTaskType, TaskComplexity
from scripts.task_type_config import TaskTypeConfigModel


class TaskStatus(Enum):
    """Status of a task in the dependency graph."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class TaskNode:
    """Node in the task dependency graph."""
    
    task_id: str
    task_type: str
    metadata: TaskTypeConfigModel
    status: TaskStatus = TaskStatus.NOT_STARTED
    
    # Time estimates for PERT
    optimistic_duration: float = 0.0
    most_likely_duration: float = 0.0
    pessimistic_duration: float = 0.0
    
    # Calculated values
    expected_duration: float = 0.0
    variance: float = 0.0
    
    # CPM values
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
    latest_start: float = 0.0
    latest_finish: float = 0.0
    slack: float = 0.0
    
    # Execution tracking
    actual_start: Optional[datetime] = None
    actual_finish: Optional[datetime] = None
    assigned_to: Optional[str] = None
    
    def __post_init__(self):
        """Calculate PERT values after initialization."""
        if self.most_likely_duration == 0.0:
            self.most_likely_duration = self.metadata.typical_duration_hours
        
        if self.optimistic_duration == 0.0:
            variance_factor = self.metadata.estimated_variance
            self.optimistic_duration = self.most_likely_duration * (1 - variance_factor)
        
        if self.pessimistic_duration == 0.0:
            variance_factor = self.metadata.estimated_variance
            self.pessimistic_duration = self.most_likely_duration * (1 + variance_factor)
        
        # PERT calculation: (O + 4M + P) / 6
        self.expected_duration = (
            self.optimistic_duration + 
            4 * self.most_likely_duration + 
            self.pessimistic_duration
        ) / 6
        
        # Variance calculation: ((P - O) / 6)²
        self.variance = ((self.pessimistic_duration - self.optimistic_duration) / 6) ** 2


class TaskDependencyGraph:
    """Manages task dependencies as a DAG with advanced analysis."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._task_nodes: Dict[str, TaskNode] = {}
    
    def add_task(self, task_node: TaskNode):
        """Add a task to the dependency graph."""
        self._task_nodes[task_node.task_id] = task_node
        self.graph.add_node(task_node.task_id, data=task_node)
    
    def add_dependency(self, from_task_id: str, to_task_id: str):
        """Add a dependency edge (from_task must complete before to_task)."""
        if from_task_id not in self._task_nodes or to_task_id not in self._task_nodes:
            raise ValueError("Both tasks must exist before adding dependency")
        
        self.graph.add_edge(from_task_id, to_task_id)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_task_id, to_task_id)
            raise ValueError("Adding this dependency would create a cycle")
    
    def get_execution_order(self) -> List[str]:
        """Get a valid execution order using topological sort."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            return []
    
    def get_parallel_groups(self) -> List[Set[str]]:
        """Get groups of tasks that can be executed in parallel."""
        groups = []
        remaining = set(self.graph.nodes())
        completed = set()
        
        while remaining:
            # Find tasks with all dependencies completed
            ready = {
                task for task in remaining
                if all(pred in completed for pred in self.graph.predecessors(task))
            }
            
            if not ready:
                # Circular dependency or error
                break
            
            groups.append(ready)
            completed.update(ready)
            remaining -= ready
        
        return groups
    
    def calculate_critical_path(self) -> Tuple[List[str], float]:
        """Calculate the critical path using CPM.
        
        Returns:
            Tuple of (critical_path_tasks, total_duration)
        """
        # Forward pass - calculate earliest times
        for task_id in nx.topological_sort(self.graph):
            task = self._task_nodes[task_id]
            
            # Find maximum earliest finish of predecessors
            max_pred_finish = 0.0
            for pred_id in self.graph.predecessors(task_id):
                pred = self._task_nodes[pred_id]
                max_pred_finish = max(max_pred_finish, pred.earliest_finish)
            
            task.earliest_start = max_pred_finish
            task.earliest_finish = task.earliest_start + task.expected_duration
        
        # Find project completion time
        project_duration = max(
            (task.earliest_finish for task in self._task_nodes.values()),
            default=0.0
        )
        
        # Backward pass - calculate latest times
        for task_id in reversed(list(nx.topological_sort(self.graph))):
            task = self._task_nodes[task_id]
            
            # Find minimum latest start of successors
            successors = list(self.graph.successors(task_id))
            if not successors:
                # This is a terminal task
                task.latest_finish = project_duration
            else:
                min_succ_start = min(
                    self._task_nodes[succ_id].latest_start
                    for succ_id in successors
                )
                task.latest_finish = min_succ_start
            
            task.latest_start = task.latest_finish - task.expected_duration
            task.slack = task.latest_start - task.earliest_start
        
        # Find critical path (tasks with zero slack)
        critical_tasks = [
            task_id for task_id, task in self._task_nodes.items()
            if abs(task.slack) < 0.001  # Float comparison tolerance
        ]
        
        # Order critical tasks
        critical_path = []
        for task_id in nx.topological_sort(self.graph):
            if task_id in critical_tasks:
                critical_path.append(task_id)
        
        return critical_path, project_duration
    
    def calculate_pert_probability(self, target_duration: float) -> float:
        """Calculate probability of completing project within target duration.
        
        Uses PERT analysis with normal distribution assumption.
        """
        critical_path, expected_duration = self.calculate_critical_path()
        
        # Calculate total variance along critical path
        total_variance = sum(
            self._task_nodes[task_id].variance
            for task_id in critical_path
        )
        
        if total_variance == 0:
            return 1.0 if target_duration >= expected_duration else 0.0
        
        # Standard deviation
        std_dev = total_variance ** 0.5
        
        # Z-score
        z = (target_duration - expected_duration) / std_dev
        
        # Approximate normal CDF
        # Using error function approximation
        import math
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    def get_blocked_tasks(self) -> List[str]:
        """Get tasks that are blocked by incomplete dependencies."""
        blocked = []
        
        for task_id, task in self._task_nodes.items():
            if task.status == TaskStatus.NOT_STARTED:
                # Check if any predecessor is not completed
                for pred_id in self.graph.predecessors(task_id):
                    if self._task_nodes[pred_id].status != TaskStatus.COMPLETED:
                        blocked.append(task_id)
                        break
        
        return blocked
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to start."""
        ready = []
        
        for task_id, task in self._task_nodes.items():
            if task.status == TaskStatus.NOT_STARTED:
                # Check if all predecessors are completed
                all_deps_done = all(
                    self._task_nodes[pred_id].status == TaskStatus.COMPLETED
                    for pred_id in self.graph.predecessors(task_id)
                )
                if all_deps_done:
                    ready.append(task_id)
        
        return ready
    
    def update_task_status(self, task_id: str, status: TaskStatus):
        """Update the status of a task."""
        if task_id in self._task_nodes:
            self._task_nodes[task_id].status = status
            
            # Update timestamps
            if status == TaskStatus.IN_PROGRESS:
                self._task_nodes[task_id].actual_start = datetime.now()
            elif status == TaskStatus.COMPLETED:
                self._task_nodes[task_id].actual_finish = datetime.now()
    
    def get_resource_allocation(self) -> Dict[str, List[str]]:
        """Get current resource allocation (tasks per assignee)."""
        allocation = {}
        
        for task_id, task in self._task_nodes.items():
            if task.assigned_to and task.status == TaskStatus.IN_PROGRESS:
                if task.assigned_to not in allocation:
                    allocation[task.assigned_to] = []
                allocation[task.assigned_to].append(task_id)
        
        return allocation
    
    def export_to_gantt_data(self) -> List[Dict[str, Any]]:
        """Export task data for Gantt chart visualization."""
        gantt_data = []
        
        for task_id in nx.topological_sort(self.graph):
            task = self._task_nodes[task_id]
            gantt_data.append({
                "id": task_id,
                "name": task.task_type,
                "start": task.earliest_start,
                "duration": task.expected_duration,
                "progress": 100 if task.status == TaskStatus.COMPLETED else 0,
                "dependencies": list(self.graph.predecessors(task_id)),
                "critical": abs(task.slack) < 0.001,
                "assignee": task.assigned_to
            })
        
        return gantt_data
    
    def visualize_graph(self, output_file: str = "task_dependency_graph.png"):
        """Create a visual representation of the task graph."""
        import matplotlib.pyplot as plt
        
        # Create layout
        pos = nx.spring_layout(self.graph)
        
        # Color nodes by status
        color_map = {
            TaskStatus.NOT_STARTED: 'lightgray',
            TaskStatus.IN_PROGRESS: 'yellow',
            TaskStatus.COMPLETED: 'lightgreen',
            TaskStatus.BLOCKED: 'lightcoral',
            TaskStatus.FAILED: 'red'
        }
        
        node_colors = [
            color_map.get(self._task_nodes[node].status, 'lightgray')
            for node in self.graph.nodes()
        ]
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=8,
            arrows=True
        )
        
        # Add title
        plt.title("Task Dependency Graph")
        plt.savefig(output_file)
        plt.close()


# Example usage
def create_example_task_graph():
    """Create an example task dependency graph."""
    graph = TaskDependencyGraph()
    
    # Create some example tasks
    # This would normally come from the task type registry
    # For now, using dummy metadata
    
    # ... implementation example ...