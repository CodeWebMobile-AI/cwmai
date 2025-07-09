"""
Comprehensive Task Monitoring System

Tracks detailed metrics about task execution, performance, quality,
dependencies, and patterns for continuous improvement.
"""

import json
import time
import psutil
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
import statistics
import subprocess
import os

from scripts.state_manager import StateManager


@dataclass
class TaskExecutionMetrics:
    """Detailed metrics for task execution."""
    task_id: str
    task_type: str
    repository: str
    
    # Timing metrics
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    queue_wait_time: float = 0.0
    
    # Resource usage
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    total_api_calls: int = 0
    api_providers_used: List[str] = field(default_factory=list)
    
    # Execution details
    retry_count: int = 0
    error_count: int = 0
    error_types: List[str] = field(default_factory=list)
    claude_execution_attempted: bool = False
    claude_execution_success: bool = False
    claude_execution_time: float = 0.0
    
    # Timestamps
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""
    parent_task_id: str
    child_task_id: str
    dependency_type: str  # "blocks", "requires", "related"
    created_at: str
    resolved_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class QualityMetrics:
    """Code quality metrics for a task."""
    task_id: str
    
    # Code quality
    linting_score: Optional[float] = None
    complexity_score: Optional[float] = None
    test_coverage_before: Optional[float] = None
    test_coverage_after: Optional[float] = None
    test_coverage_delta: Optional[float] = None
    
    # Performance impact
    performance_impact: Optional[str] = None  # "improved", "degraded", "neutral"
    performance_details: Dict[str, Any] = field(default_factory=dict)
    
    # Review/validation
    automated_tests_passed: Optional[bool] = None
    manual_review_score: Optional[int] = None  # 1-5
    issues_found: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class ExecutionStep:
    """Represents a single step in task execution."""
    step_id: str
    task_id: str
    step_name: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "in_progress"  # in_progress, completed, failed
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)




class TaskMonitoringSystem:
    """Comprehensive monitoring system for all task activities."""
    
    def __init__(self):
        """Initialize the monitoring system."""
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        
        # Storage paths
        self.monitoring_dir = Path("task_monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # In-memory caches
        self.active_tasks: Dict[str, TaskExecutionMetrics] = {}
        self.execution_steps: Dict[str, List[ExecutionStep]] = defaultdict(list)
        self.dependencies: Dict[str, List[TaskDependency]] = defaultdict(list)
        
        # Learning data
        self.task_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.failure_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.success_patterns: Dict[str, List[Dict]] = defaultdict(list)
        
        # Performance tracking
        self.worker_performance: Dict[str, Dict] = defaultdict(lambda: {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "task_types": defaultdict(int),
            "efficiency_scores": deque(maxlen=100)
        })
        
        # Predictive data
        self.task_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.completion_predictions: Dict[str, float] = {}
        
        # Load existing data
        self._load_monitoring_data()
    
    def _sanitize_task_id(self, task_id: str) -> str:
        """Sanitize task ID for safe filesystem usage."""
        return task_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    
    def _load_monitoring_data(self):
        """Load existing monitoring data from disk."""
        try:
            # Load patterns
            patterns_file = self.monitoring_dir / "learned_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.task_patterns = defaultdict(list, data.get("task_patterns", {}))
                    self.failure_patterns = defaultdict(list, data.get("failure_patterns", {}))
                    self.success_patterns = defaultdict(list, data.get("success_patterns", {}))
            
            # Load worker performance
            performance_file = self.monitoring_dir / "worker_performance.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    data = json.load(f)
                    for worker_id, perf_data in data.items():
                        self.worker_performance[worker_id].update(perf_data)
                        # Convert lists back to deques
                        if "efficiency_scores" in perf_data:
                            self.worker_performance[worker_id]["efficiency_scores"] = \
                                deque(perf_data["efficiency_scores"], maxlen=100)
            
            # Load task history
            history_file = self.monitoring_dir / "task_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for task_type, history in data.items():
                        self.task_history[task_type] = deque(history, maxlen=100)
                        
        except Exception as e:
            self.logger.error(f"Error loading monitoring data: {e}")
    
    def _save_monitoring_data(self):
        """Save monitoring data to disk."""
        try:
            # Save patterns
            patterns_data = {
                "task_patterns": dict(self.task_patterns),
                "failure_patterns": dict(self.failure_patterns),
                "success_patterns": dict(self.success_patterns),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            patterns_file = self.monitoring_dir / "learned_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save worker performance (convert deques to lists)
            performance_data = {}
            for worker_id, perf in self.worker_performance.items():
                perf_copy = dict(perf)
                if "efficiency_scores" in perf_copy:
                    perf_copy["efficiency_scores"] = list(perf_copy["efficiency_scores"])
                performance_data[worker_id] = perf_copy
            
            performance_file = self.monitoring_dir / "worker_performance.json"
            with open(performance_file, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            # Save task history (convert deques to lists)
            history_data = {
                task_type: list(history) 
                for task_type, history in self.task_history.items()
            }
            history_file = self.monitoring_dir / "task_history.json"
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving monitoring data: {e}")
    
    # Task Execution Metrics
    def start_task_monitoring(self, task_id: str, task_type: str, repository: str,
                            estimated_duration: float = 0.0) -> TaskExecutionMetrics:
        """Start monitoring a new task."""
        metrics = TaskExecutionMetrics(
            task_id=task_id,
            task_type=task_type,
            repository=repository,
            estimated_duration=estimated_duration,
            created_at=datetime.now(timezone.utc).isoformat(),
            started_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.active_tasks[task_id] = metrics
        self.logger.info(f"Started monitoring task {task_id} ({task_type})")
        
        return metrics
    
    def update_task_metrics(self, task_id: str, updates: Dict[str, Any]):
        """Update metrics for an active task."""
        if task_id in self.active_tasks:
            metrics = self.active_tasks[task_id]
            for key, value in updates.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
    
    def complete_task_monitoring(self, task_id: str, success: bool = True, 
                               error: Optional[str] = None) -> Optional[TaskExecutionMetrics]:
        """Complete monitoring for a task."""
        if task_id not in self.active_tasks:
            return None
        
        metrics = self.active_tasks[task_id]
        metrics.completed_at = datetime.now(timezone.utc).isoformat()
        
        # Calculate actual duration
        if metrics.started_at:
            start_time = datetime.fromisoformat(metrics.started_at)
            end_time = datetime.fromisoformat(metrics.completed_at)
            metrics.actual_duration = (end_time - start_time).total_seconds()
        
        # Record final metrics
        if not success and error:
            metrics.error_count += 1
            metrics.error_types.append(error)
        
        # Save to persistent storage
        self._save_task_metrics(metrics)
        
        # Update patterns and predictions
        self._update_learning_data(metrics, success)
        
        # Clean up
        del self.active_tasks[task_id]
        
        return metrics
    
    def _save_task_metrics(self, metrics: TaskExecutionMetrics):
        """Save task metrics to persistent storage."""
        metrics_file = self.monitoring_dir / f"task_{self._sanitize_task_id(metrics.task_id)}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    # Resource Monitoring
    def record_resource_usage(self, task_id: str):
        """Record current resource usage for a task."""
        if task_id not in self.active_tasks:
            return
        
        try:
            # Get current process
            process = psutil.Process()
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            if cpu_percent > self.active_tasks[task_id].peak_cpu_percent:
                self.active_tasks[task_id].peak_cpu_percent = cpu_percent
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            if memory_mb > self.active_tasks[task_id].peak_memory_mb:
                self.active_tasks[task_id].peak_memory_mb = memory_mb
                
        except Exception as e:
            self.logger.error(f"Error recording resource usage: {e}")
    
    def record_api_call(self, task_id: str, provider: str):
        """Record an API call for a task."""
        if task_id not in self.active_tasks:
            return
        
        self.active_tasks[task_id].total_api_calls += 1
        if provider not in self.active_tasks[task_id].api_providers_used:
            self.active_tasks[task_id].api_providers_used.append(provider)
    
    # Task Dependencies
    def add_dependency(self, parent_task_id: str, child_task_id: str, 
                      dependency_type: str = "blocks"):
        """Add a dependency between tasks."""
        dep = TaskDependency(
            parent_task_id=parent_task_id,
            child_task_id=child_task_id,
            dependency_type=dependency_type,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.dependencies[child_task_id].append(dep)
        
        # Save dependency
        dep_file = self.monitoring_dir / f"dependencies_{self._sanitize_task_id(child_task_id)}.json"
        deps_data = [d.to_dict() for d in self.dependencies[child_task_id]]
        with open(dep_file, 'w') as f:
            json.dump(deps_data, f, indent=2)
    
    def resolve_dependency(self, parent_task_id: str, child_task_id: str):
        """Mark a dependency as resolved."""
        if child_task_id in self.dependencies:
            for dep in self.dependencies[child_task_id]:
                if dep.parent_task_id == parent_task_id and not dep.resolved_at:
                    dep.resolved_at = datetime.now(timezone.utc).isoformat()
                    break
    
    def get_blocking_tasks(self, task_id: str) -> List[str]:
        """Get list of tasks blocking this task."""
        blocking = []
        if task_id in self.dependencies:
            for dep in self.dependencies[task_id]:
                if dep.dependency_type == "blocks" and not dep.resolved_at:
                    blocking.append(dep.parent_task_id)
        return blocking
    
    # Quality Metrics
    async def measure_quality_metrics(self, task_id: str, repository_path: str) -> QualityMetrics:
        """Measure code quality metrics for a task."""
        quality = QualityMetrics(task_id=task_id)
        
        try:
            # Run linting (example with flake8 for Python)
            if Path(repository_path).exists():
                # Get test coverage before (if tests exist)
                coverage_before = await self._get_test_coverage(repository_path)
                if coverage_before is not None:
                    quality.test_coverage_before = coverage_before
                
                # Run linting
                quality.linting_score = await self._run_linting(repository_path)
                
                # Calculate complexity
                quality.complexity_score = await self._calculate_complexity(repository_path)
                
        except Exception as e:
            self.logger.error(f"Error measuring quality metrics: {e}")
        
        # Save quality metrics
        quality_file = self.monitoring_dir / f"quality_{self._sanitize_task_id(task_id)}.json"
        with open(quality_file, 'w') as f:
            json.dump(quality.to_dict(), f, indent=2)
        
        return quality
    
    async def _get_test_coverage(self, repository_path: str) -> Optional[float]:
        """Get test coverage percentage."""
        try:
            # This is a placeholder - implement based on language/framework
            # For Python projects with pytest-cov:
            result = subprocess.run(
                ["pytest", "--cov", "--cov-report=json", "--quiet"],
                cwd=repository_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                coverage_file = Path(repository_path) / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                        return coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except:
            pass
        
        return None
    
    async def _run_linting(self, repository_path: str) -> float:
        """Run linting and return a score."""
        # Placeholder - implement based on language
        return 85.0  # Example score
    
    async def _calculate_complexity(self, repository_path: str) -> float:
        """Calculate code complexity score."""
        # Placeholder - implement cyclomatic complexity calculation
        return 15.0  # Example score
    
    # Execution Steps
    def start_execution_step(self, task_id: str, step_name: str, 
                           metadata: Optional[Dict] = None) -> ExecutionStep:
        """Start a new execution step."""
        step = ExecutionStep(
            step_id=f"{task_id}_{len(self.execution_steps[task_id])}",
            task_id=task_id,
            step_name=step_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {}
        )
        
        self.execution_steps[task_id].append(step)
        return step
    
    def complete_execution_step(self, task_id: str, step_name: str, 
                              success: bool = True, output: Optional[str] = None,
                              error: Optional[str] = None):
        """Complete an execution step."""
        if task_id in self.execution_steps:
            for step in reversed(self.execution_steps[task_id]):
                if step.step_name == step_name and not step.completed_at:
                    step.completed_at = datetime.now(timezone.utc).isoformat()
                    step.status = "completed" if success else "failed"
                    step.output = output
                    step.error = error
                    
                    # Save steps
                    self._save_execution_steps(task_id)
                    break
    
    def _save_execution_steps(self, task_id: str):
        """Save execution steps for a task."""
        steps_file = self.monitoring_dir / f"steps_{self._sanitize_task_id(task_id)}.json"
        steps_data = [step.to_dict() for step in self.execution_steps[task_id]]
        with open(steps_file, 'w') as f:
            json.dump(steps_data, f, indent=2)
    
    
    # Learning & Patterns
    def _update_learning_data(self, metrics: TaskExecutionMetrics, success: bool):
        """Update learning data based on task completion."""
        pattern = {
            "task_type": metrics.task_type,
            "repository": metrics.repository,
            "duration_ratio": metrics.actual_duration / metrics.estimated_duration if metrics.estimated_duration > 0 else 1.0,
            "retry_count": metrics.retry_count,
            "api_calls": metrics.total_api_calls,
            "success": success,
            "timestamp": metrics.completed_at
        }
        
        # Add to appropriate pattern list
        if success:
            self.success_patterns[metrics.task_type].append(pattern)
        else:
            self.failure_patterns[metrics.task_type].append(pattern)
            
            # Analyze failure pattern
            if metrics.error_types:
                failure_detail = {
                    "error_types": metrics.error_types,
                    "pattern": pattern
                }
                self.failure_patterns[f"{metrics.task_type}_errors"].append(failure_detail)
        
        # Update task history
        self.task_history[metrics.task_type].append({
            "duration": metrics.actual_duration,
            "success": success,
            "timestamp": metrics.completed_at
        })
        
        # Save patterns periodically
        if len(self.task_patterns[metrics.task_type]) % 10 == 0:
            self._save_monitoring_data()
    
    def get_common_failure_patterns(self, task_type: str) -> List[Dict]:
        """Get common failure patterns for a task type."""
        if task_type not in self.failure_patterns:
            return []
        
        # Analyze patterns
        patterns = self.failure_patterns[task_type]
        if not patterns:
            return []
        
        # Group by error types
        error_groups = defaultdict(list)
        for p in patterns:
            if isinstance(p, dict) and "error_types" in p:
                for error in p["error_types"]:
                    error_groups[error].append(p)
        
        # Find most common
        common_patterns = []
        for error_type, instances in sorted(error_groups.items(), 
                                          key=lambda x: len(x[1]), reverse=True)[:5]:
            common_patterns.append({
                "error_type": error_type,
                "frequency": len(instances),
                "avg_retry_count": statistics.mean([i["pattern"]["retry_count"] for i in instances])
            })
        
        return common_patterns
    
    # Worker Performance
    def update_worker_performance(self, worker_id: str, task_id: str, 
                                task_type: str, success: bool, duration: float):
        """Update worker performance metrics."""
        perf = self.worker_performance[worker_id]
        
        if success:
            perf["tasks_completed"] += 1
        else:
            perf["tasks_failed"] += 1
        
        perf["total_execution_time"] += duration
        perf["task_types"][task_type] += 1
        
        # Calculate efficiency score (simple version)
        if perf["tasks_completed"] > 0:
            success_rate = perf["tasks_completed"] / (perf["tasks_completed"] + perf["tasks_failed"])
            avg_duration = perf["total_execution_time"] / (perf["tasks_completed"] + perf["tasks_failed"])
            efficiency = success_rate * 100 / (avg_duration + 1)  # Higher score for faster + successful
            perf["efficiency_scores"].append(efficiency)
    
    def get_best_worker_for_task(self, task_type: str) -> Optional[str]:
        """Get the best worker for a specific task type."""
        best_worker = None
        best_score = 0
        
        for worker_id, perf in self.worker_performance.items():
            if task_type in perf["task_types"] and perf["task_types"][task_type] > 0:
                # Calculate specialization score
                total_tasks = sum(perf["task_types"].values())
                specialization = perf["task_types"][task_type] / total_tasks
                
                # Get average efficiency
                if perf["efficiency_scores"]:
                    avg_efficiency = statistics.mean(perf["efficiency_scores"])
                    score = specialization * avg_efficiency
                    
                    if score > best_score:
                        best_score = score
                        best_worker = worker_id
        
        return best_worker
    
    # Predictive Analytics
    def predict_task_duration(self, task_type: str, repository: str) -> float:
        """Predict task duration based on history."""
        if task_type not in self.task_history:
            return 14400  # Default 4 hours
        
        history = list(self.task_history[task_type])
        if not history:
            return 14400
        
        # Get recent successful tasks
        recent_durations = [
            h["duration"] for h in history[-20:] 
            if h.get("success", False)
        ]
        
        if not recent_durations:
            return 14400
        
        # Use weighted average (more recent = higher weight)
        weights = [i + 1 for i in range(len(recent_durations))]
        weighted_sum = sum(d * w for d, w in zip(recent_durations, weights))
        total_weight = sum(weights)
        
        predicted = weighted_sum / total_weight
        
        # Add buffer for uncertainty
        uncertainty = statistics.stdev(recent_durations) if len(recent_durations) > 1 else 0
        return predicted + (uncertainty * 0.5)
    
    def calculate_failure_risk(self, task_type: str) -> float:
        """Calculate risk of task failure (0-1)."""
        base_risk = 0.1  # Base 10% risk
        
        # Check failure history
        if task_type in self.failure_patterns:
            total_tasks = len(self.task_history.get(task_type, []))
            failures = len(self.failure_patterns[task_type])
            if total_tasks > 0:
                failure_rate = failures / total_tasks
                base_risk = max(base_risk, failure_rate)
        
        # Cap at 0.9 (never 100% certain of failure)
        return min(base_risk, 0.9)
    
    def get_optimal_scheduling_time(self, task_type: str) -> Dict[str, Any]:
        """Get optimal time to schedule a task."""
        # Analyze historical success rates by hour
        if task_type not in self.task_history:
            return {"hour": 2, "reason": "Default low-activity time"}
        
        history = list(self.task_history[task_type])
        if not history:
            return {"hour": 2, "reason": "No history available"}
        
        # Group by hour
        hour_success = defaultdict(lambda: {"success": 0, "total": 0})
        for task in history:
            if "timestamp" in task:
                dt = datetime.fromisoformat(task["timestamp"])
                hour = dt.hour
                hour_success[hour]["total"] += 1
                if task.get("success", False):
                    hour_success[hour]["success"] += 1
        
        # Find best hour
        best_hour = 2
        best_rate = 0
        for hour, stats in hour_success.items():
            if stats["total"] >= 3:  # Minimum sample size
                rate = stats["success"] / stats["total"]
                if rate > best_rate:
                    best_rate = rate
                    best_hour = hour
        
        return {
            "hour": best_hour,
            "success_rate": best_rate,
            "reason": f"Historical success rate: {best_rate:.1%}"
        }
    
    # Task Lifecycle Events
    def record_lifecycle_event(self, task_id: str, event: str, metadata: Optional[Dict] = None):
        """Record a lifecycle event for a task."""
        event_data = {
            "task_id": task_id,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        # Save to event log
        event_file = self.monitoring_dir / f"events_{self._sanitize_task_id(task_id)}.json"
        events = []
        if event_file.exists():
            with open(event_file, 'r') as f:
                events = json.load(f)
        
        events.append(event_data)
        
        with open(event_file, 'w') as f:
            json.dump(events, f, indent=2)
        
        # Update active task if relevant
        if task_id in self.active_tasks:
            if event == "paused":
                self.active_tasks[task_id].status = "paused"
            elif event == "resumed":
                self.active_tasks[task_id].status = "in_progress"
            elif event == "blocked":
                self.active_tasks[task_id].status = "blocked"
    
    # Reporting
    def generate_task_report(self, task_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for a task."""
        report = {
            "task_id": task_id,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Load metrics
        metrics_file = self.monitoring_dir / f"task_{self._sanitize_task_id(task_id)}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                report["metrics"] = json.load(f)
        
        # Load quality metrics
        quality_file = self.monitoring_dir / f"quality_{self._sanitize_task_id(task_id)}.json"
        if quality_file.exists():
            with open(quality_file, 'r') as f:
                report["quality"] = json.load(f)
        
        # Load execution steps
        steps_file = self.monitoring_dir / f"steps_{self._sanitize_task_id(task_id)}.json"
        if steps_file.exists():
            with open(steps_file, 'r') as f:
                report["execution_steps"] = json.load(f)
        
        # Load events
        events_file = self.monitoring_dir / f"events_{self._sanitize_task_id(task_id)}.json"
        if events_file.exists():
            with open(events_file, 'r') as f:
                report["lifecycle_events"] = json.load(f)
        
        # Load dependencies
        deps_file = self.monitoring_dir / f"dependencies_{self._sanitize_task_id(task_id)}.json"
        if deps_file.exists():
            with open(deps_file, 'r') as f:
                report["dependencies"] = json.load(f)
        
        return report
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Get high-level insights about the system."""
        insights = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_types": {},
            "worker_performance": {},
            "common_failures": {},
            "predictions": {}
        }
        
        # Task type insights
        for task_type, history in self.task_history.items():
            if history:
                recent = list(history)[-50:]
                success_count = sum(1 for h in recent if h.get("success", False))
                avg_duration = statistics.mean([h["duration"] for h in recent if "duration" in h])
                
                insights["task_types"][task_type] = {
                    "total_tasks": len(history),
                    "recent_success_rate": success_count / len(recent),
                    "avg_duration_seconds": avg_duration,
                    "predicted_next_duration": self.predict_task_duration(task_type, "")
                }
        
        # Worker insights
        for worker_id, perf in self.worker_performance.items():
            if perf["efficiency_scores"]:
                insights["worker_performance"][worker_id] = {
                    "tasks_completed": perf["tasks_completed"],
                    "tasks_failed": perf["tasks_failed"],
                    "avg_efficiency": statistics.mean(perf["efficiency_scores"]),
                    "specializations": sorted(
                        perf["task_types"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                }
        
        # Common failures
        for task_type in self.failure_patterns:
            if not task_type.endswith("_errors"):
                patterns = self.get_common_failure_patterns(task_type)
                if patterns:
                    insights["common_failures"][task_type] = patterns
        
        return insights


# Global instance
_monitoring_system = None

def get_monitoring_system() -> TaskMonitoringSystem:
    """Get or create the global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = TaskMonitoringSystem()
    return _monitoring_system