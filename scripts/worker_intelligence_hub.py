"""
Worker Intelligence Hub

Central coordination point for parallel worker intelligence, learning, and optimization.
Provides task distribution, performance tracking, and cross-worker coordination.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from worker_logging_config import WorkerLogger, create_worker_logger, CorrelationIDManager
from redis import get_redis_client


class WorkerStatus(Enum):
    """Worker status states."""
    STARTING = "starting"
    AVAILABLE = "available"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    STOPPING = "stopping"
    OFFLINE = "offline"


class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerCapability:
    """Represents a worker's capability in a specific area."""
    name: str
    proficiency: float  # 0.0 to 1.0
    success_rate: float
    avg_duration: float
    task_count: int
    last_used: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerProfile:
    """Comprehensive worker profile for intelligence tracking."""
    worker_id: str
    worker_type: str
    status: WorkerStatus
    capabilities: Dict[str, WorkerCapability]
    current_task_id: Optional[str] = None
    last_heartbeat: Optional[float] = None
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_duration: float = 0.0
    avg_task_duration: float = 0.0
    success_rate: float = 0.0
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['capabilities'] = {k: v.to_dict() for k, v in self.capabilities.items()}
        return data


@dataclass
class TaskInfo:
    """Information about a task in the system."""
    task_id: str
    task_type: str
    priority: int
    assigned_worker: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = 0.0
    claimed_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        return data


class WorkerIntelligenceHub:
    """Central hub for worker intelligence and coordination."""
    
    def __init__(self, hub_id: str = "main_hub", redis_client=None):
        self.hub_id = hub_id
        self.redis = redis_client
        self.logger = create_worker_logger(f"hub_{hub_id}", "intelligence_hub")
        
        # In-memory state (with Redis backup)
        self.workers: Dict[str, WorkerProfile] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue: deque = deque()
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.system_metrics: Dict[str, Any] = {}
        
        # Learning data
        self.task_type_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.worker_learning_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.heartbeat_timeout = 60.0  # seconds
        self.cleanup_interval = 300.0  # seconds
        self.learning_window = 1000  # number of tasks for learning
        
        self._running = False
        self._cleanup_task = None
        
        self.logger.worker_start(specialization="intelligence_coordination")
    
    async def start(self):
        """Start the intelligence hub."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize Redis connection if provided
        if self.redis:
            await self._load_state_from_redis()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.intelligence_event("Hub started", {
            "hub_id": self.hub_id,
            "redis_enabled": self.redis is not None
        })
    
    async def stop(self):
        """Stop the intelligence hub."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save state to Redis if available
        if self.redis:
            await self._save_state_to_redis()
        
        self.logger.worker_stop("normal")
    
    def register_worker(self, worker_id: str, worker_type: str, 
                       capabilities: Optional[List[str]] = None) -> bool:
        """Register a new worker with the hub."""
        if worker_id in self.workers:
            self.logger.coordination_event("Worker re-registration", {
                "worker_id": worker_id,
                "worker_type": worker_type
            })
            return False
        
        # Create worker profile
        worker_capabilities = {}
        if capabilities:
            for cap in capabilities:
                worker_capabilities[cap] = WorkerCapability(
                    name=cap,
                    proficiency=0.5,  # Start with neutral proficiency
                    success_rate=0.0,
                    avg_duration=0.0,
                    task_count=0,
                    last_used=0.0
                )
        
        profile = WorkerProfile(
            worker_id=worker_id,
            worker_type=worker_type,
            status=WorkerStatus.AVAILABLE,
            capabilities=worker_capabilities,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.workers[worker_id] = profile
        
        self.logger.coordination_event("Worker registered", {
            "worker_id": worker_id,
            "worker_type": worker_type,
            "capabilities": capabilities or []
        })
        
        return True
    
    def unregister_worker(self, worker_id: str, reason: str = "normal"):
        """Unregister a worker from the hub."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        
        # Handle any current task
        if worker.current_task_id and worker.current_task_id in self.tasks:
            task = self.tasks[worker.current_task_id]
            task.status = TaskStatus.PENDING
            task.assigned_worker = None
            task.claimed_at = None
            self.task_queue.appendleft(task)  # Put back at front for immediate reassignment
        
        # Update status and remove
        worker.status = WorkerStatus.OFFLINE
        del self.workers[worker_id]
        
        self.logger.coordination_event("Worker unregistered", {
            "worker_id": worker_id,
            "reason": reason
        })
        
        return True
    
    def heartbeat(self, worker_id: str, status: Optional[str] = None, 
                 metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Process worker heartbeat."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        worker.last_heartbeat = time.time()
        worker.updated_at = time.time()
        
        # Update status if provided
        if status:
            try:
                worker.status = WorkerStatus(status)
            except ValueError:
                pass
        
        # Update metrics if provided
        if metrics:
            if 'cpu_usage' in metrics:
                worker.cpu_usage = metrics['cpu_usage']
            if 'memory_usage' in metrics:
                worker.memory_usage = metrics['memory_usage']
        
        return True
    
    def submit_task(self, task_type: str, priority: int = 1, 
                   correlation_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a new task to the hub."""
        task_id = str(uuid.uuid4())
        
        if correlation_id is None:
            correlation_id = CorrelationIDManager.generate_id()
        
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            created_at=time.time(),
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task)
        
        self.logger.coordination_event("Task submitted", {
            "task_id": task_id,
            "task_type": task_type,
            "priority": priority,
            "correlation_id": correlation_id
        })
        
        return task_id
    
    def claim_task(self, worker_id: str, 
                   preferred_types: Optional[List[str]] = None) -> Optional[TaskInfo]:
        """Claim the best available task for a worker."""
        if worker_id not in self.workers:
            return None
        
        worker = self.workers[worker_id]
        
        # Find the best task for this worker
        best_task = None
        best_score = -1
        
        # Sort tasks by priority and creation time
        available_tasks = [task for task in self.task_queue 
                          if task.status == TaskStatus.PENDING]
        available_tasks.sort(key=lambda t: (-t.priority, t.created_at))
        
        for task in available_tasks:
            score = self._calculate_task_score(worker, task, preferred_types)
            if score > best_score:
                best_score = score
                best_task = task
        
        if best_task:
            # Claim the task
            best_task.status = TaskStatus.CLAIMED
            best_task.assigned_worker = worker_id
            best_task.claimed_at = time.time()
            
            worker.current_task_id = best_task.task_id
            worker.status = WorkerStatus.BUSY
            worker.updated_at = time.time()
            
            # Remove from queue
            self.task_queue.remove(best_task)
            
            self.logger.coordination_event("Task claimed", {
                "task_id": best_task.task_id,
                "worker_id": worker_id,
                "task_type": best_task.task_type,
                "score": best_score
            })
        
        return best_task
    
    def start_task(self, task_id: str, worker_id: str) -> bool:
        """Mark a task as started."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.assigned_worker != worker_id:
            return False
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        
        self.logger.coordination_event("Task started", {
            "task_id": task_id,
            "worker_id": worker_id,
            "task_type": task.task_type
        })
        
        return True
    
    def complete_task(self, task_id: str, worker_id: str, 
                     result: Optional[str] = None) -> bool:
        """Mark a task as completed and update learning data."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.assigned_worker != worker_id:
            return False
        
        # Update task
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result
        
        if task.started_at:
            task.duration = task.completed_at - task.started_at
        
        # Update worker profile
        worker = self.workers[worker_id]
        worker.successful_tasks += 1
        worker.total_tasks += 1
        worker.current_task_id = None
        worker.status = WorkerStatus.AVAILABLE
        worker.updated_at = time.time()
        
        if task.duration:
            worker.total_duration += task.duration
            worker.avg_task_duration = worker.total_duration / worker.total_tasks
        
        worker.success_rate = worker.successful_tasks / worker.total_tasks
        
        # Update capability proficiency
        if task.task_type in worker.capabilities:
            capability = worker.capabilities[task.task_type]
            capability.task_count += 1
            capability.success_rate = (capability.success_rate * (capability.task_count - 1) + 1.0) / capability.task_count
            
            if task.duration:
                capability.avg_duration = (capability.avg_duration * (capability.task_count - 1) + task.duration) / capability.task_count
            
            capability.proficiency = min(1.0, capability.proficiency + 0.1 * capability.success_rate)
            capability.last_used = time.time()
        
        # Learn from success
        self._learn_from_task_completion(task, worker, success=True)
        
        self.logger.intelligence_event("Task completed successfully", {
            "task_id": task_id,
            "worker_id": worker_id,
            "duration": task.duration,
            "worker_success_rate": worker.success_rate
        })
        
        return True
    
    def fail_task(self, task_id: str, worker_id: str, error: str, 
                 retry: bool = True) -> bool:
        """Mark a task as failed and handle retry logic."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.assigned_worker != worker_id:
            return False
        
        # Update task
        task.error = error
        task.retry_count += 1
        
        if task.started_at:
            task.duration = time.time() - task.started_at
        
        # Update worker profile
        worker = self.workers[worker_id]
        worker.failed_tasks += 1
        worker.total_tasks += 1
        worker.error_count += 1
        worker.last_error = error
        worker.current_task_id = None
        worker.status = WorkerStatus.AVAILABLE
        worker.updated_at = time.time()
        
        if task.duration:
            worker.total_duration += task.duration
            worker.avg_task_duration = worker.total_duration / worker.total_tasks
        
        worker.success_rate = worker.successful_tasks / worker.total_tasks
        
        # Update capability proficiency (reduce for failure)
        if task.task_type in worker.capabilities:
            capability = worker.capabilities[task.task_type]
            capability.task_count += 1
            capability.success_rate = (capability.success_rate * (capability.task_count - 1) + 0.0) / capability.task_count
            capability.proficiency = max(0.0, capability.proficiency - 0.05)
        
        # Decide on retry
        if retry and task.retry_count <= task.max_retries:
            task.status = TaskStatus.PENDING
            task.assigned_worker = None
            task.claimed_at = None
            task.started_at = None
            self.task_queue.append(task)  # Re-queue for retry
            
            self.logger.coordination_event("Task failed, retrying", {
                "task_id": task_id,
                "worker_id": worker_id,
                "error": error,
                "retry_count": task.retry_count
            })
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            
            self.logger.coordination_event("Task failed permanently", {
                "task_id": task_id,
                "worker_id": worker_id,
                "error": error,
                "retry_count": task.retry_count
            })
        
        # Learn from failure
        self._learn_from_task_completion(task, worker, success=False)
        
        return True
    
    def get_worker_status(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of one or all workers."""
        if worker_id:
            if worker_id in self.workers:
                return self.workers[worker_id].to_dict()
            return {}
        
        return {
            "total_workers": len(self.workers),
            "workers": {wid: worker.to_dict() for wid, worker in self.workers.items()},
            "status_counts": self._get_status_counts()
        }
    
    def get_task_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of one or all tasks."""
        if task_id:
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()
            return {}
        
        return {
            "total_tasks": len(self.tasks),
            "queue_length": len(self.task_queue),
            "task_status_counts": self._get_task_status_counts(),
            "recent_tasks": [task.to_dict() for task in list(self.tasks.values())[-10:]]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "workers": {
                "total": len(self.workers),
                "active": len([w for w in self.workers.values() if w.status in [WorkerStatus.AVAILABLE, WorkerStatus.BUSY]]),
                "busy": len([w for w in self.workers.values() if w.status == WorkerStatus.BUSY]),
                "average_success_rate": sum(w.success_rate for w in self.workers.values()) / max(len(self.workers), 1),
                "average_task_duration": sum(w.avg_task_duration for w in self.workers.values()) / max(len(self.workers), 1)
            },
            "tasks": {
                "total": len(self.tasks),
                "queue_length": len(self.task_queue),
                "completed": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
                "failed": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
                "success_rate": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]) / max(len(self.tasks), 1)
            },
            "task_types": self.task_type_stats,
            "system": self.system_metrics
        }
    
    def _calculate_task_score(self, worker: WorkerProfile, task: TaskInfo, 
                             preferred_types: Optional[List[str]] = None) -> float:
        """Calculate how well a worker matches a task."""
        score = 0.0
        
        # Priority bonus
        score += task.priority * 10
        
        # Capability match
        if task.task_type in worker.capabilities:
            capability = worker.capabilities[task.task_type]
            score += capability.proficiency * 50
            score += capability.success_rate * 30
            
            # Recency bonus (recently used capabilities are likely to be faster)
            if capability.last_used > 0:
                recency = max(0, 3600 - (time.time() - capability.last_used)) / 3600
                score += recency * 10
        
        # Preferred type bonus
        if preferred_types and task.task_type in preferred_types:
            score += 20
        
        # Worker performance bonus
        score += worker.success_rate * 20
        
        # Load balancing (prefer workers with fewer tasks)
        if worker.status == WorkerStatus.AVAILABLE:
            score += 15
        elif worker.status == WorkerStatus.IDLE:
            score += 25
        
        # Penalty for recent failures
        if worker.error_count > 0:
            score -= min(worker.error_count * 5, 25)
        
        return score
    
    def _learn_from_task_completion(self, task: TaskInfo, worker: WorkerProfile, success: bool):
        """Learn from task completion to improve future assignments."""
        task_type = task.task_type
        
        # Update task type statistics
        if task_type not in self.task_type_stats:
            self.task_type_stats[task_type] = {
                'total_attempts': 0,
                'successful_attempts': 0,
                'failed_attempts': 0,
                'average_duration': 0.0,
                'best_workers': [],
                'common_errors': defaultdict(int)
            }
        
        stats = self.task_type_stats[task_type]
        stats['total_attempts'] += 1
        
        if success:
            stats['successful_attempts'] += 1
            if task.duration:
                current_avg = stats['average_duration']
                stats['average_duration'] = (current_avg * (stats['successful_attempts'] - 1) + task.duration) / stats['successful_attempts']
            
            # Track best workers for this task type
            worker_perf = next((w for w in stats['best_workers'] if w['worker_id'] == worker.worker_id), None)
            if worker_perf:
                worker_perf['successes'] += 1
                worker_perf['success_rate'] = worker_perf['successes'] / (worker_perf['successes'] + worker_perf['failures'])
            else:
                stats['best_workers'].append({
                    'worker_id': worker.worker_id,
                    'worker_type': worker.worker_type,
                    'successes': 1,
                    'failures': 0,
                    'success_rate': 1.0
                })
            
            # Keep only top 5 workers
            stats['best_workers'].sort(key=lambda x: x['success_rate'], reverse=True)
            stats['best_workers'] = stats['best_workers'][:5]
        else:
            stats['failed_attempts'] += 1
            if task.error:
                stats['common_errors'][task.error] += 1
            
            # Update worker failure count in best_workers
            worker_perf = next((w for w in stats['best_workers'] if w['worker_id'] == worker.worker_id), None)
            if worker_perf:
                worker_perf['failures'] += 1
                worker_perf['success_rate'] = worker_perf['successes'] / (worker_perf['successes'] + worker_perf['failures'])
    
    def _get_status_counts(self) -> Dict[str, int]:
        """Get counts of workers by status."""
        counts = defaultdict(int)
        for worker in self.workers.values():
            counts[worker.status.value] += 1
        return dict(counts)
    
    def _get_task_status_counts(self) -> Dict[str, int]:
        """Get counts of tasks by status."""
        counts = defaultdict(int)
        for task in self.tasks.values():
            counts[task.status.value] += 1
        return dict(counts)
    
    async def _cleanup_loop(self):
        """Background cleanup of stale workers and tasks."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = time.time()
                
                # Clean up stale workers
                stale_workers = []
                for worker_id, worker in self.workers.items():
                    if (worker.last_heartbeat and 
                        current_time - worker.last_heartbeat > self.heartbeat_timeout):
                        stale_workers.append(worker_id)
                
                for worker_id in stale_workers:
                    self.unregister_worker(worker_id, "heartbeat_timeout")
                
                self.logger.coordination_event("Cleanup completed", {
                    "stale_workers_removed": len(stale_workers),
                    "active_workers": len(self.workers),
                    "pending_tasks": len(self.task_queue)
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.intelligence_event("Cleanup error", {"error": str(e)})
    
    async def _load_state_from_redis(self):
        """Load hub state from Redis."""
        try:
            # Load workers
            workers_data = await self.redis.get(f"intelligence_hub:{self.hub_id}:workers")
            if workers_data:
                workers_dict = json.loads(workers_data)
                for worker_id, worker_data in workers_dict.items():
                    # Reconstruct WorkerProfile
                    capabilities = {}
                    for cap_name, cap_data in worker_data.get('capabilities', {}).items():
                        capabilities[cap_name] = WorkerCapability(**cap_data)
                    
                    worker_data['status'] = WorkerStatus(worker_data['status'])
                    worker_data['capabilities'] = capabilities
                    self.workers[worker_id] = WorkerProfile(**worker_data)
            
            # Load task type statistics
            stats_data = await self.redis.get(f"intelligence_hub:{self.hub_id}:task_stats")
            if stats_data:
                self.task_type_stats = json.loads(stats_data)
            
            self.logger.intelligence_event("State loaded from Redis", {
                "workers_loaded": len(self.workers),
                "task_types_loaded": len(self.task_type_stats)
            })
            
        except Exception as e:
            self.logger.intelligence_event("Failed to load state from Redis", {"error": str(e)})
    
    async def _save_state_to_redis(self):
        """Save hub state to Redis."""
        try:
            # Save workers
            workers_dict = {worker_id: worker.to_dict() for worker_id, worker in self.workers.items()}
            await self.redis.set(
                f"intelligence_hub:{self.hub_id}:workers",
                json.dumps(workers_dict, default=str),
                ex=3600  # 1 hour expiry
            )
            
            # Save task type statistics
            await self.redis.set(
                f"intelligence_hub:{self.hub_id}:task_stats",
                json.dumps(self.task_type_stats, default=str),
                ex=3600
            )
            
            self.logger.intelligence_event("State saved to Redis", {
                "workers_saved": len(self.workers),
                "task_types_saved": len(self.task_type_stats)
            })
            
        except Exception as e:
            self.logger.intelligence_event("Failed to save state to Redis", {"error": str(e)})


# Global intelligence hub instance
_intelligence_hub: Optional[WorkerIntelligenceHub] = None

async def get_intelligence_hub(hub_id: str = "main_hub") -> WorkerIntelligenceHub:
    """Get or create the global intelligence hub."""
    global _intelligence_hub
    
    if _intelligence_hub is None:
        try:
            redis_client = await get_redis_client()
        except:
            redis_client = None
        
        _intelligence_hub = WorkerIntelligenceHub(hub_id, redis_client)
        await _intelligence_hub.start()
    
    return _intelligence_hub


# Example usage
if __name__ == "__main__":
    async def demo():
        # Create intelligence hub
        hub = WorkerIntelligenceHub("demo_hub")
        await hub.start()
        
        # Register workers
        hub.register_worker("worker_001", "continuous", ["github_issues", "code_review"])
        hub.register_worker("worker_002", "swarm", ["documentation", "testing"])
        
        # Submit tasks
        task1_id = hub.submit_task("github_issues", priority=2)
        task2_id = hub.submit_task("code_review", priority=1)
        task3_id = hub.submit_task("documentation", priority=3)
        
        # Worker claims and completes task
        task = hub.claim_task("worker_001", ["github_issues"])
        if task:
            hub.start_task(task.task_id, "worker_001")
            await asyncio.sleep(1)  # Simulate work
            hub.complete_task(task.task_id, "worker_001", "Issue resolved")
        
        # Get status
        status = hub.get_performance_metrics()
        print(f"Performance metrics: {json.dumps(status, indent=2, default=str)}")
        
        await hub.stop()
    
    asyncio.run(demo())