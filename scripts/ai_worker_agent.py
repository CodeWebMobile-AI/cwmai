"""
AI Worker Agent - Self-organizing intelligent worker with learning capabilities
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from scripts.work_item_types import WorkItem, TaskPriority


@dataclass
class TaskOutcome:
    """Record of a completed task for learning"""
    task_id: str
    task_type: str
    repository: Optional[str]
    success: bool
    duration: float
    confidence_predicted: float
    confidence_actual: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_score: float = 1.0  # 0-1 score based on outcome quality


@dataclass
class WorkerCapability:
    """Dynamic capability tracking for a worker"""
    task_type: str
    success_rate: float = 0.5
    avg_duration: float = 1.0
    total_tasks: int = 0
    confidence_multiplier: float = 1.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update(self, outcome: TaskOutcome):
        """Update capability based on task outcome"""
        # Exponential moving average for smoother learning
        alpha = 0.3  # Learning rate
        
        self.success_rate = alpha * (1.0 if outcome.success else 0.0) + (1 - alpha) * self.success_rate
        self.avg_duration = alpha * outcome.duration + (1 - alpha) * self.avg_duration
        
        # Update confidence multiplier based on prediction accuracy
        confidence_error = abs(outcome.confidence_predicted - outcome.confidence_actual)
        self.confidence_multiplier = alpha * (1.0 - confidence_error) + (1 - alpha) * self.confidence_multiplier
        
        self.total_tasks += 1
        self.last_updated = datetime.now(timezone.utc)


class AIWorkerAgent:
    """Intelligent worker agent with self-evaluation and learning capabilities"""
    
    def __init__(self, worker_id: str, primary_specialization: Optional[str] = None):
        self.worker_id = worker_id
        self.primary_specialization = primary_specialization
        self.logger = logging.getLogger(f"AIWorkerAgent.{worker_id}")
        
        # Capability tracking
        self.capabilities: Dict[str, WorkerCapability] = defaultdict(
            lambda: WorkerCapability(task_type="unknown")
        )
        
        # Task history for learning
        self.task_history: List[TaskOutcome] = []
        self.max_history = 1000
        
        # Performance metrics
        self.total_tasks_completed = 0
        self.current_load = 0.0
        self.max_concurrent_tasks = 3
        self.active_tasks: Dict[str, datetime] = {}
        
        # Learning parameters
        self.exploration_rate = 0.2  # Probability of taking unfamiliar tasks
        self.confidence_threshold = 0.3  # Minimum confidence to bid
        
        # Knowledge sharing
        self.shared_knowledge: Dict[str, Any] = {}
        
        self.logger.info(f"AI Worker {worker_id} initialized with specialization: {primary_specialization}")
    
    async def evaluate_task(self, work_item: WorkItem) -> Tuple[float, float, str]:
        """
        Evaluate ability to handle a task
        
        Returns:
            Tuple of (confidence, estimated_time, reasoning)
        """
        confidence = 0.0
        estimated_time = 1.0
        reasoning_parts = []
        
        # 1. Check primary specialization match
        if self.primary_specialization:
            if work_item.repository == self.primary_specialization:
                confidence += 0.5
                reasoning_parts.append(f"Primary specialization match: {self.primary_specialization}")
            elif self.primary_specialization == "system_tasks" and not work_item.repository:
                confidence += 0.4
                reasoning_parts.append("System tasks specialist for repo-less task")
        
        # 2. Check capability history
        capability_key = f"{work_item.task_type}:{work_item.repository or 'none'}"
        if capability_key in self.capabilities:
            cap = self.capabilities[capability_key]
            confidence += cap.success_rate * 0.3
            estimated_time = cap.avg_duration
            reasoning_parts.append(f"Experience: {cap.total_tasks} similar tasks, {cap.success_rate:.1%} success")
        else:
            # Check for related capabilities
            for key, cap in self.capabilities.items():
                if work_item.task_type in key or (work_item.repository and work_item.repository in key):
                    confidence += cap.success_rate * 0.1
                    reasoning_parts.append(f"Related experience in {key}")
        
        # 3. Current load factor
        load_penalty = self.current_load / self.max_concurrent_tasks
        confidence *= (1.0 - load_penalty * 0.5)
        estimated_time *= (1.0 + load_penalty)
        
        if self.current_load > 0:
            reasoning_parts.append(f"Current load: {self.current_load}/{self.max_concurrent_tasks}")
        
        # 4. Task complexity analysis
        if work_item.description:
            desc_lower = work_item.description.lower()
            
            # Boost confidence for familiar keywords
            familiar_keywords = ["refactor", "test", "optimize", "fix", "update", "implement"]
            matches = sum(1 for kw in familiar_keywords if kw in desc_lower)
            if matches > 0:
                confidence += matches * 0.05
                reasoning_parts.append(f"Familiar with {matches} task aspects")
            
            # Adjust for complexity indicators
            if any(word in desc_lower for word in ["complex", "difficult", "challenging"]):
                estimated_time *= 1.5
                reasoning_parts.append("Complex task detected")
        
        # 5. Apply confidence multiplier from learning
        avg_multiplier = np.mean([cap.confidence_multiplier for cap in self.capabilities.values()]) if self.capabilities else 1.0
        confidence *= avg_multiplier
        
        # 6. Exploration bonus for learning
        if confidence < self.confidence_threshold and np.random.random() < self.exploration_rate:
            confidence = self.confidence_threshold + 0.1
            reasoning_parts.append("Exploring new task type for learning")
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No specific experience"
        
        return confidence, estimated_time, reasoning
    
    async def submit_bid(self, work_item: WorkItem) -> Optional[Dict[str, Any]]:
        """Submit a bid for a work item"""
        confidence, estimated_time, reasoning = await self.evaluate_task(work_item)
        
        # Don't bid if confidence is too low (unless exploring)
        if confidence < self.confidence_threshold:
            self.logger.debug(f"Not bidding on {work_item.id}: confidence {confidence:.2f} below threshold")
            return None
        
        bid = {
            "worker_id": self.worker_id,
            "work_item_id": work_item.id,
            "confidence": confidence,
            "estimated_time": estimated_time,
            "reasoning": reasoning,
            "current_load": self.current_load,
            "total_experience": self.total_tasks_completed,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Submitting bid for {work_item.id}: confidence={confidence:.2f}, time={estimated_time:.1f}h")
        return bid
    
    async def start_task(self, work_item: WorkItem) -> None:
        """Mark task as started"""
        self.active_tasks[work_item.id] = datetime.now(timezone.utc)
        self.current_load = len(self.active_tasks) / self.max_concurrent_tasks
        self.logger.info(f"Started task {work_item.id}, current load: {self.current_load:.1%}")
    
    async def complete_task(self, work_item: WorkItem, success: bool, 
                          actual_duration: float, feedback_score: float = 1.0) -> None:
        """Record task completion and learn from outcome"""
        if work_item.id not in self.active_tasks:
            self.logger.warning(f"Completing unknown task: {work_item.id}")
            return
        
        # Calculate predicted confidence for this task
        predicted_confidence, _, _ = await self.evaluate_task(work_item)
        
        # Create outcome record
        outcome = TaskOutcome(
            task_id=work_item.id,
            task_type=work_item.task_type,
            repository=work_item.repository,
            success=success,
            duration=actual_duration,
            confidence_predicted=predicted_confidence,
            confidence_actual=feedback_score if success else 0.0,
            feedback_score=feedback_score
        )
        
        # Update capabilities
        capability_key = f"{work_item.task_type}:{work_item.repository or 'none'}"
        if capability_key not in self.capabilities:
            self.capabilities[capability_key] = WorkerCapability(task_type=capability_key)
        
        self.capabilities[capability_key].update(outcome)
        
        # Store in history
        self.task_history.append(outcome)
        if len(self.task_history) > self.max_history:
            self.task_history.pop(0)
        
        # Update metrics
        del self.active_tasks[work_item.id]
        self.current_load = len(self.active_tasks) / self.max_concurrent_tasks
        self.total_tasks_completed += 1
        
        self.logger.info(
            f"Completed task {work_item.id}: success={success}, "
            f"duration={actual_duration:.1f}h, feedback={feedback_score:.2f}"
        )
    
    async def share_knowledge(self, knowledge_key: str, knowledge_value: Any) -> None:
        """Share learned knowledge with other workers"""
        self.shared_knowledge[knowledge_key] = {
            "value": knowledge_value,
            "worker_id": self.worker_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": self._calculate_knowledge_confidence()
        }
    
    def _calculate_knowledge_confidence(self) -> float:
        """Calculate confidence in shared knowledge based on experience"""
        if not self.capabilities:
            return 0.5
        
        avg_success = np.mean([cap.success_rate for cap in self.capabilities.values()])
        experience_factor = min(1.0, self.total_tasks_completed / 100)
        
        return avg_success * experience_factor
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "worker_id": self.worker_id,
            "specialization": self.primary_specialization,
            "current_load": self.current_load,
            "active_tasks": len(self.active_tasks),
            "total_completed": self.total_tasks_completed,
            "capabilities": {
                k: {
                    "success_rate": v.success_rate,
                    "avg_duration": v.avg_duration,
                    "total_tasks": v.total_tasks,
                    "confidence_multiplier": v.confidence_multiplier
                }
                for k, v in self.capabilities.items()
            },
            "overall_success_rate": np.mean([cap.success_rate for cap in self.capabilities.values()]) if self.capabilities else 0.0,
            "exploration_rate": self.exploration_rate
        }
    
    async def adjust_exploration_rate(self, system_performance: float) -> None:
        """Dynamically adjust exploration rate based on system performance"""
        if system_performance > 0.8:
            # System doing well, can afford more exploration
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
        elif system_performance < 0.6:
            # System struggling, focus on known strengths
            self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
    
    def should_collaborate(self, work_item: WorkItem) -> bool:
        """Determine if this task would benefit from collaboration"""
        # Complex tasks benefit from collaboration
        if work_item.estimated_cycles > 5:
            return True
        
        # Tasks we're uncertain about
        capability_key = f"{work_item.task_type}:{work_item.repository or 'none'}"
        if capability_key in self.capabilities:
            if self.capabilities[capability_key].success_rate < 0.6:
                return True
        
        # New task types we haven't seen much
        if capability_key not in self.capabilities or self.capabilities[capability_key].total_tasks < 3:
            return True
        
        return False