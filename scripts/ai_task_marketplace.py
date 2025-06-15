"""
AI Task Marketplace - Intelligent task distribution through competitive bidding
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import numpy as np

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.ai_worker_agent import AIWorkerAgent
from scripts.redis_work_queue import RedisWorkQueue


@dataclass
class TaskBid:
    """Bid submitted by a worker for a task"""
    worker_id: str
    work_item_id: str
    confidence: float
    estimated_time: float
    reasoning: str
    current_load: float
    total_experience: int
    timestamp: datetime
    
    def score(self, weights: Dict[str, float]) -> float:
        """Calculate bid score with configurable weights"""
        score = (
            self.confidence * weights.get("confidence", 0.4) +
            (1.0 / (1.0 + self.estimated_time)) * weights.get("speed", 0.2) +
            (1.0 - self.current_load) * weights.get("availability", 0.2) +
            min(1.0, self.total_experience / 100) * weights.get("experience", 0.2)
        )
        return score


class AITaskMarketplace:
    """Marketplace for intelligent task distribution"""
    
    def __init__(self, redis_queue: Optional[RedisWorkQueue] = None):
        self.logger = logging.getLogger("AITaskMarketplace")
        self.redis_queue = redis_queue
        
        # Registered workers
        self.workers: Dict[str, AIWorkerAgent] = {}
        
        # Active auctions
        self.active_auctions: Dict[str, List[TaskBid]] = {}
        
        # Auction parameters
        self.auction_duration = 5.0  # seconds
        self.min_bids_required = 1
        
        # Scoring weights (can be adjusted based on system performance)
        self.bid_weights = {
            "confidence": 0.4,
            "speed": 0.2,
            "availability": 0.2,
            "experience": 0.2
        }
        
        # Performance tracking
        self.task_outcomes: List[Dict[str, Any]] = []
        self.assignment_history: Dict[str, List[str]] = {}  # task_type -> [worker_ids]
        
        # Collaboration tracking
        self.collaboration_groups: Dict[str, List[str]] = {}
        
        self.logger.info("AI Task Marketplace initialized")
    
    async def register_worker(self, worker: AIWorkerAgent) -> None:
        """Register a worker in the marketplace"""
        self.workers[worker.worker_id] = worker
        self.logger.info(f"Registered worker: {worker.worker_id}")
    
    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker from the marketplace"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            self.logger.info(f"Unregistered worker: {worker_id}")
    
    async def submit_task(self, work_item: WorkItem) -> Optional[str]:
        """
        Submit a task to the marketplace and get the best worker
        
        Returns:
            Worker ID of the winning bidder, or None if no suitable worker
        """
        self.logger.info(f"Starting auction for task: {work_item.id} ({work_item.task_type})")
        
        # Initialize auction
        self.active_auctions[work_item.id] = []
        
        # Collect bids from all workers
        bid_tasks = []
        for worker_id, worker in self.workers.items():
            bid_tasks.append(self._collect_bid(worker, work_item))
        
        # Wait for bids with timeout
        bids = await asyncio.gather(*bid_tasks, return_exceptions=True)
        
        # Process valid bids
        valid_bids = []
        for bid in bids:
            if isinstance(bid, dict) and bid is not None:
                valid_bids.append(TaskBid(
                    worker_id=bid["worker_id"],
                    work_item_id=bid["work_item_id"],
                    confidence=bid["confidence"],
                    estimated_time=bid["estimated_time"],
                    reasoning=bid["reasoning"],
                    current_load=bid["current_load"],
                    total_experience=bid["total_experience"],
                    timestamp=datetime.fromisoformat(bid["timestamp"])
                ))
        
        self.active_auctions[work_item.id] = valid_bids
        
        # Select winner
        winner_id = await self._select_winner(work_item, valid_bids)
        
        # Clean up auction
        del self.active_auctions[work_item.id]
        
        if winner_id:
            # Record assignment
            task_type = work_item.task_type
            if task_type not in self.assignment_history:
                self.assignment_history[task_type] = []
            self.assignment_history[task_type].append(winner_id)
            
            # Notify winner to start task
            await self.workers[winner_id].start_task(work_item)
            
            self.logger.info(f"Task {work_item.id} assigned to worker {winner_id}")
        else:
            self.logger.warning(f"No suitable worker found for task {work_item.id}")
        
        return winner_id
    
    async def _collect_bid(self, worker: AIWorkerAgent, work_item: WorkItem) -> Optional[Dict[str, Any]]:
        """Collect bid from a worker with timeout"""
        try:
            return await asyncio.wait_for(
                worker.submit_bid(work_item),
                timeout=self.auction_duration
            )
        except asyncio.TimeoutError:
            self.logger.debug(f"Worker {worker.worker_id} timeout on bid for {work_item.id}")
            return None
        except Exception as e:
            self.logger.error(f"Worker {worker.worker_id} bid error: {e}")
            return None
    
    async def _select_winner(self, work_item: WorkItem, bids: List[TaskBid]) -> Optional[str]:
        """Select winning bid using intelligent scoring"""
        if not bids:
            return None
        
        # Check if task needs collaboration
        collaboration_needed = await self._assess_collaboration_need(work_item, bids)
        
        if collaboration_needed and len(bids) >= 2:
            # Select multiple workers for collaboration
            return await self._select_collaboration_team(work_item, bids)
        
        # Score all bids
        bid_scores = [(bid, bid.score(self.bid_weights)) for bid in bids]
        
        # Add historical performance bonus
        for bid, base_score in bid_scores:
            if bid.worker_id in self.assignment_history.get(work_item.task_type, []):
                # Bonus for workers who have done this task type before
                history_bonus = 0.1
                bid_scores[bid_scores.index((bid, base_score))] = (bid, base_score + history_bonus)
        
        # Sort by score
        bid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select winner
        winner_bid, winner_score = bid_scores[0]
        
        self.logger.info(
            f"Winner for {work_item.id}: {winner_bid.worker_id} "
            f"(score: {winner_score:.3f}, confidence: {winner_bid.confidence:.2f})"
        )
        
        # Log runner-ups for analysis
        if len(bid_scores) > 1:
            runner_up = bid_scores[1]
            self.logger.debug(
                f"Runner-up: {runner_up[0].worker_id} (score: {runner_up[1]:.3f})"
            )
        
        return winner_bid.worker_id
    
    async def _assess_collaboration_need(self, work_item: WorkItem, bids: List[TaskBid]) -> bool:
        """Determine if task would benefit from collaboration"""
        # High complexity tasks
        if work_item.estimated_cycles > 5:
            return True
        
        # No single worker is very confident
        max_confidence = max(bid.confidence for bid in bids) if bids else 0
        if max_confidence < 0.6 and len(bids) >= 2:
            return True
        
        # Task type known to benefit from collaboration
        collaborative_types = ["NEW_PROJECT", "COMPLEX_INTEGRATION", "SYSTEM_REFACTOR"]
        if work_item.task_type in collaborative_types:
            return True
        
        return False
    
    async def _select_collaboration_team(self, work_item: WorkItem, bids: List[TaskBid]) -> Optional[str]:
        """Select team of workers for collaboration"""
        # For now, return the most confident worker as lead
        # In full implementation, this would coordinate multiple workers
        if not bids:
            return None
        
        lead_worker = max(bids, key=lambda b: b.confidence)
        
        # Record collaboration group
        self.collaboration_groups[work_item.id] = [bid.worker_id for bid in bids[:3]]
        
        self.logger.info(
            f"Collaboration team for {work_item.id}: "
            f"lead={lead_worker.worker_id}, team={self.collaboration_groups[work_item.id]}"
        )
        
        return lead_worker.worker_id
    
    async def report_task_outcome(self, work_item_id: str, worker_id: str, 
                                success: bool, duration: float, quality_score: float = 1.0) -> None:
        """Report task completion outcome for learning"""
        outcome = {
            "work_item_id": work_item_id,
            "worker_id": worker_id,
            "success": success,
            "duration": duration,
            "quality_score": quality_score,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.task_outcomes.append(outcome)
        
        # Update bid weights based on outcomes
        await self._adjust_bid_weights(outcome)
        
        self.logger.info(
            f"Task outcome reported: {work_item_id} by {worker_id} - "
            f"success={success}, quality={quality_score:.2f}"
        )
    
    async def _adjust_bid_weights(self, outcome: Dict[str, Any]) -> None:
        """Dynamically adjust bid scoring weights based on outcomes"""
        # Simple adjustment based on recent performance
        recent_outcomes = self.task_outcomes[-20:]  # Last 20 tasks
        
        if len(recent_outcomes) >= 10:
            success_rate = sum(1 for o in recent_outcomes if o["success"]) / len(recent_outcomes)
            
            if success_rate < 0.7:
                # Increase weight on confidence and experience
                self.bid_weights["confidence"] = min(0.5, self.bid_weights["confidence"] + 0.02)
                self.bid_weights["experience"] = min(0.3, self.bid_weights["experience"] + 0.01)
                self.bid_weights["speed"] = max(0.1, self.bid_weights["speed"] - 0.02)
                self.bid_weights["availability"] = max(0.1, self.bid_weights["availability"] - 0.01)
            elif success_rate > 0.9:
                # System doing well, can optimize for speed
                self.bid_weights["speed"] = min(0.3, self.bid_weights["speed"] + 0.01)
                self.bid_weights["confidence"] = max(0.3, self.bid_weights["confidence"] - 0.01)
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get current marketplace statistics"""
        total_tasks = len(self.task_outcomes)
        successful_tasks = sum(1 for o in self.task_outcomes if o["success"])
        
        worker_stats = {}
        for worker_id, worker in self.workers.items():
            worker_stats[worker_id] = worker.get_performance_metrics()
        
        return {
            "active_workers": len(self.workers),
            "active_auctions": len(self.active_auctions),
            "total_tasks_processed": total_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "bid_weights": self.bid_weights,
            "worker_performance": worker_stats,
            "collaboration_groups": len(self.collaboration_groups),
            "task_type_distribution": {
                task_type: len(workers) 
                for task_type, workers in self.assignment_history.items()
            }
        }
    
    async def rebalance_workers(self) -> None:
        """Rebalance worker exploration rates based on system performance"""
        stats = self.get_marketplace_stats()
        system_success_rate = stats["success_rate"]
        
        for worker in self.workers.values():
            await worker.adjust_exploration_rate(system_success_rate)
        
        self.logger.info(f"Rebalanced workers with system success rate: {system_success_rate:.2%}")