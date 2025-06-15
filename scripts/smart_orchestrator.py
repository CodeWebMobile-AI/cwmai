"""
Smart Orchestrator - Meta-learning orchestrator for intelligent task distribution
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from scripts.ai_worker_agent import AIWorkerAgent
from scripts.ai_task_marketplace import AITaskMarketplace
from scripts.worker_capability_store import WorkerCapabilityStore
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.redis_work_queue import RedisWorkQueue


class SmartOrchestrator:
    """Meta-learning orchestrator that manages the entire intelligent system"""
    
    def __init__(self, num_workers: int = 5, redis_queue: Optional[RedisWorkQueue] = None):
        self.logger = logging.getLogger("SmartOrchestrator")
        self.num_workers = num_workers
        
        # Core components
        self.marketplace = AITaskMarketplace(redis_queue)
        self.capability_store = WorkerCapabilityStore()
        self.redis_queue = redis_queue
        
        # Worker pool
        self.workers: Dict[str, AIWorkerAgent] = {}
        
        # System state
        self.is_running = False
        self.performance_window = []  # Recent performance for adaptive behavior
        self.max_performance_window = 50
        
        # Adaptive parameters
        self.worker_spawn_threshold = 0.9  # Spawn new workers if load > this
        self.worker_remove_threshold = 0.3  # Remove workers if load < this
        self.rebalance_interval = 300  # seconds
        
        # Learning state
        self.task_patterns: Dict[str, Dict[str, Any]] = {}
        self.optimal_configs: Dict[str, Any] = {}
        
        self.logger.info(f"Smart Orchestrator initialized with {num_workers} workers")
    
    async def initialize(self, specializations: Optional[List[str]] = None) -> None:
        """Initialize the orchestrator and spawn initial workers"""
        self.logger.info("Initializing Smart Orchestrator...")
        
        # Default specializations if none provided
        if not specializations:
            specializations = ["system_tasks"] + [None] * (self.num_workers - 1)
        
        # Spawn initial workers
        for i in range(self.num_workers):
            spec = specializations[i] if i < len(specializations) else None
            worker_id = f"smart_worker_{i+1}"
            
            # Load previous capabilities if available
            capabilities = await self.capability_store.load_worker_capabilities(worker_id)
            
            worker = AIWorkerAgent(worker_id, spec)
            
            # Restore capabilities if available
            if capabilities and "capabilities" in capabilities:
                for cap_key, cap_data in capabilities["capabilities"].items():
                    if cap_key in worker.capabilities:
                        worker.capabilities[cap_key].__dict__.update(cap_data)
            
            self.workers[worker_id] = worker
            await self.marketplace.register_worker(worker)
            
            self.logger.info(f"Spawned worker {worker_id} with specialization: {spec}")
        
        # Start background tasks
        self.is_running = True
        asyncio.create_task(self._rebalance_loop())
        asyncio.create_task(self._save_capabilities_loop())
        asyncio.create_task(self._analyze_patterns_loop())
        
        self.logger.info("Smart Orchestrator initialized successfully")
    
    async def process_work_item(self, work_item: WorkItem) -> Optional[str]:
        """Process a work item through the intelligent system"""
        # Enrich work item with pattern analysis
        work_item = await self._enrich_work_item(work_item)
        
        # Submit to marketplace for bidding
        winner_id = await self.marketplace.submit_task(work_item)
        
        if winner_id:
            # Update task patterns
            pattern_key = f"{work_item.task_type}:{work_item.repository or 'none'}"
            if pattern_key not in self.task_patterns:
                self.task_patterns[pattern_key] = {
                    "count": 0,
                    "workers": [],
                    "avg_duration": 0,
                    "success_rate": 0
                }
            
            self.task_patterns[pattern_key]["count"] += 1
            self.task_patterns[pattern_key]["workers"].append(winner_id)
        
        return winner_id
    
    async def _enrich_work_item(self, work_item: WorkItem) -> WorkItem:
        """Enrich work item with additional context from learning"""
        pattern_key = f"{work_item.task_type}:{work_item.repository or 'none'}"
        
        if pattern_key in self.task_patterns:
            pattern = self.task_patterns[pattern_key]
            
            # Add pattern insights to metadata
            if not work_item.metadata:
                work_item.metadata = {}
            
            work_item.metadata["pattern_insights"] = {
                "historical_count": pattern["count"],
                "avg_duration": pattern["avg_duration"],
                "success_rate": pattern["success_rate"],
                "frequent_workers": self._get_frequent_workers(pattern["workers"])
            }
            
            # Adjust estimated cycles based on history
            if pattern["avg_duration"] > 0:
                work_item.estimated_cycles = int(pattern["avg_duration"])
        
        return work_item
    
    def _get_frequent_workers(self, worker_list: List[str], top_n: int = 3) -> List[str]:
        """Get most frequent workers from a list"""
        from collections import Counter
        counter = Counter(worker_list)
        return [worker for worker, _ in counter.most_common(top_n)]
    
    async def report_task_completion(self, work_item: WorkItem, worker_id: str, 
                                   success: bool, duration: float, quality_score: float = 1.0) -> None:
        """Report task completion and trigger learning"""
        # Report to marketplace
        await self.marketplace.report_task_outcome(
            work_item.id, worker_id, success, duration, quality_score
        )
        
        # Update worker
        if worker_id in self.workers:
            await self.workers[worker_id].complete_task(
                work_item, success, duration, quality_score
            )
        
        # Update performance window
        self.performance_window.append({
            "success": success,
            "quality": quality_score,
            "duration": duration,
            "timestamp": datetime.now(timezone.utc)
        })
        
        if len(self.performance_window) > self.max_performance_window:
            self.performance_window.pop(0)
        
        # Update task patterns
        pattern_key = f"{work_item.task_type}:{work_item.repository or 'none'}"
        if pattern_key in self.task_patterns:
            pattern = self.task_patterns[pattern_key]
            
            # Update success rate
            success_count = sum(1 for w in self.performance_window if w["success"])
            pattern["success_rate"] = success_count / len(self.performance_window)
            
            # Update average duration
            pattern["avg_duration"] = (pattern["avg_duration"] * (pattern["count"] - 1) + duration) / pattern["count"]
        
        # Check if we need to adapt
        await self._check_adaptation_needed()
    
    async def _check_adaptation_needed(self) -> None:
        """Check if system needs to adapt based on performance"""
        if len(self.performance_window) < 10:
            return
        
        recent_success_rate = sum(1 for p in self.performance_window[-10:] if p["success"]) / 10
        overall_load = sum(w.current_load for w in self.workers.values()) / len(self.workers)
        
        # Spawn new worker if overloaded and performing well
        if overall_load > self.worker_spawn_threshold and recent_success_rate > 0.7:
            await self._spawn_adaptive_worker()
        
        # Remove underutilized worker if load is low
        elif overall_load < self.worker_remove_threshold and len(self.workers) > 2:
            await self._remove_underutilized_worker()
    
    async def _spawn_adaptive_worker(self) -> None:
        """Spawn a new worker based on current needs"""
        # Analyze which task types need more workers
        task_loads = {}
        for pattern_key, pattern in self.task_patterns.items():
            if pattern["count"] > 0:
                task_loads[pattern_key] = pattern["count"] / sum(p["count"] for p in self.task_patterns.values())
        
        # Find the most loaded task type
        if task_loads:
            most_needed = max(task_loads.items(), key=lambda x: x[1])[0]
            specialization = most_needed.split(":")[1] if ":" in most_needed else None
        else:
            specialization = None
        
        # Create new worker
        worker_id = f"smart_worker_adaptive_{len(self.workers) + 1}"
        worker = AIWorkerAgent(worker_id, specialization)
        
        self.workers[worker_id] = worker
        await self.marketplace.register_worker(worker)
        
        self.logger.info(f"Spawned adaptive worker {worker_id} with specialization: {specialization}")
    
    async def _remove_underutilized_worker(self) -> None:
        """Remove the least utilized worker"""
        if len(self.workers) <= 2:  # Keep minimum workers
            return
        
        # Find worker with lowest utilization
        worker_utils = [(w_id, w.current_load, w.total_tasks_completed) 
                       for w_id, w in self.workers.items()]
        
        # Sort by current load and total tasks (prefer removing inexperienced idle workers)
        worker_utils.sort(key=lambda x: (x[1], x[2]))
        
        worker_to_remove = worker_utils[0][0]
        
        # Don't remove system_tasks worker
        if self.workers[worker_to_remove].primary_specialization == "system_tasks":
            if len(worker_utils) > 1:
                worker_to_remove = worker_utils[1][0]
            else:
                return
        
        # Remove worker
        await self.marketplace.unregister_worker(worker_to_remove)
        del self.workers[worker_to_remove]
        
        self.logger.info(f"Removed underutilized worker: {worker_to_remove}")
    
    async def _rebalance_loop(self) -> None:
        """Periodic rebalancing of the system"""
        while self.is_running:
            await asyncio.sleep(self.rebalance_interval)
            
            try:
                # Rebalance marketplace weights
                await self.marketplace.rebalance_workers()
                
                # Analyze and share insights
                insights = await self.capability_store.get_learning_insights()
                
                # Share insights with workers
                for insight_type, insight_data in insights.items():
                    await self.capability_store.share_knowledge(
                        f"system_insight_{insight_type}",
                        {"type": insight_type, "data": insight_data}
                    )
                
                self.logger.info("System rebalancing completed")
            except Exception as e:
                self.logger.error(f"Error in rebalance loop: {e}")
    
    async def _save_capabilities_loop(self) -> None:
        """Periodically save worker capabilities"""
        while self.is_running:
            await asyncio.sleep(60)  # Save every minute
            
            try:
                for worker_id, worker in self.workers.items():
                    capabilities = worker.get_performance_metrics()
                    await self.capability_store.save_worker_capabilities(worker_id, capabilities)
                
                # Save system performance
                system_stats = self.get_system_stats()
                await self.capability_store.save_system_performance(system_stats)
                
            except Exception as e:
                self.logger.error(f"Error saving capabilities: {e}")
    
    async def _analyze_patterns_loop(self) -> None:
        """Analyze task patterns and optimize system configuration"""
        while self.is_running:
            await asyncio.sleep(600)  # Analyze every 10 minutes
            
            try:
                # Analyze task patterns
                pattern_analysis = await self._analyze_task_patterns()
                
                # Share pattern insights
                await self.capability_store.share_knowledge(
                    "task_pattern_analysis",
                    pattern_analysis
                )
                
                # Find optimal configurations
                optimal_config = await self._find_optimal_configuration()
                if optimal_config:
                    self.optimal_configs[datetime.now(timezone.utc).isoformat()] = optimal_config
                
                self.logger.info("Pattern analysis completed")
            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {e}")
    
    async def _analyze_task_patterns(self) -> Dict[str, Any]:
        """Analyze task patterns for insights"""
        analysis = {
            "common_patterns": [],
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Find most common task patterns
        sorted_patterns = sorted(
            self.task_patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        for pattern_key, pattern_data in sorted_patterns[:5]:
            analysis["common_patterns"].append({
                "pattern": pattern_key,
                "count": pattern_data["count"],
                "success_rate": pattern_data["success_rate"],
                "avg_duration": pattern_data["avg_duration"]
            })
        
        # Identify bottlenecks (low success rate or long duration)
        for pattern_key, pattern_data in self.task_patterns.items():
            if pattern_data["count"] > 5:  # Enough data
                if pattern_data["success_rate"] < 0.7:
                    analysis["bottlenecks"].append({
                        "pattern": pattern_key,
                        "issue": "low_success_rate",
                        "success_rate": pattern_data["success_rate"]
                    })
                elif pattern_data["avg_duration"] > 5.0:  # 5 hours
                    analysis["bottlenecks"].append({
                        "pattern": pattern_key,
                        "issue": "long_duration",
                        "avg_duration": pattern_data["avg_duration"]
                    })
        
        # Generate recommendations
        if analysis["bottlenecks"]:
            for bottleneck in analysis["bottlenecks"]:
                if bottleneck["issue"] == "low_success_rate":
                    # Find experts for this pattern
                    experts = await self.capability_store.find_experts(
                        bottleneck["pattern"].split(":")[0]
                    )
                    
                    analysis["recommendations"].append({
                        "pattern": bottleneck["pattern"],
                        "recommendation": "assign_to_experts",
                        "experts": experts[:3]
                    })
                elif bottleneck["issue"] == "long_duration":
                    analysis["recommendations"].append({
                        "pattern": bottleneck["pattern"],
                        "recommendation": "break_into_subtasks",
                        "reason": "Task takes too long, consider decomposition"
                    })
        
        return analysis
    
    async def _find_optimal_configuration(self) -> Dict[str, Any]:
        """Find optimal system configuration based on performance"""
        if len(self.performance_window) < 20:
            return {}
        
        recent_performance = self.performance_window[-20:]
        success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
        avg_quality = sum(p["quality"] for p in recent_performance) / len(recent_performance)
        avg_duration = sum(p["duration"] for p in recent_performance) / len(recent_performance)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_workers": len(self.workers),
            "worker_specializations": {
                w_id: w.primary_specialization 
                for w_id, w in self.workers.items()
            },
            "performance_metrics": {
                "success_rate": success_rate,
                "avg_quality": avg_quality,
                "avg_duration": avg_duration
            },
            "marketplace_weights": self.marketplace.bid_weights.copy(),
            "exploration_rates": {
                w_id: w.exploration_rate 
                for w_id, w in self.workers.items()
            }
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        marketplace_stats = self.marketplace.get_marketplace_stats()
        
        return {
            "orchestrator": {
                "num_workers": len(self.workers),
                "is_running": self.is_running,
                "performance_window_size": len(self.performance_window),
                "task_patterns_tracked": len(self.task_patterns)
            },
            "marketplace": marketplace_stats,
            "workers": {
                worker_id: worker.get_performance_metrics()
                for worker_id, worker in self.workers.items()
            },
            "recent_performance": {
                "success_rate": sum(1 for p in self.performance_window if p["success"]) / len(self.performance_window) if self.performance_window else 0,
                "avg_quality": sum(p["quality"] for p in self.performance_window) / len(self.performance_window) if self.performance_window else 0
            }
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator"""
        self.logger.info("Shutting down Smart Orchestrator...")
        
        self.is_running = False
        
        # Save final capabilities
        for worker_id, worker in self.workers.items():
            capabilities = worker.get_performance_metrics()
            await self.capability_store.save_worker_capabilities(worker_id, capabilities)
        
        # Save final system state
        system_stats = self.get_system_stats()
        await self.capability_store.save_system_performance(system_stats)
        
        self.logger.info("Smart Orchestrator shutdown complete")