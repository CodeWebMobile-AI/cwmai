"""
Ultimate Orchestrator - Combining All Benefits Without Redis

Integrates the best features from:
- SmartOrchestrator: Bidding system, worker capabilities, adaptive spawning
- EnhancedContinuousOrchestrator: MCP integration, fallback mechanisms
- HybridContinuousOrchestrator: Dynamic charter, outcome learning, strategic intelligence

All powered by SQLite instead of Redis.
"""

import asyncio
import logging
import time
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import random
import traceback

# Base imports
from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis, WorkerStatus, WorkerState
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.hybrid_state_manager import HybridStateManager

# Dynamic components
from scripts.dynamic_charter import DynamicCharter
from scripts.outcome_learning import OutcomeLearningSystem
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.dynamic_validator import DynamicTaskValidator

# Utilities
from scripts.ai_brain_factory import AIBrainFactory
from scripts.state_manager import StateManager


@dataclass
class WorkerBid:
    """A bid from a worker for a task."""
    worker_id: str
    task_id: str
    confidence: float  # 0.0 to 1.0
    estimated_time: float  # seconds
    reasoning: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerCapability:
    """Tracks worker capabilities and performance."""
    worker_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    task_type_experience: Dict[str, int] = field(default_factory=dict)
    repository_experience: Dict[str, int] = field(default_factory=dict)
    average_completion_time: float = 0.0
    specializations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks


class UltimateOrchestrator(ContinuousOrchestratorNoRedis):
    """
    The Ultimate Orchestrator combining all benefits without Redis.
    
    Features:
    - Competitive bidding system for task assignment
    - Worker capability tracking and specialization
    - Adaptive worker spawning based on load
    - Dynamic charter evolution
    - Outcome-based learning
    - Task pattern recognition
    - Self-improving configuration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Ultimate Orchestrator."""
        # Default configuration
        default_config = {
            "enable_bidding": True,
            "enable_adaptive": True,
            "enable_learning": True,
            "enable_charter": True,
            "enable_mcp": True,
            "bidding_timeout": 5.0,
            "min_workers": 2,
            "max_workers": 10,
            "load_threshold_spawn": 0.8,
            "load_threshold_remove": 0.3,
            "enable_parallel": True,
            "enable_research": True,
            "enable_round_robin": False
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize base orchestrator
        super().__init__(
            max_workers=self.config["max_workers"],
            enable_parallel=self.config["enable_parallel"],
            enable_research=self.config["enable_research"],
            enable_round_robin=self.config["enable_round_robin"]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize SQLite database
        self._init_database()
        
        # Initialize components
        self._init_components()
        
        # Metrics
        self.bidding_metrics = {
            "total_auctions": 0,
            "successful_bids": 0,
            "failed_bids": 0,
            "average_bid_time": 0.0,
            "worker_assignments": {}
        }
        
        # Active auctions
        self.active_auctions: Dict[str, List[WorkerBid]] = {}
        
        print(f"✓ Ultimate Orchestrator initialized with config: {self.config}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        self.db_path = "ultimate_orchestrator.db"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        cursor = self.conn.cursor()
        
        # Worker capabilities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS worker_capabilities (
                worker_id TEXT PRIMARY KEY,
                capabilities JSON,
                total_tasks INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                specializations JSON,
                last_updated TIMESTAMP
            )
        """)
        
        # Task patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_patterns (
                pattern_key TEXT PRIMARY KEY,
                task_type TEXT,
                repository TEXT,
                avg_duration REAL,
                success_rate REAL,
                worker_affinity JSON
            )
        """)
        
        # Bid history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bid_history (
                bid_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                worker_id TEXT,
                confidence REAL,
                estimated_time REAL,
                actual_time REAL,
                success BOOLEAN,
                timestamp TIMESTAMP
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT,
                value REAL,
                metadata JSON,
                timestamp TIMESTAMP
            )
        """)
        
        self.conn.commit()
        
        print("✓ SQLite database initialized")
    
    def _init_components(self):
        """Initialize intelligent components."""
        # Dynamic components (if enabled)
        if self.config["enable_charter"]:
            self.charter_system = DynamicCharter(self.ai_brain)
            self.logger.info("Dynamic charter system initialized")
        
        if self.config["enable_learning"]:
            self.learning_system = OutcomeLearningSystem(self.ai_brain)
            self.task_validator = DynamicTaskValidator(self.ai_brain, self.charter_system if self.config["enable_charter"] else None)
            self.logger.info("Learning systems initialized")
        
        # Worker capabilities cache
        self.worker_capabilities: Dict[str, WorkerCapability] = {}
        self._load_worker_capabilities()
    
    def _load_worker_capabilities(self):
        """Load worker capabilities from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM worker_capabilities")
        
        for row in cursor.fetchall():
            capability = WorkerCapability(
                worker_id=row["worker_id"],
                total_tasks=row["total_tasks"]
            )
            
            # Load JSON fields
            if row["capabilities"]:
                caps = json.loads(row["capabilities"])
                capability.task_type_experience = caps.get("task_type_experience", {})
                capability.repository_experience = caps.get("repository_experience", {})
                capability.average_completion_time = caps.get("average_completion_time", 0.0)
                capability.successful_tasks = caps.get("successful_tasks", 0)
                capability.failed_tasks = caps.get("failed_tasks", 0)
            
            if row["specializations"]:
                capability.specializations = json.loads(row["specializations"])
            
            self.worker_capabilities[row["worker_id"]] = capability
    
    async def find_single_work_item(self) -> Optional[WorkItem]:
        """Find work using intelligent task generation and pattern matching."""
        # First try base implementation
        work_item = await super().find_single_work_item()
        
        if work_item and self.config["enable_learning"]:
            # Enhance with pattern matching
            pattern_key = f"{work_item.task_type}:{work_item.repository or 'system'}"
            pattern = self._get_task_pattern(pattern_key)
            
            if pattern:
                # Add pattern insights to metadata
                work_item.metadata["pattern_insights"] = {
                    "avg_duration": pattern["avg_duration"],
                    "success_rate": pattern["success_rate"],
                    "best_workers": pattern.get("worker_affinity", {})
                }
        
        return work_item
    
    async def _assign_work_to_worker(self, work_item: WorkItem) -> Optional[str]:
        """Assign work to a worker using bidding or direct assignment."""
        if not self.config["enable_bidding"]:
            # Use traditional round-robin assignment
            idle_workers = [w_id for w_id, w in self.workers.items() 
                          if w.status == WorkerStatus.IDLE]
            return idle_workers[0] if idle_workers else None
        
        # Start bidding process
        auction_id = work_item.id
        self.active_auctions[auction_id] = []
        
        # Notify workers about the task
        await self._broadcast_task_to_workers(work_item)
        
        # Wait for bids
        await asyncio.sleep(self.config["bidding_timeout"])
        
        # Select best bid
        best_worker = self._select_best_bid(auction_id, work_item)
        
        # Clean up auction
        del self.active_auctions[auction_id]
        
        # Record metrics
        self.bidding_metrics["total_auctions"] += 1
        if best_worker:
            self.bidding_metrics["successful_bids"] += 1
            self.bidding_metrics["worker_assignments"][best_worker] = \
                self.bidding_metrics["worker_assignments"].get(best_worker, 0) + 1
        else:
            self.bidding_metrics["failed_bids"] += 1
        
        return best_worker
    
    async def _broadcast_task_to_workers(self, work_item: WorkItem):
        """Broadcast task to all idle workers for bidding."""
        for worker_id, worker in self.workers.items():
            if worker.status == WorkerStatus.IDLE:
                # In a real implementation, this would send a message to the worker
                # For now, we'll simulate worker bidding
                await self._simulate_worker_bid(worker_id, work_item)
    
    async def _simulate_worker_bid(self, worker_id: str, work_item: WorkItem):
        """Simulate a worker creating a bid for a task."""
        capability = self.worker_capabilities.get(worker_id, WorkerCapability(worker_id))
        
        # Calculate confidence based on experience
        task_experience = capability.task_type_experience.get(work_item.task_type, 0)
        repo_experience = capability.repository_experience.get(work_item.repository or "system", 0)
        
        confidence = min(1.0, (task_experience * 0.1 + repo_experience * 0.05 + capability.success_rate * 0.5))
        
        # Estimate time based on history
        if capability.average_completion_time > 0:
            estimated_time = capability.average_completion_time * (2.0 - confidence)
        else:
            estimated_time = 60.0  # Default 1 minute
        
        # Add some randomness to simulate different worker assessments
        confidence *= random.uniform(0.8, 1.2)
        confidence = max(0.1, min(1.0, confidence))
        
        bid = WorkerBid(
            worker_id=worker_id,
            task_id=work_item.id,
            confidence=confidence,
            estimated_time=estimated_time,
            reasoning=f"Experience: {task_experience} tasks of this type, {repo_experience} in this repo"
        )
        
        self.active_auctions[work_item.id].append(bid)
    
    def _select_best_bid(self, auction_id: str, work_item: WorkItem) -> Optional[str]:
        """Select the best bid for a task."""
        bids = self.active_auctions.get(auction_id, [])
        
        if not bids:
            return None
        
        # Score bids based on confidence, time estimate, and worker load
        best_bid = None
        best_score = -1
        
        for bid in bids:
            # Calculate score: high confidence, low time estimate
            score = bid.confidence / (bid.estimated_time / 60.0)  # Normalize by minutes
            
            # Boost score for workers with fewer current tasks
            worker = self.workers.get(bid.worker_id)
            if worker and worker.status == WorkerStatus.IDLE:
                score *= 1.5
            
            # Consider task priority
            if work_item.priority == TaskPriority.CRITICAL:
                # For critical tasks, prioritize confidence
                score = bid.confidence * 2.0
            
            if score > best_score:
                best_score = score
                best_bid = bid
        
        if best_bid:
            # Record bid in history
            self._record_bid(best_bid, work_item)
            return best_bid.worker_id
        
        return None
    
    def _record_bid(self, bid: WorkerBid, work_item: WorkItem):
        """Record bid in database for learning."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO bid_history (task_id, worker_id, confidence, estimated_time, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (bid.task_id, bid.worker_id, bid.confidence, bid.estimated_time, datetime.now(timezone.utc)))
        self.conn.commit()
    
    async def _worker_loop(self, worker_id: str):
        """Enhanced worker loop with bidding support."""
        self.logger.info(f"Worker {worker_id} started (Ultimate mode)")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Initialize worker capability
        if worker_id not in self.worker_capabilities:
            self.worker_capabilities[worker_id] = WorkerCapability(worker_id)
        
        while not self.shutdown_requested:
            try:
                if self.pause_requested:
                    await asyncio.sleep(1)
                    continue
                
                # Update worker status
                self.workers[worker_id].status = WorkerStatus.IDLE
                
                # Find work (either through bidding or direct assignment)
                work_item = await self.find_single_work_item()
                
                if not work_item:
                    await asyncio.sleep(5)
                    continue
                
                # If bidding is enabled, wait for assignment
                if self.config["enable_bidding"]:
                    assigned_worker = await self._assign_work_to_worker(work_item)
                    if assigned_worker != worker_id:
                        # Another worker won the bid
                        continue
                
                # Update worker status
                self.workers[worker_id].status = WorkerStatus.WORKING
                
                # Execute work
                start_time = time.time()
                result = await self._execute_work_item(worker_id, work_item)
                duration = time.time() - start_time
                
                # Update capabilities
                self._update_worker_capability(worker_id, work_item, result, duration)
                
                # Learn from outcome if enabled
                if self.config["enable_learning"] and hasattr(self, 'learning_system'):
                    await self.learning_system.record_outcome(
                        work_item.to_dict(),
                        result
                    )
                
                # Update bid history with actual result
                if self.config["enable_bidding"]:
                    self._update_bid_history(work_item.id, worker_id, duration, result['success'])
                
                # Reset error count on success
                if result['success']:
                    consecutive_errors = 0
                
                await asyncio.sleep(1)
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Worker {worker_id} error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(f"Worker {worker_id} shutting down due to errors")
                    break
                
                await asyncio.sleep(min(60, 5 * (2 ** (consecutive_errors - 1))))
        
        self.workers[worker_id].status = WorkerStatus.SHUTTING_DOWN
        self.logger.info(f"Worker {worker_id} stopped")
    
    def _update_worker_capability(self, worker_id: str, work_item: WorkItem, 
                                 result: Dict[str, Any], duration: float):
        """Update worker capability based on task result."""
        capability = self.worker_capabilities.get(worker_id)
        if not capability:
            capability = WorkerCapability(worker_id)
            self.worker_capabilities[worker_id] = capability
        
        # Update counts
        capability.total_tasks += 1
        if result['success']:
            capability.successful_tasks += 1
        else:
            capability.failed_tasks += 1
        
        # Update experience
        task_type = work_item.task_type
        repo = work_item.repository or "system"
        
        capability.task_type_experience[task_type] = \
            capability.task_type_experience.get(task_type, 0) + 1
        capability.repository_experience[repo] = \
            capability.repository_experience.get(repo, 0) + 1
        
        # Update average completion time
        if capability.average_completion_time == 0:
            capability.average_completion_time = duration
        else:
            capability.average_completion_time = \
                (capability.average_completion_time * (capability.total_tasks - 1) + duration) / capability.total_tasks
        
        # Update specializations
        if capability.task_type_experience[task_type] >= 5 and task_type not in capability.specializations:
            capability.specializations.append(task_type)
        
        # Save to database
        self._save_worker_capability(capability)
    
    def _save_worker_capability(self, capability: WorkerCapability):
        """Save worker capability to database."""
        cursor = self.conn.cursor()
        
        capabilities_json = json.dumps({
            "task_type_experience": capability.task_type_experience,
            "repository_experience": capability.repository_experience,
            "average_completion_time": capability.average_completion_time,
            "successful_tasks": capability.successful_tasks,
            "failed_tasks": capability.failed_tasks
        })
        
        cursor.execute("""
            INSERT OR REPLACE INTO worker_capabilities 
            (worker_id, capabilities, total_tasks, success_rate, specializations, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            capability.worker_id,
            capabilities_json,
            capability.total_tasks,
            capability.success_rate,
            json.dumps(capability.specializations),
            datetime.now(timezone.utc)
        ))
        self.conn.commit()
    
    def _update_bid_history(self, task_id: str, worker_id: str, actual_time: float, success: bool):
        """Update bid history with actual results."""
        cursor = self.conn.cursor()
        # First find the most recent bid
        cursor.execute("""
            SELECT bid_id FROM bid_history 
            WHERE task_id = ? AND worker_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (task_id, worker_id))
        
        row = cursor.fetchone()
        if row:
            # Update the specific bid
            cursor.execute("""
                UPDATE bid_history 
                SET actual_time = ?, success = ?
                WHERE bid_id = ?
            """, (actual_time, success, row[0]))
            self.conn.commit()
    
    def _get_task_pattern(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Get task pattern from database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM task_patterns WHERE pattern_key = ?
        """, (pattern_key,))
        
        row = cursor.fetchone()
        if row:
            return {
                "avg_duration": row["avg_duration"],
                "success_rate": row["success_rate"],
                "worker_affinity": json.loads(row["worker_affinity"]) if row["worker_affinity"] else {}
            }
        return None
    
    async def _adaptive_worker_management(self):
        """Manage worker count based on system load."""
        # Wait a bit before starting adaptive management
        await asyncio.sleep(10)
        
        while not self.shutdown_requested:
            try:
                # Calculate system load
                total_workers = len(self.workers)
                busy_workers = sum(1 for w in self.workers.values() 
                                 if w.status == WorkerStatus.WORKING)
                
                if total_workers > 0:
                    load = busy_workers / total_workers
                    
                    # Check if there's pending work
                    pending_work = len(self.active_auctions) > 0 if hasattr(self, 'active_auctions') else False
                    
                    # Spawn new worker if load is high
                    if load > self.config["load_threshold_spawn"] and total_workers < self.config["max_workers"]:
                        worker_id = f"worker_{total_workers:02d}"
                        self.workers[worker_id] = WorkerState(
                            id=worker_id,
                            status=WorkerStatus.IDLE
                        )
                        self.worker_tasks[worker_id] = asyncio.create_task(
                            self._worker_loop(worker_id)
                        )
                        self.logger.info(f"Spawned new worker {worker_id} due to high load ({load:.2%})")
                    
                    # Remove idle worker if load is low AND no pending work
                    elif load < self.config["load_threshold_remove"] and total_workers > self.config["min_workers"] and not pending_work:
                        # Find an idle worker to remove
                        for worker_id, worker in list(self.workers.items()):
                            if worker.status == WorkerStatus.IDLE and total_workers > self.config["min_workers"]:
                                # Mark for shutdown
                                worker.status = WorkerStatus.SHUTTING_DOWN
                                self.logger.info(f"Removing worker {worker_id} due to low load ({load:.2%})")
                                total_workers -= 1
                                break
                
                # Record load metrics
                self._record_metric("system_load", load, {
                    "total_workers": total_workers,
                    "busy_workers": busy_workers
                })
                
            except Exception as e:
                self.logger.error(f"Error in adaptive worker management: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    def _record_metric(self, metric_type: str, value: float, metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO performance_metrics (metric_type, value, metadata, timestamp)
            VALUES (?, ?, ?, ?)
        """, (metric_type, value, json.dumps(metadata or {}), datetime.now(timezone.utc)))
        self.conn.commit()
    
    async def start(self):
        """Start the ultimate orchestrator."""
        self.logger.info("Starting Ultimate Orchestrator...")
        
        # Start adaptive worker management if enabled
        if self.config["enable_adaptive"]:
            asyncio.create_task(self._adaptive_worker_management())
        
        # Start base orchestrator
        await super().start()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        base_status = await super().get_status()
        
        # Add ultimate orchestrator specific status
        ultimate_status = {
            **base_status,
            "ultimate_features": {
                "bidding_enabled": self.config["enable_bidding"],
                "adaptive_enabled": self.config["enable_adaptive"],
                "learning_enabled": self.config["enable_learning"],
                "charter_enabled": self.config["enable_charter"]
            },
            "bidding_metrics": self.bidding_metrics,
            "worker_capabilities": {
                worker_id: {
                    "total_tasks": cap.total_tasks,
                    "success_rate": cap.success_rate,
                    "specializations": cap.specializations
                }
                for worker_id, cap in self.worker_capabilities.items()
            }
        }
        
        return ultimate_status
    
    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()