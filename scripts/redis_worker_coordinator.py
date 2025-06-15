"""
Redis Worker Coordinator

Enables real-time coordination between workers using Redis Pub/Sub.
Provides instant communication, task redistribution, and emergency controls.
"""

import json
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from enum import Enum

from redis_integration.redis_client import RedisClient
from redis_integration.redis_pubsub_manager import RedisPubSubManager
from scripts.mcp_redis_integration import MCPRedisIntegration


class WorkerEvent(Enum):
    """Types of worker coordination events."""
    # Worker lifecycle
    WORKER_STARTED = "worker_started"
    WORKER_STOPPED = "worker_stopped"
    WORKER_HEARTBEAT = "worker_heartbeat"
    
    # Task events
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_REASSIGN = "task_reassign"
    
    # Coordination events
    EMERGENCY_STOP = "emergency_stop"
    PRIORITY_CHANGE = "priority_change"
    LOAD_BALANCE = "load_balance"
    
    # System events
    METRIC_UPDATE = "metric_update"
    CONFIG_UPDATE = "config_update"
    ALERT = "alert"


class RedisWorkerCoordinator:
    """Coordinates workers using Redis Pub/Sub for real-time communication."""
    
    def __init__(self, redis_client: RedisClient = None, worker_id: str = None):
        """Initialize worker coordinator.
        
        Args:
            redis_client: Redis client instance
            worker_id: ID of this worker (None for orchestrator)
        """
        self.redis_client = redis_client
        self.worker_id = worker_id or "orchestrator"
        self.logger = logging.getLogger(__name__)
        
        # Pub/Sub manager
        self.pubsub: Optional[RedisPubSubManager] = None
        
        # Event handlers
        self.event_handlers: Dict[WorkerEvent, List[Callable]] = {
            event: [] for event in WorkerEvent
        }
        
        # Channels
        self.global_channel = "cwmai:workers:global"
        self.worker_channel = f"cwmai:workers:{self.worker_id}"
        self.orchestrator_channel = "cwmai:workers:orchestrator"
        
        # Worker registry
        self.active_workers: Dict[str, Dict[str, Any]] = {}
        self.worker_heartbeats: Dict[str, float] = {}
        
        # Coordination state
        self.emergency_stop_active = False
        self.load_balancing_active = False
        
        self._initialized = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
        # MCP-Redis integration
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
    
    async def initialize(self):
        """Initialize Pub/Sub connections and start monitoring."""
        if self._initialized:
            return
        
        try:
            # Get Redis client if not provided
            if not self.redis_client:
                from redis_integration.redis_client import get_redis_client
                self.redis_client = await get_redis_client()
            
            # Initialize Pub/Sub manager
            self.pubsub = RedisPubSubManager(self.redis_client)
            await self.pubsub.start()
            
            # Subscribe to channels
            await self._subscribe_to_channels()
            
            # Start heartbeat if we're a worker
            if self.worker_id != "orchestrator":
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start monitoring task
            self._monitor_task = asyncio.create_task(self._monitor_workers())
            
            # Announce our presence
            await self._announce_worker_started()
            
            # Initialize MCP-Redis if enabled
            if self._use_mcp:
                try:
                    self.mcp_redis = MCPRedisIntegration()
                    await self.mcp_redis.initialize()
                    self.logger.info("MCP-Redis integration enabled for worker coordinator")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                    self._use_mcp = False
            
            self._initialized = True
            self.logger.info(f"Worker coordinator initialized for {self.worker_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize worker coordinator: {e}")
            raise
    
    async def _subscribe_to_channels(self):
        """Subscribe to relevant Pub/Sub channels."""
        # Global channel for all workers
        await self.pubsub.subscribe(self.global_channel, self._handle_global_message)
        
        # Worker-specific channel
        if self.worker_id != "orchestrator":
            await self.pubsub.subscribe(self.worker_channel, self._handle_worker_message)
        
        # Orchestrator listens to all worker events
        if self.worker_id == "orchestrator":
            await self.pubsub.subscribe_pattern("cwmai:workers:*", self._handle_worker_broadcast_wrapper)
    
    async def _handle_worker_broadcast_wrapper(self, channel: str, message: Dict[str, Any]):
        """Wrapper for pattern subscription handler."""
        await self._handle_worker_broadcast("cwmai:workers:*", channel, message)
    
    async def _handle_global_message(self, message: Dict[str, Any]):
        """Handle messages on the global channel."""
        try:
            event_type = WorkerEvent(message.get('event'))
            await self._dispatch_event(event_type, message)
        except Exception as e:
            self.logger.error(f"Error handling global message: {e}")
    
    async def _handle_worker_message(self, message: Dict[str, Any]):
        """Handle messages directed to this worker."""
        try:
            event_type = WorkerEvent(message.get('event'))
            await self._dispatch_event(event_type, message)
        except Exception as e:
            self.logger.error(f"Error handling worker message: {e}")
    
    async def _handle_worker_broadcast(self, pattern: str, channel: str, message: Dict[str, Any]):
        """Handle broadcasts from all workers (orchestrator only)."""
        try:
            # Extract worker ID from channel
            parts = channel.split(':')
            if len(parts) >= 3:
                source_worker = parts[2]
                
                # Update worker registry
                event_type = WorkerEvent(message.get('event'))
                
                if event_type == WorkerEvent.WORKER_STARTED:
                    self.active_workers[source_worker] = message.get('data', {})
                    self.worker_heartbeats[source_worker] = datetime.now(timezone.utc).timestamp()
                    
                elif event_type == WorkerEvent.WORKER_STOPPED:
                    self.active_workers.pop(source_worker, None)
                    self.worker_heartbeats.pop(source_worker, None)
                    
                elif event_type == WorkerEvent.WORKER_HEARTBEAT:
                    self.worker_heartbeats[source_worker] = datetime.now(timezone.utc).timestamp()
                    if source_worker in self.active_workers:
                        self.active_workers[source_worker].update(message.get('data', {}))
                
                # Dispatch to handlers
                await self._dispatch_event(event_type, message)
                
        except Exception as e:
            self.logger.error(f"Error handling worker broadcast: {e}")
    
    async def _dispatch_event(self, event_type: WorkerEvent, message: Dict[str, Any]):
        """Dispatch event to registered handlers."""
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    def register_handler(self, event_type: WorkerEvent, handler: Callable):
        """Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Callback function
        """
        self.event_handlers[event_type].append(handler)
    
    async def broadcast_event(self, event_type: WorkerEvent, data: Any = None, 
                            target_worker: Optional[str] = None):
        """Broadcast an event to workers.
        
        Args:
            event_type: Type of event
            data: Event data
            target_worker: Optional specific worker to target
        """
        # Don't try to initialize during initialization to prevent recursion
        if not self._initialized and event_type != WorkerEvent.WORKER_STARTED:
            await self.initialize()
        
        message = {
            'event': event_type.value,
            'source': self.worker_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': data
        }
        
        # Choose channel
        if target_worker:
            channel = f"cwmai:workers:{target_worker}"
        else:
            channel = self.global_channel
        
        # Publish message
        await self.pubsub.publish(channel, message)
    
    async def _announce_worker_started(self):
        """Announce that this worker has started."""
        if self.worker_id != "orchestrator":
            worker_info = {
                'worker_id': self.worker_id,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'capabilities': self._get_worker_capabilities()
            }
            
            await self.broadcast_event(WorkerEvent.WORKER_STARTED, worker_info)
    
    async def _announce_worker_stopped(self):
        """Announce that this worker is stopping."""
        if self.worker_id != "orchestrator":
            await self.broadcast_event(WorkerEvent.WORKER_STOPPED, {
                'worker_id': self.worker_id,
                'stopped_at': datetime.now(timezone.utc).isoformat()
            })
    
    def _get_worker_capabilities(self) -> Dict[str, Any]:
        """Get this worker's capabilities."""
        # Override in subclass to provide specific capabilities
        return {
            'max_concurrent_tasks': 5,
            'supported_task_types': ['*'],
            'specializations': []
        }
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while True:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
                # Send heartbeat with current status
                status = {
                    'worker_id': self.worker_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'active_tasks': 0,  # Override in implementation
                    'cpu_usage': 0.0,   # Override in implementation
                    'memory_usage': 0.0  # Override in implementation
                }
                
                await self.broadcast_event(WorkerEvent.WORKER_HEARTBEAT, status)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    async def _monitor_workers(self):
        """Monitor worker health and detect failures."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.worker_id == "orchestrator":
                    now = datetime.now(timezone.utc).timestamp()
                    dead_workers = []
                    
                    # Check for dead workers (no heartbeat for 2 minutes)
                    for worker_id, last_heartbeat in self.worker_heartbeats.items():
                        if now - last_heartbeat > 120:
                            dead_workers.append(worker_id)
                    
                    # Handle dead workers
                    for worker_id in dead_workers:
                        self.logger.warning(f"Worker {worker_id} appears to be dead")
                        self.active_workers.pop(worker_id, None)
                        self.worker_heartbeats.pop(worker_id, None)
                        
                        # Notify about dead worker
                        await self.broadcast_event(WorkerEvent.ALERT, {
                            'type': 'worker_failure',
                            'worker_id': worker_id,
                            'message': f"Worker {worker_id} has not sent heartbeat for 2 minutes"
                        })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
    
    async def request_task_reassignment(self, task_id: str, reason: str):
        """Request reassignment of a task to another worker."""
        await self.broadcast_event(WorkerEvent.TASK_REASSIGN, {
            'task_id': task_id,
            'current_worker': self.worker_id,
            'reason': reason,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop for all workers."""
        self.emergency_stop_active = True
        
        await self.broadcast_event(WorkerEvent.EMERGENCY_STOP, {
            'reason': reason,
            'initiated_by': self.worker_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def update_task_priority(self, task_id: str, new_priority: str):
        """Update priority of a task across all workers."""
        await self.broadcast_event(WorkerEvent.PRIORITY_CHANGE, {
            'task_id': task_id,
            'new_priority': new_priority,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def request_load_balancing(self):
        """Request load balancing across workers."""
        if self.worker_id == "orchestrator":
            self.load_balancing_active = True
            
            # Analyze current load
            worker_loads = {}
            for worker_id, info in self.active_workers.items():
                worker_loads[worker_id] = info.get('active_tasks', 0)
            
            await self.broadcast_event(WorkerEvent.LOAD_BALANCE, {
                'worker_loads': worker_loads,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    
    def get_active_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active workers."""
        return self.active_workers.copy()
    
    def get_worker_health(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get health status of a specific worker."""
        if worker_id not in self.active_workers:
            return None
        
        now = datetime.now(timezone.utc).timestamp()
        last_heartbeat = self.worker_heartbeats.get(worker_id, 0)
        
        return {
            'worker_id': worker_id,
            'healthy': (now - last_heartbeat) < 60,  # Healthy if heartbeat within 1 minute
            'last_heartbeat': last_heartbeat,
            'info': self.active_workers[worker_id]
        }
    
    # MCP-Redis Enhanced Coordination Methods
    async def find_optimal_worker(self, task_description: str, task_type: str) -> Optional[str]:
        """Use MCP-Redis to find the optimal worker for a task."""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to simple load balancing
            worker_loads = await self._calculate_worker_loads()
            if worker_loads:
                return min(worker_loads, key=worker_loads.get)
            return None
        
        try:
            result = await self.mcp_redis.execute(f"""
                Find the optimal worker for task:
                Description: {task_description}
                Type: {task_type}
                
                Consider:
                - Current worker loads and capacity
                - Worker specializations and past performance
                - Task type success rates per worker
                - Current worker health and response times
                - Geographic or resource constraints
                
                Active workers: {json.dumps(list(self.active_workers.keys()))}
                
                Return the worker_id of the best match, or null if none suitable.
            """)
            
            if isinstance(result, str) and result in self.active_workers:
                return result
            elif isinstance(result, dict) and 'worker_id' in result:
                return result['worker_id']
            else:
                # Fallback
                worker_loads = await self._calculate_worker_loads()
                if worker_loads:
                    return min(worker_loads, key=worker_loads.get)
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding optimal worker: {e}")
            return None
    
    async def predict_worker_failures(self) -> List[Dict[str, Any]]:
        """Use MCP-Redis to predict potential worker failures."""
        if not self._use_mcp or not self.mcp_redis:
            return []
        
        try:
            predictions = await self.mcp_redis.execute(f"""
                Predict potential worker failures based on:
                - Heartbeat patterns and missed heartbeats
                - Task failure rates trending upward
                - Memory/CPU usage patterns if available
                - Response time degradation
                - Error log patterns
                
                Active workers: {json.dumps(self.active_workers)}
                Recent heartbeats: {json.dumps(self.worker_heartbeats)}
                
                For each at-risk worker provide:
                - worker_id
                - failure_probability (0-1)
                - predicted_time_to_failure (minutes)
                - failure_indicators (list)
                - recommended_actions (list)
            """)
            
            return predictions if isinstance(predictions, list) else []
            
        except Exception as e:
            self.logger.error(f"Error predicting worker failures: {e}")
            return []
    
    async def optimize_load_distribution(self) -> Dict[str, Any]:
        """Use MCP-Redis to optimize task distribution across workers."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            optimization = await self.mcp_redis.execute(f"""
                Optimize load distribution across workers:
                
                Current state:
                - Active workers: {json.dumps(self.active_workers)}
                - Worker loads: {json.dumps(await self._calculate_worker_loads())}
                
                Provide:
                - Optimal task distribution strategy
                - Which workers should handle which task types
                - Recommended worker scaling (add/remove workers)
                - Load balancing thresholds
                - Task reassignment recommendations
                - Estimated efficiency improvement
            """)
            
            return optimization if isinstance(optimization, dict) else {"optimization": optimization}
            
        except Exception as e:
            self.logger.error(f"Error optimizing load distribution: {e}")
            return {"error": str(e)}
    
    async def analyze_coordination_patterns(self) -> Dict[str, Any]:
        """Use MCP-Redis to analyze worker coordination patterns."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            analysis = await self.mcp_redis.execute("""
                Analyze worker coordination patterns:
                - Communication efficiency between workers
                - Task handoff success rates
                - Coordination bottlenecks
                - Optimal team compositions for different task types
                - Worker collaboration patterns
                - Emergency response effectiveness
                - Load balancing effectiveness over time
                
                Provide insights and recommendations for improvement.
            """)
            
            return analysis if isinstance(analysis, dict) else {"analysis": analysis}
            
        except Exception as e:
            self.logger.error(f"Error analyzing coordination: {e}")
            return {"error": str(e)}
    
    async def suggest_worker_configuration(self, expected_workload: Dict[str, int]) -> Dict[str, Any]:
        """Use MCP-Redis to suggest optimal worker configuration."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            suggestions = await self.mcp_redis.execute(f"""
                Suggest optimal worker configuration for expected workload:
                {json.dumps(expected_workload)}
                
                Consider:
                - Task type requirements and complexity
                - Historical performance data
                - Resource constraints
                - Cost optimization
                - Redundancy needs
                
                Provide:
                - Recommended number of workers per type
                - Optimal worker specifications
                - Scaling triggers and thresholds
                - Estimated resource usage
                - Cost-benefit analysis
            """)
            
            return suggestions if isinstance(suggestions, dict) else {"suggestions": suggestions}
            
        except Exception as e:
            self.logger.error(f"Error suggesting configuration: {e}")
            return {"error": str(e)}
    
    async def get_coordination_insights(self) -> Dict[str, Any]:
        """Get AI-powered insights about worker coordination."""
        if not self._use_mcp or not self.mcp_redis:
            return {
                "active_workers": len(self.active_workers),
                "message": "MCP-Redis not available for insights"
            }
        
        try:
            insights = await self.mcp_redis.execute(f"""
                Provide coordination insights:
                - Overall coordination efficiency score (0-100)
                - Top coordination bottlenecks
                - Worker communication patterns
                - Task distribution fairness
                - Resource utilization efficiency
                - Collaboration effectiveness
                - System resilience assessment
                - Specific improvement recommendations
                
                Current state:
                - Active workers: {len(self.active_workers)}
                - Emergency stops: {self.emergency_stop_active}
                - Load balancing: {self.load_balancing_active}
            """)
            
            return insights if isinstance(insights, dict) else {"insights": insights}
            
        except Exception as e:
            self.logger.error(f"Error getting coordination insights: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources."""
        # Announce shutdown
        await self._announce_worker_stopped()
        
        # Cancel tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop Pub/Sub
        if self.pubsub:
            await self.pubsub.stop()
        
        self._initialized = False
        self.logger.info(f"Worker coordinator cleaned up for {self.worker_id}")