"""
Worker Capability Store - Persistent storage and sharing of worker capabilities
"""
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
import asyncio

try:
    from redis_integration.redis_client import RedisClient
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class WorkerCapabilityStore:
    """Persistent store for worker capabilities and shared knowledge"""
    
    def __init__(self, redis_client: Optional[Any] = None, storage_path: str = "worker_capabilities"):
        self.logger = logging.getLogger("WorkerCapabilityStore")
        self.redis_client = redis_client
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Cache for quick access
        self.capability_cache: Dict[str, Dict[str, Any]] = {}
        self.knowledge_base: Dict[str, Any] = {}
        
        # Redis keys
        self.capability_key_prefix = "cwmai:worker:capability:"
        self.knowledge_key_prefix = "cwmai:knowledge:"
        self.performance_key = "cwmai:system:performance"
        
        self.logger.info("Worker Capability Store initialized")
    
    async def save_worker_capabilities(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        """Save worker capabilities to persistent storage"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Add metadata
        capabilities["last_updated"] = timestamp
        capabilities["worker_id"] = worker_id
        
        # Save to cache
        self.capability_cache[worker_id] = capabilities
        
        # Save to Redis if available
        if REDIS_AVAILABLE and self.redis_client:
            try:
                key = f"{self.capability_key_prefix}{worker_id}"
                await self.redis_client.set(key, json.dumps(capabilities), ex=86400)  # 24h expiry
                self.logger.debug(f"Saved capabilities for {worker_id} to Redis")
            except Exception as e:
                self.logger.error(f"Failed to save to Redis: {e}")
        
        # Save to file as backup
        file_path = self.storage_path / f"{worker_id}_capabilities.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(capabilities, f, indent=2)
            self.logger.debug(f"Saved capabilities for {worker_id} to file")
        except Exception as e:
            self.logger.error(f"Failed to save to file: {e}")
    
    async def load_worker_capabilities(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Load worker capabilities from storage"""
        # Check cache first
        if worker_id in self.capability_cache:
            return self.capability_cache[worker_id]
        
        # Try Redis
        if REDIS_AVAILABLE and self.redis_client:
            try:
                key = f"{self.capability_key_prefix}{worker_id}"
                data = await self.redis_client.get(key)
                if data:
                    capabilities = json.loads(data)
                    self.capability_cache[worker_id] = capabilities
                    return capabilities
            except Exception as e:
                self.logger.error(f"Failed to load from Redis: {e}")
        
        # Try file
        file_path = self.storage_path / f"{worker_id}_capabilities.json"
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    capabilities = json.load(f)
                self.capability_cache[worker_id] = capabilities
                return capabilities
            except Exception as e:
                self.logger.error(f"Failed to load from file: {e}")
        
        return None
    
    async def share_knowledge(self, knowledge_key: str, knowledge_data: Dict[str, Any]) -> None:
        """Share knowledge across all workers"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        knowledge_entry = {
            "data": knowledge_data,
            "timestamp": timestamp,
            "key": knowledge_key
        }
        
        # Update knowledge base
        self.knowledge_base[knowledge_key] = knowledge_entry
        
        # Save to Redis for real-time sharing
        if REDIS_AVAILABLE and self.redis_client:
            try:
                redis_key = f"{self.knowledge_key_prefix}{knowledge_key}"
                await self.redis_client.set(redis_key, json.dumps(knowledge_entry), ex=3600)  # 1h expiry
                
                # Publish to knowledge channel for real-time updates
                await self.redis_client.publish(
                    "cwmai:knowledge:updates",
                    json.dumps({"key": knowledge_key, "timestamp": timestamp})
                )
            except Exception as e:
                self.logger.error(f"Failed to share knowledge via Redis: {e}")
        
        self.logger.info(f"Shared knowledge: {knowledge_key}")
    
    async def get_shared_knowledge(self, knowledge_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve shared knowledge"""
        # Check cache
        if knowledge_key in self.knowledge_base:
            return self.knowledge_base[knowledge_key]
        
        # Try Redis
        if REDIS_AVAILABLE and self.redis_client:
            try:
                redis_key = f"{self.knowledge_key_prefix}{knowledge_key}"
                data = await self.redis_client.get(redis_key)
                if data:
                    knowledge = json.loads(data)
                    self.knowledge_base[knowledge_key] = knowledge
                    return knowledge
            except Exception as e:
                self.logger.error(f"Failed to retrieve knowledge from Redis: {e}")
        
        return None
    
    async def get_all_worker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all workers"""
        metrics = {}
        
        # From cache
        for worker_id, capabilities in self.capability_cache.items():
            metrics[worker_id] = self._extract_metrics(capabilities)
        
        # From Redis
        if REDIS_AVAILABLE and self.redis_client:
            try:
                pattern = f"{self.capability_key_prefix}*"
                keys = await self.redis_client.keys(pattern)
                
                for key in keys:
                    worker_id = key.decode().replace(self.capability_key_prefix, "")
                    if worker_id not in metrics:
                        data = await self.redis_client.get(key)
                        if data:
                            capabilities = json.loads(data)
                            metrics[worker_id] = self._extract_metrics(capabilities)
            except Exception as e:
                self.logger.error(f"Failed to get metrics from Redis: {e}")
        
        return metrics
    
    def _extract_metrics(self, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from capabilities"""
        return {
            "overall_success_rate": capabilities.get("overall_success_rate", 0),
            "total_completed": capabilities.get("total_completed", 0),
            "current_load": capabilities.get("current_load", 0),
            "specialization": capabilities.get("specialization", "unknown"),
            "last_updated": capabilities.get("last_updated", "unknown")
        }
    
    async def save_system_performance(self, performance_data: Dict[str, Any]) -> None:
        """Save overall system performance metrics"""
        timestamp = datetime.now(timezone.utc).isoformat()
        performance_data["timestamp"] = timestamp
        
        # Save to Redis
        if REDIS_AVAILABLE and self.redis_client:
            try:
                await self.redis_client.zadd(
                    self.performance_key,
                    {json.dumps(performance_data): datetime.now(timezone.utc).timestamp()}
                )
                # Keep only last 1000 entries
                await self.redis_client.zremrangebyrank(self.performance_key, 0, -1001)
            except Exception as e:
                self.logger.error(f"Failed to save performance to Redis: {e}")
        
        # Save to file
        file_path = self.storage_path / "system_performance.jsonl"
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(performance_data) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save performance to file: {e}")
    
    async def get_performance_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system performance history"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        history = []
        
        # From Redis
        if REDIS_AVAILABLE and self.redis_client:
            try:
                start_score = cutoff_time.timestamp()
                end_score = datetime.now(timezone.utc).timestamp()
                
                entries = await self.redis_client.zrangebyscore(
                    self.performance_key,
                    start_score,
                    end_score
                )
                
                for entry in entries:
                    try:
                        data = json.loads(entry)
                        history.append(data)
                    except:
                        pass
            except Exception as e:
                self.logger.error(f"Failed to get performance from Redis: {e}")
        
        # Sort by timestamp
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return history
    
    async def find_experts(self, task_type: str, min_success_rate: float = 0.7) -> List[str]:
        """Find workers who are experts at a specific task type"""
        experts = []
        
        all_metrics = await self.get_all_worker_metrics()
        
        for worker_id, capabilities in self.capability_cache.items():
            # Check if worker has experience with this task type
            worker_capabilities = capabilities.get("capabilities", {})
            
            for cap_key, cap_data in worker_capabilities.items():
                if task_type in cap_key and isinstance(cap_data, dict):
                    success_rate = cap_data.get("success_rate", 0)
                    total_tasks = cap_data.get("total_tasks", 0)
                    
                    # Expert criteria: high success rate and experience
                    if success_rate >= min_success_rate and total_tasks >= 5:
                        experts.append({
                            "worker_id": worker_id,
                            "success_rate": success_rate,
                            "total_tasks": total_tasks,
                            "avg_duration": cap_data.get("avg_duration", 0)
                        })
        
        # Sort by success rate and experience
        experts.sort(key=lambda x: (x["success_rate"], x["total_tasks"]), reverse=True)
        
        return [e["worker_id"] for e in experts]
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Analyze worker learning patterns and provide insights"""
        insights = {
            "fastest_learners": [],
            "most_versatile": [],
            "specialists": [],
            "struggling_areas": [],
            "collaboration_success": []
        }
        
        all_metrics = await self.get_all_worker_metrics()
        
        for worker_id, capabilities in self.capability_cache.items():
            worker_caps = capabilities.get("capabilities", {})
            
            # Versatility score (number of different task types)
            versatility = len(worker_caps)
            
            # Average success rate across all capabilities  
            if worker_caps:
                avg_success = sum(
                    cap.get("success_rate", 0) 
                    for cap in worker_caps.values() 
                    if isinstance(cap, dict)
                ) / len(worker_caps)
            else:
                avg_success = 0
            
            # Learning rate (improvement over time)
            # This would need historical data to calculate properly
            
            # Identify specialists (high success in specific areas)
            for cap_key, cap_data in worker_caps.items():
                if isinstance(cap_data, dict) and cap_data.get("success_rate", 0) > 0.9:
                    if cap_data.get("total_tasks", 0) > 10:
                        insights["specialists"].append({
                            "worker_id": worker_id,
                            "specialty": cap_key,
                            "success_rate": cap_data["success_rate"]
                        })
            
            # Most versatile workers
            if versatility > 5:
                insights["most_versatile"].append({
                    "worker_id": worker_id,
                    "task_types": versatility,
                    "avg_success": avg_success
                })
        
        return insights