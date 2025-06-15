"""
Redis-based Task Persistence System

Distributed task persistence using Redis for scalable task deduplication,
history tracking, and multi-worker coordination.
"""

import asyncio
import json
import time
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.redis_integration.redis_client import RedisClient, get_redis_client
from scripts.mcp_redis_integration import MCPRedisIntegration


def safe_decode(value):
    """Safely decode bytes to string, handling both bytes and string inputs."""
    if isinstance(value, bytes):
        return value.decode()
    return value


@dataclass
class RedisCompletedTask:
    """Redis-compatible completed task record."""
    work_item_id: str
    title: str
    task_type: str
    repository: Optional[str]
    description_hash: str
    completed_at: str  # ISO format string for Redis
    issue_number: Optional[int] = None
    value_created: float = 0.0
    execution_result: Optional[str] = None  # JSON string

    def to_redis_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format."""
        data = {
            'work_item_id': self.work_item_id,
            'title': self.title,
            'task_type': self.task_type,
            'description_hash': self.description_hash,
            'completed_at': self.completed_at,
            'value_created': str(self.value_created)
        }
        
        if self.repository:
            data['repository'] = self.repository
        if self.issue_number is not None:
            data['issue_number'] = str(self.issue_number)
        if self.execution_result:
            data['execution_result'] = self.execution_result
            
        return data

    @classmethod
    def from_redis_hash(cls, data: Dict[bytes, bytes]) -> 'RedisCompletedTask':
        """Create from Redis hash data."""
        # Decode bytes to strings (handle both bytes and strings)
        decoded_data = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else k
            value = v.decode() if isinstance(v, bytes) else v
            decoded_data[key] = value
        
        return cls(
            work_item_id=decoded_data['work_item_id'],
            title=decoded_data['title'],
            task_type=decoded_data['task_type'],
            repository=decoded_data.get('repository'),
            description_hash=decoded_data['description_hash'],
            completed_at=decoded_data['completed_at'],
            issue_number=int(decoded_data['issue_number']) if 'issue_number' in decoded_data else None,
            value_created=float(decoded_data.get('value_created', 0.0)),
            execution_result=decoded_data.get('execution_result')
        )


class RedisTaskPersistence:
    """Redis-based task persistence for distributed systems."""
    
    def __init__(self, redis_client: RedisClient = None, namespace: str = "task_persistence"):
        """Initialize Redis task persistence.
        
        Args:
            redis_client: Redis client instance
            namespace: Redis key namespace
        """
        self.redis = redis_client
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        
        # Redis key patterns
        self.completed_tasks_key = f"{namespace}:completed"
        self.title_hashes_key = f"{namespace}:title_hashes"
        self.description_hashes_key = f"{namespace}:desc_hashes"
        self.skip_stats_key = f"{namespace}:skip_stats"
        self.skip_cooldowns_key = f"{namespace}:skip_cooldowns"
        self.task_history_key = f"{namespace}:history"
        self.stats_key = f"{namespace}:stats"
        
        # Task-specific cooldown periods (in hours)
        self.task_cooldowns = {
            'TESTING': 12,
            'FEATURE': 72,
            'BUG_FIX': 48,
            'DOCUMENTATION': 48,
            'RESEARCH': 168,
            'SYSTEM_IMPROVEMENT': 168,
            'MAINTENANCE': 72,
            'NEW_PROJECT': 720,
            'INTEGRATION': 168,
            'REPOSITORY_HEALTH': 24
        }
        
        # Configuration
        self.default_cooldown_hours = 24
        self.base_skip_cooldown = 300  # 5 minutes
        self.max_stored_tasks = 10000
        self.history_ttl = 30 * 24 * 3600  # 30 days
        
        self._initialized = False
        
        # MCP-Redis integration (optional)
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
    
    async def _ensure_redis(self):
        """Ensure Redis client is available."""
        if not self.redis:
            self.redis = await get_redis_client()
        if not self._initialized:
            await self._initialize_indexes()
            
            # Initialize MCP-Redis if enabled
            if self._use_mcp:
                try:
                    self.mcp_redis = MCPRedisIntegration()
                    await self.mcp_redis.initialize()
                    self.logger.info("MCP-Redis integration enabled for task persistence")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                    self._use_mcp = False
            
            self._initialized = True
    
    async def _initialize_indexes(self):
        """Initialize Redis indexes and data structures."""
        # Ensure sorted sets exist - no need to initialize empty sorted sets
        # Redis will create them automatically when we add the first element
        self.logger.info("Redis task persistence initialized")
    
    async def record_completed_task(self, work_item: WorkItem, execution_result: Dict[str, Any]) -> bool:
        """Record a completed task in Redis.
        
        Args:
            work_item: The completed work item
            execution_result: Result of task execution
            
        Returns:
            True if recorded successfully
        """
        await self._ensure_redis()
        
        try:
            # Create task record
            completed_task = RedisCompletedTask(
                work_item_id=work_item.id,
                title=work_item.title,
                task_type=work_item.task_type,
                repository=work_item.repository,
                description_hash=self._hash_description(work_item.description),
                completed_at=datetime.now(timezone.utc).isoformat(),
                issue_number=execution_result.get('issue_number'),
                value_created=execution_result.get('value_created', 0.0),
                execution_result=json.dumps(execution_result) if execution_result else None
            )
            
            # Use pipeline for atomic operations
            async with self.redis.pipeline() as pipe:
                # Store completed task
                task_key = f"{self.completed_tasks_key}:{work_item.id}"
                pipe.hset(task_key, mapping=completed_task.to_redis_hash())
                pipe.expire(task_key, self.history_ttl)
                
                # Add to indexes
                pipe.sadd(self.title_hashes_key, self._hash_title(work_item.title))
                pipe.sadd(self.description_hashes_key, completed_task.description_hash)
                
                # Add to history sorted set (score = timestamp)
                timestamp = time.time()
                pipe.zadd(self.task_history_key, {work_item.id: timestamp})
                
                # Update stats
                pipe.hincrby(self.stats_key, 'total_completed', 1)
                pipe.hincrby(self.stats_key, f'type_{work_item.task_type}', 1)
                if work_item.repository:
                    pipe.hincrby(self.stats_key, f'repo_{work_item.repository}', 1)
                pipe.hincrbyfloat(self.stats_key, 'total_value', execution_result.get('value_created', 0.0))
                
                # Maintain history size limit
                pipe.zremrangebyrank(self.task_history_key, 0, -self.max_stored_tasks - 1)
                
                await pipe.execute()
            
            self.logger.info(f"✅ Recorded completed task in Redis: {work_item.title}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error recording completed task: {e}")
            return False
    
    async def record_skipped_task(self, task_title: str, reason: str = "duplicate") -> None:
        """Record a skipped task to track problematic patterns.
        
        Args:
            task_title: Title of the skipped task
            reason: Reason for skipping
        """
        await self._ensure_redis()
        
        try:
            skip_key = f"{self.skip_stats_key}:{self._hash_title(task_title)}"
            cooldown_key = f"{self.skip_cooldowns_key}:{self._hash_title(task_title)}"
            
            # Get current skip count first (outside pipeline)
            skip_count_data = await self.redis.hget(skip_key, 'count')
            if skip_count_data:
                # Handle both bytes and string
                skip_count_str = skip_count_data.decode() if isinstance(skip_count_data, bytes) else skip_count_data
                skip_count = int(skip_count_str)
            else:
                skip_count = 0
            skip_count += 1  # Increment for this skip
            
            async with self.redis.pipeline() as pipe:
                # Increment skip count
                pipe.hincrby(skip_key, 'count', 1)
                pipe.hset(skip_key, 'last_skip', datetime.now(timezone.utc).isoformat())
                pipe.rpush(f"{skip_key}:reasons", reason)
                pipe.ltrim(f"{skip_key}:reasons", -100, -1)  # Keep last 100 reasons
                
                # Exponential backoff for frequently skipped tasks
                if skip_count > 10:
                    cooldown = min(3600, self.base_skip_cooldown * (2 ** ((skip_count - 10) // 5)))
                else:
                    cooldown = self.base_skip_cooldown
                
                # Set cooldown expiration
                pipe.setex(cooldown_key, cooldown, '1')
                
                # Set TTL on skip stats
                pipe.expire(skip_key, 7 * 24 * 3600)  # 7 days
                
                await pipe.execute()
            
            if skip_count > 10:
                self.logger.warning(
                    f"⚠️ Task '{task_title}' skipped {skip_count} times. "
                    f"Cooldown: {cooldown // 60} minutes"
                )
                
        except Exception as e:
            self.logger.error(f"Error recording skipped task: {e}")
    
    async def is_duplicate_task(self, work_item: WorkItem) -> bool:
        """Check if a work item is a duplicate using Redis.
        
        Args:
            work_item: Work item to check
            
        Returns:
            True if this is a duplicate task
        """
        await self._ensure_redis()
        
        try:
            # Check skip cooldown first
            cooldown_key = f"{self.skip_cooldowns_key}:{self._hash_title(work_item.title)}"
            if await self.redis.exists(cooldown_key):
                self.logger.debug(f"🔄 Task in skip cooldown: {work_item.title}")
                return True
            
            # Check title duplicates
            title_hash = self._hash_title(work_item.title)
            if await self.redis.sismember(self.title_hashes_key, title_hash):
                # Check if any matching tasks are in cooldown
                if await self._check_cooldown_by_title(work_item):
                    return True
            
            # Check description duplicates
            desc_hash = self._hash_description(work_item.description)
            if await self.redis.sismember(self.description_hashes_key, desc_hash):
                # Check if any matching tasks are in cooldown
                if await self._check_cooldown_by_description(work_item, desc_hash):
                    return True
            
            # Check repository-specific duplicates
            if work_item.repository:
                if await self._check_repository_duplicates(work_item):
                    return True
            
            return False
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error checking duplicate task: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.logger.error(f"Work item details: type={work_item.task_type}, title={work_item.title[:50] if work_item.title else 'None'}")
            return False
    
    async def _check_cooldown_by_title(self, work_item: WorkItem) -> bool:
        """Check if tasks with matching title are in cooldown."""
        # Get recent tasks from history
        cutoff_time = time.time() - (self.task_cooldowns.get(work_item.task_type, self.default_cooldown_hours) * 3600)
        
        # Get task IDs from recent history
        task_ids = await self.redis.zrangebyscore(
            self.task_history_key,
            cutoff_time,
            '+inf'
        )
        
        # Check each task
        for task_id in task_ids[-100:]:  # Check last 100 tasks max
            task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
            task_key = f"{self.completed_tasks_key}:{task_id_str}"
            task_title = await self.redis.hget(task_key, 'title')
            
            if task_title:
                title_str = task_title.decode() if isinstance(task_title, bytes) else task_title
                if title_str == work_item.title:
                    self.logger.debug(f"🔄 Duplicate title found in cooldown: {work_item.title}")
                    return True
        
        return False
    
    async def _check_cooldown_by_description(self, work_item: WorkItem, desc_hash: str) -> bool:
        """Check if tasks with matching description are in cooldown."""
        cutoff_time = time.time() - (self.task_cooldowns.get(work_item.task_type, self.default_cooldown_hours) * 3600)
        
        task_ids = await self.redis.zrangebyscore(
            self.task_history_key,
            cutoff_time,
            '+inf'
        )
        
        for task_id in task_ids[-100:]:
            task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
            task_key = f"{self.completed_tasks_key}:{task_id_str}"
            task_desc_hash = await self.redis.hget(task_key, 'description_hash')
            
            if task_desc_hash:
                desc_hash_str = task_desc_hash.decode() if isinstance(task_desc_hash, bytes) else task_desc_hash
                if desc_hash_str == desc_hash:
                    self.logger.debug(f"🔄 Duplicate description found in cooldown: {work_item.title}")
                    return True
        
        return False
    
    async def _check_repository_duplicates(self, work_item: WorkItem) -> bool:
        """Check for repository-specific duplicates."""
        cutoff_time = time.time() - (self.task_cooldowns.get(work_item.task_type, self.default_cooldown_hours) * 3600)
        
        task_ids = await self.redis.zrangebyscore(
            self.task_history_key,
            cutoff_time,
            '+inf'
        )
        
        for task_id in task_ids[-50:]:  # Check last 50 tasks
            task_id_str = safe_decode(task_id)
            task_key = f"{self.completed_tasks_key}:{task_id_str}"
            task_data = await self.redis.hgetall(task_key)
            
            if not task_data:
                continue
                
            # Decode and check
            task_repo = safe_decode(task_data.get(b'repository', b''))
            task_type = safe_decode(task_data.get(b'task_type', b''))
            task_title = safe_decode(task_data.get(b'title', b''))
            
            if (task_repo == work_item.repository and 
                task_type == work_item.task_type and
                self._calculate_title_similarity(task_title, work_item.title) > 0.7):
                self.logger.debug(f"🔄 Similar repository task found in cooldown: {work_item.title}")
                return True
        
        return False
    
    async def get_task_history(self, repository: Optional[str] = None, 
                              task_type: Optional[str] = None,
                              hours_back: int = 168) -> List[Dict[str, Any]]:
        """Get history of completed tasks from Redis.
        
        Args:
            repository: Filter by repository
            task_type: Filter by task type
            hours_back: Hours of history to retrieve
            
        Returns:
            List of completed tasks
        """
        await self._ensure_redis()
        
        try:
            cutoff_time = time.time() - (hours_back * 3600)
            
            # Get task IDs from history
            task_ids = await self.redis.zrangebyscore(
                self.task_history_key,
                cutoff_time,
                '+inf',
                withscores=True
            )
            
            tasks = []
            for task_id, score in task_ids:
                task_id_str = safe_decode(task_id)
                task_key = f"{self.completed_tasks_key}:{task_id_str}"
                task_data = await self.redis.hgetall(task_key)
                
                if not task_data:
                    continue
                
                # Decode and filter
                task = {safe_decode(k): safe_decode(v) for k, v in task_data.items()}
                
                if repository and task.get('repository') != repository:
                    continue
                if task_type and task.get('task_type') != task_type:
                    continue
                
                # Parse numeric fields
                if 'value_created' in task:
                    task['value_created'] = float(task['value_created'])
                if 'issue_number' in task:
                    task['issue_number'] = int(task['issue_number'])
                if 'execution_result' in task:
                    task['execution_result'] = json.loads(task['execution_result'])
                
                tasks.append(task)
            
            # Sort by completion time (most recent first)
            tasks.sort(key=lambda t: t.get('completed_at', ''), reverse=True)
            
            return tasks
            
        except Exception as e:
            self.logger.error(f"Error getting task history: {e}")
            return []
    
    async def get_completion_stats(self) -> Dict[str, Any]:
        """Get task completion statistics from Redis.
        
        Returns:
            Statistics dictionary
        """
        await self._ensure_redis()
        
        try:
            stats_data = await self.redis.hgetall(self.stats_key)
            if not stats_data:
                return {
                    'total_tasks': 0,
                    'total_value_created': 0.0,
                    'task_types': {},
                    'repositories': {},
                    'avg_value_per_task': 0.0
                }
            
            # Decode stats
            stats = {}
            task_types = {}
            repositories = {}
            
            for key, value in stats_data.items():
                key_str = safe_decode(key)
                value_str = safe_decode(value)
                
                if key_str == 'total_completed':
                    stats['total_tasks'] = int(value_str)
                elif key_str == 'total_value':
                    stats['total_value_created'] = float(value_str)
                elif key_str.startswith('type_'):
                    task_type = key_str[5:]
                    task_types[task_type] = int(value_str)
                elif key_str.startswith('repo_'):
                    repo = key_str[5:]
                    repositories[repo] = int(value_str)
            
            total_tasks = stats.get('total_tasks', 0)
            total_value = stats.get('total_value_created', 0.0)
            
            return {
                'total_tasks': total_tasks,
                'total_value_created': round(total_value, 2),
                'task_types': task_types,
                'repositories': repositories,
                'avg_value_per_task': round(total_value / max(total_tasks, 1), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting completion stats: {e}")
            return {}
    
    async def cleanup_old_tasks(self, max_age_days: int = 30):
        """Clean up old tasks from Redis.
        
        Args:
            max_age_days: Maximum age of tasks to keep
        """
        await self._ensure_redis()
        
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            # Get old task IDs
            old_task_ids = await self.redis.zrangebyscore(
                self.task_history_key,
                0,
                cutoff_time
            )
            
            if old_task_ids:
                async with self.redis.pipeline() as pipe:
                    # Remove from history
                    pipe.zremrangebyscore(self.task_history_key, 0, cutoff_time)
                    
                    # Delete task data
                    for task_id in old_task_ids:
                        task_id_str = safe_decode(task_id)
                        task_key = f"{self.completed_tasks_key}:{task_id_str}"
                        pipe.delete(task_key)
                    
                    await pipe.execute()
                
                self.logger.info(f"🧹 Cleaned up {len(old_task_ids)} old tasks from Redis")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old tasks: {e}")
    
    def _hash_title(self, title: str) -> str:
        """Create hash of task title."""
        import hashlib
        if not title:
            return ""
        title_str = str(title)  # Ensure it's a string
        normalized = title_str.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _hash_description(self, description: str) -> str:
        """Create hash of task description."""
        import hashlib
        if not description:
            return ""
        desc_str = str(description)  # Ensure it's a string
        normalized = desc_str.lower().strip()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'for', 'to', 'of', 'in', 'on', 'at'}
        words = [word for word in normalized.split() if word not in common_words]
        normalized = ' '.join(sorted(words))
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    # MCP-Redis Enhanced Methods
    async def find_semantic_duplicates(self, work_item: WorkItem, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find semantically similar tasks using MCP-Redis natural language search."""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to basic duplicate check
            is_dup = await self.is_duplicate_task(work_item)
            return [{"is_duplicate": is_dup}] if is_dup else []
        
        try:
            # Use natural language to find semantic duplicates
            result = await self.mcp_redis.execute(f"""
                Find tasks that are semantically similar to:
                Title: {work_item.title}
                Description: {work_item.description}
                Type: {work_item.task_type}
                
                Consider:
                - Semantic similarity (not just exact matches)
                - Paraphrasing and synonyms
                - Same intent but different wording
                - Tasks completed in the last {self.task_cooldowns.get(work_item.task_type, self.default_cooldown_hours)} hours
                - Return similarity score > {threshold}
                - Include task details and completion time
            """)
            
            return result if isinstance(result, list) else []
            
        except Exception as e:
            self.logger.error(f"Error finding semantic duplicates: {e}")
            # Fallback to basic check
            is_dup = await self.is_duplicate_task(work_item)
            return [{"is_duplicate": is_dup}] if is_dup else []
    
    async def analyze_duplicate_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in duplicate task creation using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            # Return basic stats
            stats = await self.get_persistence_stats()
            return {"basic_stats": stats}
        
        try:
            analysis = await self.mcp_redis.execute("""
                Analyze duplicate task patterns:
                - Which task types are most frequently duplicated?
                - What time patterns exist for duplicate creation?
                - Which repositories have the most duplicates?
                - What are common phrases in duplicated tasks?
                - Identify root causes of duplication
                - Suggest prevention strategies
                - Calculate optimal cooldown periods per task type
            """)
            
            return analysis if isinstance(analysis, dict) else {"analysis": analysis}
            
        except Exception as e:
            self.logger.error(f"Error analyzing duplicate patterns: {e}")
            return {"error": str(e)}
    
    async def optimize_cooldown_periods(self) -> Dict[str, int]:
        """Use MCP-Redis to optimize cooldown periods based on historical data."""
        if not self._use_mcp or not self.mcp_redis:
            return self.task_cooldowns
        
        try:
            optimized = await self.mcp_redis.execute("""
                Analyze task completion patterns and optimize cooldown periods:
                - For each task type, find the optimal cooldown to prevent duplicates
                - Consider task value and completion frequency
                - Balance between preventing duplicates and allowing valid repeats
                - Factor in repository-specific patterns
                - Return recommended cooldown hours for each task type
            """)
            
            if isinstance(optimized, dict):
                # Update cooldowns with optimized values
                for task_type, hours in optimized.items():
                    if task_type in self.task_cooldowns and isinstance(hours, (int, float)):
                        self.task_cooldowns[task_type] = int(hours)
                        self.logger.info(f"Optimized cooldown for {task_type}: {hours} hours")
            
            return self.task_cooldowns
            
        except Exception as e:
            self.logger.error(f"Error optimizing cooldowns: {e}")
            return self.task_cooldowns
    
    async def get_task_value_insights(self) -> Dict[str, Any]:
        """Get insights about task value creation using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            insights = await self.mcp_redis.execute("""
                Analyze task value creation:
                - Which task types generate the most value?
                - What's the average value per task type?
                - Identify high-value task patterns
                - Find correlations between task attributes and value
                - Recommend focus areas for maximum impact
                - Show value trends over time
            """)
            
            return insights if isinstance(insights, dict) else {"insights": insights}
            
        except Exception as e:
            self.logger.error(f"Error getting value insights: {e}")
            return {"error": str(e)}
    
    async def predict_duplicate_likelihood(self, work_item: WorkItem) -> float:
        """Predict the likelihood of a task being a duplicate using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            # Simple check
            is_dup = await self.is_duplicate_task(work_item)
            return 1.0 if is_dup else 0.0
        
        try:
            result = await self.mcp_redis.execute(f"""
                Predict duplicate likelihood for:
                Title: {work_item.title}
                Type: {work_item.task_type}
                Repository: {work_item.repository or 'None'}
                
                Based on:
                - Historical duplicate patterns
                - Task type frequency
                - Title similarity to recent tasks
                - Repository-specific patterns
                - Time since similar tasks
                
                Return a probability score between 0.0 and 1.0
            """)
            
            if isinstance(result, (int, float)):
                return float(result)
            elif isinstance(result, dict) and 'probability' in result:
                return float(result['probability'])
            else:
                # Fallback to simple check
                is_dup = await self.is_duplicate_task(work_item)
                return 1.0 if is_dup else 0.0
                
        except Exception as e:
            self.logger.error(f"Error predicting duplicate: {e}")
            is_dup = await self.is_duplicate_task(work_item)
            return 1.0 if is_dup else 0.0
    
    async def cleanup_duplicates(self) -> Dict[str, Any]:
        """Clean up duplicate entries and optimize storage using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            result = await self.mcp_redis.execute("""
                Clean up duplicate task entries:
                - Identify exact duplicates in completed tasks
                - Find near-duplicates that can be merged
                - Remove redundant hash entries
                - Optimize storage by consolidating similar tasks
                - Maintain data integrity and history
                - Return cleanup statistics
            """)
            
            return result if isinstance(result, dict) else {"result": result}
            
        except Exception as e:
            self.logger.error(f"Error cleaning duplicates: {e}")
            return {"error": str(e)}