"""
MCP-Redis Integration for CWMAI
Provides natural language interfaces to Redis operations using MCP
"""

import asyncio
import logging
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from scripts.mcp_client import MCPClient, MCPServer
from scripts.mcp_integration import MCPIntegrationHub
from scripts.work_item_types import WorkItem, TaskPriority

logger = logging.getLogger(__name__)


class MCPRedisIntegration:
    """High-level Redis operations through MCP using natural language"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.hub: Optional[MCPIntegrationHub] = None
        self.server_name = "redis"
        self._initialized = False
        
    async def initialize(self):
        """Initialize MCP-Redis connection"""
        if self._initialized:
            return
            
        try:
            # Create integration hub
            self.hub = MCPIntegrationHub()
            
            # Find npx executable with full path
            npx_path = shutil.which("npx") or "npx"
            
            # Set up environment with PATH for node
            import os
            env = os.environ.copy()
            env["REDIS_URL"] = self.redis_url
            # Ensure node is in PATH
            if "/usr/local/share/nvm/versions/node" not in env.get("PATH", ""):
                env["PATH"] = "/usr/local/share/nvm/versions/node/v22.16.0/bin:" + env.get("PATH", "")
            
            # Add Redis server
            redis_server = MCPServer(
                name=self.server_name,
                command=[npx_path, "-y", "@modelcontextprotocol/server-redis"],
                env=env
            )
            self.hub.client.add_server(redis_server)
            
            # Connect to Redis server
            await self.hub.client.connect(self.server_name)
            
            self._initialized = True
            logger.info(f"MCP-Redis initialized with URL: {self.redis_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP-Redis: {e}")
            raise
    
    async def execute(self, command: str) -> Any:
        """Execute a natural language Redis command"""
        if not self._initialized:
            await self.initialize()
            
        # Parse command to determine which tool to use
        cmd_parts = command.strip().upper().split()
        if not cmd_parts:
            raise ValueError("Empty command")
            
        cmd = cmd_parts[0]
        
        try:
            if cmd == "PING":
                # Use GET with a dummy key to test connection
                result = await self.hub.client.call_tool(f"{self.server_name}:get", {
                    "key": "_mcp_ping_test"
                })
                return "PONG"
            elif cmd == "SET" and len(cmd_parts) >= 3:
                key = cmd_parts[1]
                value = " ".join(cmd_parts[2:])
                result = await self.hub.client.call_tool(f"{self.server_name}:set", {
                    "key": key,
                    "value": value
                })
                return result
            elif cmd == "GET" and len(cmd_parts) >= 2:
                key = cmd_parts[1]
                result = await self.hub.client.call_tool(f"{self.server_name}:get", {
                    "key": key
                })
                return result
            elif cmd == "DEL" and len(cmd_parts) >= 2:
                keys = cmd_parts[1:]
                result = await self.hub.client.call_tool(f"{self.server_name}:delete", {
                    "key": keys if len(keys) > 1 else keys[0]
                })
                return result
            elif cmd == "KEYS" and len(cmd_parts) >= 2:
                pattern = cmd_parts[1]
                result = await self.hub.client.call_tool(f"{self.server_name}:list", {
                    "pattern": pattern
                })
                return result
            else:
                raise ValueError(f"Unsupported Redis command: {cmd}")
                
        except Exception as e:
            logger.error(f"Error executing Redis command '{command}': {e}")
            raise
    
    # Work Queue Operations
    async def add_work_item(self, work_item: WorkItem) -> Dict[str, Any]:
        """Add a work item to the queue using natural language"""
        command = f"""
        Add this work item to Redis:
        - Store in stream 'cwmai:work_queue:{work_item.priority.name.lower()}'
        - Fields:
          - id: {work_item.id}
          - task_type: {work_item.task_type}
          - title: {work_item.title}
          - description: {work_item.description}
          - priority: {work_item.priority.name}
          - repository: {work_item.repository or 'none'}
          - created_at: {work_item.created_at.isoformat() if work_item.created_at else datetime.now().isoformat()}
          - metadata: {json.dumps(work_item.metadata) if work_item.metadata else '{}'}
        - Return the stream entry ID
        """
        return await self.execute(command)
    
    async def get_work_for_worker(self, worker_id: str, specialization: Optional[str] = None, count: int = 1) -> List[Dict[str, Any]]:
        """Get work items for a specific worker"""
        command = f"""
        Get up to {count} work items for worker '{worker_id}':
        - Check streams in priority order: critical, high, medium, low, background
        - Use consumer group 'cwmai_workers'
        - If specialization is '{specialization}', only return matching items
        - For each item, acknowledge it and mark as assigned to '{worker_id}'
        - Return the work items with all fields
        """
        return await self.execute(command)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get work queue statistics"""
        command = """
        Get statistics for all work queues:
        - For each priority stream (critical, high, medium, low, background):
          - Count total items
          - Count pending items
          - Get oldest and newest item timestamps
        - Calculate total items across all queues
        - Return as structured data
        """
        return await self.execute(command)
    
    # State Management Operations
    async def save_state(self, key: str, state: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Save system state to Redis"""
        state_json = json.dumps(state)
        ttl_part = f"with TTL of {ttl} seconds" if ttl else "without expiration"
        
        command = f"""
        Save this state to Redis:
        - Key: {key}
        - Value: {state_json}
        - Store {ttl_part}
        - Return success status
        """
        result = await self.execute(command)
        return bool(result)
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve system state from Redis"""
        command = f"""
        Get the state stored at key '{key}':
        - Return the JSON value
        - If key doesn't exist, return null
        """
        result = await self.execute(command)
        if result:
            try:
                return json.loads(result) if isinstance(result, str) else result
            except:
                return result
        return None
    
    # Worker Management Operations
    async def update_worker_status(self, worker_id: str, status: str, current_task: Optional[str] = None) -> bool:
        """Update worker status in Redis"""
        task_part = f", current_task: '{current_task}'" if current_task else ""
        
        command = f"""
        Update worker status:
        - Store in hash 'cwmai:workers:{worker_id}'
        - Set fields:
          - status: {status}
          - last_update: {datetime.now().isoformat()}
          {task_part}
        - Set expiration to 5 minutes
        - Return success status
        """
        result = await self.execute(command)
        return bool(result)
    
    async def get_all_workers(self) -> List[Dict[str, Any]]:
        """Get all active workers"""
        command = """
        Find all active workers:
        - Search for keys matching 'cwmai:workers:*'
        - For each worker, get all hash fields
        - Only include workers updated in the last 5 minutes
        - Return list of worker data
        """
        return await self.execute(command)
    
    # Search and Query Operations
    async def find_similar_tasks(self, description: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar tasks using Redis search capabilities"""
        command = f"""
        Search for tasks similar to '{description}':
        - Search in all work queue streams
        - Use text similarity on title and description fields
        - Limit to {limit} results
        - Return tasks ordered by similarity score
        """
        return await self.execute(command)
    
    async def query_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Query tasks by status"""
        command = f"""
        Find all tasks with status '{status}':
        - Search across all work queue streams
        - Filter by status field
        - Include task details and timestamps
        - Return sorted by created_at descending
        """
        return await self.execute(command)
    
    # Advanced Operations
    async def cleanup_old_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up old completed tasks"""
        command = f"""
        Clean up old tasks:
        - For each work queue stream
        - Remove entries older than {older_than_hours} hours
        - Only remove if status is 'completed' or 'failed'
        - Return count of removed items
        """
        result = await self.execute(command)
        return int(result) if result else 0
    
    async def requeue_stuck_tasks(self, timeout_minutes: int = 30) -> int:
        """Requeue tasks that have been stuck"""
        command = f"""
        Find and requeue stuck tasks:
        - Check all work queue streams
        - Find tasks assigned but not completed for over {timeout_minutes} minutes
        - Reset their assigned_worker field
        - Move them back to pending state
        - Return count of requeued tasks
        """
        result = await self.execute(command)
        return int(result) if result else 0
    
    # Stream Operations
    async def create_event_stream(self, event: Dict[str, Any], stream_name: str = "cwmai:events") -> str:
        """Add event to event stream"""
        event_json = json.dumps(event)
        command = f"""
        Add event to stream '{stream_name}':
        - Event data: {event_json}
        - Add timestamp field with current time
        - Return the stream entry ID
        """
        return await self.execute(command)
    
    async def get_recent_events(self, stream_name: str = "cwmai:events", count: int = 100) -> List[Dict[str, Any]]:
        """Get recent events from stream"""
        command = f"""
        Get last {count} events from stream '{stream_name}':
        - Read in reverse order (newest first)
        - Include all fields
        - Parse JSON fields if present
        - Return as list of events
        """
        return await self.execute(command)
    
    # Monitoring Operations
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics from Redis"""
        command = """
        Gather system metrics:
        - Count total tasks by priority
        - Count active workers
        - Get average task completion time from recent completed tasks
        - Get queue depths for each priority
        - Calculate task throughput (completed in last hour)
        - Return as structured metrics data
        """
        return await self.execute(command)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        command = """
        Perform health check:
        - Test Redis connectivity
        - Check memory usage
        - Count total keys
        - Check for any blocked clients
        - Return health status and metrics
        """
        return await self.execute(command)
    
    # Utility Methods
    async def clear_all_queues(self) -> bool:
        """Clear all work queues (use with caution!)"""
        command = """
        Clear all work queues:
        - Delete all streams matching 'cwmai:work_queue:*'
        - Clear worker states
        - Reset consumer groups
        - Return success status
        """
        result = await self.execute(command)
        return bool(result)
    
    async def backup_state(self, backup_key: str) -> bool:
        """Backup current system state"""
        command = f"""
        Create system backup:
        - Copy all cwmai:* keys to backup:{backup_key}:*
        - Include streams, hashes, and regular keys
        - Set 7 day expiration on backup
        - Return success status
        """
        result = await self.execute(command)
        return bool(result)
    
    async def close(self):
        """Close MCP-Redis connection"""
        if self.hub:
            await self.hub.close()
        self._initialized = False
        logger.info("MCP-Redis connection closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Convenience functions for direct usage
async def get_mcp_redis() -> MCPRedisIntegration:
    """Get initialized MCP-Redis instance"""
    redis = MCPRedisIntegration()
    await redis.initialize()
    return redis


# Example usage and testing
async def test_mcp_redis():
    """Test MCP-Redis integration"""
    async with MCPRedisIntegration() as redis:
        # Test health check
        health = await redis.health_check()
        print(f"Redis health: {health}")
        
        # Test adding a work item
        from scripts.work_item_types import WorkItem, TaskPriority
        test_item = WorkItem(
            id="test-123",
            task_type="test",
            title="Test MCP-Redis Integration",
            description="Testing the new MCP-Redis integration",
            priority=TaskPriority.MEDIUM
        )
        
        result = await redis.add_work_item(test_item)
        print(f"Added work item: {result}")
        
        # Test getting queue stats
        stats = await redis.get_queue_stats()
        print(f"Queue stats: {stats}")
        
        # Test state management
        test_state = {"version": "1.0", "status": "testing"}
        saved = await redis.save_state("test:state", test_state)
        print(f"Saved state: {saved}")
        
        retrieved = await redis.get_state("test:state")
        print(f"Retrieved state: {retrieved}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_mcp_redis())