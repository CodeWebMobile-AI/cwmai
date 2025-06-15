"""
MCP-Based Task Persistence System

Uses MySQL MCP for reliable long-term task storage and Memory MCP
for fast caching, providing a robust persistence layer.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from scripts.task_persistence import TaskPersistence
from scripts.mcp_integration import MCPIntegrationHub
from scripts.work_item_types import WorkItem, TaskPriority


@dataclass
class TaskRecord:
    """Represents a task record in the database."""
    id: str
    title: str
    description: str
    task_type: str
    priority: str
    status: str
    repository: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    github_issue_number: Optional[int] = None
    metadata: Optional[Dict] = None
    

class MCPTaskPersistence(TaskPersistence):
    """Task persistence using MCP MySQL and Memory."""
    
    def __init__(self, mcp_hub: Optional[MCPIntegrationHub] = None):
        """Initialize MCP-based task persistence.
        
        Args:
            mcp_hub: Optional pre-initialized MCP integration hub
        """
        super().__init__()
        self.mcp_hub = mcp_hub
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
    async def _ensure_initialized(self):
        """Ensure MCP and database are initialized."""
        if self._initialized:
            return
            
        if not self.mcp_hub:
            self.mcp_hub = MCPIntegrationHub()
            await self.mcp_hub.initialize(servers=['mysql', 'memory'])
        
        # Create tables if they don't exist
        if self.mcp_hub.mysql:
            await self._create_tables()
            
        self._initialized = True
        
    async def _create_tables(self):
        """Create necessary database tables."""
        try:
            # Tasks table
            await self.mcp_hub.mysql.create_table("tasks", {
                "id": "VARCHAR(255) PRIMARY KEY",
                "title": "VARCHAR(500) NOT NULL",
                "description": "TEXT",
                "task_type": "VARCHAR(50)",
                "priority": "VARCHAR(20)",
                "status": "VARCHAR(50) DEFAULT 'pending'",
                "repository": "VARCHAR(255)",
                "github_issue_number": "INT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
                "completed_at": "TIMESTAMP NULL",
                "metadata": "JSON"
            })
            
            # Task history table
            await self.mcp_hub.mysql.create_table("task_history", {
                "id": "INT AUTO_INCREMENT PRIMARY KEY",
                "task_id": "VARCHAR(255)",
                "action": "VARCHAR(50)",
                "old_status": "VARCHAR(50)",
                "new_status": "VARCHAR(50)",
                "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "metadata": "JSON",
                "INDEX": "idx_task_id (task_id)"
            })
            
            # Task metrics table
            await self.mcp_hub.mysql.create_table("task_metrics", {
                "id": "INT AUTO_INCREMENT PRIMARY KEY",
                "date": "DATE",
                "tasks_created": "INT DEFAULT 0",
                "tasks_completed": "INT DEFAULT 0",
                "tasks_failed": "INT DEFAULT 0",
                "avg_completion_time": "FLOAT",
                "value_created": "FLOAT",
                "UNIQUE KEY": "idx_date (date)"
            })
            
            self.logger.info("✅ Database tables created/verified")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            
    async def save_task(self, task: Dict[str, Any]) -> bool:
        """Save a task to persistent storage.
        
        Args:
            task: Task data to save
            
        Returns:
            Success status
        """
        await self._ensure_initialized()
        
        # Save to parent's storage first (Redis/file)
        parent_result = await super().save_task(task)
        
        if not self.mcp_hub or not self.mcp_hub.mysql:
            return parent_result
            
        try:
            # Prepare task record
            record = {
                "id": task.get("id"),
                "title": task.get("title"),
                "description": task.get("description"),
                "task_type": task.get("task_type"),
                "priority": task.get("priority", "medium"),
                "status": task.get("status", "pending"),
                "repository": task.get("repository"),
                "github_issue_number": task.get("github_issue_number"),
                "metadata": json.dumps(task.get("metadata", {}))
            }
            
            # Insert or update in MySQL
            existing = await self._get_task_from_mysql(task["id"])
            if existing:
                # Update existing task
                await self.mcp_hub.mysql.execute_query(
                    """UPDATE tasks SET title=?, description=?, status=?, 
                       github_issue_number=?, updated_at=NOW() WHERE id=?""",
                    [record["title"], record["description"], record["status"],
                     record["github_issue_number"], record["id"]]
                )
            else:
                # Insert new task
                await self.mcp_hub.mysql.insert_record("tasks", record)
                
            # Cache in memory for fast access
            if self.mcp_hub.memory:
                await self.mcp_hub.memory.store_context(
                    key=f"task_{task['id']}",
                    value=task,
                    metadata={"type": "task", "status": task.get("status")}
                )
                
            # Record history
            await self._record_task_history(task["id"], "save", None, task.get("status"))
            
            self.logger.info(f"💾 Saved task {task['id']} to MySQL")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving task to MySQL: {e}")
            return parent_result
            
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task data or None
        """
        await self._ensure_initialized()
        
        # Check memory cache first
        if self.mcp_hub and self.mcp_hub.memory:
            try:
                cached = await self.mcp_hub.memory.retrieve_context(f"task_{task_id}")
                if cached:
                    return cached
            except:
                pass
        
        # Check MySQL
        if self.mcp_hub and self.mcp_hub.mysql:
            task = await self._get_task_from_mysql(task_id)
            if task:
                # Cache it
                if self.mcp_hub.memory:
                    await self.mcp_hub.memory.store_context(f"task_{task_id}", task)
                return task
        
        # Fallback to parent's storage
        return await super().get_task(task_id)
        
    async def _get_task_from_mysql(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task from MySQL."""
        try:
            rows = await self.mcp_hub.mysql.execute_query(
                "SELECT * FROM tasks WHERE id = ?",
                [task_id]
            )
            
            if rows:
                row = rows[0]
                return {
                    "id": row["id"],
                    "title": row["title"],
                    "description": row["description"],
                    "task_type": row["task_type"],
                    "priority": row["priority"],
                    "status": row["status"],
                    "repository": row["repository"],
                    "github_issue_number": row["github_issue_number"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                    "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }
                
        except Exception as e:
            self.logger.error(f"Error getting task from MySQL: {e}")
            
        return None
        
    async def list_tasks(self, status: Optional[str] = None, 
                        repository: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """List tasks with optional filtering.
        
        Args:
            status: Filter by status
            repository: Filter by repository
            limit: Maximum tasks to return
            
        Returns:
            List of tasks
        """
        await self._ensure_initialized()
        
        if not self.mcp_hub or not self.mcp_hub.mysql:
            return await super().list_tasks(status, repository, limit)
            
        try:
            # Build query
            query = "SELECT * FROM tasks WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
                
            if repository:
                query += " AND repository = ?"
                params.append(repository)
                
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = await self.mcp_hub.mysql.execute_query(query, params)
            
            tasks = []
            for row in rows:
                task = {
                    "id": row["id"],
                    "title": row["title"],
                    "description": row["description"],
                    "task_type": row["task_type"],
                    "priority": row["priority"],
                    "status": row["status"],
                    "repository": row["repository"],
                    "github_issue_number": row["github_issue_number"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }
                tasks.append(task)
                
            return tasks
            
        except Exception as e:
            self.logger.error(f"Error listing tasks from MySQL: {e}")
            return await super().list_tasks(status, repository, limit)
            
    async def update_task_status(self, task_id: str, status: str, 
                               metadata: Optional[Dict] = None) -> bool:
        """Update task status.
        
        Args:
            task_id: Task ID
            status: New status
            metadata: Optional metadata updates
            
        Returns:
            Success status
        """
        await self._ensure_initialized()
        
        # Update parent storage
        parent_result = await super().update_task_status(task_id, status, metadata)
        
        if not self.mcp_hub or not self.mcp_hub.mysql:
            return parent_result
            
        try:
            # Get current status
            current = await self._get_task_from_mysql(task_id)
            old_status = current["status"] if current else None
            
            # Update in MySQL
            if status == "completed":
                await self.mcp_hub.mysql.execute_query(
                    "UPDATE tasks SET status=?, completed_at=NOW() WHERE id=?",
                    [status, task_id]
                )
            else:
                await self.mcp_hub.mysql.execute_query(
                    "UPDATE tasks SET status=? WHERE id=?",
                    [status, task_id]
                )
                
            # Update cache
            if self.mcp_hub.memory:
                cached = await self.mcp_hub.memory.retrieve_context(f"task_{task_id}")
                if cached:
                    cached["status"] = status
                    if metadata:
                        cached["metadata"].update(metadata)
                    await self.mcp_hub.memory.store_context(f"task_{task_id}", cached)
                    
            # Record history
            await self._record_task_history(task_id, "status_change", old_status, status)
            
            # Update metrics
            await self._update_metrics(status)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating task status in MySQL: {e}")
            return parent_result
            
    async def _record_task_history(self, task_id: str, action: str, 
                                 old_status: Optional[str], new_status: Optional[str]):
        """Record task history."""
        try:
            await self.mcp_hub.mysql.insert_record("task_history", {
                "task_id": task_id,
                "action": action,
                "old_status": old_status,
                "new_status": new_status,
                "metadata": json.dumps({"timestamp": datetime.now().isoformat()})
            })
        except Exception as e:
            self.logger.error(f"Error recording task history: {e}")
            
    async def _update_metrics(self, status: str):
        """Update task metrics."""
        try:
            today = datetime.now().date().isoformat()
            
            # Check if today's record exists
            rows = await self.mcp_hub.mysql.execute_query(
                "SELECT * FROM task_metrics WHERE date = ?",
                [today]
            )
            
            if rows:
                # Update existing record
                if status == "completed":
                    await self.mcp_hub.mysql.execute_query(
                        "UPDATE task_metrics SET tasks_completed = tasks_completed + 1 WHERE date = ?",
                        [today]
                    )
                elif status == "failed":
                    await self.mcp_hub.mysql.execute_query(
                        "UPDATE task_metrics SET tasks_failed = tasks_failed + 1 WHERE date = ?",
                        [today]
                    )
            else:
                # Create new record
                metrics = {
                    "date": today,
                    "tasks_created": 0,
                    "tasks_completed": 1 if status == "completed" else 0,
                    "tasks_failed": 1 if status == "failed" else 0
                }
                await self.mcp_hub.mysql.insert_record("task_metrics", metrics)
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
            
    async def get_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get task metrics for the specified period.
        
        Args:
            days: Number of days to include
            
        Returns:
            Metrics data
        """
        await self._ensure_initialized()
        
        if not self.mcp_hub or not self.mcp_hub.mysql:
            return {}
            
        try:
            rows = await self.mcp_hub.mysql.execute_query(
                """SELECT * FROM task_metrics 
                   WHERE date >= DATE_SUB(CURDATE(), INTERVAL ? DAY)
                   ORDER BY date DESC""",
                [days]
            )
            
            metrics = {
                "daily_metrics": [],
                "totals": {
                    "tasks_created": 0,
                    "tasks_completed": 0,
                    "tasks_failed": 0
                }
            }
            
            for row in rows:
                daily = {
                    "date": row["date"].isoformat(),
                    "tasks_created": row["tasks_created"],
                    "tasks_completed": row["tasks_completed"],
                    "tasks_failed": row["tasks_failed"]
                }
                metrics["daily_metrics"].append(daily)
                
                metrics["totals"]["tasks_created"] += row["tasks_created"]
                metrics["totals"]["tasks_completed"] += row["tasks_completed"]
                metrics["totals"]["tasks_failed"] += row["tasks_failed"]
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {}
            
    async def close(self):
        """Close MCP connections."""
        if self.mcp_hub and self._initialized:
            await self.mcp_hub.close()