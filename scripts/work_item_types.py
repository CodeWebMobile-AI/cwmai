"""
Work Item Types for Continuous AI System

Shared types to avoid circular imports between modules.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid


class TaskPriority(Enum):
    """Task priority levels for continuous execution."""
    CRITICAL = 1    # Execute immediately
    HIGH = 2        # Execute within minutes
    MEDIUM = 3      # Execute within hours
    LOW = 4         # Execute when nothing else
    BACKGROUND = 5  # Background processing


@dataclass
class WorkItem:
    """A unit of work in the continuous system."""
    id: str
    task_type: str
    title: str
    description: str
    priority: TaskPriority
    repository: Optional[str] = None
    estimated_cycles: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'task_type': self.task_type,
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'repository': self.repository,
            'estimated_cycles': self.estimated_cycles,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'assigned_worker': self.assigned_worker,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class WorkOpportunity:
    """A discovered work opportunity."""
    source: str
    type: str
    priority: TaskPriority
    title: str
    description: str
    repository: Optional[str] = None
    metadata: Dict[str, Any] = None
    estimated_cycles: int = 1
    
    def to_work_item(self) -> WorkItem:
        """Convert to a WorkItem."""
        return WorkItem(
            id=f"work_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            task_type=self.type,
            title=self.title,
            description=self.description,
            priority=self.priority,
            repository=self.repository,
            estimated_cycles=self.estimated_cycles,
            metadata=self.metadata or {}
        )