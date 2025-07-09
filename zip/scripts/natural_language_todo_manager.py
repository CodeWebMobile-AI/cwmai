"""
Natural Language Todo Manager

Core todo management system with persistence and natural language integration.
Manages the actual todo list data and operations.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
import re
from module_installer import module_installer

# Try to auto-install fuzzywuzzy if not available
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    if module_installer.try_install('fuzzywuzzy', 'fuzzywuzzy[speedup]'):
        try:
            from fuzzywuzzy import fuzz, process
            FUZZY_AVAILABLE = True
        except ImportError:
            FUZZY_AVAILABLE = False
    else:
        FUZZY_AVAILABLE = False


class TodoStatus(Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TodoItem:
    """Represents a single todo item."""
    id: str
    description: str
    created_at: datetime
    status: TodoStatus = TodoStatus.PENDING
    priority: str = "medium"
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    reminder: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        if self.due_date:
            data['due_date'] = self.due_date.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.reminder:
            data['reminder'] = self.reminder.isoformat()
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TodoItem':
        """Create from dictionary."""
        # Convert datetime strings
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['status'] = TodoStatus(data['status'])
        
        if data.get('due_date'):
            data['due_date'] = datetime.fromisoformat(data['due_date'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if data.get('reminder'):
            data['reminder'] = datetime.fromisoformat(data['reminder'])
            
        return cls(**data)
        
    def matches_query(self, query: str) -> bool:
        """Check if todo matches a search query."""
        query_lower = query.lower()
        
        # Check description
        if query_lower in self.description.lower():
            return True
            
        # Check tags
        for tag in self.tags:
            if query_lower in tag.lower():
                return True
                
        # Check notes
        if query_lower in self.notes.lower():
            return True
            
        return False


class NaturalLanguageTodoManager:
    """Manages todo items with natural language support."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the todo manager."""
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path or Path("todos.json")
        self.todos: Dict[str, TodoItem] = {}
        
        # Load existing todos
        self._load_todos()
        
        # Track recent operations for undo
        self.operation_history: List[Dict[str, Any]] = []
        self.max_history = 50
        
    def _load_todos(self):
        """Load todos from storage."""
        if not self.storage_path.exists():
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            for todo_data in data.get('todos', []):
                todo = TodoItem.from_dict(todo_data)
                self.todos[todo.id] = todo
                
            self.logger.info(f"Loaded {len(self.todos)} todos from storage")
            
        except Exception as e:
            self.logger.error(f"Failed to load todos: {e}")
            
    def _save_todos(self):
        """Save todos to storage."""
        try:
            data = {
                'todos': [todo.to_dict() for todo in self.todos.values()],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save todos: {e}")
            
    def _generate_id(self) -> str:
        """Generate unique todo ID."""
        return str(uuid.uuid4())[:8]
        
    def _record_operation(self, operation: str, data: Dict[str, Any]):
        """Record operation for undo functionality."""
        self.operation_history.append({
            'operation': operation,
            'timestamp': datetime.now(timezone.utc),
            'data': data
        })
        
        # Limit history size
        if len(self.operation_history) > self.max_history:
            self.operation_history.pop(0)
            
    def add_todo(
        self,
        description: str,
        priority: str = "medium",
        due_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        reminder: Optional[datetime] = None
    ) -> TodoItem:
        """Add a new todo item."""
        todo = TodoItem(
            id=self._generate_id(),
            description=description,
            created_at=datetime.now(timezone.utc),
            priority=priority,
            due_date=due_date,
            tags=tags or [],
            reminder=reminder
        )
        
        self.todos[todo.id] = todo
        self._save_todos()
        
        self._record_operation('add', {'todo_id': todo.id})
        self.logger.info(f"Added todo: {todo.id} - {description}")
        
        return todo
        
    def complete_todo(self, todo_ref: str) -> Optional[TodoItem]:
        """Mark a todo as completed."""
        todo = self.find_todo(todo_ref)
        
        if not todo:
            self.logger.warning(f"Todo not found: {todo_ref}")
            return None
            
        if todo.status == TodoStatus.COMPLETED:
            self.logger.info(f"Todo already completed: {todo.id}")
            return todo
            
        # Record previous state for undo
        self._record_operation('complete', {
            'todo_id': todo.id,
            'previous_status': todo.status.value
        })
        
        todo.status = TodoStatus.COMPLETED
        todo.completed_at = datetime.now(timezone.utc)
        self._save_todos()
        
        self.logger.info(f"Completed todo: {todo.id} - {todo.description}")
        return todo
        
    def update_todo(
        self,
        todo_ref: str,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        due_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        status: Optional[TodoStatus] = None
    ) -> Optional[TodoItem]:
        """Update a todo item."""
        todo = self.find_todo(todo_ref)
        
        if not todo:
            self.logger.warning(f"Todo not found: {todo_ref}")
            return None
            
        # Record previous state
        previous_state = todo.to_dict()
        
        # Update fields
        if description is not None:
            todo.description = description
        if priority is not None:
            todo.priority = priority
        if due_date is not None:
            todo.due_date = due_date
        if tags is not None:
            todo.tags = tags
        if notes is not None:
            todo.notes = notes
        if status is not None:
            todo.status = status
            if status == TodoStatus.COMPLETED:
                todo.completed_at = datetime.now(timezone.utc)
                
        self._save_todos()
        
        self._record_operation('update', {
            'todo_id': todo.id,
            'previous_state': previous_state
        })
        
        self.logger.info(f"Updated todo: {todo.id}")
        return todo
        
    def delete_todo(self, todo_ref: str) -> Optional[TodoItem]:
        """Delete a todo item."""
        todo = self.find_todo(todo_ref)
        
        if not todo:
            self.logger.warning(f"Todo not found: {todo_ref}")
            return None
            
        # Record for undo
        self._record_operation('delete', {
            'todo': todo.to_dict()
        })
        
        del self.todos[todo.id]
        self._save_todos()
        
        self.logger.info(f"Deleted todo: {todo.id} - {todo.description}")
        return todo
        
    def find_todo(self, reference: str) -> Optional[TodoItem]:
        """Find a todo by ID or fuzzy match description."""
        # Try exact ID match first
        if reference in self.todos:
            return self.todos[reference]
            
        # Try fuzzy matching on descriptions if available
        if FUZZY_AVAILABLE and self.todos:
            descriptions = {
                todo.id: todo.description
                for todo in self.todos.values()
                if todo.status != TodoStatus.COMPLETED
            }
            
            if descriptions:
                # Use fuzzy matching
                matches = process.extract(
                    reference,
                    descriptions.values(),
                    scorer=fuzz.token_sort_ratio,
                    limit=1
                )
                
                if matches and matches[0][1] > 70:  # 70% similarity threshold
                    # Find the todo with this description
                    matched_desc = matches[0][0]
                    for todo_id, desc in descriptions.items():
                        if desc == matched_desc:
                            return self.todos[todo_id]
        elif self.todos and not FUZZY_AVAILABLE:
            # Simple substring matching fallback
            reference_lower = reference.lower()
            for todo in self.todos.values():
                if todo.status != TodoStatus.COMPLETED:
                    if reference_lower in todo.description.lower():
                        return todo
                            
        return None
        
    def list_todos(
        self,
        status: Optional[TodoStatus] = None,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        due_before: Optional[datetime] = None,
        search_query: Optional[str] = None
    ) -> List[TodoItem]:
        """List todos with optional filters."""
        todos = list(self.todos.values())
        
        # Apply filters
        if status is not None:
            todos = [t for t in todos if t.status == status]
            
        if priority is not None:
            todos = [t for t in todos if t.priority == priority]
            
        if tag is not None:
            todos = [t for t in todos if tag in t.tags]
            
        if due_before is not None:
            todos = [t for t in todos if t.due_date and t.due_date <= due_before]
            
        if search_query is not None:
            todos = [t for t in todos if t.matches_query(search_query)]
            
        # Sort by priority and due date
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        
        def sort_key(todo):
            pri = priority_order.get(todo.priority, 99)
            due = todo.due_date.timestamp() if todo.due_date else float('inf')
            return (pri, due)
            
        todos.sort(key=sort_key)
        
        return todos
        
    def get_todos_for_date(self, date: datetime) -> List[TodoItem]:
        """Get todos due on a specific date."""
        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        return [
            todo for todo in self.todos.values()
            if todo.due_date and date_start <= todo.due_date < date_end
            and todo.status != TodoStatus.COMPLETED
        ]
        
    def get_overdue_todos(self) -> List[TodoItem]:
        """Get overdue todos."""
        now = datetime.now(timezone.utc)
        
        return [
            todo for todo in self.todos.values()
            if todo.due_date and todo.due_date < now
            and todo.status != TodoStatus.COMPLETED
        ]
        
    def get_upcoming_reminders(self, hours: int = 24) -> List[TodoItem]:
        """Get todos with reminders in the next N hours."""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=hours)
        
        return [
            todo for todo in self.todos.values()
            if todo.reminder and now <= todo.reminder <= future
            and todo.status != TodoStatus.COMPLETED
        ]
        
    def undo_last_operation(self) -> bool:
        """Undo the last operation."""
        if not self.operation_history:
            return False
            
        operation = self.operation_history.pop()
        op_type = operation['operation']
        data = operation['data']
        
        try:
            if op_type == 'add':
                # Delete the added todo
                todo_id = data['todo_id']
                if todo_id in self.todos:
                    del self.todos[todo_id]
                    
            elif op_type == 'delete':
                # Restore the deleted todo
                todo_data = data['todo']
                todo = TodoItem.from_dict(todo_data)
                self.todos[todo.id] = todo
                
            elif op_type == 'update':
                # Restore previous state
                todo_id = data['todo_id']
                if todo_id in self.todos:
                    previous_state = data['previous_state']
                    self.todos[todo_id] = TodoItem.from_dict(previous_state)
                    
            elif op_type == 'complete':
                # Restore previous status
                todo_id = data['todo_id']
                if todo_id in self.todos:
                    self.todos[todo_id].status = TodoStatus(data['previous_status'])
                    self.todos[todo_id].completed_at = None
                    
            self._save_todos()
            self.logger.info(f"Undid operation: {op_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to undo operation: {e}")
            return False
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get todo statistics."""
        todos = list(self.todos.values())
        
        total = len(todos)
        completed = sum(1 for t in todos if t.status == TodoStatus.COMPLETED)
        pending = sum(1 for t in todos if t.status == TodoStatus.PENDING)
        overdue = len(self.get_overdue_todos())
        
        # Priority breakdown
        priority_counts = {}
        for todo in todos:
            if todo.status != TodoStatus.COMPLETED:
                priority_counts[todo.priority] = priority_counts.get(todo.priority, 0) + 1
                
        # Completion rate by time period
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        completed_this_week = sum(
            1 for t in todos
            if t.completed_at and t.completed_at >= week_ago
        )
        
        completed_this_month = sum(
            1 for t in todos
            if t.completed_at and t.completed_at >= month_ago
        )
        
        return {
            'total': total,
            'completed': completed,
            'pending': pending,
            'overdue': overdue,
            'completion_rate': completed / total if total > 0 else 0,
            'priority_breakdown': priority_counts,
            'completed_this_week': completed_this_week,
            'completed_this_month': completed_this_month
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = NaturalLanguageTodoManager()
    
    # Add some todos
    todo1 = manager.add_todo(
        "Review API documentation",
        priority="high",
        due_date=datetime.now(timezone.utc) + timedelta(days=1),
        tags=["documentation", "api"]
    )
    
    todo2 = manager.add_todo(
        "Call John about project",
        priority="medium",
        reminder=datetime.now(timezone.utc) + timedelta(hours=3)
    )
    
    # List todos
    print("\nAll pending todos:")
    for todo in manager.list_todos(status=TodoStatus.PENDING):
        print(f"- [{todo.priority}] {todo.description}")
        
    # Complete a todo
    manager.complete_todo("API")  # Fuzzy match
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Undo last operation
    manager.undo_last_operation()
    print("\nAfter undo:")
    for todo in manager.list_todos(status=TodoStatus.PENDING):
        print(f"- [{todo.priority}] {todo.description}")