"""
State Manager - SQLite-backed implementation

This module provides backward compatibility while using SQLite for better reliability.
It wraps SQLiteStateManager to provide the same interface as the old StateManager.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

from scripts.sqlite_state_manager import SQLiteStateManager


class StateManager:
    """State manager with SQLite backend for better reliability."""
    
    def __init__(self):
        """Initialize state manager with SQLite backend."""
        self.logger = logging.getLogger(__name__)
        
        # Use SQLite backend in local_repos/state directory
        state_dir = Path("local_repos/state")
        state_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = state_dir / "cwmai_state.db"
        self.sqlite = SQLiteStateManager(str(db_path))
        
        # For backward compatibility
        self.state_file = state_dir / "system_state.json"
        self.task_state_file = state_dir / "task_state.json"
        
        # Load initial state
        self.state = self.load_state()
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            # Handle objects with to_dict method (like TodoItem)
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            # Handle other objects by converting to dict
            return self._make_serializable(obj.__dict__)
        else:
            # Return as-is for basic types
            return obj
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state from SQLite."""
        # Try SQLite first
        state_json = self.sqlite.get("system_state")
        if state_json:
            return json.loads(state_json)
        
        # Fallback to file if exists (migration)
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                # Save to SQLite
                self.sqlite.set("system_state", json.dumps(state))
                return state
            except Exception as e:
                self.logger.error(f"Error loading state file: {e}")
        
        return {}
    
    def load_state(self) -> Dict[str, Any]:
        """Load state - alias for get_state."""
        return self.get_state()
    
    def save_state(self) -> None:
        """Save current state to SQLite."""
        if hasattr(self, 'state') and self.state is not None:
            try:
                # Convert any non-serializable objects
                state_copy = self._make_serializable(self.state)
                self.sqlite.set("system_state", json.dumps(state_copy))
                # Also save to file for backward compatibility
                try:
                    with open(self.state_file, 'w') as f:
                        json.dump(state_copy, f, indent=2)
                except Exception as e:
                    self.logger.debug(f"Could not save to file (using SQLite): {e}")
            except Exception as e:
                self.logger.error(f"Failed to persist state: {e}")
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state with partial updates."""
        current_state = self.get_state()
        
        # Deep merge updates into current state
        def deep_merge(base: Dict, update: Dict) -> Dict:
            """Recursively merge update dict into base dict."""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        self.state = deep_merge(current_state, updates)
        self.save_state()
    
    def load_task_state(self) -> Dict[str, Any]:
        """Load task state from SQLite."""
        # Try SQLite first
        task_json = self.sqlite.get("task_state")
        if task_json:
            return json.loads(task_json)
        
        # Fallback to file if exists (migration)
        if self.task_state_file.exists():
            try:
                with open(self.task_state_file, 'r') as f:
                    state = json.load(f)
                # Save to SQLite
                self.sqlite.set("task_state", json.dumps(state))
                return state
            except Exception as e:
                self.logger.error(f"Error loading task state file: {e}")
        
        return {"tasks": {}, "completed_tasks": []}
    
    def save_task_state(self, task_state: Dict[str, Any]) -> None:
        """Save task state to SQLite."""
        self.sqlite.set("task_state", json.dumps(task_state))
        # Also save to file for backward compatibility
        try:
            with open(self.task_state_file, 'w') as f:
                json.dump(task_state, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Could not save task state to file (using SQLite): {e}")
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get task history."""
        task_state = self.load_task_state()
        return task_state.get('completed_tasks', [])
    
    def add_completed_task(self, task: Dict[str, Any]) -> None:
        """Add a completed task to history."""
        task_state = self.load_task_state()
        if 'completed_tasks' not in task_state:
            task_state['completed_tasks'] = []
        
        # Add timestamp if not present
        if 'completed_at' not in task:
            task['completed_at'] = datetime.now(timezone.utc).isoformat()
        
        task_state['completed_tasks'].append(task)
        
        # Limit history size
        max_history = 1000
        if len(task_state['completed_tasks']) > max_history:
            task_state['completed_tasks'] = task_state['completed_tasks'][-max_history:]
        
        self.save_task_state(task_state)
    
    def save_orchestrator_state(self, orch_state: Dict[str, Any]) -> None:
        """Save orchestrator state."""
        self.sqlite.set("orchestrator_state", json.dumps(orch_state))
        
        # Also save to file for monitoring tools
        orch_file = Path("local_repos/state/continuous_orchestrator_state.json")
        try:
            with open(orch_file, 'w') as f:
                json.dump(orch_state, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Could not save orchestrator state to file: {e}")
    
    def load_orchestrator_state(self) -> Dict[str, Any]:
        """Load orchestrator state."""
        orch_json = self.sqlite.get("orchestrator_state")
        if orch_json:
            return json.loads(orch_json)
        return {}
    
    # Redis-compatibility methods (delegate to SQLite)
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return self.sqlite.get(key)
    
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key-value pair."""
        return self.sqlite.set(key, value, ex)
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        return self.sqlite.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.sqlite.exists(key)
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'sqlite'):
            self.sqlite.close()