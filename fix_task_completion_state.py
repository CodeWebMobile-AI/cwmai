#!/usr/bin/env python3
"""
Fix for task completion state persistence.

This patch ensures that when tasks are completed, they are properly saved to:
1. task_state.json - for task tracking
2. system_state.json - for overall system state
3. continuous_orchestrator_state.json - for orchestrator metrics
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis
from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator

# Patch the _execute_work_item method to save state after completion
original_execute_work_item = EnhancedHybridOrchestrator._execute_work_item

async def patched_execute_work_item(self, worker_id: str, work_item):
    """Execute work item and properly save state after completion."""
    # Call original method
    result = await original_execute_work_item(self, worker_id, work_item)
    
    # If task completed successfully, update and save state
    if result.get('success', False):
        try:
            # Update task state
            task_state = self.state_manager.load_state()
            
            # Initialize completed_tasks if not exists
            if 'completed_tasks' not in task_state:
                task_state['completed_tasks'] = []
            
            # Add completed task
            task_state['completed_tasks'].append({
                'id': work_item.id,
                'task_type': work_item.task_type,
                'title': work_item.title,
                'repository': work_item.repository,
                'completed_at': result.get('completed_at', ''),
                'duration': result.get('duration', 0),
                'worker_id': worker_id
            })
            
            # Remove from active tasks if present
            if 'tasks' in task_state and work_item.id in task_state['tasks']:
                del task_state['tasks'][work_item.id]
            
            # Save updated state
            self.state_manager.save_state(task_state)
            self.logger.info(f"✅ Saved completed task {work_item.id} to state")
            
            # Also update orchestrator state
            orch_state = {
                'metrics': self.metrics,
                'work_queue': [],
                'completed_work': getattr(self, 'completed_work', []),
                'workers': {
                    worker_id: {
                        'status': worker.status.value if hasattr(worker, 'status') else 'active',
                        'total_completed': getattr(worker, 'total_completed', 0),
                        'total_errors': getattr(worker, 'total_errors', 0)
                    } for worker_id, worker in self.workers.items()
                },
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            # Save to continuous_orchestrator_state.json
            with open('continuous_orchestrator_state.json', 'w') as f:
                json.dump(orch_state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save task completion state: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    return result

# Apply patch
EnhancedHybridOrchestrator._execute_work_item = patched_execute_work_item

print("✅ Task completion state persistence fix applied!")
print("\nThis fix ensures that:")
print("1. Completed tasks are saved to task_state.json")
print("2. System state is updated with completed tasks")
print("3. Orchestrator metrics are persisted")
print("\nTo use this fix, import it at the start of your hybrid orchestrator:")
print("import fix_task_completion_state")

# Also create a method to add to StateManager
def add_completed_task(self, task_data):
    """Add a completed task to the state."""
    state = self.load_state()
    
    if 'completed_tasks' not in state:
        state['completed_tasks'] = []
    
    # Add timestamp if not present
    if 'completed_at' not in task_data:
        task_data['completed_at'] = datetime.now(timezone.utc).isoformat()
    
    state['completed_tasks'].append(task_data)
    
    # Remove from active tasks if present
    if 'tasks' in state and task_data.get('id') in state['tasks']:
        del state['tasks'][task_data['id']]
    
    return self.save_state(state)

# Add the method to StateManager
from scripts.state_manager import StateManager
from datetime import datetime, timezone
import json

StateManager.add_completed_task = add_completed_task

print("\n✅ Also added 'add_completed_task' method to StateManager")