#!/usr/bin/env python3
"""
Fix script for Claude completion notification.

This script updates the ClaudeExecutionQueue to notify the task persistence
system when Claude completes execution, ensuring tasks are properly marked
as completed only after actual execution.
"""

import os
import sys
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the file before modifying."""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up {filepath} to {backup_path}")
    return backup_path

def fix_claude_execution_queue():
    """Fix claude_execution_queue.py to notify task persistence on completion."""
    filepath = "scripts/claude_execution_queue.py"
    print(f"\n🔧 Fixing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    backup_file(filepath)
    
    # Add import for task persistence
    if "from scripts.task_persistence import TaskPersistence" not in content:
        import_section = """import uuid

from scripts.task_persistence import TaskPersistence"""
        content = content.replace("import uuid", import_section)
        print("✅ Added TaskPersistence import")
    
    # Add task_persistence initialization in __init__
    init_old = """        self.logger = logging.getLogger(__name__)
        self.queue_file = queue_file
        self._queue: List[ClaudeTask] = []
        self._processing: Dict[str, ClaudeTask] = {}
        self._completed: List[ClaudeTask] = []
        self._lock = asyncio.Lock()"""
    
    init_new = """        self.logger = logging.getLogger(__name__)
        self.queue_file = queue_file
        self._queue: List[ClaudeTask] = []
        self._processing: Dict[str, ClaudeTask] = {}
        self._completed: List[ClaudeTask] = []
        self._lock = asyncio.Lock()
        
        # Initialize task persistence for completion notifications
        self.task_persistence = TaskPersistence()"""
    
    if init_old in content:
        content = content.replace(init_old, init_new)
        print("✅ Added task_persistence initialization")
    
    # Fix the complete_task method to notify task persistence
    complete_old = """        duration = (task.completed_at - task.started_at).total_seconds()
        self.logger.info(f"Task {task_id} completed in {duration:.1f}s")"""
    
    complete_new = """        duration = (task.completed_at - task.started_at).total_seconds()
        self.logger.info(f"Task {task_id} completed in {duration:.1f}s")
        
        # Notify task persistence that this task is now complete
        try:
            # Extract work_item_id from the task or result
            work_item_id = None
            if result and 'task_id' in result:
                work_item_id = result['task_id']
            elif task.result and 'task_id' in task.result:
                work_item_id = task.result['task_id']
            else:
                # Try to find work_item_id in in-progress tasks
                # by matching repository and issue number
                in_progress_tasks = self.task_persistence.in_progress_tasks
                for wid, task_data in in_progress_tasks.items():
                    if (task_data.get('repository') == task.repository and 
                        task_data.get('issue_number') == task.issue_number):
                        work_item_id = wid
                        break
            
            if work_item_id:
                # Mark the task as completed in task persistence
                completion_result = {
                    'success': True,
                    'completed': True,
                    'claude_completed': True,
                    'issue_number': task.issue_number,
                    'repository': task.repository,
                    'execution_time': duration,
                    'claude_result': result
                }
                
                if hasattr(self.task_persistence, 'complete_in_progress_task'):
                    self.task_persistence.complete_in_progress_task(work_item_id, completion_result)
                    self.logger.info(f"✅ Notified task persistence of completion for work_item {work_item_id}")
                else:
                    self.logger.warning("Task persistence doesn't have complete_in_progress_task method")
            else:
                self.logger.warning(f"Could not find work_item_id for completed Claude task {task_id}")
                
        except Exception as e:
            self.logger.error(f"Error notifying task persistence of completion: {e}")
            # Don't fail the completion if notification fails"""
    
    if complete_old in content:
        content = content.replace(complete_old, complete_new)
        print("✅ Updated complete_task method with persistence notification")
    
    # Also update fail_task to notify persistence of failures
    fail_section = """                self.logger.error(f"Task {task_id} failed permanently: {error}")
            
            self._save_queue()"""
    
    fail_new = """                self.logger.error(f"Task {task_id} failed permanently: {error}")
                
                # Notify task persistence of failure
                try:
                    # Find work_item_id similar to complete_task
                    work_item_id = None
                    in_progress_tasks = self.task_persistence.in_progress_tasks
                    for wid, task_data in in_progress_tasks.items():
                        if (task_data.get('repository') == task.repository and 
                            task_data.get('issue_number') == task.issue_number):
                            work_item_id = wid
                            break
                    
                    if work_item_id:
                        failure_result = {
                            'success': False,
                            'completed': True,
                            'claude_failed': True,
                            'error': error,
                            'issue_number': task.issue_number,
                            'repository': task.repository
                        }
                        
                        if hasattr(self.task_persistence, 'complete_in_progress_task'):
                            self.task_persistence.complete_in_progress_task(work_item_id, failure_result)
                            self.logger.info(f"❌ Notified task persistence of failure for work_item {work_item_id}")
                            
                except Exception as e:
                    self.logger.error(f"Error notifying task persistence of failure: {e}")
            
            self._save_queue()"""
    
    if fail_section in content:
        content = content.replace(fail_section, fail_new)
        print("✅ Updated fail_task method with persistence notification")
    
    # Add a method to link work_item_id with Claude tasks
    if "def link_work_item" not in content:
        link_method = '''
    def link_work_item(self, claude_task_id: str, work_item_id: str):
        """Link a Claude task with its work item ID for tracking.
        
        Args:
            claude_task_id: ID of the Claude task
            work_item_id: ID of the work item
        """
        # Store mapping in task metadata
        for task in self._queue:
            if task.id == claude_task_id:
                if not task.result:
                    task.result = {}
                task.result['work_item_id'] = work_item_id
                self._save_queue()
                return
        
        for task_id, task in self._processing.items():
            if task.id == claude_task_id:
                if not task.result:
                    task.result = {}
                task.result['work_item_id'] = work_item_id
                self._save_queue()
                return'''
        
        # Add before the last method or at the end
        last_method_pos = content.rfind("def get_processing_tasks")
        if last_method_pos != -1:
            # Find the end of this method
            end_pos = content.find("\n\n", last_method_pos)
            if end_pos != -1:
                content = content[:end_pos] + "\n" + link_method + content[end_pos:]
            else:
                content += "\n" + link_method
            print("✅ Added link_work_item method")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {filepath}")

def update_local_issue_creator():
    """Update local_issue_creator.py to link work items with Claude tasks."""
    filepath = "scripts/local_issue_creator.py"
    print(f"\n🔧 Updating {filepath} to link work items...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    backup_file(filepath)
    
    # Find where Claude tasks are added to the queue
    link_pattern = """claude_task_id = claude_task.id
                        
                        print(f"✅ Claude task {claude_task.id} queued for parallel execution")"""
    
    link_replacement = """claude_task_id = claude_task.id
                        
                        # Link the Claude task with the work item for tracking
                        self.claude_queue.link_work_item(claude_task_id, work_item.id)
                        
                        print(f"✅ Claude task {claude_task.id} queued for parallel execution")"""
    
    if link_pattern in content:
        content = content.replace(link_pattern, link_replacement)
        print("✅ Added work item linking for new issues")
    
    # Also fix for existing issues
    existing_pattern = """claude_task_id = claude_task.id
                        
                        print(f"✅ Claude task {claude_task.id} queued for existing issue #{existing_issue}")"""
    
    existing_replacement = """claude_task_id = claude_task.id
                        
                        # Link the Claude task with the work item for tracking
                        self.claude_queue.link_work_item(claude_task_id, work_item.id)
                        
                        print(f"✅ Claude task {claude_task.id} queued for existing issue #{existing_issue}")"""
    
    if existing_pattern in content:
        content = content.replace(existing_pattern, existing_replacement)
        print("✅ Added work item linking for existing issues")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {filepath}")

def main():
    """Run all fixes."""
    print("🚀 Starting Claude completion notification fix...")
    print("=" * 60)
    
    # Check we're in the right directory
    if not os.path.exists("scripts/claude_execution_queue.py"):
        print("❌ Error: Must run from project root directory")
        sys.exit(1)
    
    # Apply fixes
    fix_claude_execution_queue()
    update_local_issue_creator()
    
    print("\n" + "=" * 60)
    print("✅ All fixes applied successfully!")
    print("\n📝 Summary of changes:")
    print("1. ClaudeExecutionQueue now notifies TaskPersistence when tasks complete")
    print("2. LocalIssueCreator links work items with Claude tasks for tracking")
    print("3. Task completion is properly tracked through the entire lifecycle")
    print("\n🔄 Next steps:")
    print("1. Run fix_task_completion_tracking.py first if not already done")
    print("2. Test the system to ensure tasks are only marked complete after Claude execution")
    print("3. Monitor the in_progress_tasks.json file to see tasks moving through stages")

if __name__ == "__main__":
    main()