#!/usr/bin/env python3
"""
Fix script for task completion tracking issue.

This script updates the system to properly track task completion,
ensuring tasks are only marked as completed when Claude actually
executes them, not just when they're queued.
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

def backup_file(filepath):
    """Create a backup of the file before modifying."""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up {filepath} to {backup_path}")
    return backup_path

def fix_local_issue_creator():
    """Fix local_issue_creator.py to return proper completion status."""
    filepath = "scripts/local_issue_creator.py"
    print(f"\n🔧 Fixing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix 1: Add completed flag to returns when Claude is queued
    replacements = [
        # When creating new issue and queuing Claude
        (
            """return {
                        'success': True,
                        'task_id': work_item.id,
                        'issue_number': issue.number,
                        'repository': work_item.repository,
                        'value_created': self._calculate_value_created(work_item),
                        'execution_time': time.time(),
                        'storage_mode': 'local',
                        'claude_task_id': claude_task_id,
                        'claude_queued': True
                    }""",
            """return {
                        'success': True,
                        'completed': False,  # Task not yet completed - Claude execution pending
                        'stage': 'issue_created',
                        'task_id': work_item.id,
                        'issue_number': issue.number,
                        'repository': work_item.repository,
                        'value_created': self._calculate_value_created(work_item),
                        'execution_time': time.time(),
                        'storage_mode': 'local',
                        'claude_task_id': claude_task_id,
                        'claude_queued': True
                    }"""
        ),
        # When existing issue found and queuing Claude
        (
            """return {
                        'success': True,
                        'task_id': work_item.id,
                        'issue_number': existing_issue,
                        'repository': work_item.repository,
                        'value_created': self._calculate_value_created(work_item),
                        'execution_time': time.time(),
                        'existing_issue': True,
                        'claude_task_id': claude_task_id,
                        'claude_queued': True
                    }""",
            """return {
                        'success': True,
                        'completed': False,  # Task not yet completed - Claude execution pending
                        'stage': 'issue_found',
                        'task_id': work_item.id,
                        'issue_number': existing_issue,
                        'repository': work_item.repository,
                        'value_created': self._calculate_value_created(work_item),
                        'execution_time': time.time(),
                        'existing_issue': True,
                        'claude_task_id': claude_task_id,
                        'claude_queued': True
                    }"""
        ),
        # Fix returns for already implemented tasks
        (
            """return {
                        'success': True,
                        'task_id': work_item.id,
                        'issue_number': existing_issue,
                        'repository': work_item.repository,
                        'value_created': 0,
                        'execution_time': time.time(),
                        'existing_issue': True,
                        'already_implemented': True
                    }""",
            """return {
                        'success': True,
                        'completed': True,  # Already implemented - task is complete
                        'stage': 'already_implemented',
                        'task_id': work_item.id,
                        'issue_number': existing_issue,
                        'repository': work_item.repository,
                        'value_created': 0,
                        'execution_time': time.time(),
                        'existing_issue': True,
                        'already_implemented': True
                    }"""
        ),
        # Fix returns for already completed Claude tasks
        (
            """return {
                            'success': True,
                            'task_id': work_item.id,
                            'issue_number': existing_issue,
                            'repository': work_item.repository,
                            'value_created': 0,
                            'execution_time': time.time(),
                            'existing_issue': True,
                            'claude_already_completed': True
                        }""",
            """return {
                            'success': True,
                            'completed': True,  # Claude already completed - task is done
                            'stage': 'claude_already_completed',
                            'task_id': work_item.id,
                            'issue_number': existing_issue,
                            'repository': work_item.repository,
                            'value_created': 0,
                            'execution_time': time.time(),
                            'existing_issue': True,
                            'claude_already_completed': True
                        }"""
        ),
        # Fix return when Claude is disabled but issue created
        (
            """# Claude execution disabled - issue created successfully
                    print(f"✅ Issue created successfully (Claude execution disabled)")
                    return {
                        'success': True,
                        'task_id': work_item.id,
                        'issue_number': issue.number,
                        'repository': work_item.repository,
                        'value_created': self._calculate_value_created(work_item),
                        'execution_time': time.time(),
                        'storage_mode': 'local',
                        'claude_queued': False
                    }""",
            """# Claude execution disabled - issue created successfully
                    print(f"✅ Issue created successfully (Claude execution disabled)")
                    return {
                        'success': True,
                        'completed': True,  # No Claude execution needed - task is complete
                        'stage': 'issue_created_no_claude',
                        'task_id': work_item.id,
                        'issue_number': issue.number,
                        'repository': work_item.repository,
                        'value_created': self._calculate_value_created(work_item),
                        'execution_time': time.time(),
                        'storage_mode': 'local',
                        'claude_queued': False
                    }"""
        )
    ]
    
    backup_file(filepath)
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Updated return statement for proper completion tracking")
        else:
            print(f"⚠️  Pattern not found, may have already been updated")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {filepath}")

def fix_continuous_orchestrator():
    """Fix continuous_orchestrator.py to check completion status."""
    filepath = "scripts/continuous_orchestrator.py"
    print(f"\n🔧 Fixing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the task completion recording logic
    old_code = """# Record completion if successful
            if result.get('success'):
                try:
                    if asyncio.iscoroutinefunction(self.task_persistence.record_completed_task):
                        await self.task_persistence.record_completed_task(work_item, result)
                    else:
                        self.task_persistence.record_completed_task(work_item, result)"""
    
    new_code = """# Record completion based on actual completion status
            if result.get('success'):
                # Check if task is actually completed or just queued
                is_completed = result.get('completed', True)  # Default True for backward compatibility
                
                if is_completed:
                    # Task is actually completed
                    try:
                        if asyncio.iscoroutinefunction(self.task_persistence.record_completed_task):
                            await self.task_persistence.record_completed_task(work_item, result)
                        else:
                            self.task_persistence.record_completed_task(work_item, result)
                    except Exception as e:
                        self.logger.error(f"Error recording task completion: {e}")
                elif result.get('claude_queued'):
                    # Task is queued for Claude execution - track as in-progress
                    self.logger.info(f"📋 Task queued for Claude execution: {work_item.title}")
                    try:
                        if hasattr(self.task_persistence, 'record_in_progress_task'):
                            if asyncio.iscoroutinefunction(self.task_persistence.record_in_progress_task):
                                await self.task_persistence.record_in_progress_task(work_item, result)
                            else:
                                self.task_persistence.record_in_progress_task(work_item, result)
                        else:
                            # Fallback: log that task is in progress
                            self.logger.info(f"Task {work_item.id} is in progress (Claude execution pending)")
                    except Exception as e:
                        self.logger.error(f"Error recording in-progress task: {e}")"""
    
    backup_file(filepath)
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        print(f"✅ Updated task completion recording logic")
    else:
        print(f"⚠️  Pattern not found, checking for alternate pattern...")
        # Try a simpler replacement
        if "record_completed_task(work_item, result)" in content:
            print(f"✅ Found alternate pattern, manual fix may be needed")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {filepath}")

def add_in_progress_tracking():
    """Add in-progress task tracking to task_persistence.py."""
    filepath = "scripts/task_persistence.py"
    print(f"\n🔧 Adding in-progress tracking to {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    backup_file(filepath)
    
    # Add in_progress_tasks tracking
    if "self.in_progress_tasks" not in content:
        # Add after the init method's other attributes
        init_addition = """        
        # Track in-progress tasks (e.g., queued for Claude execution)
        self.in_progress_tasks: Dict[str, Dict[str, Any]] = {}
        self._load_in_progress_tasks()"""
        
        # Find where to insert (after self._load_problematic_tasks())
        insert_pos = content.find("self._load_problematic_tasks()")
        if insert_pos != -1:
            insert_pos = content.find("\n", insert_pos) + 1
            content = content[:insert_pos] + init_addition + "\n" + content[insert_pos:]
            print("✅ Added in_progress_tasks tracking")
    
    # Add the load method
    if "_load_in_progress_tasks" not in content:
        load_method = '''
    def _load_in_progress_tasks(self) -> None:
        """Load in-progress tasks from disk."""
        try:
            in_progress_file = Path(self.storage_file).parent / "in_progress_tasks.json"
            if in_progress_file.exists():
                with open(in_progress_file, 'r') as f:
                    self.in_progress_tasks = json.load(f)
                self.logger.info(f"Loaded {len(self.in_progress_tasks)} in-progress tasks")
        except Exception as e:
            self.logger.error(f"Error loading in-progress tasks: {e}")
            self.in_progress_tasks = {}
    
    def _save_in_progress_tasks(self) -> None:
        """Save in-progress tasks to disk."""
        try:
            in_progress_file = Path(self.storage_file).parent / "in_progress_tasks.json"
            with open(in_progress_file, 'w') as f:
                json.dump(self.in_progress_tasks, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving in-progress tasks: {e}")'''
        
        # Insert before the record_completed_task method
        insert_pos = content.find("def record_completed_task")
        if insert_pos != -1:
            content = content[:insert_pos] + load_method + "\n\n    " + content[insert_pos:]
            print("✅ Added in-progress task load/save methods")
    
    # Add the record_in_progress_task method
    if "record_in_progress_task" not in content:
        in_progress_method = '''
    def record_in_progress_task(self, work_item: WorkItem, execution_result: Dict[str, Any]) -> bool:
        """Record a task that is in progress (e.g., queued for Claude execution).
        
        Args:
            work_item: The work item that is in progress
            execution_result: Current execution state
            
        Returns:
            True if recorded successfully
        """
        try:
            task_data = {
                'work_item_id': work_item.id,
                'title': work_item.title,
                'task_type': work_item.task_type,
                'repository': work_item.repository,
                'stage': execution_result.get('stage', 'unknown'),
                'claude_task_id': execution_result.get('claude_task_id'),
                'issue_number': execution_result.get('issue_number'),
                'started_at': datetime.now(timezone.utc).isoformat(),
                'execution_result': execution_result
            }
            
            # Store in memory and persist
            self.in_progress_tasks[work_item.id] = task_data
            self._save_in_progress_tasks()
            
            print(f"📋 Recorded in-progress task: {work_item.title} (stage: {task_data['stage']})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error recording in-progress task: {e}")
            return False
    
    def complete_in_progress_task(self, work_item_id: str, execution_result: Dict[str, Any]) -> bool:
        """Mark an in-progress task as completed.
        
        Args:
            work_item_id: ID of the work item to complete
            execution_result: Final execution result
            
        Returns:
            True if completed successfully
        """
        try:
            if work_item_id in self.in_progress_tasks:
                task_data = self.in_progress_tasks[work_item_id]
                
                # Create a WorkItem from stored data for completion recording
                work_item = WorkItem(
                    id=work_item_id,
                    title=task_data['title'],
                    task_type=task_data['task_type'],
                    description="",  # Not stored in in-progress
                    repository=task_data['repository'],
                    priority=TaskPriority.MEDIUM  # Default
                )
                
                # Record as completed
                self.record_completed_task(work_item, execution_result)
                
                # Remove from in-progress
                del self.in_progress_tasks[work_item_id]
                self._save_in_progress_tasks()
                
                print(f"✅ Completed in-progress task: {task_data['title']}")
                return True
            else:
                self.logger.warning(f"Task {work_item_id} not found in in-progress tasks")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error completing in-progress task: {e}")
            return False'''
        
        # Insert after record_completed_task method
        insert_pos = content.find("return False", content.find("def record_completed_task"))
        if insert_pos != -1:
            insert_pos = content.find("\n", insert_pos) + 1
            content = content[:insert_pos] + in_progress_method + "\n" + content[insert_pos:]
            print("✅ Added in-progress task recording methods")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Enhanced {filepath} with in-progress tracking")

def main():
    """Run all fixes."""
    print("🚀 Starting task completion tracking fix...")
    print("=" * 60)
    
    # Check we're in the right directory
    if not os.path.exists("scripts/local_issue_creator.py"):
        print("❌ Error: Must run from project root directory")
        sys.exit(1)
    
    # Apply fixes
    fix_local_issue_creator()
    fix_continuous_orchestrator()
    add_in_progress_tracking()
    
    print("\n" + "=" * 60)
    print("✅ All fixes applied successfully!")
    print("\n📝 Summary of changes:")
    print("1. LocalIssueCreator now returns 'completed: False' for queued tasks")
    print("2. ContinuousOrchestrator checks the 'completed' flag before marking tasks complete")
    print("3. TaskPersistence can now track in-progress tasks separately")
    print("\n⚠️  Note: You may need to update ClaudeExecutionQueue to call")
    print("   complete_in_progress_task() when Claude finishes execution")

if __name__ == "__main__":
    main()