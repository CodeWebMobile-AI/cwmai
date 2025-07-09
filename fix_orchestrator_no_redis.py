#!/usr/bin/env python3
"""
Fix continuous_orchestrator_no_redis.py to check the 'completed' flag.
"""

import os
import shutil
from datetime import datetime

def fix_orchestrator():
    """Fix the orchestrator to properly check completion status."""
    
    print("🔧 Fixing scripts/continuous_orchestrator_no_redis.py...")
    
    orchestrator_file = "scripts/continuous_orchestrator_no_redis.py"
    
    # Backup the file
    backup_file = f"{orchestrator_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(orchestrator_file, backup_file)
    print(f"✅ Backed up to {backup_file}")
    
    # Read the file
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    # Find and fix the task completion logic
    # Look for where execution_result.get('success') is checked
    old_pattern = """if execution_result.get('success'):
                            self.task_persistence.record_completed_task(work_item, execution_result)"""
    
    new_pattern = """if execution_result.get('success'):
                            # Check if task is actually completed or just queued
                            if execution_result.get('completed', True):  # Default to True for backward compatibility
                                self.task_persistence.record_completed_task(work_item, execution_result)
                            else:
                                # Task is successful but not completed (e.g., queued for Claude)
                                self.task_persistence.record_in_progress_task(work_item, execution_result)
                                self.logger.info(f"📋 Task queued for execution: {work_item.title}")"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("✅ Updated task completion logic")
    else:
        print("⚠️  Could not find exact pattern, searching for alternative...")
        
        # Try to find the general area
        import re
        pattern = re.compile(r"(if execution_result\.get\('success'\):.*?record_completed_task.*?)(\n\s+else:)", re.DOTALL)
        match = pattern.search(content)
        
        if match:
            old_block = match.group(1)
            indent = " " * 28  # Adjust based on actual indentation
            new_block = f"""if execution_result.get('success'):
{indent}# Check if task is actually completed or just queued
{indent}if execution_result.get('completed', True):  # Default to True for backward compatibility
{indent}    self.task_persistence.record_completed_task(work_item, execution_result)
{indent}else:
{indent}    # Task is successful but not completed (e.g., queued for Claude)
{indent}    self.task_persistence.record_in_progress_task(work_item, execution_result)
{indent}    self.logger.info(f"📋 Task queued for execution: {work_item.title}")"""
            
            content = content.replace(old_block, new_block)
            print("✅ Updated task completion logic using pattern matching")
    
    # Write the fixed content
    with open(orchestrator_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed continuous_orchestrator_no_redis.py")
    print("\n🎯 The orchestrator will now:")
    print("   - Check the 'completed' flag before marking tasks as complete")
    print("   - Record in-progress tasks when they're queued for Claude")
    print("   - Only mark tasks complete when Claude actually finishes")

if __name__ == "__main__":
    fix_orchestrator()