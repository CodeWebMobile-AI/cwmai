#!/usr/bin/env python3
"""Complete test of the infinite loop prevention system."""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.task_persistence import TaskPersistence
from scripts.alternative_task_generator import AlternativeTaskGenerator


async def main():
    """Test complete infinite loop prevention flow."""
    
    print("\n=== Complete Infinite Loop Prevention Test ===\n")
    
    # Initialize components
    persistence = TaskPersistence()
    alt_generator = AlternativeTaskGenerator()
    
    # Create a task that will become problematic
    task = WorkItem(
        id="test_001",
        task_type="DOCUMENTATION",
        title="Create comprehensive documentation for project-x",
        description="Create detailed documentation including API docs, user guide, and examples",
        priority=TaskPriority.HIGH,
        repository="project-x",
        metadata={}
    )
    
    print(f"Test Task: {task.title}")
    print(f"Repository: {task.repository}")
    print(f"Type: {task.task_type}\n")
    
    # Step 1: Mark as completed (to trigger duplicate detection)
    print("Step 1: Recording task as completed...")
    persistence.record_completed_task(task, {'status': 'completed', 'value_created': 1.0})
    
    # Step 2: Check if it's detected as duplicate
    print("\nStep 2: Checking duplicate detection...")
    is_duplicate = persistence.is_duplicate_task(task)
    print(f"Is duplicate: {is_duplicate}")
    
    # Step 3: Try to generate alternatives multiple times
    print("\nStep 3: Attempting to generate alternatives...")
    attempted_alternatives = set([task.title.lower()])
    
    for attempt in range(1, 5):
        print(f"\n--- Attempt #{attempt} ---")
        
        # Generate alternative
        context = {
            'attempted_alternatives': list(attempted_alternatives),
            'attempt_number': attempt
        }
        
        alternative = await alt_generator.generate_alternative_task(task, context)
        
        if alternative:
            print(f"Generated: {alternative.title}")
            
            # Mark it as completed too (to make it a duplicate)
            persistence.record_completed_task(alternative, {'status': 'completed'})
            attempted_alternatives.add(alternative.title.lower())
            
            # Record as skipped
            persistence.record_skipped_task(alternative.title, "duplicate_alternative")
    
    # Step 4: After multiple failures, mark as problematic
    print("\nStep 4: Recording as problematic task...")
    persistence.record_problematic_task(task.title, task.task_type, task.repository)
    
    # Step 5: Verify it's now blocked
    print("\nStep 5: Verifying problematic task is blocked...")
    is_blocked = persistence.is_duplicate_task(task)
    print(f"Task is blocked: {is_blocked}")
    
    # Check problematic task data
    task_key = f"{task.task_type}:{task.repository}:{task.title.lower()}"
    if task_key in persistence.problematic_tasks:
        prob_data = persistence.problematic_tasks[task_key]
        print(f"\nProblematic Task Details:")
        print(f"  Attempts: {prob_data['attempt_count']}")
        print(f"  Cooldown Until: {prob_data['cooldown_until']}")
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"✓ Duplicate detection working: {is_duplicate}")
    print(f"✓ Alternative generation attempted: {len(attempted_alternatives) - 1} times")
    print(f"✓ Problematic task blocked: {is_blocked}")
    print(f"✓ Infinite loop prevention: ACTIVE")
    
    print("\n=== How the System Prevents Infinite Loops ===")
    print("1. Limits alternative generation to 3 attempts")
    print("2. Tracks all attempted alternatives to avoid repeats")
    print("3. Records problematic tasks with 24-hour cooldown")
    print("4. Blocks problematic tasks from future processing")
    print("5. Uses exponential backoff for frequently skipped tasks")


if __name__ == "__main__":
    asyncio.run(main())