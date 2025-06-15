#!/usr/bin/env python3
"""
Test script to verify duplicate task handling improvements.
"""

import asyncio
import sys
sys.path.append('/workspaces/cwmai')

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.task_persistence import TaskPersistence
from scripts.alternative_task_generator import AlternativeTaskGenerator
from scripts.ai_brain import IntelligentAIBrain
from datetime import datetime, timezone


async def test_duplicate_handling():
    """Test the duplicate handling system."""
    print("=== Testing Duplicate Task Handling ===\n")
    
    # Initialize components
    task_persistence = TaskPersistence()
    system_state = {'projects': {}}
    ai_brain = IntelligentAIBrain(system_state, {})
    alt_generator = AlternativeTaskGenerator(ai_brain)
    
    # Create a test task that would be a duplicate
    test_task = WorkItem(
        id="test_task_1",
        task_type="DOCUMENTATION",
        title="Improve ai-creative-studio documentation",
        description="Update and improve the documentation for ai-creative-studio",
        priority=TaskPriority.MEDIUM,
        repository="ai-creative-studio",
        estimated_cycles=2,
        metadata={'test': True}
    )
    
    print("1. Testing duplicate detection:")
    is_dup = task_persistence.is_duplicate_task(test_task)
    print(f"   - Is 'Improve ai-creative-studio documentation' a duplicate? {is_dup}")
    
    if is_dup:
        print("\n2. Testing alternative task generation:")
        print("   - Generating alternative task...")
        alternative = await alt_generator.generate_alternative_task(
            test_task,
            context={'repository': test_task.repository}
        )
        
        if alternative:
            print(f"   - Alternative task: {alternative.title}")
            print(f"   - Task type: {alternative.task_type}")
            print(f"   - Description: {alternative.description[:100]}...")
        else:
            print("   - Failed to generate alternative task")
    
    # Test with another common duplicate
    test_task2 = WorkItem(
        id="test_task_2",
        task_type="DOCUMENTATION",
        title="Update documentation for moderncms-with-ai-powered-content-recommendations changes",
        description="Update docs for the CMS",
        priority=TaskPriority.MEDIUM,
        repository="moderncms-with-ai-powered-content-recommendations",
        estimated_cycles=2
    )
    
    print("\n3. Testing another duplicate:")
    is_dup2 = task_persistence.is_duplicate_task(test_task2)
    print(f"   - Is 'Update documentation for moderncms...' a duplicate? {is_dup2}")
    
    # Test the skip tracking
    print("\n4. Testing skip tracking:")
    completed_tasks = task_persistence.get_completed_tasks()
    skipped_tasks = task_persistence.get_skipped_tasks()
    print(f"   - Completed tasks: {len(completed_tasks)}")
    print(f"   - Skipped tasks: {len(skipped_tasks)}")
    
    # Show some skipped task details
    if skipped_tasks:
        print("\n   Recent skipped tasks:")
        for title, info in list(skipped_tasks.items())[:5]:
            print(f"   - {title}: skipped {info.get('count', 0)} times, reason: {info.get('reason', 'unknown')}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_duplicate_handling())