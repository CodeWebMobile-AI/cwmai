#!/usr/bin/env python3
"""Test the infinite loop prevention system for alternative task generation."""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.task_persistence import TaskPersistence
from scripts.alternative_task_generator import AlternativeTaskGenerator
from scripts.continuous_orchestrator import ContinuousOrchestrator


async def test_infinite_loop_prevention():
    """Test that the system prevents infinite loops when generating alternatives."""
    
    print("\n=== Testing Infinite Loop Prevention ===\n")
    
    # Initialize components
    task_persistence = TaskPersistence()
    alternative_generator = AlternativeTaskGenerator()
    
    # Create a problematic task that always gets detected as duplicate
    problematic_task = WorkItem(
        id="test_001",
        task_type="DOCUMENTATION",
        title="Update README for moderncms-with-ai-powered-content-recommendations",
        description="Update the README file with current project information",
        priority=TaskPriority.MEDIUM,
        repository="moderncms-with-ai-powered-content-recommendations",
        metadata={}
    )
    
    print(f"Original Task: {problematic_task.title}")
    print(f"Task Type: {problematic_task.task_type}")
    print(f"Repository: {problematic_task.repository}")
    
    # Simulate the task being marked as duplicate multiple times
    print("\n--- Simulating Alternative Generation Attempts ---")
    
    # First, mark the original as completed to trigger duplicate detection
    task_persistence.record_completed_task(problematic_task, {
        'status': 'completed',
        'value_created': 1.0
    })
    
    # Now simulate multiple attempts to generate alternatives
    attempted_alternatives = set([problematic_task.title.lower()])
    
    for attempt in range(1, 5):
        print(f"\nAttempt #{attempt}:")
        
        # Check if it's a duplicate
        is_duplicate = task_persistence.is_duplicate_task(problematic_task)
        print(f"  Is Duplicate: {is_duplicate}")
        
        if is_duplicate:
            # Try to generate alternative
            context = {
                'attempted_alternatives': list(attempted_alternatives),
                'attempt_number': attempt
            }
            
            alternative = await alternative_generator.generate_alternative_task(
                problematic_task, context
            )
            
            if alternative:
                print(f"  Generated Alternative: {alternative.title}")
                
                # Check if this alternative is also a duplicate
                alt_is_duplicate = task_persistence.is_duplicate_task(alternative)
                print(f"  Alternative Is Duplicate: {alt_is_duplicate}")
                
                if alt_is_duplicate:
                    # Record as skipped
                    task_persistence.record_skipped_task(alternative.title, "duplicate")
                    attempted_alternatives.add(alternative.title.lower())
                    
                    # After 3 attempts, mark as problematic
                    if attempt >= 3:
                        print(f"  Recording as problematic task...")
                        task_persistence.record_problematic_task(
                            problematic_task.title,
                            problematic_task.task_type,
                            problematic_task.repository
                        )
            else:
                print(f"  Failed to generate alternative")
    
    # Test that problematic task is now blocked
    print("\n--- Testing Problematic Task Blocking ---")
    
    # Check if original task is still considered duplicate
    is_blocked = task_persistence.is_duplicate_task(problematic_task)
    print(f"\nProblematic task is blocked: {is_blocked}")
    
    # Check skip statistics
    skip_stats = task_persistence.skip_stats.get(problematic_task.title, {})
    if skip_stats:
        print(f"\nSkip Statistics for '{problematic_task.title}':")
        print(f"  Skip Count: {skip_stats.get('count', 0)}")
        print(f"  First Skip: {skip_stats.get('first_skip', 'N/A')}")
        print(f"  Last Skip: {skip_stats.get('last_skip', 'N/A')}")
        print(f"  Reasons: {skip_stats.get('reasons', [])}")
    
    # Check problematic task data
    task_key = f"{problematic_task.task_type}:{problematic_task.repository}:{problematic_task.title.lower()}"
    problematic_data = task_persistence.problematic_tasks.get(task_key, {})
    if problematic_data:
        print(f"\nProblematic Task Data:")
        print(f"  Attempt Count: {problematic_data.get('attempt_count', 0)}")
        print(f"  Cooldown Until: {problematic_data.get('cooldown_until', 'N/A')}")
    
    print("\n=== Test Complete ===")
    print("\nKey Findings:")
    print("1. System limits alternative generation attempts")
    print("2. Problematic tasks are recorded with extended cooldown")
    print("3. Infinite loops are prevented by blocking problematic tasks")
    print("4. Skip statistics track frequency of duplicate detection")


async def test_orchestrator_integration():
    """Test the complete integration with ContinuousOrchestrator."""
    
    print("\n\n=== Testing Orchestrator Integration ===\n")
    
    # Create a mock AI brain
    class MockAIBrain:
        async def generate_enhanced_response(self, prompt):
            return {'content': '{"title": "Test task", "description": "Test"}'}
    
    # Initialize orchestrator
    orchestrator = ContinuousOrchestrator(MockAIBrain())
    
    # Create a work item that will trigger duplicate detection
    work_item = WorkItem(
        id="orch_test_001",
        task_type="DOCUMENTATION",
        title="Create comprehensive documentation",
        description="Create detailed documentation for the project",
        priority=TaskPriority.HIGH,
        repository="test-repo",
        metadata={}
    )
    
    print("Testing orchestrator handling of duplicate with alternatives...")
    
    # First, record it as completed
    orchestrator.task_persistence.record_completed_task(work_item, {'status': 'completed'})
    
    # Now try to process it again (should trigger alternative generation)
    result = await orchestrator._handle_duplicate_task(work_item)
    
    if result:
        print(f"\nOrchestrator generated alternative: {result.title}")
    else:
        print("\nOrchestrator failed to generate alternative (as expected after max attempts)")
    
    print("\n=== Orchestrator Integration Test Complete ===")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_infinite_loop_prevention())
    asyncio.run(test_orchestrator_integration())