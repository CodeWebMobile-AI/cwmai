#!/usr/bin/env python3
"""
Test the continuous orchestrator integration with staged improvements
"""

import os
import sys
import asyncio

# Set environment variables
os.environ['SELF_IMPROVEMENT_STAGING_ENABLED'] = 'true'
os.environ['SELF_IMPROVEMENT_AUTO_VALIDATE'] = 'true'
os.environ['SELF_IMPROVEMENT_AUTO_APPLY_VALIDATED'] = 'false'
os.environ['SELF_IMPROVEMENT_MAX_DAILY'] = '3'

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from continuous_orchestrator import ContinuousOrchestrator
from work_item_types import WorkItem, TaskPriority
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


async def test_improvement_task():
    """Test executing a system improvement task."""
    print("🧪 Testing Continuous Orchestrator Integration\n")
    
    # Create orchestrator (single worker for testing)
    orchestrator = ContinuousOrchestrator(max_workers=1, enable_parallel=False)
    
    # Create a system improvement work item
    work_item = WorkItem(
        id="test_improvement_001",
        task_type="SYSTEM_IMPROVEMENT",
        title="Test staged improvement integration",
        description="Testing if staged improvements work with continuous orchestrator",
        priority=TaskPriority.HIGH,
        repository=None,  # System-wide
        estimated_cycles=1
    )
    
    print("📋 Created work item:", work_item.title)
    print(f"   Type: {work_item.task_type}")
    print(f"   Priority: {work_item.priority.name}\n")
    
    # Execute the improvement task
    print("🚀 Executing improvement task...\n")
    
    try:
        result = await orchestrator._execute_system_improvement_task(work_item)
        
        print("\n✅ Task completed successfully!")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Staged: {result.get('staged', 0)} improvements")
        print(f"   Validated: {result.get('validated', 0)} improvements")
        print(f"   Applied: {result.get('applied', 0)} improvements")
        print(f"   Total opportunities: {result.get('total_opportunities', 0)}")
        print(f"   Confidence score: {result.get('confidence_score', 0):.2f}")
        print(f"   Value created: {result.get('value_created', 0):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error executing task: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    print("="*60)
    print("CONTINUOUS ORCHESTRATOR INTEGRATION TEST")
    print("="*60)
    print()
    
    success = await test_improvement_task()
    
    print("\n" + "="*60)
    print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)