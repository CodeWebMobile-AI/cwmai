#!/usr/bin/env python3
"""
Test complete flow to ensure no JSON serialization errors.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.unified_task_state import UnifiedTaskState
from scripts.in_memory_work_queue import InMemoryWorkQueue
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.gemini_deep_research import MarketOpportunity
from dataclasses import asdict


async def test_complete_flow():
    """Test the complete flow with MarketOpportunity objects."""
    print("🧪 Testing Complete Flow with JSON Serialization")
    print("=" * 50)
    
    try:
        # 1. Test UnifiedTaskState
        print("\n1. Testing UnifiedTaskState...")
        task_state = UnifiedTaskState()
        
        # Create a work item with MarketOpportunity in metadata
        opp = MarketOpportunity(
            problem_description="Test problem",
            affected_audience="Test users",
            current_solutions="Current solutions",
            opportunity="Test opportunity",
            monetization_model="SaaS",
            market_size="$1B",
            competition_level="Medium",
            validation_sources=["source1", "source2"],
            key_insights=["insight1"],
            implementation_difficulty="Medium",
            potential_revenue="$100k/month",
            unique_advantages=["advantage1"],
            search_evidence={"test": "data"}
        )
        
        work_item = WorkItem(
            id="test_123",
            task_type="NEW_PROJECT",
            title="Test Project",
            description="Test Description",
            priority=TaskPriority.HIGH,
            metadata={
                "full_opportunity": asdict(opp),  # Convert to dict
                "raw_opportunity": opp  # This should be handled by custom encoder
            }
        )
        
        # Register the task
        task_state.register_task(work_item)
        
        # Save state
        task_state._save_state()
        print("   ✓ Task registered and state saved successfully")
        
        # 2. Test InMemoryWorkQueue
        print("\n2. Testing InMemoryWorkQueue...")
        work_queue = InMemoryWorkQueue()
        
        # Add work item with MarketOpportunity
        # Convert WorkItem to dict for enqueue
        work_dict = {
            'id': work_item.id,
            'task_type': work_item.task_type,
            'title': work_item.title,
            'description': work_item.description,
            'priority': work_item.priority.value,
            'metadata': work_item.metadata
        }
        work_queue.enqueue(work_dict)
        print("   ✓ Work item added to queue")
        
        # Save queue to disk (this tests JSON serialization)
        work_queue._save_to_disk()
        print("   ✓ Queue saved to disk successfully")
        
        # 3. Test loading back
        print("\n3. Testing Load Back...")
        
        # Create new instances
        new_task_state = UnifiedTaskState()
        new_task_state._load_state()
        print("   ✓ Task state loaded successfully")
        
        new_queue = InMemoryWorkQueue()
        new_queue._load_from_disk()
        print("   ✓ Queue loaded successfully")
        
        print("\n✅ All flow tests passed - JSON serialization working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_complete_flow())
    sys.exit(0 if result else 1)