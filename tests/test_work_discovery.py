#!/usr/bin/env python3
"""
Test work discovery specifically to find the enum issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

async def test_work_discovery():
    """Test work discovery in isolation."""
    print("=== Testing Work Discovery ===")
    
    try:
        from intelligent_work_finder import IntelligentWorkFinder
        from ai_brain import IntelligentAIBrain
        from work_item_types import TaskPriority, WorkOpportunity
        
        # Create minimal state
        system_state = {'projects': {}}
        ai_brain = IntelligentAIBrain(system_state, {})
        
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        print("✓ Work finder initialized")
        
        # Test research opportunities specifically
        print("Testing research opportunities discovery...")
        research_ops = await work_finder._discover_research_opportunities()
        print(f"✓ Found {len(research_ops)} research opportunities")
        
        # Test converting to work items
        print("Testing work item conversion...")
        for i, opp in enumerate(research_ops):
            print(f"  Opportunity {i+1}: {opp.type} - {opp.title}")
            work_item = opp.to_work_item()
            print(f"    → Work item: {work_item.task_type} - {work_item.title}")
            print(f"    → Priority: {work_item.priority}")
        
        # Test full discovery
        print("\nTesting full work discovery...")
        all_work = await work_finder.discover_work(max_items=3)
        print(f"✓ Discovered {len(all_work)} work items total")
        
        for work_item in all_work:
            print(f"  - {work_item.task_type}: {work_item.title} (priority: {work_item.priority})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in work discovery test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_work_discovery())
    print(f"\nTest {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)