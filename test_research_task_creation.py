#!/usr/bin/env python3
"""
Test that research tasks are being created when workers are idle.
"""

import asyncio
import os
import sys

# Set environment variables
os.environ['DISCOVERY_INTERVAL_SECONDS'] = '10'  # Quick discovery for testing

async def test_research_tasks():
    """Test research task creation."""
    from scripts.intelligent_work_finder import IntelligentWorkFinder
    from scripts.ai_brain import IntelligentAIBrain
    from scripts.state_manager import StateManager
    
    # Initialize components
    state_manager = StateManager()
    system_state = state_manager.load_state_with_repository_discovery()
    ai_brain = IntelligentAIBrain(system_state, {})
    
    # Create work finder
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    print("Testing research task creation...")
    print("=" * 60)
    
    # Test 1: Discover work with high capacity (simulating idle workers)
    print("\nTest 1: High capacity (10 slots available)")
    work_items = await work_finder.discover_work(max_items=10, current_workload=0)
    
    print(f"Found {len(work_items)} work items:")
    research_count = 0
    for item in work_items:
        is_research = (item.repository is None or 
                      item.metadata.get('no_repository_needed', False) or
                      item.task_type in ['MARKET_RESEARCH', 'PORTFOLIO_ANALYSIS', 'PROJECT_CREATION'])
        if is_research:
            research_count += 1
            print(f"  ✅ RESEARCH: {item.task_type} - {item.title}")
        else:
            print(f"  📦 Repository: {item.task_type} - {item.title} ({item.repository})")
    
    print(f"\nResearch tasks: {research_count}/{len(work_items)}")
    
    # Test 2: Check _create_research_work_items directly
    print("\n" + "=" * 60)
    print("Test 2: Direct research task creation")
    
    try:
        research_opportunities = await work_finder._create_research_work_items(count=3)
        print(f"Created {len(research_opportunities)} research opportunities:")
        
        for opp in research_opportunities:
            print(f"  - {opp.type}: {opp.title}")
            print(f"    Metadata: {opp.metadata}")
            
    except Exception as e:
        print(f"Error creating research tasks: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check if research tasks can be converted to work items
    print("\n" + "=" * 60)
    print("Test 3: Convert research opportunities to work items")
    
    if research_opportunities:
        for opp in research_opportunities[:1]:  # Just test one
            work_item = opp.to_work_item()
            print(f"Work Item:")
            print(f"  ID: {work_item.id}")
            print(f"  Type: {work_item.task_type}")
            print(f"  Title: {work_item.title}")
            print(f"  Repository: {work_item.repository}")
            print(f"  Metadata: {work_item.metadata}")
    
    return research_count > 0

async def main():
    """Run the test."""
    try:
        success = await test_research_tasks()
        if success:
            print("\n✅ Research task creation is working!")
            return 0
        else:
            print("\n❌ No research tasks were created")
            return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))