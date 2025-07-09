#!/usr/bin/env python3
"""
Test that MarketOpportunity serialization is fixed in the full flow.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.market_discovery import MarketDiscovery
from scripts.intelligent_work_finder import IntelligentWorkFinder
from scripts.ai_brain import IntelligentAIBrain


async def test_serialization():
    """Test that the full flow works without serialization errors."""
    print("🧪 Testing MarketOpportunity Serialization Fix")
    print("=" * 50)
    
    try:
        # 1. Test Market Discovery
        print("\n1. Testing Market Discovery...")
        market_discovery = MarketDiscovery()
        problems = await market_discovery.discover_problems(count=1)
        
        if problems:
            print(f"   ✓ Discovered {len(problems)} problems")
            problem = problems[0]
            
            # Check that the problem has the special attributes
            has_full_opp = hasattr(problem, '_full_opportunity')
            has_phases = hasattr(problem, '_research_phases')
            print(f"   ✓ Has _full_opportunity: {has_full_opp}")
            print(f"   ✓ Has _research_phases: {has_phases}")
        else:
            print("   ✗ No problems discovered")
            return False
            
        # 2. Test Intelligent Work Finder
        print("\n2. Testing Intelligent Work Finder...")
        ai_brain = IntelligentAIBrain()
        system_state = {'projects': {}, 'repositories': {}}
        
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        work_items = await work_finder.discover_work(max_items=1)
        
        if work_items:
            print(f"   ✓ Generated {len(work_items)} work items")
            work_item = work_items[0]
            
            # Try to serialize the work item
            try:
                serialized = json.dumps({
                    'id': work_item.id,
                    'title': work_item.title,
                    'metadata': work_item.metadata
                }, indent=2)
                print("   ✓ Work item serialization successful")
                
                # Check if research phases made it through
                if 'research_phases' in work_item.metadata:
                    phases = work_item.metadata['research_phases']
                    print(f"   ✓ Research phases included: {list(phases.keys())[:3]}...")
                    
                    # Check phase_5_final_selection specifically
                    if 'phase_5_final_selection' in phases:
                        phase5 = phases['phase_5_final_selection']
                        if isinstance(phase5, list) and phase5:
                            item = phase5[0]
                            is_dict = isinstance(item, dict)
                            print(f"   ✓ Phase 5 items are dicts: {is_dict}")
                        else:
                            print("   ⚠️ Phase 5 is empty or not a list")
                            
            except Exception as e:
                print(f"   ✗ Serialization failed: {e}")
                return False
                
        else:
            print("   ✗ No work items generated")
            return False
            
        print("\n✅ All serialization tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_serialization())
    sys.exit(0 if result else 1)