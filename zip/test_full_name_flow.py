#!/usr/bin/env python3
"""
Test full flow of creative name generation
"""

import asyncio
import logging
from scripts.intelligent_work_finder import IntelligentWorkFinder, ResearchCache
from scripts.ai_brain import IntelligentAIBrain
from scripts.project_creator import ProjectCreator

logging.basicConfig(level=logging.INFO)

async def test_full_flow():
    """Test the full flow from research to project creation."""
    
    # Initialize components
    ai_brain = IntelligentAIBrain()
    system_state = {
        'projects': {},
        'repositories': {}
    }
    
    # Test research cache
    print("\n1️⃣ Testing Research Cache...")
    cache = ResearchCache(system_state=system_state)
    
    # Get a problem from cache (this will trigger research if needed)
    problem = await cache.get_next_problem()
    
    if problem:
        print(f"\n✅ Got cached problem:")
        print(f"   Creative Name: '{problem.name}'")
        print(f"   Problem: {problem.problem[:80]}...")
        print(f"   Features: {len(problem.features)} features")
        
        # Test work finder
        print("\n2️⃣ Testing Work Finder...")
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        
        # Generate a new project opportunity
        opportunities = await work_finder._generate_new_project_opportunity()
        
        if opportunities:
            opp = opportunities[0]
            print(f"\n✅ Generated work opportunity:")
            print(f"   Title: {opp.title}")
            print(f"   Type: {opp.type}")
            
            # Check metadata for project name
            metadata = opp.metadata
            if 'selected_project' in metadata:
                project_name = metadata['selected_project'].get('project_name', '')
                print(f"   Project Name: '{project_name}'")
                print(f"   Is Creative: {'✅' if project_name and '-solution-' not in project_name else '❌'}")
            else:
                print("   ❌ No selected_project in metadata")
        else:
            print("   ❌ No opportunities generated")
    else:
        print("   ❌ No problems in cache and research failed")

if __name__ == "__main__":
    asyncio.run(test_full_flow())