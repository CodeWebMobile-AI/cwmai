#!/usr/bin/env python3
"""
Test project creation with no fallbacks - only deep research.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.project_creator import ProjectCreator


async def test_no_fallback():
    """Test project creation using only deep research."""
    print("🚀 Testing Project Creation (Deep Research Only - No Fallbacks)")
    print("=" * 60)
    
    creator = ProjectCreator()
    
    try:
        # Create a project using deep research
        print("\nUsing Gemini Deep Research with Google Search grounding...")
        print("This will analyze 25 comprehensive market sectors...")
        print("\n⏳ This may take 2-3 minutes as it performs deep market analysis...\n")
        
        project = await creator.create_innovative_project({})
        
        if project:
            print(f"\n✅ Successfully created project: {project['name']}")
            print(f"\nDescription: {project['description']}")
            print(f"\nProblem: {project.get('problem_statement', 'N/A')}")
            print(f"Target Market: {project.get('target_market', 'N/A')}")
            print(f"Monetization: {project.get('monetization_model', 'N/A')}")
            print(f"Revenue Potential: {project.get('revenue_potential', 'N/A')}")
            
            if 'market_research' in project:
                research = project['market_research']
                if 'opportunities' in research:
                    print(f"\n📊 Based on {len(research['opportunities'])} validated market opportunities")
        else:
            print("❌ No project created")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_no_fallback())