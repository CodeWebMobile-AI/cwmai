#!/usr/bin/env python3
"""
Test what projects are generated from the fallback method.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.market_research_engine import MarketResearchEngine
from scripts.ai_brain import IntelligentAIBrain


async def test_fallback_projects():
    """Test project generation using fallback methods."""
    print("🔍 Testing Project Generation (Fallback Method)")
    print("=" * 60)
    
    # Initialize
    ai_brain = IntelligentAIBrain()
    engine = MarketResearchEngine(ai_brain)
    
    print("\nUsing traditional market analysis methods...")
    print("This will analyze:")
    print("  • Real-world problem spaces")
    print("  • GitHub trends") 
    print("  • Emerging technologies")
    print("  • Developer community needs\n")
    
    try:
        # Force fallback by calling the method directly
        opportunities = await engine._fallback_opportunity_discovery()
        
        print(f"✅ Found {len(opportunities)} opportunities\n")
        
        # Display opportunities
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp.problem_description}")
            print(f"   Audience: {opp.affected_audience}")
            print(f"   Competition: {opp.competition_level}")
            print()
        
        # Generate projects from these
        print("\nGenerating projects from opportunities...")
        
        projects = await engine.generate_project_opportunities({}, max_opportunities=5)
        
        print(f"\n✅ Generated {len(projects)} projects:\n")
        
        for i, proj in enumerate(projects, 1):
            print(f"Project {i}: {proj.title}")
            print(f"Description: {proj.description}")
            print(f"Target: {proj.target_market}")
            print(f"Tech: {', '.join(proj.tech_stack[:3]) if hasattr(proj, 'tech_stack') else 'N/A'}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    await test_fallback_projects()


if __name__ == "__main__":
    asyncio.run(main())