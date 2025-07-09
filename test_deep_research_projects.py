#!/usr/bin/env python3
"""
Test what projects Gemini Deep Research generates from the 25 market sectors.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.market_research_engine import MarketResearchEngine
from scripts.ai_brain import IntelligentAIBrain


async def test_deep_research_projects():
    """Test what projects are generated from Gemini Deep Research."""
    print("🔬 Testing Gemini Deep Research Project Generation")
    print("=" * 80)
    
    # Initialize
    ai_brain = IntelligentAIBrain()
    
    # Create market research engine
    research_engine = MarketResearchEngine(ai_brain)
    
    print(f"\n📊 Analyzing {len(GeminiDeepResearch.DEFAULT_FOCUS_AREAS)} market sectors:")
    for i, area in enumerate(GeminiDeepResearch.DEFAULT_FOCUS_AREAS[:10], 1):
        print(f"  {i}. {area}")
    print(f"  ... and {len(GeminiDeepResearch.DEFAULT_FOCUS_AREAS) - 10} more sectors\n")
    
    try:
        # Step 1: Get market opportunities from deep research
        print("1️⃣ Discovering market opportunities with Gemini Deep Research...")
        start_time = datetime.now()
        
        opportunities = await research_engine.discover_market_opportunities()
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Found {len(opportunities)} opportunities in {duration:.1f}s\n")
        
        # Display opportunities
        print("Market Opportunities Found:")
        print("-" * 80)
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"\n{i}. {opp.problem_description[:100]}...")
            print(f"   Audience: {opp.affected_audience}")
            print(f"   Market: {opp.market_size}")
            print(f"   Revenue: {opp.potential_revenue}")
            print(f"   Competition: {opp.competition_level}")
        
        # Step 2: Generate projects from these opportunities
        print("\n\n2️⃣ Generating projects from validated opportunities...")
        
        # Mock portfolio
        mock_portfolio = {}
        
        project_opportunities = await research_engine.generate_project_opportunities(
            existing_portfolio=mock_portfolio,
            max_opportunities=5
        )
        
        print(f"\n✅ Generated {len(project_opportunities)} projects\n")
        
        print("Projects Generated:")
        print("=" * 80)
        for i, proj in enumerate(project_opportunities, 1):
            print(f"\nProject {i}: {proj.title}")
            print(f"Description: {proj.description}")
            print(f"Problem: {proj.problem_statement}")
            print(f"Target Market: {proj.target_market}")
            print(f"Monetization: {getattr(proj, 'monetization_model', 'N/A')}")
            print(f"Revenue: {getattr(proj, 'revenue_potential', 'N/A')}")
            print(f"Tech Stack: {proj.tech_stack}")
            print("-" * 60)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    await test_deep_research_projects()


if __name__ == "__main__":
    print("🚀 Gemini Deep Research Project Test")
    print("This will show what projects are generated from analyzing 25 market sectors")
    print("using Google Search grounding for validation\n")
    
    asyncio.run(main())