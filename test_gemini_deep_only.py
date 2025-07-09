#!/usr/bin/env python3
"""
Test only the Gemini Deep Research functionality.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch


async def test_deep_research():
    """Test Gemini Deep Research directly."""
    print("🔬 Testing Gemini Deep Research")
    print("=" * 60)
    
    researcher = GeminiDeepResearch()
    
    print(f"\nAnalyzing {len(researcher.DEFAULT_FOCUS_AREAS)} sectors:")
    for i, area in enumerate(researcher.DEFAULT_FOCUS_AREAS[:5], 1):
        print(f"  {i}. {area}")
    print(f"  ... and {len(researcher.DEFAULT_FOCUS_AREAS) - 5} more\n")
    
    try:
        opportunities = await researcher.conduct_deep_market_research()
        
        print(f"\n✅ Found {len(opportunities)} opportunities:\n")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp.problem_description[:80]}...")
            print(f"   Audience: {opp.affected_audience}")
            print(f"   Market: {opp.market_size}")
            print(f"   Revenue: {opp.potential_revenue}")
            print(f"   Competition: {opp.competition_level}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


asyncio.run(test_deep_research())