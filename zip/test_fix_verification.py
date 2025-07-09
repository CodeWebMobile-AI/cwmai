#!/usr/bin/env python3
"""
Quick test to verify the schema complexity fix works.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research import GeminiDeepResearchOptimized
from scripts.ai_brain import IntelligentAIBrain


async def test_fixed_research():
    """Test that the fixed version works."""
    print("Testing Fixed Research System")
    print("="*60)
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ ERROR: GOOGLE_API_KEY not set")
        return False
    
    # Initialize
    ai_brain = IntelligentAIBrain()
    research = GeminiDeepResearchOptimized(ai_brain)
    
    # Verify research_phases_data exists
    if hasattr(research, 'research_phases_data'):
        print("✅ research_phases_data attribute exists")
    else:
        print("❌ research_phases_data attribute missing")
        return False
    
    # Test Phase 0
    print("\nTesting Phase 0: Market Timing...")
    market_timing = await research._phase_0_market_timing()
    
    if market_timing:
        print("✅ Phase 0 completed successfully")
        print(f"   Found {len(market_timing.get('trending_areas', []))} trending areas")
    else:
        print("❌ Phase 0 failed")
        return False
    
    # Test Phase 1 with fixed discover_count
    print("\nTesting Phase 1: Problem Discovery...")
    problems = await research._phase_1_problem_discovery(market_timing, 10)
    
    if problems:
        print(f"✅ Phase 1 completed successfully")
        print(f"   Discovered {len(problems)} problems (requested up to 20)")
        
        # Show a few examples
        for i, problem in enumerate(problems[:3]):
            print(f"\n   Problem {i+1}:")
            print(f"   - {problem.get('problem_description', '')[:100]}...")
            print(f"   - Audience: {problem.get('affected_audience', '')}")
    else:
        print("❌ Phase 1 failed")
        return False
    
    # Test full research with count=2
    print("\n\nTesting Full Research Pipeline (count=2)...")
    print("-"*60)
    
    opportunities = await research.research_market_opportunities(
        count=2,
        existing_portfolio={}
    )
    
    if opportunities:
        print(f"\n✅ Research completed successfully!")
        print(f"   Found {len(opportunities)} opportunities")
        
        for i, opp in enumerate(opportunities):
            print(f"\n   Opportunity {i+1}:")
            print(f"   Problem: {opp.problem_description[:100]}...")
            print(f"   Market Size: {opp.market_size}")
            print(f"   Revenue: {opp.potential_revenue}")
    else:
        print("❌ Research failed")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_fixed_research())
    if success:
        print("\n✨ All fixes verified! The system is working correctly.")
    else:
        print("\n❌ Fix verification failed")