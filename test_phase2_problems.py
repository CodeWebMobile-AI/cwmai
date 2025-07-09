#!/usr/bin/env python3
"""
Test just the problem discovery phase.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch


async def test_phase2():
    """Test problem discovery phase."""
    print("🔬 Testing Phase 2: Problem Discovery")
    print("=" * 60)
    
    researcher = GeminiDeepResearch()
    
    # Mock market data from phase 1
    market_data = {
        "explored_sectors": [
            {
                "sector": "Small business operations",
                "problems_pain_points": "Cash flow management, inventory tracking"
            }
        ],
        "market_gaps": ["Affordable accounting tools"]
    }
    
    try:
        # Call phase 2 directly
        problems = await researcher._discover_problems_with_search(market_data)
        
        print(f"\n✅ Found {len(problems)} problems")
        
        if problems:
            print("\nFirst problem:")
            problem = problems[0]
            for key, value in problem.items():
                print(f"  {key}: {str(value)[:100]}...")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_phase2())