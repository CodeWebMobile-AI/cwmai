#!/usr/bin/env python3
"""
Minimal test of deep research with fewer sectors.
"""

import asyncio
import os
import sys
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch


async def test_minimal():
    """Test with minimal sectors."""
    print("🔬 Testing Gemini Deep Research (Minimal)")
    print("=" * 60)
    
    researcher = GeminiDeepResearch()
    
    # Test with just 2 sectors for faster results
    test_sectors = [
        "Small business operations",
        "Health and wellness"
    ]
    
    try:
        print(f"\nAnalyzing {len(test_sectors)} sectors for quick test...")
        
        opportunities = await researcher.conduct_deep_market_research(
            focus_areas=test_sectors
        )
        
        print(f"\n✅ Research completed!")
        print(f"Found {len(opportunities)} opportunities\n")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp.problem_description}")
            print(f"   Target: {opp.affected_audience}")
            print(f"   Revenue: {opp.potential_revenue}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_minimal())