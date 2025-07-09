#!/usr/bin/env python3
"""
Test just the first phase of Gemini Deep Research.
"""

import asyncio
import os
import sys
import logging

# Set up logging to see debug output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch


async def test_phase1():
    """Test just the market exploration phase."""
    print("🔬 Testing Phase 1: Market Exploration")
    print("=" * 60)
    
    researcher = GeminiDeepResearch()
    
    # Test with just a few focus areas for debugging
    test_areas = [
        "Small business operations",
        "Health and wellness",
        "Education and learning"
    ]
    
    print(f"\nExploring {len(test_areas)} test areas:")
    for area in test_areas:
        print(f"  • {area}")
    
    try:
        # Call phase 1 directly
        result = await researcher._explore_and_expand_markets(test_areas)
        
        print(f"\n✅ Phase 1 completed successfully!")
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"\n{key}:")
                if isinstance(value, list):
                    print(f"  - {len(value)} items")
                    if value and len(value) > 0:
                        print(f"  - First item: {value[0]}")
                elif isinstance(value, dict):
                    print(f"  - Dict with {len(value)} keys")
                else:
                    print(f"  - {type(value)}: {str(value)[:100]}...")
                    
    except Exception as e:
        print(f"❌ Error in Phase 1: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_phase1())