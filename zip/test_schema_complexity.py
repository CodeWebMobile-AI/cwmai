#!/usr/bin/env python3
"""
Test to verify if schema complexity (maxItems) is causing the Gemini error.
We'll test with different maxItems values to find the limit.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research import GeminiDeepResearchOptimized
from scripts.ai_brain import IntelligentAIBrain


async def test_schema_with_max_items(max_items: int):
    """Test if a schema with specific maxItems works."""
    print(f"\n{'='*60}")
    print(f"Testing with maxItems: {max_items}")
    print('='*60)
    
    # Initialize
    ai_brain = IntelligentAIBrain()
    research = GeminiDeepResearchOptimized(ai_brain)
    
    # Create test prompt and schema
    prompt = f"""
    Discover real problems people face based on current market trends.
    
    Return {max_items} specific problems with clear audiences.
    Focus on problems with transaction potential and passive income possibilities.
    """
    
    schema = {
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "problem_description": {"type": "string"},
                        "affected_audience": {"type": "string"}
                    },
                    "required": ["problem_description", "affected_audience"]
                },
                "maxItems": max_items
            }
        },
        "required": ["problems"]
    }
    
    try:
        result = await research._call_gemini_structured(prompt, schema)
        if result and 'problems' in result:
            print(f"✅ SUCCESS with maxItems={max_items}")
            print(f"   Returned {len(result['problems'])} problems")
            return True
        else:
            print(f"❌ FAILED with maxItems={max_items} - No result")
            return False
    except Exception as e:
        print(f"❌ FAILED with maxItems={max_items}")
        print(f"   Error: {str(e)}")
        return False


async def find_max_items_limit():
    """Binary search to find the maximum working maxItems value."""
    print("Finding maximum supported maxItems value...")
    
    # Test specific values first
    test_values = [10, 20, 30, 40, 50]
    results = {}
    
    for value in test_values:
        success = await test_schema_with_max_items(value)
        results[value] = success
        
        # If we found a failure, do binary search between last success and this failure
        if not success and value > 10:
            print(f"\nFound failure at {value}, doing binary search...")
            
            # Find the last working value
            last_working = 10
            for v in sorted(results.keys()):
                if results[v] and v < value:
                    last_working = v
            
            # Binary search between last_working and value
            low = last_working
            high = value
            
            while high - low > 1:
                mid = (low + high) // 2
                print(f"\nTesting binary search value: {mid}")
                if await test_schema_with_max_items(mid):
                    low = mid
                else:
                    high = mid
            
            print(f"\n{'='*60}")
            print(f"BINARY SEARCH RESULT: Maximum working maxItems = {low}")
            print(f"{'='*60}")
            return low
    
    # If all passed, the limit is higher than 50
    print(f"\n{'='*60}")
    print(f"All tests passed! maxItems can be at least 50")
    print(f"{'='*60}")
    return 50


async def test_discover_count_calculation():
    """Test the actual discover_count calculation that's failing."""
    print("\n" + "="*60)
    print("Testing actual discover_count calculation")
    print("="*60)
    
    # Test with count=10 (what the system is using)
    target_count = 10
    discover_count = min(50, target_count * 10)
    
    print(f"With count={target_count}:")
    print(f"  discover_count = min(50, {target_count} * 10) = {discover_count}")
    
    # Test if this works
    success = await test_schema_with_max_items(discover_count)
    
    if not success:
        # Try with reduced multiplier
        print("\nTrying reduced multipliers...")
        for multiplier in [5, 4, 3, 2]:
            new_discover_count = min(50, target_count * multiplier)
            print(f"\n  Multiplier {multiplier}: discover_count = {new_discover_count}")
            if await test_schema_with_max_items(new_discover_count):
                print(f"\n✅ RECOMMENDATION: Use multiplier {multiplier} (discover_count = {new_discover_count})")
                return new_discover_count
    
    return discover_count


async def main():
    """Run all tests."""
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ ERROR: GOOGLE_API_KEY not set")
        print("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        return
    
    print("Testing Gemini Schema Complexity Limits")
    print("="*60)
    
    # First, test the exact scenario that's failing
    print("1. Testing the exact failing scenario (maxItems=50)...")
    await test_schema_with_max_items(50)
    
    # Find the actual limit
    print("\n2. Finding the maximum supported maxItems...")
    max_limit = await find_max_items_limit()
    
    # Test the discover_count calculation
    print("\n3. Testing discover_count calculation...")
    recommended_count = await test_discover_count_calculation()
    
    print("\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    print(f"Maximum working maxItems: {max_limit}")
    print(f"Recommended discover_count: {recommended_count}")
    print("\nSuggested fix in gemini_deep_research.py:")
    print(f"  discover_count = min({max_limit}, target_count * {recommended_count // 10})")


if __name__ == "__main__":
    asyncio.run(main())