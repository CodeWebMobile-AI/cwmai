#!/usr/bin/env python3
"""
Test script to verify the simplified schema fix for Gemini deep research.
This tests that the final ranking phase now works without schema complexity errors.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research import GeminiDeepResearch

async def test_schema_fix():
    """Test that the simplified schema allows opportunities to be returned."""
    
    print("Testing Simplified Schema Fix for Gemini Deep Research")
    print("=" * 60)
    
    # Initialize the research module
    research = GeminiDeepResearch()
    
    # Test with a small count to make it faster
    count = 1
    print(f"Requesting {count} opportunity with simplified schema")
    print("\nNote: This will run the full research pipeline (~8-10 minutes)")
    print("Watch for the final ranking phase...")
    
    try:
        # Run the research
        opportunities = await research.conduct_deep_market_research(count=count)
        
        print(f"\n✅ Research completed successfully!")
        print(f"Returned {len(opportunities)} opportunities")
        
        if opportunities:
            print("\n🎉 SUCCESS: Schema fix worked! Opportunities returned:")
            for i, opp in enumerate(opportunities, 1):
                print(f"\n--- Opportunity {i} ---")
                print(f"Problem: {opp.problem_description[:100]}...")
                print(f"Audience: {opp.affected_audience}")
                print(f"Revenue: {opp.potential_revenue}")
                print(f"Monetization: {opp.monetization_model}")
                if hasattr(opp, 'key_features') and opp.key_features:
                    print(f"Key Features: {', '.join(opp.key_features[:3])}")
        else:
            print("\n⚠️ No opportunities returned (but no schema error!)")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        if "too many states" in str(e) or "INVALID_ARGUMENT" in str(e):
            print("\n🔴 Schema is still too complex! Further simplification needed.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we have the Gemini API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Run: export $(cat ../.env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_schema_fix())