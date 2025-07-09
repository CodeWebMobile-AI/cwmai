#!/usr/bin/env python3
"""
Test script to verify the single opportunity selection fix works correctly.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research import GeminiDeepResearch

async def test_single_opportunity():
    """Test that we get a single opportunity even if pre-build validation fails."""
    
    print("Testing Single Opportunity Selection Fix")
    print("=" * 60)
    
    # Initialize the research module
    research = GeminiDeepResearch()
    
    # Set count to 1 explicitly
    count = 1
    print(f"Requesting {count} opportunity")
    
    # Run the research
    opportunities = await research.conduct_deep_market_research(count=count)
    
    print(f"\n✅ Research returned {len(opportunities)} opportunities")
    
    if opportunities:
        opp = opportunities[0]
        print(f"\nOpportunity: {opp.problem_description}")
        print(f"Target: {opp.affected_audience}")
        
        # Check if pre-build validation was added
        if hasattr(opp, 'pre_build_validation') and opp.pre_build_validation:
            val = opp.pre_build_validation
            print(f"\nPre-build Validation:")
            print(f"  Score: {val.get('validation_score', 'N/A')}/10")
            print(f"  Domain Available: {val.get('domain_available', 'N/A')}")
            print(f"  Proceed: {val.get('proceed_with_build', 'N/A')}")
            if val.get('suggested_domains'):
                print(f"  Alternative domains: {', '.join(val.get('suggested_domains', [])[:3])}")
        else:
            print("\nNo pre-build validation data attached")
    else:
        print("\n❌ No opportunities returned (this should not happen with the fix)")

if __name__ == "__main__":
    # Ensure we have the Gemini API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Run: export $(cat ../.env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_single_opportunity())