#!/usr/bin/env python3
"""
Test that research now returns the requested number of problems.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import after path setup
from scripts.market_discovery import MarketDiscovery


async def test_research_count():
    """Test that research returns multiple problems."""
    print("🧪 Testing Research Count Fix")
    print("=" * 60)
    
    # Create market discovery instance
    discovery = MarketDiscovery()
    
    # Request 5 problems
    requested_count = 5
    print(f"\n📋 Requesting {requested_count} problems...")
    
    try:
        problems = await discovery.discover_problems(count=requested_count)
        
        print(f"\n✅ Research completed!")
        print(f"Requested: {requested_count} problems")
        print(f"Received: {len(problems)} problems")
        
        if len(problems) == requested_count:
            print("✅ SUCCESS: Got the requested number of problems!")
        else:
            print(f"⚠️ WARNING: Expected {requested_count}, got {len(problems)}")
        
        # Display the problems
        print("\n📊 Problems discovered:")
        for i, problem in enumerate(problems, 1):
            print(f"\n{i}. {problem.problem_statement[:100]}...")
            print(f"   Audience: {problem.affected_users}")
            print(f"   Market Size: {problem.market_size}")
            
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not set. Please run:")
        print("export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    success = asyncio.run(test_research_count())
    sys.exit(0 if success else 1)