#!/usr/bin/env python3
"""
Test creative project name generation
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.market_discovery import MarketDiscovery
from scripts.intelligent_work_finder import ResearchCache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_creative_names():
    """Test that creative names are generated during research."""
    print("\n🧪 Testing Creative Project Name Generation")
    print("=" * 60)
    
    # Create market discovery
    market_discovery = MarketDiscovery({})
    
    # Discover problems (this should generate creative names)
    print("\n1️⃣ Discovering market problems...")
    problems = await market_discovery.discover_problems(count=3)
    
    print(f"\n✅ Discovered {len(problems)} problems")
    
    # Check each problem has creative name
    for i, problem in enumerate(problems):
        print(f"\n📦 Problem {i+1}:")
        print(f"   Statement: {problem.problem_statement[:80]}...")
        print(f"   Audience: {problem.affected_users}")
        
        # Check if opportunity has project_name
        if hasattr(problem, '_full_opportunity') and problem._full_opportunity:
            opp = problem._full_opportunity
            if hasattr(opp, 'project_name'):
                print(f"   ✨ Creative Name: '{opp.project_name}'")
            else:
                print(f"   ❌ ERROR: No project_name field in opportunity!")
        else:
            print(f"   ❌ ERROR: No _full_opportunity attached!")
    
    # Test research cache
    print("\n\n2️⃣ Testing Research Cache...")
    cache = ResearchCache(system_state={})
    
    # Try to cache problems
    try:
        cached = await cache.batch_research(count=2)
        print(f"\n✅ Cached {len(cached)} problems")
        
        for i, cp in enumerate(cached):
            print(f"\n📦 Cached Problem {i+1}:")
            print(f"   Name: '{cp.name}'")
            print(f"   Problem: {cp.problem[:80]}...")
    except Exception as e:
        print(f"\n❌ Cache error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_creative_names())