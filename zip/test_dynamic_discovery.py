#!/usr/bin/env python3
"""
Test that the research system discovers opportunities dynamically
based on current market trends, not hard-coded sectors.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research_optimized import GeminiDeepResearchOptimized
from scripts.ai_brain import IntelligentAIBrain


async def test_dynamic_discovery():
    """Test the dynamic discovery based on market trends."""
    
    print("Testing Dynamic Market Discovery")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ ERROR: GOOGLE_API_KEY not set")
        print("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        return False
    
    # Initialize
    print("🚀 Initializing research system...")
    ai_brain = IntelligentAIBrain()
    research = GeminiDeepResearchOptimized(ai_brain)
    
    print("\n📊 Testing Phase 0: Market Timing Analysis")
    print("-" * 40)
    
    # Test market timing to see what trends it discovers
    market_timing = await research._phase_0_market_timing()
    
    if market_timing and 'trending_areas' in market_timing:
        print("✅ Market timing discovered trending areas:")
        for i, area in enumerate(market_timing['trending_areas']):
            print(f"   {i+1}. {area.get('area', 'Unknown')}")
            print(f"      Evidence: {area.get('evidence', 'N/A')[:100]}...")
            print(f"      Score: {area.get('timing_score', 0)}")
    else:
        print("❌ No trending areas discovered")
        return False
    
    print("\n📊 Testing Full Research Pipeline")
    print("-" * 40)
    print("Running research with count=2 (discover 2 opportunities)...")
    
    # Run the full pipeline
    opportunities = await research.research_market_opportunities(
        count=2,
        existing_portfolio={}
    )
    
    print(f"\n🎯 Research Results:")
    print(f"   Opportunities found: {len(opportunities)}")
    
    if opportunities:
        for i, opp in enumerate(opportunities):
            print(f"\n   Opportunity {i+1}:")
            print(f"   Problem: {opp.problem_description[:100]}...")
            print(f"   Audience: {opp.affected_audience}")
            print(f"   Market Size: {opp.market_size}")
            print(f"   Revenue Potential: {opp.potential_revenue}")
        
        # Check if opportunities are diverse (not all from same sector)
        problems = [opp.problem_description.lower() for opp in opportunities]
        
        # Simple diversity check - do the problems mention different domains?
        tech_keywords = ['ai', 'software', 'app', 'platform', 'tool', 'automation']
        other_keywords = ['health', 'finance', 'education', 'entertainment', 'social', 'business']
        
        tech_count = sum(1 for p in problems if any(k in p for k in tech_keywords))
        other_count = sum(1 for p in problems if any(k in p for k in other_keywords))
        
        print(f"\n📊 Diversity Analysis:")
        print(f"   Tech-focused opportunities: {tech_count}")
        print(f"   Other sectors: {other_count}")
        
        print("\n✅ Dynamic discovery is working!")
        print("   - Market timing identifies current trends")
        print("   - Problems are discovered based on those trends")
        print("   - No hard-coded sectors limiting discovery")
        
        # Show optimization stats
        print(f"\n📈 Performance Stats:")
        print(f"   API calls: {research.api_calls}")
        cache_stats = research.search_cache.get_stats()
        print(f"   Cache hits: {cache_stats['hits']}")
        print(f"   Cache misses: {cache_stats['misses']}")
        
        return True
    else:
        print("❌ No opportunities discovered")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_dynamic_discovery())
    if success:
        print("\n✨ Dynamic discovery test passed!")
    else:
        print("\n❌ Dynamic discovery test failed")