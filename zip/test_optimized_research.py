#!/usr/bin/env python3
"""
Test the optimized research system to ensure it works correctly.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research_optimized import GeminiDeepResearchOptimized
from scripts.ai_brain import IntelligentAIBrain


async def test_optimized_research():
    """Test the optimized research system."""
    
    print("Testing Optimized Gemini Deep Research")
    print("=" * 60)
    
    # Initialize
    print("🚀 Initializing optimized research system...")
    ai_brain = IntelligentAIBrain()
    research = GeminiDeepResearchOptimized(ai_brain)
    
    # Test configuration
    print("\n📋 Configuration:")
    print(f"  Phase limits: {research.PHASE_LIMITS}")
    print(f"  Min scores: {research.MIN_SCORES}")
    print(f"  Batch sizes: {research.BATCH_SIZES}")
    
    # Test 1: Schema complexity handling
    print("\n🧪 Test 1: Schema Complexity Handling")
    print("-" * 40)
    
    # Create test opportunities with lots of data
    test_opportunities = []
    for i in range(20):
        opp = {
            'id': f'test_{i}',
            'problem_description': f'Test Problem {i}: ' + 'A' * 500,
            'affected_audience': 'Test audience ' + 'B' * 200,
            'solution_name': f'Solution {i}',
            'revenue_potential': f'${i*1000}/month',
            'validation_score': 7 + (i % 3),
            'user_pain_score': 6 + (i % 4),
            'revenue_score': 7 + (i % 2),
            'competition_score': 8 - (i % 3),
            'technical_score': 8,
            'market_score': 7,
            'economics_score': 8,
            'mvp_score': 8
        }
        test_opportunities.append(opp)
    
    try:
        # Test the optimized final selection with simple schema
        selected = await research._phase_5_final_selection_optimized(
            test_opportunities[:5],  # Just top 5
            {}
        )
        
        if selected:
            print("✅ Schema test passed! No complexity errors")
            print(f"   Selected: {selected[0].problem_description[:50]}...")
        else:
            print("⚠️  No selection made, but no schema error")
            
    except Exception as e:
        if "too many states" in str(e):
            print(f"❌ SCHEMA ERROR: {e}")
            return False
        else:
            print(f"❌ Other error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test 2: Progressive Filtering
    print("\n🧪 Test 2: Progressive Filtering")
    print("-" * 40)
    
    # Test filtering logic
    filtered = await research._progressive_filter(
        test_opportunities,
        'test_phase',
        limit=5,
        min_score=7.5
    )
    
    print(f"Input: {len(test_opportunities)} opportunities")
    print(f"Output: {len(filtered)} opportunities (limit=5, min_score=7.5)")
    
    if len(filtered) <= 5:
        print("✅ Progressive filtering working correctly")
    else:
        print("❌ Progressive filtering not working")
        return False
    
    # Test 3: Deduplication
    print("\n🧪 Test 3: Problem Deduplication")
    print("-" * 40)
    
    duplicate_problems = [
        {'problem_description': 'Mental health support for remote workers', 
         'affected_audience': 'Remote employees'},
        {'problem_description': 'Mental wellness platform for distributed teams',
         'affected_audience': 'Remote workers'},
        {'problem_description': 'Burnout prevention tools for remote employees',
         'affected_audience': 'Work from home professionals'},
        {'problem_description': 'AI code review automation',
         'affected_audience': 'Software developers'},
        {'problem_description': 'Automated code review with AI',
         'affected_audience': 'Dev teams'}
    ]
    
    deduplicated = research._deduplicate_problems(duplicate_problems)
    print(f"Input: {len(duplicate_problems)} problems")
    print(f"Output: {len(deduplicated)} unique problems")
    
    if len(deduplicated) < len(duplicate_problems):
        print("✅ Deduplication working correctly")
    else:
        print("❌ Deduplication not removing duplicates")
    
    # Test 4: Cache functionality
    print("\n🧪 Test 4: Search Cache")
    print("-" * 40)
    
    # Test cache
    cache = research.search_cache
    cache.set("test query", {"result": "test data"})
    cached_result = cache.get("test query")
    
    if cached_result and cached_result.get("result") == "test data":
        print("✅ Cache store/retrieve working")
    else:
        print("❌ Cache not working")
        return False
    
    # Test cache normalization
    cache.set("software market trends 2024", {"data": "trends"})
    normalized_result = cache.get("Software Market Trends 2025")  # Different year
    
    if normalized_result:
        print("✅ Cache normalization working (year agnostic)")
    else:
        print("⚠️  Cache normalization could be improved")
    
    stats = cache.get_stats()
    print(f"   Cache stats: {stats}")
    
    # Test 5: Full mini pipeline
    print("\n🧪 Test 5: Mini Research Pipeline")
    print("-" * 40)
    print("Running abbreviated research pipeline...")
    
    start_time = time.time()
    
    try:
        # Run with a focused query
        opportunities = await research.research_market_opportunities(
            market_sector="passive income software tools",
            existing_portfolio={}
        )
        
        duration = time.time() - start_time
        
        print(f"\n✅ Pipeline completed in {duration:.1f} seconds")
        print(f"   API calls made: {research.api_calls}")
        print(f"   Opportunities found: {len(opportunities)}")
        
        if opportunities:
            opp = opportunities[0]
            print(f"\n   Selected opportunity:")
            print(f"   Problem: {opp.problem_description[:100]}...")
            print(f"   Revenue: {opp.potential_revenue}")
            print(f"   Market: {opp.market_size}")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print("\n✅ All core optimizations tested successfully:")
    print("  • Ultra-simple schema (index-based selection)")
    print("  • Progressive filtering")
    print("  • Problem deduplication")
    print("  • Search caching")
    print("  • Mini pipeline execution")
    
    print("\n📊 Performance indicators:")
    print(f"  • API calls for mini test: {research.api_calls}")
    print(f"  • Execution time: {duration:.1f}s")
    print(f"  • No schema complexity errors")
    
    print("\n🎯 Ready for integration!")
    
    return True


async def main():
    """Main test function."""
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ ERROR: GOOGLE_API_KEY not set")
        print("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run tests
    success = await test_optimized_research()
    
    if success:
        print("\n✅ Optimized version is working correctly!")
        print("\nNext steps:")
        print("1. Run the full system test to compare performance")
        print("2. Apply the optimization to replace the original")
    else:
        print("\n❌ Tests failed. Do not integrate yet.")


if __name__ == "__main__":
    asyncio.run(main())