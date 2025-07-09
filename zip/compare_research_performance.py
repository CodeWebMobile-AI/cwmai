#!/usr/bin/env python3
"""
Compare performance between original and optimized research systems.
"""

import asyncio
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

# Import both versions
from scripts.gemini_deep_research import GeminiDeepResearch as OriginalResearch
from scripts.gemini_deep_research_optimized import GeminiDeepResearchOptimized
from scripts.ai_brain import IntelligentAIBrain


async def test_original():
    """Test the original research system."""
    print("\n" + "="*60)
    print("TESTING ORIGINAL RESEARCH SYSTEM")
    print("="*60)
    
    ai_brain = IntelligentAIBrain()
    research = OriginalResearch(ai_brain)
    
    start_time = time.time()
    api_calls_start = 0  # We'll estimate based on phases
    
    try:
        # Run research
        opportunities = await research.research_market_opportunities(
            market_sector="AI automation tools",
            existing_portfolio={}
        )
        
        duration = time.time() - start_time
        
        # Count phases processed
        phases_data = research.research_phases_data
        total_processed = 0
        for phase, data in phases_data.items():
            if data and isinstance(data, list):
                total_processed += len(data)
        
        # Estimate API calls (1 per opportunity per phase on average)
        estimated_api_calls = total_processed
        
        print(f"\n✅ Original system completed")
        print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"   Opportunities found: {len(opportunities)}")
        print(f"   Estimated API calls: ~{estimated_api_calls}")
        print(f"   Problems discovered: {len(phases_data.get('phase_1_problems_discovered', []))}")
        print(f"   Problems validated: {len(phases_data.get('phase_2_problems_validated', []))}")
        
        return {
            'duration': duration,
            'opportunities': len(opportunities),
            'api_calls': estimated_api_calls,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ Original system failed: {e}")
        if "too many states" in str(e):
            print("   Schema complexity error occurred!")
        return {
            'duration': time.time() - start_time,
            'opportunities': 0,
            'api_calls': 0,
            'success': False,
            'error': str(e)
        }


async def test_optimized():
    """Test the optimized research system."""
    print("\n" + "="*60)
    print("TESTING OPTIMIZED RESEARCH SYSTEM")
    print("="*60)
    
    ai_brain = IntelligentAIBrain()
    research = GeminiDeepResearchOptimized(ai_brain)
    
    start_time = time.time()
    
    try:
        # Run research
        opportunities = await research.research_market_opportunities(
            market_sector="AI automation tools",
            existing_portfolio={}
        )
        
        duration = time.time() - start_time
        
        print(f"\n✅ Optimized system completed")
        print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"   Opportunities found: {len(opportunities)}")
        print(f"   API calls: {research.api_calls}")
        
        # Show cache stats
        cache_stats = research.search_cache.get_stats()
        print(f"\n   Cache performance:")
        print(f"     Hits: {cache_stats['hits']}")
        print(f"     Misses: {cache_stats['misses']}")
        print(f"     Hit rate: {cache_stats['hit_rate']}")
        
        # Show filtering stats
        print(f"\n   Progressive filtering:")
        active = len(research.active_opportunities)
        total_scored = len(research.scores)
        print(f"     Total opportunities processed: {total_scored}")
        print(f"     Early terminations: {total_scored - active}")
        
        return {
            'duration': duration,
            'opportunities': len(opportunities),
            'api_calls': research.api_calls,
            'cache_stats': cache_stats,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ Optimized system failed: {e}")
        return {
            'duration': time.time() - start_time,
            'opportunities': 0,
            'api_calls': research.api_calls,
            'success': False,
            'error': str(e)
        }


async def main():
    """Run comparison tests."""
    print("Research System Performance Comparison")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("\n❌ ERROR: GOOGLE_API_KEY not set")
        print("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Test both systems
    print("\nRunning tests sequentially to avoid rate limits...")
    
    # Test optimized first (it's faster)
    optimized_results = await test_optimized()
    
    # Wait a bit to avoid rate limits
    print("\nWaiting 5 seconds before testing original system...")
    await asyncio.sleep(5)
    
    # Test original
    original_results = await test_original()
    
    # Compare results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    if optimized_results['success'] and original_results['success']:
        # Calculate improvements
        time_reduction = (original_results['duration'] - optimized_results['duration']) / original_results['duration'] * 100
        api_reduction = (original_results['api_calls'] - optimized_results['api_calls']) / original_results['api_calls'] * 100
        
        print(f"\n📊 Performance Improvements:")
        print(f"   Time: {original_results['duration']:.1f}s → {optimized_results['duration']:.1f}s ({time_reduction:.1f}% faster)")
        print(f"   API calls: {original_results['api_calls']} → {optimized_results['api_calls']} ({api_reduction:.1f}% fewer)")
        
        print(f"\n🚀 Optimization Benefits:")
        print(f"   • {time_reduction:.0f}% faster execution")
        print(f"   • {api_reduction:.0f}% fewer API calls")
        print(f"   • Search caching enabled")
        print(f"   • Progressive filtering active")
        print(f"   • No schema complexity errors")
        
    elif not original_results['success'] and optimized_results['success']:
        print(f"\n✅ Optimized version succeeded where original failed!")
        print(f"   Original error: {original_results.get('error', 'Unknown')}")
        print(f"   Optimized completed in {optimized_results['duration']:.1f}s with {optimized_results['api_calls']} API calls")
        
    else:
        print(f"\n⚠️  Could not complete full comparison")
        if not optimized_results['success']:
            print(f"   Optimized error: {optimized_results.get('error', 'Unknown')}")
        if not original_results['success']:
            print(f"   Original error: {original_results.get('error', 'Unknown')}")
    
    print("\n" + "="*60)
    print("Comparison complete!")


if __name__ == "__main__":
    asyncio.run(main())