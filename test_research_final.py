#!/usr/bin/env python3
"""
Final Research Engine Test with All Fixes and Optimizations

This test includes:
- All comprehensive fixes (timeouts, retries, circuit breakers, etc.)
- Performance optimizations (parallel execution, batching, etc.)
- Detailed monitoring and metrics
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import json

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('research_engine_final.log', mode='w')
    ]
)

# Reduce noise
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('github').setLevel(logging.WARNING)

async def test_final_research_engine():
    """Run final research engine test"""
    print("=" * 80)
    print("FINAL RESEARCH ENGINE TEST")
    print("With comprehensive fixes and optimizations")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print("Logging to: research_engine_final.log")
    print("=" * 80)
    
    try:
        # Import required modules
        from scripts.research_evolution_engine import ResearchEvolutionEngine
        from scripts.state_manager import StateManager
        from scripts.ai_brain_factory import AIBrainFactory
        from scripts.research_engine_fixes import apply_research_engine_fixes
        from scripts.research_engine_optimized import apply_research_optimizations
        
        print("\n1. Creating minimal components...")
        
        # Create state manager
        state_manager = StateManager()
        print("   ✓ State manager created")
        
        # Create AI brain
        ai_brain = AIBrainFactory.create_for_research()
        print("   ✓ AI brain created")
        
        print("\n2. Creating research engine...")
        research_engine = ResearchEvolutionEngine(
            state_manager=state_manager,
            ai_brain=ai_brain,
            task_generator=None,
            self_improver=None,
            outcome_learning=None
        )
        print("   ✓ Research engine created")
        
        print("\n3. Applying fixes and optimizations...")
        
        # Apply comprehensive fixes first
        research_engine = apply_research_engine_fixes(research_engine)
        print("   ✓ Comprehensive fixes applied")
        
        # Apply performance optimizations
        research_engine = apply_research_optimizations(research_engine)
        print("   ✓ Performance optimizations applied")
        
        print("\n   Applied enhancements:")
        print("   - Timeout handling (60s per operation)")
        print("   - Retry logic with exponential backoff")
        print("   - Circuit breaker pattern")
        print("   - Parallel research execution")
        print("   - Batch insight extraction")
        print("   - Limited queries per topic")
        print("   - Early failure termination")
        
        print("\n4. Executing optimized research cycle...")
        print("-" * 80)
        
        cycle_start = datetime.now()
        
        # Set minimal context
        research_engine.last_trigger_context = {
            'reason': "Final optimized test",
            'test_mode': True
        }
        
        # Execute single cycle
        results = await research_engine.execute_research_cycle()
        
        cycle_end = datetime.now()
        duration = (cycle_end - cycle_start).total_seconds()
        
        print(f"\n✅ Cycle completed in {duration:.1f} seconds")
        
        # Display results
        print("\n📊 RESULTS:")
        print(f"   Status: {results.get('status')}")
        print(f"   Duration: {results.get('duration_seconds', 0):.1f}s")
        print(f"   Research conducted: {len(results.get('research_conducted', []))}")
        print(f"   Insights extracted: {len(results.get('insights_extracted', []))}")
        print(f"   Tasks generated: {len(results.get('tasks_generated', []))}")
        
        if results.get('errors'):
            print(f"\n⚠️ Errors encountered:")
            for error in results['errors']:
                print(f"   - {error}")
        
        # Get performance metrics
        if hasattr(research_engine, 'get_enhanced_metrics'):
            metrics = research_engine.get_enhanced_metrics()
            
            print("\n📈 PERFORMANCE METRICS:")
            print(f"   Total operations: {metrics['total_operations']}")
            print(f"   Success rate: {metrics['success_rate']:.1%}")
            print(f"   Average duration: {metrics['average_duration']:.2f}s")
            print(f"   Timeouts: {metrics['timeout_operations']}")
            print(f"   Retries: {metrics['retry_operations']}")
            
            # Operation breakdown
            if metrics['recent_operations']:
                print("\n   Recent operations:")
                for op in metrics['recent_operations'][-5:]:
                    status = "✓" if op['success'] else "✗"
                    print(f"   {status} {op['operation']}: {op['duration']:.1f}s")
        
        # Analyze research quality
        if results.get('research_conducted'):
            print("\n📝 RESEARCH QUALITY:")
            for i, research in enumerate(results['research_conducted'][:3], 1):
                print(f"\n   Research {i}:")
                print(f"   - Topic: {research.get('topic', {}).get('topic', 'Unknown')}")
                print(f"   - Quality score: {research.get('quality_score', 0):.2f}")
                print(f"   - Area: {research.get('area', 'Unknown')}")
        
        # Save results
        output_data = {
            'test_summary': {
                'duration': duration,
                'status': results.get('status'),
                'timestamp': datetime.now().isoformat()
            },
            'results': results,
            'metrics': metrics if 'metrics' in locals() else {}
        }
        
        with open('research_final_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📄 Results saved to: research_final_results.json")
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {type(e).__name__}: {e}")
        logging.error("Test failed", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_final_research_engine())