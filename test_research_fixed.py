#!/usr/bin/env python3
"""
Test Research Engine with Comprehensive Fixes Applied

This test runs the research engine with:
- Proper timeout handling on all operations
- Retry logic with exponential backoff
- Circuit breaker pattern for provider failures
- Error boundaries and graceful degradation
- Comprehensive monitoring and metrics
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import json

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('research_engine_fixed.log', mode='w')
    ]
)

# Reduce noise from some libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

async def test_fixed_research_engine():
    """Run research engine test with all fixes applied"""
    print("=" * 80)
    print("RESEARCH ENGINE TEST WITH COMPREHENSIVE FIXES")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print("Logging to: research_engine_fixed.log")
    print("=" * 80)
    
    try:
        # Import required modules
        from scripts.research_evolution_engine import ResearchEvolutionEngine
        from scripts.state_manager import StateManager
        from scripts.ai_brain_factory import AIBrainFactory
        from scripts.intelligent_task_generator import IntelligentTaskGenerator
        from scripts.safe_self_improver import SafeSelfImprover
        from scripts.outcome_learning import OutcomeLearningSystem
        from scripts.dynamic_charter import DynamicCharter
        from scripts.research_engine_fixes import apply_research_engine_fixes
        
        print("\n1. Initializing components...")
        logging.info("=== INITIALIZATION PHASE ===")
        
        # Create state manager
        print("   Creating state manager...")
        state_manager = StateManager()
        print("   ✓ State manager created")
        
        # Create AI brain
        print("   Creating AI brain...")
        ai_brain = AIBrainFactory.create_for_research()
        print("   ✓ AI brain created")
        
        # Create task generator
        print("   Creating task generator...")
        charter_system = DynamicCharter(ai_brain)
        task_generator = IntelligentTaskGenerator(
            ai_brain,
            charter_system,
            None  # outcome_learning will be set later
        )
        print("   ✓ Task generator created")
        
        # Create self improver
        print("   Creating self improver...")
        self_improver = SafeSelfImprover(".", 24)
        print("   ✓ Self improver created")
        
        # Create outcome learning
        print("   Creating outcome learning system...")
        outcome_learning = OutcomeLearningSystem(ai_brain)
        task_generator.learning_system = outcome_learning
        print("   ✓ Outcome learning created")
        
        print("\n2. Creating research engine...")
        logging.info("=== CREATING RESEARCH ENGINE ===")
        
        research_engine = ResearchEvolutionEngine(
            state_manager=state_manager,
            ai_brain=ai_brain,
            task_generator=task_generator,
            self_improver=self_improver,
            outcome_learning=outcome_learning
        )
        
        # Configure for testing
        research_engine.config.update({
            "max_concurrent_research": 2,
            "max_research_per_cycle": 3,
            "min_insight_confidence": 0.5,
            "auto_implement_threshold": 0.9,
            "enable_external_agent_research": False,
        })
        
        print("   ✓ Research engine created")
        
        print("\n3. Applying comprehensive fixes...")
        logging.info("=== APPLYING FIXES ===")
        
        # Apply all fixes
        research_engine = apply_research_engine_fixes(research_engine)
        print("   ✓ Fixes applied:")
        print("      - Timeout handling on all operations")
        print("      - Retry logic with exponential backoff")
        print("      - Circuit breaker pattern")
        print("      - Error boundaries and fallbacks")
        print("      - Comprehensive monitoring")
        
        print("\n4. Running research cycles with monitoring...")
        print("-" * 80)
        
        # Run 2 research cycles
        for i in range(2):
            print(f"\n🔄 RESEARCH CYCLE {i+1}")
            logging.info(f"=== STARTING RESEARCH CYCLE {i+1} ===")
            
            cycle_start = datetime.now()
            
            # Set trigger context
            research_engine.last_trigger_context = {
                'reason': f"Test cycle {i+1} with fixes",
                'test_mode': True,
                'cycle_number': i+1
            }
            
            try:
                # Execute research cycle
                print(f"   Starting cycle at {cycle_start.strftime('%H:%M:%S')}...")
                
                results = await research_engine.execute_research_cycle()
                
                cycle_end = datetime.now()
                duration = (cycle_end - cycle_start).total_seconds()
                
                print(f"   ✓ Cycle completed in {duration:.1f} seconds")
                
                # Analyze results
                print("\n   📊 CYCLE RESULTS:")
                print(f"      Status: {results.get('status', 'unknown')}")
                print(f"      Research conducted: {len(results.get('research_conducted', []))}")
                print(f"      Insights extracted: {len(results.get('insights_extracted', []))}")
                print(f"      Tasks generated: {len(results.get('tasks_generated', []))}")
                print(f"      Implementations: {len(results.get('implementations', []))}")
                
                if results.get('errors'):
                    print(f"      ⚠️ Errors: {results['errors']}")
                
                # Get enhanced metrics
                if hasattr(research_engine, 'get_enhanced_metrics'):
                    metrics = research_engine.get_enhanced_metrics()
                    print("\n   📈 PERFORMANCE METRICS:")
                    print(f"      Total operations: {metrics['total_operations']}")
                    print(f"      Success rate: {metrics['success_rate']:.1%}")
                    print(f"      Average duration: {metrics['average_duration']:.2f}s")
                    print(f"      Timeouts: {metrics['timeout_operations']}")
                    print(f"      Retries: {metrics['retry_operations']}")
                    print(f"      Circuit breaker trips: {metrics['circuit_breaker_trips']}")
                    
                    if metrics['provider_failures']:
                        print(f"      Provider failures: {metrics['provider_failures']}")
                    
                    if metrics['error_types']:
                        print(f"      Error types: {metrics['error_types']}")
                
                # Log detailed results
                logging.info(f"Cycle {i+1} results: {json.dumps(results, indent=2)}")
                
                # Brief pause between cycles
                if i < 1:
                    print(f"\n   Waiting 5 seconds before next cycle...")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                print(f"   ❌ ERROR: {type(e).__name__}: {e}")
                logging.error(f"Cycle {i+1} error", exc_info=True)
                
                # Still get metrics on failure
                if hasattr(research_engine, 'get_enhanced_metrics'):
                    metrics = research_engine.get_enhanced_metrics()
                    print(f"\n   Metrics after failure:")
                    print(f"      Failed operations: {metrics['failed_operations']}")
                    print(f"      Error types: {metrics.get('error_types', {})}")
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        # Get final statistics
        stats = research_engine.cycle_statistics
        print(f"Total cycles attempted: {research_engine.research_cycles}")
        print(f"Successful cycles: {stats['successful_cycles']}")
        print(f"Failed cycles: {stats['failed_cycles']}")
        print(f"Total research conducted: {stats['total_research_conducted']}")
        print(f"Total insights extracted: {stats['total_insights_extracted']}")
        print(f"Total tasks generated: {stats['total_tasks_generated']}")
        
        # Get final enhanced metrics
        if hasattr(research_engine, 'get_enhanced_metrics'):
            final_metrics = research_engine.get_enhanced_metrics()
            print(f"\n📊 FINAL PERFORMANCE METRICS:")
            print(f"   Total operations executed: {final_metrics['total_operations']}")
            print(f"   Overall success rate: {final_metrics['success_rate']:.1%}")
            print(f"   Average operation duration: {final_metrics['average_duration']:.2f}s")
            print(f"   Total timeouts: {final_metrics['timeout_operations']}")
            print(f"   Total retries: {final_metrics['retry_operations']}")
            print(f"   Circuit breaker activations: {final_metrics['circuit_breaker_trips']}")
            
            # Save metrics
            with open('research_engine_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=2)
            print(f"\n📄 Metrics saved to: research_engine_metrics.json")
        
        print("\n✅ Test completed successfully!")
        print(f"Check 'research_engine_fixed.log' for detailed logs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {type(e).__name__}: {e}")
        logging.error("Test failed", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_research_engine())