#!/usr/bin/env python3
"""
Test Research Evolution Engine with detailed logging
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('research_engine_test.log', mode='w')
    ]
)

# Reduce noise from some libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

async def test_research_engine():
    """Run research engine test with detailed monitoring"""
    print("=" * 80)
    print("RESEARCH ENGINE DETAILED TEST")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print("Logging to: research_engine_test.log")
    print("=" * 80)
    
    try:
        # Use the logged version
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from scripts.research_evolution_engine_logged import ResearchEvolutionEngine
        from scripts.state_manager import StateManager
        from scripts.ai_brain_factory import AIBrainFactory
        from scripts.intelligent_task_generator import IntelligentTaskGenerator
        from scripts.safe_self_improver import SafeSelfImprover
        from scripts.outcome_learning import OutcomeLearningSystem
        from scripts.dynamic_charter import DynamicCharter
        
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
            "max_concurrent_research": 2,  # Limit concurrency for testing
            "max_research_per_cycle": 3,   # Limit research items
            "min_insight_confidence": 0.5,
            "auto_implement_threshold": 0.9,  # Higher threshold for safety
            "enable_external_agent_research": False,  # Disable external for now
        })
        
        print("   ✓ Research engine created and configured")
        
        print("\n3. Running research cycles...")
        print("-" * 80)
        
        # Run 3 research cycles with monitoring
        for i in range(3):
            print(f"\n🔄 RESEARCH CYCLE {i+1}")
            logging.info(f"=== STARTING RESEARCH CYCLE {i+1} ===")
            
            cycle_start = datetime.now()
            
            # Set trigger context
            research_engine.last_trigger_context = {
                'reason': f"Test cycle {i+1}",
                'test_mode': True,
                'cycle_number': i+1
            }
            
            try:
                # Execute research cycle with timeout
                print(f"   Starting cycle at {cycle_start.strftime('%H:%M:%S')}...")
                
                results = await asyncio.wait_for(
                    research_engine.execute_research_cycle(),
                    timeout=120.0  # 2 minute timeout per cycle
                )
                
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
                
                # Log detailed results
                logging.info(f"Cycle {i+1} results: {json.dumps(results, indent=2)}")
                
                # Brief pause between cycles
                if i < 2:
                    print(f"\n   Waiting 5 seconds before next cycle...")
                    await asyncio.sleep(5)
                    
            except asyncio.TimeoutError:
                print(f"   ❌ TIMEOUT: Cycle {i+1} exceeded 2 minutes!")
                logging.error(f"Cycle {i+1} timed out after 120 seconds")
                
            except Exception as e:
                print(f"   ❌ ERROR: {type(e).__name__}: {e}")
                logging.error(f"Cycle {i+1} error", exc_info=True)
        
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
        
        # Get research state
        research_state = research_engine.get_research_state()
        print(f"\nResearch knowledge store entries: {len(research_state.get('knowledge_store', {}))}")
        
        print("\n✅ Test completed successfully!")
        print(f"Check 'research_engine_test.log' for detailed logs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {type(e).__name__}: {e}")
        logging.error("Test failed", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_research_engine())