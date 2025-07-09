#!/usr/bin/env python3
"""
Simplified Research Engine Test with minimal logging
"""
import asyncio
import sys
import os
import logging
from datetime import datetime
import json
import time

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('research_test_simple.log', mode='w')
    ]
)

# Patch the ResearchEvolutionEngine to add logging
def add_logging_to_research():
    """Monkey patch the research engine to add logging"""
    from scripts.research_evolution_engine import ResearchEvolutionEngine
    
    # Save original methods
    original_execute_cycle = ResearchEvolutionEngine.execute_research_cycle
    original_analyze = ResearchEvolutionEngine._analyze_system_needs
    original_select = ResearchEvolutionEngine._select_research_topics
    original_execute = ResearchEvolutionEngine._execute_research
    original_extract = ResearchEvolutionEngine._extract_insights
    original_generate = ResearchEvolutionEngine._generate_implementation_tasks
    
    # Create wrapped methods with logging
    async def logged_execute_cycle(self):
        logging.info("=== EXECUTE_RESEARCH_CYCLE STARTED ===")
        start_time = time.time()
        try:
            result = await original_execute_cycle(self)
            duration = time.time() - start_time
            logging.info(f"=== EXECUTE_RESEARCH_CYCLE COMPLETED in {duration:.2f}s ===")
            logging.info(f"Result status: {result.get('status')}")
            logging.info(f"Research conducted: {len(result.get('research_conducted', []))}")
            logging.info(f"Insights extracted: {len(result.get('insights_extracted', []))}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"=== EXECUTE_RESEARCH_CYCLE FAILED after {duration:.2f}s ===")
            logging.error(f"Error: {type(e).__name__}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    async def logged_analyze(self):
        logging.info("  -> _analyze_system_needs starting...")
        start_time = time.time()
        try:
            result = await original_analyze(self)
            duration = time.time() - start_time
            logging.info(f"  <- _analyze_system_needs completed in {duration:.2f}s")
            logging.info(f"     Result type: {type(result)}, Keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"  <- _analyze_system_needs FAILED after {duration:.2f}s: {e}")
            raise
    
    async def logged_select(self, system_needs):
        logging.info("  -> _select_research_topics starting...")
        start_time = time.time()
        try:
            result = await original_select(self, system_needs)
            duration = time.time() - start_time
            logging.info(f"  <- _select_research_topics completed in {duration:.2f}s")
            logging.info(f"     Selected {len(result) if isinstance(result, list) else '?'} topics")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"  <- _select_research_topics FAILED after {duration:.2f}s: {e}")
            raise
    
    async def logged_execute(self, topics):
        logging.info(f"  -> _execute_research starting with {len(topics) if isinstance(topics, list) else '?'} topics...")
        start_time = time.time()
        try:
            result = await original_execute(self, topics)
            duration = time.time() - start_time
            logging.info(f"  <- _execute_research completed in {duration:.2f}s")
            logging.info(f"     Results: {len(result) if isinstance(result, list) else '?'} items")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"  <- _execute_research FAILED after {duration:.2f}s: {e}")
            raise
    
    async def logged_extract(self, research_results):
        logging.info(f"  -> _extract_insights starting with {len(research_results) if isinstance(research_results, list) else '?'} results...")
        start_time = time.time()
        try:
            result = await original_extract(self, research_results)
            duration = time.time() - start_time
            logging.info(f"  <- _extract_insights completed in {duration:.2f}s")
            logging.info(f"     Insights: {len(result) if isinstance(result, list) else '?'} items")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"  <- _extract_insights FAILED after {duration:.2f}s: {e}")
            raise
    
    async def logged_generate(self, insights):
        logging.info(f"  -> _generate_implementation_tasks starting with {len(insights) if isinstance(insights, list) else '?'} insights...")
        start_time = time.time()
        try:
            result = await original_generate(self, insights)
            duration = time.time() - start_time
            logging.info(f"  <- _generate_implementation_tasks completed in {duration:.2f}s")
            logging.info(f"     Tasks: {len(result) if isinstance(result, list) else '?'} items")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"  <- _generate_implementation_tasks FAILED after {duration:.2f}s: {e}")
            raise
    
    # Apply patches
    ResearchEvolutionEngine.execute_research_cycle = logged_execute_cycle
    ResearchEvolutionEngine._analyze_system_needs = logged_analyze
    ResearchEvolutionEngine._select_research_topics = logged_select
    ResearchEvolutionEngine._execute_research = logged_execute
    ResearchEvolutionEngine._extract_insights = logged_extract
    ResearchEvolutionEngine._generate_implementation_tasks = logged_generate
    
    logging.info("Research engine logging patches applied")

async def test_research_engine():
    """Run research engine test"""
    print("=" * 80)
    print("SIMPLIFIED RESEARCH ENGINE TEST")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print("Logging to: research_test_simple.log")
    print("=" * 80)
    
    try:
        # Apply logging patches
        add_logging_to_research()
        
        from scripts.research_evolution_engine import ResearchEvolutionEngine
        from scripts.state_manager import StateManager
        from scripts.ai_brain_factory import AIBrainFactory
        
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
            task_generator=None,  # Minimal setup
            self_improver=None,
            outcome_learning=None
        )
        
        # Configure for minimal testing
        research_engine.config.update({
            "max_concurrent_research": 1,
            "max_research_per_cycle": 2,
            "enable_external_agent_research": False,
        })
        
        print("   ✓ Research engine created")
        
        print("\n3. Executing single research cycle...")
        print("-" * 80)
        
        cycle_start = datetime.now()
        
        # Execute with timeout
        results = await asyncio.wait_for(
            research_engine.execute_research_cycle(),
            timeout=180.0  # 3 minute timeout
        )
        
        cycle_end = datetime.now()
        duration = (cycle_end - cycle_start).total_seconds()
        
        print(f"\n✅ Cycle completed in {duration:.1f} seconds")
        print(f"Status: {results.get('status')}")
        print(f"Research conducted: {len(results.get('research_conducted', []))}")
        print(f"Insights extracted: {len(results.get('insights_extracted', []))}")
        print(f"Tasks generated: {len(results.get('tasks_generated', []))}")
        
        if results.get('errors'):
            print(f"⚠️ Errors: {results['errors']}")
        
        # Save results
        with open('research_cycle_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: research_cycle_results.json")
        
    except asyncio.TimeoutError:
        print(f"\n❌ TIMEOUT: Research cycle exceeded 3 minutes!")
        logging.error("Research cycle timed out after 180 seconds")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        logging.error("Test failed", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_research_engine())