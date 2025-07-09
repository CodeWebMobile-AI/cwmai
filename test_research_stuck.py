#!/usr/bin/env python3
"""
Quick test to see if research engine is stuck
"""
import asyncio
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

async def test_research():
    print("Testing if research engine gets stuck...")
    
    try:
        from scripts.research_evolution_engine import ResearchEvolutionEngine
        from scripts.state_manager import StateManager
        from scripts.ai_brain_factory import AIBrainFactory
        
        print("1. Initializing components...")
        state_manager = StateManager()
        ai_brain = AIBrainFactory.create_for_research()
        
        print("2. Creating research engine...")
        research_engine = ResearchEvolutionEngine(
            state_manager=state_manager,
            ai_brain=ai_brain,
            task_generator=None,
            self_improver=None,
            outcome_learning=None
        )
        
        print("3. Starting research cycle...")
        print("   If this hangs, the research engine is stuck")
        
        # Add timeout
        results = await asyncio.wait_for(
            research_engine.execute_research_cycle(),
            timeout=30.0  # 30 second timeout
        )
        
        print(f"✅ Research completed!")
        print(f"   Results type: {type(results)}")
        print(f"   Keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
        
    except asyncio.TimeoutError:
        print("❌ TIMEOUT: Research engine is stuck!")
        print("   The execute_research_cycle() method is not completing")
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_research())