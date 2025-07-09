#!/usr/bin/env python3
"""
Fast evolutionary run - bypasses ensemble for speed.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def fast_run():
    """Run a fast evolutionary test."""
    from scripts.evolutionary_improvement_engine import EvolutionaryImprovementEngine
    from scripts.state_manager import StateManager
    
    print("🚀 FAST EVOLUTIONARY TEST - Direct Engine Only")
    print("="*60)
    
    state_manager = StateManager()
    engine = EvolutionaryImprovementEngine(state_manager)
    
    # Reduce population for speed
    engine.population_size = 3
    engine.elite_size = 1
    
    start_time = datetime.now()
    
    try:
        # Initialize with simple improvements
        print("\n1. Initializing population...")
        await engine.initialize_population()
        print(f"✓ Population size: {len(engine.improvement_population)}")
        
        # Run 3 generations
        for gen in range(1, 4):
            print(f"\n2. Generation {gen}...")
            
            # Simple fitness assignment
            for i, imp in enumerate(engine.improvement_population):
                imp['fitness'] = 0.5 + (i * 0.1)
                imp['generation'] = gen
            
            # Evolve
            await engine.evolve_generation()
            
            # Show population
            print(f"   Population after evolution: {len(engine.improvement_population)}")
            if engine.fitness_history:
                if engine.fitness_history and isinstance(engine.fitness_history[-1], dict):
                print(f"   Average fitness: {engine.fitness_history[-1]['average_fitness']:.3f}")
            elif engine.fitness_history:
                print(f"   Average fitness: {engine.fitness_history[-1]:.3f}")
            
            # Quick pause
            await asyncio.sleep(1)
        
        # Apply best
        print("\n3. Applying best improvement...")
        result = await engine.apply_best_improvement()
        
        if result['success']:
            print(f"✅ Created: {result['file_path']}")
        else:
            print(f"❌ Failed: {result.get('reason', 'Unknown')}")
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Test completed in {elapsed:.1f} seconds")
        print(f"Generations: {engine.generation_count}")
        print(f"Fitness history: {[f'{f:.3f}' for f in engine.fitness_history]}")
        
        # Save results
        results = {
            'success': True,
            'runtime_seconds': elapsed,
            'generations': engine.generation_count,
            'fitness_history': engine.fitness_history,
            'final_population_size': len(engine.improvement_population),
            'improvements_created': result['success'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open('evolutionary_fast_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to: evolutionary_fast_results.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        with open('evolutionary_fast_error.json', 'w') as f:
            json.dump({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(fast_run())