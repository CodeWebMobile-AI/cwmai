#!/usr/bin/env python3
"""
Test that the evolution system generates real code
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.knowledge_driven_evolutionary_engine import KnowledgeDrivenEvolutionaryEngine
from scripts.state_manager import StateManager
from scripts.evolutionary_code_generator_fix import RealCodeEvolutionFix

async def test_real_code_generation():
    """Test generating a single improvement with real code."""
    
    print("🧪 Testing Real Code Generation in Evolution System\n")
    
    # Create engine
    state_manager = StateManager()
    engine = KnowledgeDrivenEvolutionaryEngine(state_manager)
    
    # Create a test improvement
    test_improvement = {
        'name': 'test_cache_manager',
        'type': 'cache',
        'description': 'A simple cache manager for testing',
        'approach': 'Implement LRU cache with get/set operations',
        'metrics': ['hit_rate', 'miss_rate'],
        'integration_points': ['memory_store']
    }
    
    print("1. Testing placeholder detection...")
    
    # Generate placeholder code (what the base system does)
    placeholder_code = engine._generate_code_for_improvement(test_improvement)
    print(f"Generated code marked as needing real implementation: {test_improvement.get('needs_real_code', False)}")
    print(f"Error message: {test_improvement.get('code_generation_error', 'None')}")
    
    if test_improvement.get('needs_real_code', False):
        print("\n✅ Correctly detected placeholder code\n")
        
        print("2. Testing real code regeneration...")
        # Try to regenerate real code
        success = await engine._regenerate_real_code(test_improvement)
        
        if success:
            print("✅ Successfully generated real code!")
            print(f"Validated: {test_improvement.get('validated', False)}")
            
            # Show a snippet of the generated code
            code = test_improvement.get('code', '')
            lines = code.split('\n')
            print("\nGenerated code snippet:")
            print("=" * 50)
            for i, line in enumerate(lines[:20]):
                print(f"{i+1:3d} | {line}")
            if len(lines) > 20:
                print("... (truncated)")
            print("=" * 50)
            
            # Check test results
            if 'test_results' in test_improvement:
                results = test_improvement['test_results']
                print(f"\nTest Results:")
                print(f"  Tests run: {results.get('tests_run', 0)}")
                print(f"  Tests passed: {results.get('tests_passed', 0)}")
                print(f"  All passed: {results.get('all_passed', False)}")
        else:
            print("❌ Failed to generate real code")
    else:
        print("❌ Failed to detect placeholder code")
    
    print("\n3. Testing fitness evaluation...")
    
    # Test fitness for placeholder vs real code
    placeholder_fitness = await engine.evaluate_fitness({
        'name': 'test1',
        'code': 'def apply(self, context): return {"success": True}',
        'type': 'test'
    })
    
    real_fitness = await engine.evaluate_fitness({
        'name': 'test2',
        'code': test_improvement.get('code', ''),
        'validated': True,
        'type': 'test'
    })
    
    print(f"Placeholder code fitness: {placeholder_fitness:.3f}")
    print(f"Real code fitness: {real_fitness:.3f}")
    print(f"Fitness ratio (real/placeholder): {real_fitness/placeholder_fitness:.1f}x")
    
    print("\n✨ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_real_code_generation())