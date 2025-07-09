#!/usr/bin/env python3
"""
Smart Evolutionary Improvements System

This is the enhanced version that uses adaptive parameters, dynamic discovery,
and meta-learning instead of hard-coded values.
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime
import logging
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import smart components
from scripts.evolutionary_config import EvolutionaryConfig
from scripts.adaptive_parameters import AdaptiveParameterEngine
from scripts.improvement_discovery import ImprovementDiscoveryEngine
from scripts.evolutionary_memory import EvolutionaryMemory
from scripts.evolutionary_plugins import EvolutionaryPluginManager


async def run_smart_evolutionary():
    """Run the smart evolutionary system with adaptive intelligence."""
    # Initialize smart components
    config = EvolutionaryConfig()
    memory = EvolutionaryMemory()
    plugin_manager = EvolutionaryPluginManager()
    
    print("🧠 SMART EVOLUTIONARY IMPROVEMENTS SYSTEM")
    print("="*60)
    print("✨ Features:")
    print("  • Adaptive parameter tuning")
    print("  • Dynamic improvement type discovery")
    print("  • Meta-learning from history")
    print("  • Plugin-based extensibility")
    print("  • Context-aware strategies")
    print("="*60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run smart evolutionary improvements")
    parser.add_argument('--minutes', type=int, default=10,
                       help='Runtime in minutes (default: 10)')
    parser.add_argument('--profile', type=str, default='adaptive',
                       choices=['aggressive', 'conservative', 'balanced', 'adaptive'],
                       help='Strategy profile (default: adaptive)')
    parser.add_argument('--learn-from-history', action='store_true',
                       help='Use historical data to initialize parameters')
    parser.add_argument('--discover-types', action='store_true', default=True,
                       help='Enable dynamic improvement type discovery')
    parser.add_argument('--plugins', type=str, nargs='*',
                       help='Additional plugin directories to load')
    
    args = parser.parse_args()
    
    # Load additional plugins if specified
    if args.plugins:
        for plugin_dir in args.plugins:
            plugin_manager.plugin_dirs.append(plugin_dir)
        plugin_manager.discover_and_load_plugins()
    
    print(f"\n🔧 CONFIGURATION:")
    print(f"Profile: {args.profile}")
    print(f"Runtime: {args.minutes} minutes")
    print(f"Learning from history: {args.learn_from_history}")
    print(f"Type discovery: {args.discover_types}")
    print(f"Loaded plugins: {len(plugin_manager.list_plugins())}")
    
    # Show loaded plugins
    plugin_info = plugin_manager.get_plugin_info()
    for ptype, count in plugin_info["plugin_counts"].items():
        if count > 0:
            print(f"  • {ptype}: {count} plugins")
    
    print("="*60)
    
    # Import runner after loading configuration
    from run_evolutionary_improvements import EvolutionaryImprovementRunner
    
    # Create runner
    runner = EvolutionaryImprovementRunner(runtime_minutes=args.minutes)
    
    # Inject smart components
    runner.config = config
    runner.memory = memory
    runner.plugin_manager = plugin_manager
    runner.adaptive_engine = AdaptiveParameterEngine(config)
    runner.discovery_engine = ImprovementDiscoveryEngine(config)
    
    # Initialize from memory if requested
    if args.learn_from_history:
        await initialize_from_memory(runner, memory, config)
    
    # Apply profile settings
    profile = config.get_profile(args.profile)
    if args.profile != 'adaptive':
        # Apply profile multipliers
        apply_profile(runner, profile)
    else:
        # Use adaptive parameters
        runner.use_adaptive = True
        print("\n🤖 Using adaptive parameter optimization")
        print("Parameters will adjust automatically based on performance")
    
    # Enable type discovery
    if args.discover_types:
        runner.enable_discovery = True
        print("\n🔍 Dynamic type discovery enabled")
        print("New improvement types will be discovered during evolution")
    
    # Override evolutionary engine methods with smart versions
    inject_smart_methods(runner)
    
    print("\n🚀 Starting smart evolution...\n")
    
    start_time = datetime.now()
    
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        await runner._final_summary()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get summary even on error
        try:
            await runner._final_summary()
        except:
            pass
    finally:
        # Save learning data
        memory.save_session()
        config.save_config()
        
        # Save adaptation history
        if hasattr(runner, 'adaptive_engine'):
            runner.adaptive_engine.save_adaptation_history(
                f"adaptation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️ Total runtime: {elapsed:.1f} seconds")
    
    # Show learning summary
    print("\n📊 LEARNING SUMMARY:")
    memory_summary = memory.get_memory_summary()
    print(f"Sessions analyzed: {memory_summary['total_sessions']}")
    print(f"Successful patterns: {memory_summary['successful_patterns']}")
    print(f"Known failures: {memory_summary['known_failures']}")
    
    if memory_summary.get('best_configuration'):
        print(f"\n🏆 Best configuration found:")
        best = memory_summary['best_configuration']
        print(f"  Performance: {best['performance']:.3f}")
        print(f"  Parameters: {json.dumps(best['parameters'], indent=4)}")


async def initialize_from_memory(runner, memory, config):
    """Initialize runner from historical memory."""
    print("\n📚 Loading from evolutionary memory...")
    
    # Get best parameters for current context
    context = {
        "fitness_stagnation": 0,
        "diversity_score": 0.5,
        "success_trend": 0.5
    }
    
    best_params = memory.get_best_parameters_for_context(context)
    if best_params:
        print("Found historical best parameters:")
        for param, value in best_params.items():
            if hasattr(runner.evolutionary_engine, param):
                setattr(runner.evolutionary_engine, param, value)
                print(f"  • {param}: {value}")
    
    # Load successful patterns
    for imp_type in config.get_improvement_types()[:5]:
        patterns = memory.get_successful_improvement_patterns(imp_type)
        if patterns:
            print(f"\nSuccessful patterns for {imp_type}:")
            for pattern in patterns[:3]:
                print(f"  • {pattern['pattern']} (fitness: {pattern['avg_fitness']:.3f})")


def apply_profile(runner, profile):
    """Apply a strategy profile to the runner."""
    print(f"\n📋 Applying profile settings:")
    
    # Apply multipliers
    if "population_multiplier" in profile:
        new_pop = int(runner.evolutionary_engine.population_size * profile["population_multiplier"])
        runner.evolutionary_engine.population_size = new_pop
        print(f"  • Population size: {new_pop}")
    
    if "mutation_multiplier" in profile:
        new_mut = runner.evolutionary_engine.mutation_rate * profile["mutation_multiplier"]
        runner.evolutionary_engine.mutation_rate = min(0.95, new_mut)
        print(f"  • Mutation rate: {new_mut:.2f}")
    
    if "exploration_bias" in profile:
        # This affects crossover vs mutation preference
        runner.evolutionary_engine.crossover_rate = 1.0 - profile["exploration_bias"]
        print(f"  • Exploration bias: {profile['exploration_bias']:.2f}")


def inject_smart_methods(runner):
    """Inject smart methods into the runner."""
    
    # Store original methods
    runner._original_evolve = runner._evolve_generation
    runner._original_generate = runner.evolutionary_engine._generate_improvement_candidate
    
    # Create smart wrapper methods
    async def smart_evolve_generation():
        """Smart evolution with adaptive parameters."""
        # Collect generation metrics
        generation_metrics = {
            "generation": runner.generations_completed,
            "fitness_scores": [imp.get("fitness", 0) for imp in runner.evolutionary_engine.improvement_population],
            "diversity_score": calculate_diversity(runner.evolutionary_engine.improvement_population),
            "success_rate": calculate_success_rate(runner.improvements_created),
            "generation_time": 0,  # Will be updated
            "best_fitness": max(
                (imp.get("fitness", 0) for imp in runner.evolutionary_engine.improvement_population),
                default=0
            )
        }
        
        # Adapt parameters if enabled
        if hasattr(runner, 'use_adaptive') and runner.use_adaptive:
            new_params = runner.adaptive_engine.adapt_parameters(generation_metrics)
            
            # Apply new parameters
            for param, value in new_params.items():
                if hasattr(runner.evolutionary_engine, param):
                    setattr(runner.evolutionary_engine, param, value)
                    
            print(f"\n🔄 Adapted parameters for generation {runner.generations_completed + 1}:")
            for param, value in new_params.items():
                print(f"  • {param}: {value}")
        
        # Call original evolution
        await runner._original_evolve()
        
        # Discover new types if enabled
        if hasattr(runner, 'enable_discovery') and runner.enable_discovery:
            discoveries = await runner.discovery_engine.discover_improvement_types(
                runner.state_manager.get_state()
            )
            
            if discoveries:
                print(f"\n🔍 Discovered {len(discoveries)} new improvement types:")
                for disc in discoveries[:3]:
                    print(f"  • {disc['type']} (confidence: {disc['confidence']:.2f})")
    
    async def smart_generate_candidate(needs):
        """Smart improvement generation using plugins and discovery."""
        # Check if we should use a plugin
        if runner.plugin_manager and hasattr(runner, 'use_adaptive'):
            generators = runner.plugin_manager.list_plugins("generator")
            if generators and runner.generations_completed % 3 == 0:
                # Use plugin every 3rd generation for diversity
                plugin_name = generators[runner.generations_completed % len(generators)]
                try:
                    context = {"needs": needs, "generation": runner.generations_completed}
                    improvement = await runner.plugin_manager.execute_plugin(
                        "generator", plugin_name, context
                    )
                    if improvement:
                        print(f"  💡 Used plugin: {plugin_name}")
                        return improvement
                except:
                    pass
        
        # Use discovered types
        improvement_types = runner.config.get_improvement_types(include_discovered=True)
        
        # Call original with expanded types
        original_types = runner.evolutionary_engine.available_types if hasattr(
            runner.evolutionary_engine, 'available_types'
        ) else None
        
        # Temporarily set expanded types
        runner.evolutionary_engine.available_types = improvement_types
        result = await runner._original_generate(needs)
        
        # Restore original
        if original_types:
            runner.evolutionary_engine.available_types = original_types
            
        return result
    
    # Inject methods
    runner._evolve_generation = smart_evolve_generation
    runner.evolutionary_engine._generate_improvement_candidate = smart_generate_candidate


def calculate_diversity(population):
    """Calculate population diversity score using the diversity enhancer."""
    if len(population) < 2:
        return 0.0
    
    # Import and use diversity enhancer directly
    try:
        from scripts.system_aligned_diversity_enhancer import SystemAlignedDiversityEnhancer
        enhancer = SystemAlignedDiversityEnhancer()
        return enhancer.calculate_diversity_score(population)
    except:
        pass
    
    # Fallback to simple calculation
    types = [imp.get("type", "unknown") for imp in population]
    unique_types = len(set(types))
    
    approaches = [imp.get("approach", "default") for imp in population]
    hashable_approaches = []
    for approach in approaches:
        if isinstance(approach, dict):
            hashable_approaches.append(str(sorted(approach.items())))
        elif isinstance(approach, list):
            hashable_approaches.append(tuple(approach))
        else:
            hashable_approaches.append(approach)
    unique_approaches = len(set(hashable_approaches))
    
    diversity = (unique_types / len(types) + unique_approaches / len(approaches)) / 2
    return min(1.0, diversity)


def calculate_success_rate(improvements_created):
    """Calculate recent success rate."""
    if not improvements_created:
        return 0.5
        
    recent = improvements_created[-10:]  # Last 10 improvements
    successes = sum(1 for imp in recent if imp.get("verification", {}).get("is_deployable", False))
    
    return successes / len(recent) if recent else 0.5


if __name__ == "__main__":
    # Set up logging for detailed output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce sklearn verbosity
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    
    asyncio.run(run_smart_evolutionary())