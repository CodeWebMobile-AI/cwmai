#!/usr/bin/env python3
"""
Run Ultra God Mode Controller
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.ultra_god_mode_controller import UltraGodModeController
from scripts.ai_brain_factory import AIBrainFactory
from scripts.state_manager import StateManager


async def run_ultra_god_mode():
    """Run Ultra God Mode with real components."""
    print("=== Starting Ultra God Mode Controller ===\n")
    
    # Initialize AI Brain
    print("Initializing AI Brain...")
    ai_brain = AIBrainFactory.create_for_production()
    
    # Initialize State Manager
    print("Initializing State Manager...")
    state_manager = StateManager()
    
    # Create Ultra God Mode Controller
    print("Creating Ultra God Mode Controller...")
    controller = UltraGodModeController(ai_brain, state_manager)
    
    # Initialize the system
    await controller.initialize()
    
    print("\nUltra God Mode Ready!")
    print("This system will:")
    print("- Analyze current capabilities and identify gaps")
    print("- Research solutions and generate new tools")
    print("- Run experiments to test improvements")
    print("- Optimize existing components")
    print("- Create new system versions\n")
    
    # Run one improvement cycle
    print("Running improvement cycle...")
    results = await controller.run_improvement_cycle()
    
    print(f"\n=== Cycle Results ===")
    print(f"Research conducted: {results['research']['topics_researched']} topics")
    print(f"New capabilities: {len(results['new_capabilities'])}")
    print(f"Experiments run: {results['experiments']['hypotheses_tested']}")
    print(f"Optimizations made: {len(results['optimizations'])}")
    print(f"Total improvements: {results['total_improvements']}")
    
    if results.get('version'):
        print(f"\nNew version created: {results['version']['version']}")
        print(f"Status: {results['version']['status']}")
    
    # Get improvement summary
    summary = controller.get_improvement_summary()
    print(f"\n=== Overall Summary ===")
    print(f"Total capabilities: {summary['capabilities']['total']}")
    print(f"Capability gaps: {len(summary['capabilities']['gaps_identified'])}")
    print(f"Pending versions: {len(summary['pending_versions'])}")
    
    # Option to run continuous loop
    response = input("\nRun continuous improvement loop? (y/n): ")
    if response.lower() == 'y':
        max_cycles = input("Max cycles (press Enter for unlimited): ")
        max_cycles = int(max_cycles) if max_cycles else None
        await controller.continuous_improvement_loop(max_cycles)
    
    print("\nUltra God Mode complete!")


if __name__ == "__main__":
    # Run with proper environment
    asyncio.run(run_ultra_god_mode())