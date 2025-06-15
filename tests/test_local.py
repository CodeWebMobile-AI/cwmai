#!/usr/bin/env python
"""Local testing script with preset intensity level"""
import asyncio
import os
from scripts.dynamic_god_mode_controller import DynamicGodModeController, GodModeConfig, IntensityLevel
from scripts.ai_brain_factory import AIBrainFactory

async def test_local():
    # Initialize AI Brain
    print("Initializing AI Brain for local testing...")
    ai_brain = AIBrainFactory.create_for_production()
    
    # Set intensity level directly (change as needed)
    intensity = IntensityLevel.BALANCED  # or CONSERVATIVE, AGGRESSIVE, EXPERIMENTAL
    
    config = GodModeConfig(
        intensity=intensity,
        enable_self_modification=False,
        enable_multi_repo=True,
        enable_predictive=True,
        enable_quantum=False,
        max_parallel_operations=3
    )
    
    print(f"✓ Configured with {intensity.value} intensity")
    
    # Initialize controller
    controller = DynamicGodModeController(config, ai_brain)
    
    # Run a single cycle for testing
    print("\nRunning test cycle...")
    result = await controller.run_god_mode_cycle()
    
    print("\n✓ Test cycle completed")
    print(f"Operations executed: {len(result.get('operations', []))}")
    
    return result

if __name__ == "__main__":
    # Load environment
    os.system('export $(cat .env.local | grep -v "^#" | xargs)')
    
    # Run test
    asyncio.run(test_local())