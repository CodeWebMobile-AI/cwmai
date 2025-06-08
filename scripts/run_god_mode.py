import asyncio
import json
from god_mode_controller import GodModeController, GodModeConfig, GodModeIntensity

async def run_god_mode():
    # Get intensity from input or default
    intensity_str = 'experimental' or 'balanced'
    intensity = GodModeIntensity(intensity_str)
    
    print(f'Starting God Mode Controller (Intensity: {intensity.value})')
    
    # Create configuration
    config = GodModeConfig(
        intensity=intensity,
        enable_self_modification=True,
        enable_multi_repo=True,
        enable_predictive=True,
        enable_quantum=True,
        safety_threshold=0.8 if intensity != GodModeIntensity.EXPERIMENTAL else 0.6
    )
    
    # Initialize controller
    controller = GodModeController(config)
    
    try:
        # Run one cycle
        results = await controller.run_god_mode_cycle()
        
        # Save results
        with open('../god_mode_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\nCycle completed successfully!')
        print(f'Operations: {len(results["operations"])}')
        print(f'Errors: {len(results["errors"])}')
        
        # Show metrics
        print('\nMetrics:')
        for metric, value in results['metrics'].items():
            print(f'  {metric}: {value}')
        
    except Exception as e:
        print(f'Error during God Mode execution: {e}')
        await controller.emergency_shutdown()
        raise
    
    return results

# Run the god mode cycle
asyncio.run(run_god_mode())
