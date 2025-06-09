import asyncio
import json
from ai_brain_factory import AIBrainFactory
from dynamic_god_mode_controller import DynamicGodModeController  
from god_mode_controller import GodModeConfig, IntensityLevel

async def run_god_mode():
    # Get intensity from input or default
    intensity_str = 'experimental' or 'balanced'
    intensity = IntensityLevel(intensity_str)
    
    print(f'Starting Dynamic God Mode Controller (Intensity: {intensity.value})')
    
    # Initialize AI Brain using factory
    print('Initializing AI Brain for workflow...')
    ai_brain = AIBrainFactory.create_for_workflow()
    print('✓ AI Brain ready with workflow optimization')
    
    # Create configuration
    config = GodModeConfig(
        intensity=intensity,
        enable_self_modification=True,
        enable_multi_repo=True,
        enable_predictive=True,
        enable_quantum=True,
        safety_threshold=0.8 if intensity != IntensityLevel.EXPERIMENTAL else 0.6
    )
    
    # Initialize dynamic controller with AI brain
    controller = DynamicGodModeController(config, ai_brain)
    
    try:
        # Run one cycle
        results = await controller.run_god_mode_cycle()
        
        # Save results
        with open('../god_mode_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\nCycle completed successfully!')
        print(f'Operations: {len(results["operations"])}')
        print(f'Errors: {len(results["errors"])}')
        print(f'Tasks generated: {results.get("tasks_generated", 0)}')
        print(f'Tasks validated: {results.get("tasks_validated", 0)}')
        
        # Show metrics
        print('\nMetrics:')
        for metric, value in results['metrics'].items():
            print(f'  {metric}: {value}')
        
        # Show learnings
        if results.get('learnings'):
            print('\nLearnings:')
            for learning in results['learnings'][:5]:
                print(f'  - {learning.get("type", "unknown")}: {learning.get("impact", "N/A")}')
        
    except Exception as e:
        print(f'Error during Dynamic God Mode execution: {e}')
        await controller.emergency_shutdown()
        raise
    
    return results

# Run the god mode cycle
asyncio.run(run_god_mode())
