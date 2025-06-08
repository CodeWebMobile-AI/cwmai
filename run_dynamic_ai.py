#!/usr/bin/env python3
"""
Run the Dynamic AI Development Orchestrator

This script runs the fully dynamic AI system with no hardcoded values.
All decisions are made through AI reasoning.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.ai_brain_factory import AIBrainFactory
from scripts.dynamic_god_mode_controller import DynamicGodModeController
from scripts.god_mode_controller import GodModeConfig, IntensityLevel


async def main():
    """Run the dynamic AI system."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          DYNAMIC AI DEVELOPMENT ORCHESTRATOR                     ║
║                                                                  ║
║  Fully autonomous AI system that:                                ║
║  • Creates software projects from Laravel React starter kit      ║
║  • Manages a portfolio of applications                           ║
║  • Learns from outcomes to improve                               ║
║  • Uses only AI reasoning - no hardcoded logic                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check environment
    print("Checking environment...")
    
    required_keys = {
        'ANTHROPIC_API_KEY': 'Required for Claude AI',
        'GITHUB_TOKEN': 'Required for GitHub operations (or CLAUDE_PAT)',
        'OPENAI_API_KEY': 'Optional for GPT models',
        'GEMINI_API_KEY': 'Optional for Gemini models'
    }
    
    missing_required = []
    for key, desc in required_keys.items():
        if 'Required' in desc and not (os.getenv(key) or (key == 'GITHUB_TOKEN' and os.getenv('CLAUDE_PAT'))):
            missing_required.append(f"{key} - {desc}")
            
    if missing_required:
        print("\n❌ Missing required environment variables:")
        for item in missing_required:
            print(f"   {item}")
        print("\nPlease set these environment variables and try again.")
        return
        
    print("✓ Environment configured\n")
    
    # Initialize AI Brain using factory
    print("Initializing AI Brain for production...")
    ai_brain = AIBrainFactory.create_for_production()
    print("✓ AI Brain ready with full capabilities\n")
    
    # Configure system
    print("Select intensity level:")
    print("1. CONSERVATIVE - Careful, validated actions")
    print("2. BALANCED - Normal operation (recommended)")
    print("3. AGGRESSIVE - Maximum capability utilization")
    print("4. EXPERIMENTAL - Cutting-edge, higher risk")
    
    choice = input("\nEnter choice (1-4) [2]: ").strip() or "2"
    
    intensity_map = {
        "1": IntensityLevel.CONSERVATIVE,
        "2": IntensityLevel.BALANCED,
        "3": IntensityLevel.AGGRESSIVE,
        "4": IntensityLevel.EXPERIMENTAL
    }
    
    intensity = intensity_map.get(choice, IntensityLevel.BALANCED)
    
    config = GodModeConfig(
        intensity=intensity,
        enable_self_modification=False,  # Safety first
        enable_multi_repo=True,
        enable_predictive=True,
        enable_quantum=False,  # No real quantum needed
        max_parallel_operations=3
    )
    
    print(f"\n✓ Configured with {intensity.value} intensity\n")
    
    # Initialize controller
    print("Initializing Dynamic God Mode Controller...")
    controller = DynamicGodModeController(config, ai_brain)
    print("✓ All systems initialized\n")
    
    # Main loop
    print("Starting autonomous operation...\n")
    
    cycle_count = 0
    total_value = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*60}")
            print(f"CYCLE {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            # Run cycle
            result = await controller.run_god_mode_cycle()
            
            # Show results
            print(f"\n✓ Cycle {cycle_count} completed in {result.get('duration', 0):.2f}s")
            
            # Show operations
            operations = result.get('operations', [])
            if operations:
                print(f"\nOperations executed: {len(operations)}")
                for op in operations:
                    task = op.get('task', {})
                    status = "✓" if op.get('result', {}).get('success') else "✗"
                    print(f"  {status} {task.get('type', 'Unknown')}: {task.get('title', 'Unknown')}")
                    
            # Show value created
            cycle_value = sum(
                op.get('result', {}).get('value_assessment', {}).get('value_score', 0)
                for op in operations
            )
            total_value += cycle_value
            
            if cycle_value > 0:
                print(f"\nValue created this cycle: {cycle_value:.2f}")
                print(f"Total value created: {total_value:.2f}")
                
            # Show learnings
            learnings = result.get('learnings', [])
            if learnings:
                print("\nLearnings:")
                for learning in learnings:
                    print(f"  • {learning.get('type', 'Unknown')}")
                    
            # Show recommendations
            recommendations = result.get('recommendations', {})
            if recommendations and isinstance(recommendations, dict):
                print("\nRecommendations for next cycle:")
                for key, value in recommendations.items():
                    if isinstance(value, list) and value:
                        print(f"  • {key}: {value[0] if value else 'None'}")
                        
            # Wait before next cycle
            print(f"\nWaiting 5 minutes before next cycle...")
            print("Press Ctrl+C to stop")
            
            await asyncio.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        
    # Final summary
    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cycles: {cycle_count}")
    print(f"Total value created: {total_value:.2f}")
    
    # Get final status
    status = await controller.get_system_status()
    
    print(f"\nProjects created: {status.get('dynamic_systems', {}).get('projects', {})}")
    print(f"Tasks generated: {status.get('dynamic_systems', {}).get('task_generation', {}).get('tasks_generated', 0)}")
    print(f"Outcomes learned: {status.get('dynamic_systems', {}).get('learning', {}).get('outcomes_recorded', 0)}")
    
    print("\n✓ Dynamic AI Orchestrator session complete")


if __name__ == "__main__":
    asyncio.run(main())