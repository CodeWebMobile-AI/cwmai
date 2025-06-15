#!/usr/bin/env python3
"""
Run the Dynamic AI Development Orchestrator

This script runs the complete production AI system orchestrating all workflows:
- Task Management (30 min cycles)
- Main AI Cycle (4 hour cycles)
- God Mode Controller (6 hour cycles)
- System Monitoring (daily cycles)
"""

import asyncio
import os
import sys
import signal
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env files
load_dotenv('.env.local')  # Load local environment first
load_dotenv()  # Then load .env as fallback

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.production_config import create_config, ExecutionMode
from scripts.production_orchestrator import ProductionOrchestrator


async def main():
    """Run the production AI orchestrator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Dynamic AI Development Orchestrator - Production System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dynamic_ai.py                    # Run in production mode
  python run_dynamic_ai.py --mode development # Run with faster cycles
  python run_dynamic_ai.py --mode test        # Run each cycle once
  python run_dynamic_ai.py --cycles task main # Run only specific cycles
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['development', 'production', 'test', 'legacy'],
        default='production',
        help='Execution mode (default: production)'
    )
    
    parser.add_argument(
        '--cycles',
        nargs='+',
        choices=['task', 'main', 'god_mode', 'monitoring'],
        help='Specific cycles to enable (default: all)'
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Run legacy God Mode only (5 min cycles)'
    )
    
    args = parser.parse_args()
    
    # Legacy mode - run old behavior
    if args.legacy or args.mode == 'legacy':
        await run_legacy_mode()
        return
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          DYNAMIC AI DEVELOPMENT ORCHESTRATOR                     ║
║                  PRODUCTION SYSTEM                               ║
║                                                                  ║
║  Complete autonomous AI system orchestrating:                    ║
║  • Task Management    - Issue/PR creation and tracking           ║
║  • Main AI Cycle     - Core development operations               ║
║  • God Mode Control  - Advanced AI capabilities                  ║
║  • System Monitoring - Health checks and reporting               ║
║                                                                  ║
║  All workflows run concurrently with proper scheduling           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Create configuration
    config = create_config(args.mode)
    
    # Apply cycle filter if specified
    if args.cycles:
        all_cycles = ['task', 'main', 'god_mode', 'monitoring']
        for cycle in all_cycles:
            if cycle not in args.cycles:
                getattr(config, f"{cycle}_cycle").enabled = False
                
    # Validate configuration
    print("\nValidating configuration...")
    if not config.validate():
        return
        
    print("✓ Configuration validated")
    print(f"\nMode: {config.mode.value}")
    print(f"Enabled cycles: {list(config.get_enabled_cycles().keys())}")
    
    # Show cycle intervals
    print("\nCycle intervals:")
    for name, cycle in config.get_enabled_cycles().items():
        hours = cycle.interval_seconds / 3600
        if hours >= 1:
            print(f"  • {name}: every {hours:.1f} hours")
        else:
            minutes = cycle.interval_seconds / 60
            print(f"  • {name}: every {minutes:.0f} minutes")
    
    # Create orchestrator
    print("\nInitializing Production Orchestrator...")
    orchestrator = ProductionOrchestrator(config)
    
    # Set up signal handlers for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            print("\n\nShutdown requested. Stopping all cycles gracefully...")
            asyncio.create_task(orchestrator.stop())
        else:
            print("\nForce shutdown requested. Exiting...")
            sys.exit(1)
            
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start orchestrator
    print("\nStarting all workflow cycles...")
    print("Press Ctrl+C to stop\n")
    
    try:
        await orchestrator.start()
        await orchestrator.wait_for_completion()
    except KeyboardInterrupt:
        pass
    finally:
        if orchestrator.running:
            await orchestrator.stop()
            
    # Print final summary
    print(f"\n{'='*60}")
    print("ORCHESTRATOR SESSION SUMMARY")
    print(f"{'='*60}")
    
    status = orchestrator.get_status()
    uptime = status['uptime_seconds']
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    
    print(f"\nRuntime: {hours}h {minutes}m")
    print(f"Total cycles executed: {status['total_cycles']}")
    print("\nCycle breakdown:")
    for cycle, count in status['cycle_counts'].items():
        if count > 0:
            print(f"  • {cycle}: {count} executions")
            
    print("\n✓ Production Orchestrator session complete")


async def run_legacy_mode():
    """Run the legacy God Mode only behavior."""
    from scripts.ai_brain_factory import AIBrainFactory
    from scripts.dynamic_god_mode_controller import DynamicGodModeController
    from scripts.god_mode_controller import GodModeConfig, IntensityLevel
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          DYNAMIC AI DEVELOPMENT ORCHESTRATOR                     ║
║                    (LEGACY MODE)                                 ║
║                                                                  ║
║  Running God Mode Controller only with 5 minute cycles           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check environment
    print("\nChecking environment...")
    
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
    print("LEGACY MODE SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cycles: {cycle_count}")
    print(f"Total value created: {total_value:.2f}")
    
    # Get final status
    status = await controller.get_system_status()
    
    print(f"\nProjects created: {status.get('dynamic_systems', {}).get('projects', {})}")
    print(f"Tasks generated: {status.get('dynamic_systems', {}).get('task_generation', {}).get('tasks_generated', 0)}")
    print(f"Outcomes learned: {status.get('dynamic_systems', {}).get('learning', {}).get('outcomes_recorded', 0)}")
    
    print("\n✓ Legacy mode session complete")


if __name__ == "__main__":
    asyncio.run(main())