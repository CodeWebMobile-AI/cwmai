#!/usr/bin/env python3
"""
Start CWMAI with Self-Amplifying Intelligence Activated

This script starts the continuous AI orchestrator with all self-amplifying
features enabled and optimized for maximum learning and improvement.
"""

import os
import sys
import asyncio
import signal
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from continuous_orchestrator import ContinuousOrchestrator


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Shutting down Self-Amplifying AI System...")
    sys.exit(0)


async def monitor_self_amplification():
    """Monitor and report on self-amplification progress."""
    print("\n📊 Self-Amplifying Intelligence Monitor Started")
    print("=" * 60)
    
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Read current metrics from system state
            try:
                import json
                with open('system_state.json', 'r') as f:
                    state = json.load(f)
                
                # Extract research metrics
                research_state = state.get('research_evolution_state', {})
                metrics = research_state.get('metrics', {})
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Self-Amplification Status:")
                print(f"  • Research Effectiveness: {metrics.get('research_effectiveness', 0):.1%}")
                print(f"  • Implementation Success: {metrics.get('implementation_success_rate', 0):.1%}")
                print(f"  • Performance Improvement: {metrics.get('performance_improvement_rate', 0):.1%}")
                print(f"  • Learning Accuracy: {metrics.get('learning_accuracy', 0):.1%}")
                
                # Check knowledge accumulation
                knowledge_path = 'research_knowledge/metadata/research_index.json'
                if os.path.exists(knowledge_path):
                    with open(knowledge_path, 'r') as f:
                        knowledge = json.load(f)
                    print(f"  • Knowledge Items: {len(knowledge.get('entries', []))}")
                
            except Exception as e:
                print(f"  ⚠️  Error reading metrics: {e}")
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")


async def main():
    """Start the self-amplifying AI system."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           CWMAI Self-Amplifying Intelligence System          ║
║                                                              ║
║  🧠 Continuous Learning: ACTIVE                              ║
║  🔄 Dynamic Triggering: ACTIVE                               ║
║  🌟 Proactive Research: ACTIVE                               ║
║  🌐 External Learning: ACTIVE                                ║
║  📈 Cross-Analysis: ACTIVE                                   ║
║                                                              ║
║  The system will now continuously:                           ║
║  • Research improvements every 20 minutes                    ║
║  • React to performance drops in 3 minutes                  ║
║  • Learn from external AI systems                            ║
║  • Discover optimization opportunities                       ║
║  • Implement high-confidence improvements                    ║
║  • Analyze patterns across all research                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Set environment for enhanced operation
    os.environ['EXECUTION_MODE'] = 'development'  # Enable all proactive features
    os.environ['SELF_AMPLIFYING'] = 'true'
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create orchestrator
    orchestrator = ContinuousOrchestrator()
    
    # Start monitoring task
    monitor_task = asyncio.create_task(monitor_self_amplification())
    
    try:
        print("\n🚀 Starting Self-Amplifying AI Orchestrator...")
        print("   Press Ctrl+C to stop\n")
        
        # Run the orchestrator
        await orchestrator.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
    except Exception as e:
        print(f"\n❌ Error in orchestrator: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cancel monitor
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Cleanup orchestrator
        await orchestrator.cleanup()
        print("\n✅ Self-Amplifying AI System stopped gracefully")


if __name__ == "__main__":
    print("🧠 Initializing CWMAI Self-Amplifying Intelligence...")
    
    # Create necessary directories
    os.makedirs('research_knowledge/raw_research', exist_ok=True)
    os.makedirs('research_knowledge/processed_insights', exist_ok=True)
    os.makedirs('research_knowledge/metadata', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the async main
    asyncio.run(main())