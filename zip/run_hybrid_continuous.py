#!/usr/bin/env python3
"""
Run Hybrid Continuous AI System

Combines the best of Continuous System (24/7 operation) with
Dynamic God Mode (strategic intelligence and learning).
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
import argparse

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Apply work distribution fixes before importing orchestrator
# Commented out - fixes already applied in orchestrator
# import scripts.apply_work_distribution_fixes

from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator


def setup_logging(log_level: str = "INFO"):
    """Set up logging for the hybrid system."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hybrid_continuous.log', mode='a')
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('github').setLevel(logging.WARNING)


class HybridRunner:
    """Runner for the hybrid continuous AI system."""
    
    def __init__(self, max_workers: int = 5, enable_parallel: bool = True,
                 enable_round_robin: bool = True):
        """Initialize the hybrid runner."""
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        self.enable_round_robin = enable_round_robin
        self.orchestrator = None
        self.logger = logging.getLogger(__name__)
        self.shutdown_requested = False
        
    async def start(self):
        """Start the hybrid continuous AI system."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING ENHANCED HYBRID CONTINUOUS AI SYSTEM")
        self.logger.info("=" * 80)
        self.logger.info("Combining 24/7 operation with strategic intelligence")
        self.logger.info("Enhanced with Research Evolution, Pattern Learning, Swarm Intelligence")
        self.logger.info(f"Workers: {self.max_workers}")
        self.logger.info(f"Parallel: {self.enable_parallel}")
        self.logger.info("Research: Handled by knowledge-driven evolution system")
        self.logger.info("=" * 80)
        
        print("""
╔═══════════════════════════════════════════════════════════════╗
║         ENHANCED HYBRID CONTINUOUS AI SYSTEM                  ║
║                                                               ║
║  🧠 Dynamic God Mode Intelligence                            ║
║  🔄 24/7 Continuous Operation                                ║
║  📚 Outcome-Based Learning                                   ║
║  🎯 Repository-First Task Generation                         ║
║  ✅ Task Validation & Correction                             ║
║  📊 Evolving Charter System                                  ║
║                                                               ║
║  ✨ NEW ENHANCEMENTS:                                        ║
║  🔬 Research Evolution Engine + Knowledge Store              ║
║  🧩 Pattern Learning Engine                                  ║
║  📈 Feedback Learning System                                 ║
║  🐝 Swarm Intelligence (5 AI Agents)                        ║
║  🎯 Multi-Objective Optimization                             ║
║  🤖 Claude Executor Pool (Auto-starts)                       ║
║                                                               ║
║  Press Ctrl+C to shutdown gracefully                         ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        # Create enhanced orchestrator
        self.orchestrator = EnhancedHybridOrchestrator(
            max_workers=self.max_workers,
            enable_parallel=self.enable_parallel,
            enable_round_robin=self.enable_round_robin
        )
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Run the orchestrator
        try:
            await self.orchestrator.run()
        except Exception as e:
            self.logger.error(f"Fatal error in orchestrator: {e}")
            raise
        finally:
            await self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        if self.orchestrator:
            asyncio.create_task(self.orchestrator.shutdown())
    
    async def shutdown(self):
        """Graceful shutdown."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        self.logger.info("Hybrid system shutdown complete")


async def run_hybrid_system(args):
    """Run the hybrid continuous system with given arguments."""
    runner = HybridRunner(
        max_workers=args.workers,
        enable_parallel=not args.no_parallel,
        enable_round_robin=not args.no_round_robin
    )
    
    await runner.start()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid Continuous AI System - 24/7 operation with strategic intelligence"
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    
    
    parser.add_argument(
        '--no-round-robin',
        action='store_true',
        help='Disable round-robin AI provider selection (defaults to enabled)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    args = parser.parse_args()
    
    # Handle status request
    if args.status:
        print("\nHybrid Continuous AI System Status")
        print("=" * 40)
        
        # Check state files
        import glob
        import json
        
        state_files = glob.glob('*.json')
        relevant_files = ['system_state.json', 'continuous_orchestrator_state.json', 
                         'task_state.json']
        
        for file in relevant_files:
            if file in state_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    if file == 'continuous_orchestrator_state.json':
                        print(f"\n✅ Orchestrator State:")
                        print(f"   Last updated: {data.get('last_updated', 'Unknown')}")
                        print(f"   Metrics: {data.get('metrics', {})}")
                    elif file == 'system_state.json':
                        projects = data.get('projects', {})
                        print(f"\n✅ System State:")
                        print(f"   Active projects: {len(projects)}")
                        print(f"   Last updated: {data.get('last_updated', 'Unknown')}")
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
        
        # Check for hybrid metrics
        try:
            with open('continuous_orchestrator_state.json', 'r') as f:
                data = json.load(f)
                if 'hybrid_metrics' in data:
                    print(f"\n✅ Hybrid Features Active:")
                    print(f"   {data['hybrid_metrics']}")
        except:
            print("\n⚠️  Hybrid features not yet active")
        
        return
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Print configuration
    print(f"\nStarting Enhanced Hybrid Continuous AI System")
    print(f"Configuration:")
    print(f"  Workers: {args.workers}")
    print(f"  Parallel: {not args.no_parallel}")
    print(f"  Round-robin: {not args.no_round_robin}")
    print(f"  Log level: {args.log_level}")
    print(f"\nEnhancements Active:")
    print(f"  ✅ Pattern Learning System")
    print(f"  ✅ Feedback Learning")
    print(f"  ✅ Swarm Intelligence")
    print(f"  ✅ Multi-Objective Optimization")
    print(f"  ✅ Claude Executor Pool ({os.getenv('CLAUDE_EXECUTORS', 3)} executors)")
    print()
    
    # Run the system
    try:
        asyncio.run(run_hybrid_system(args))
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Hybrid system shutdown complete.")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()