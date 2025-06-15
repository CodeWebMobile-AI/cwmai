#!/usr/bin/env python3
"""
Continuous 24/7 AI System Runner

Replaces interval-based scheduling with intelligent continuous operation.
The AI system never rests - always working on the highest priority task available.
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, timezone
from typing import Optional
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')  # Load local environment first
load_dotenv()  # Then load .env as fallback

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from continuous_orchestrator import ContinuousOrchestrator
from enhanced_continuous_orchestrator import EnhancedContinuousOrchestrator
from production_config import create_config, ExecutionMode


def setup_logging(log_level: str = "INFO"):
    """Set up logging for the continuous system."""
    from deduplicating_formatter import SmartDeduplicatingHandler
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler('continuous_ai.log', mode='a')
    
    # Set formatter for original handlers
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Wrap handlers with deduplication
    dedup_console = SmartDeduplicatingHandler(console_handler, dedup_window=5)
    dedup_file = SmartDeduplicatingHandler(file_handler, dedup_window=5)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[dedup_console, dedup_file]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


class ContinuousAIRunner:
    """Runner for the continuous 24/7 AI system."""
    
    def __init__(self, max_workers: int = 3, enable_parallel: bool = True, mode: str = "production", 
                 enable_research: bool = True, enable_round_robin: bool = False, 
                 enable_worker_monitor: bool = False, enable_mcp: bool = True):
        """Initialize the runner.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_parallel: Whether to enable parallel processing
            mode: Execution mode (production, development, test)
            enable_research: Whether to enable research evolution engine
            enable_round_robin: Whether to enable round-robin AI provider selection
            enable_worker_monitor: Whether to enable automatic worker monitoring
            enable_mcp: Whether to enable MCP (Model Context Protocol) integration
        """
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        self.mode = mode
        self.enable_research = enable_research
        self.enable_round_robin = enable_round_robin
        self.enable_worker_monitor = enable_worker_monitor
        self.enable_mcp = enable_mcp
        self.orchestrator: Optional[ContinuousOrchestrator] = None
        self.logger = logging.getLogger(__name__)
        self.shutdown_requested = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Load configuration
        self.config = create_config(mode)
        
    async def start(self):
        """Start the continuous AI system."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING CONTINUOUS 24/7 AI SYSTEM")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Workers: {self.max_workers}")
        self.logger.info(f"Parallel: {self.enable_parallel}")
        self.logger.info(f"Research: {self.enable_research}")
        self.logger.info(f"Round-Robin AI: {self.enable_round_robin}")
        self.logger.info(f"Worker Monitor: {self.enable_worker_monitor}")
        self.logger.info(f"MCP Integration: {self.enable_mcp}")
        self.logger.info(f"Started at: {datetime.now(timezone.utc)}")
        self.logger.info("=" * 80)
        
        # Validate configuration
        if not self.config.validate():
            self.logger.error("Configuration validation failed!")
            return False
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        try:
            # Create and start orchestrator
            if self.enable_mcp and self.mode != "test":
                # Use enhanced orchestrator with MCP support
                self.logger.info("🔧 Using Enhanced Orchestrator with MCP support")
                self.orchestrator = EnhancedContinuousOrchestrator(
                    max_workers=self.max_workers,
                    enable_parallel=self.enable_parallel,
                    enable_research=self.enable_research,
                    enable_round_robin=self.enable_round_robin,
                    enable_mcp=True
                )
            else:
                # Use standard orchestrator
                self.logger.info("📦 Using Standard Orchestrator")
                self.orchestrator = ContinuousOrchestrator(
                    max_workers=self.max_workers,
                    enable_parallel=self.enable_parallel,
                    enable_research=self.enable_research,
                    enable_round_robin=self.enable_round_robin
                )
            
            # Start worker monitor if enabled
            if self.enable_worker_monitor:
                self.monitor_task = asyncio.create_task(self._run_worker_monitor())
                self.logger.info("📊 Worker monitoring started")
            
            # Initialize the orchestrator if using enhanced version
            if hasattr(self.orchestrator, 'initialize'):
                await self.orchestrator.initialize()
            
            # Start the orchestrator (this will run indefinitely)
            if hasattr(self.orchestrator, 'run'):
                await self.orchestrator.run()
            else:
                await self.orchestrator.start()
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            await self.shutdown()
        except Exception as e:
            self.logger.error(f"Fatal error in continuous AI system: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
        return True
    
    async def shutdown(self):
        """Shutdown the continuous AI system gracefully."""
        if self.shutdown_requested:
            return
            
        self.shutdown_requested = True
        self.logger.info("🛑 SHUTTING DOWN CONTINUOUS AI SYSTEM")
        
        # Stop worker monitor if running
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.logger.info("📊 Worker monitoring stopped")
        
        if self.orchestrator:
            if hasattr(self.orchestrator, 'cleanup'):
                await self.orchestrator.cleanup()
            elif hasattr(self.orchestrator, 'stop'):
                await self.orchestrator.stop()
        
        self.logger.info("✅ Continuous AI system shutdown complete")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}")
            asyncio.create_task(self.shutdown())
        
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_status(self) -> dict:
        """Get current system status."""
        status = {
            'running': not self.shutdown_requested,
            'mode': self.mode,
            'max_workers': self.max_workers,
            'parallel_enabled': self.enable_parallel,
            'config_valid': self.config.validate() if self.config else False
        }
        
        if self.orchestrator:
            status.update(self.orchestrator.get_status())
        
        return status
    
    async def _run_worker_monitor(self):
        """Run worker monitor in background."""
        try:
            # Import here to avoid circular imports
            # Use fixed WorkerStatusMonitor to avoid Redis connection leaks
            from scripts.worker_status_monitor_fixed import WorkerStatusMonitor
            
            monitor = WorkerStatusMonitor()
            
            # Initialize the monitor with Redis connections
            try:
                await monitor.initialize()
                self.logger.info("Worker monitor initialized with Redis connections")
            except Exception as e:
                self.logger.error(f"Failed to initialize worker monitor: {e}")
                return
            
            # Run continuous monitoring with periodic updates
            update_interval = 2  # seconds - real-time monitoring
            
            # Get the WorkerMonitor logger for proper logging
            import logging
            worker_monitor_logger = logging.getLogger('WorkerMonitor')
            
            while not self.shutdown_requested:
                try:
                    # Get worker status
                    status = await monitor.get_worker_status()
                    
                    # Log summary to both console and file (always log, even with 0 workers)
                    if status:
                        # Workers is a dict, so iterate over values
                        workers_dict = status.get('workers', {})
                        if workers_dict:
                            workers = workers_dict.values()
                            active = sum(1 for w in workers if w and w.get('status') == 'working')
                            idle = sum(1 for w in workers if w and w.get('status') == 'idle')
                            total = len(workers_dict)
                        else:
                            # No workers case
                            active = 0
                            idle = 0
                            total = 0
                        
                        # Get queue and completion stats safely
                        queue_status = status.get('queue_status', {})
                        queue_total = queue_status.get('total_queued', 0)  # Fixed: was 'total', should be 'total_queued'
                        completed_total = queue_status.get('total_completed', 0)  # Fixed: was looking in wrong dict
                        
                        # Log to main logger (continuous_ai.log) - ALWAYS log, even with 0 workers
                        self.logger.info(
                            f"📊 Worker Status: {active}/{total} active, {idle} idle | "
                            f"Queue: {queue_total} tasks | "
                            f"Completed: {completed_total}"
                        )
                        
                        # Log detailed status to WorkerMonitor logger (worker_monitor.log)
                        worker_monitor_logger.info("=" * 80)
                        worker_monitor_logger.info(f"WORKER STATUS UPDATE - {status.get('timestamp', 'N/A')}")
                        worker_monitor_logger.info("=" * 80)
                        
                        # System health
                        health = status.get('system_health', {})
                        worker_monitor_logger.info("System Health:")
                        worker_monitor_logger.info(f"  Overall: {health.get('system_health', 0):.1f}%")
                        worker_monitor_logger.info(f"  Worker Health: {health.get('worker_health', 0):.1f}%")
                        worker_monitor_logger.info(f"  Queue Health: {health.get('queue_health', 0):.1f}%")
                        worker_monitor_logger.info(f"  Active Workers: {active}/{total}")
                        worker_monitor_logger.info(f"  Idle Workers: {idle}/{total}")
                        
                        # Queue details
                        worker_monitor_logger.info("\nQueue Status:")
                        worker_monitor_logger.info(f"  Total Queued: {queue_total}")
                        # Get priority breakdown from the correct location
                        priority_breakdown = queue_status.get('by_priority', {})
                        if priority_breakdown:
                            for priority, count in priority_breakdown.items():
                                worker_monitor_logger.info(f"  {priority}: {count} items")
                        else:
                            worker_monitor_logger.info("  No items in queue")
                        
                        # Worker details
                        worker_monitor_logger.info("\nWorker Details:")
                        if workers_dict:
                            for worker_id, worker_data in workers_dict.items():
                                if not worker_data:  # Skip if worker_data is None
                                    continue
                                worker_status = worker_data.get('status', 'unknown')
                                current_task = worker_data.get('current_task', 'None')
                                specialization = worker_data.get('specialization', 'general')
                                tasks_completed = worker_data.get('total_completed', 0)  # Fixed: was 'tasks_completed'
                                
                                # Handle current_task which might be a dict or string
                                if isinstance(current_task, dict):
                                    task_display = current_task.get('title', 'Unknown task')[:50]
                                elif isinstance(current_task, str) and current_task != 'None':
                                    task_display = current_task[:50]
                                else:
                                    task_display = 'None'
                                    
                                worker_monitor_logger.info(
                                    f"  {worker_id}: {worker_status} | "
                                    f"Spec: {specialization} | "
                                    f"Completed: {tasks_completed} | "
                                    f"Task: {task_display}{'...' if len(str(task_display)) >= 50 else ''}"
                                )
                        else:
                            worker_monitor_logger.info("  No workers currently active")
                        
                        # Active tasks summary - active_tasks is a list, not a dict
                        active_tasks = status.get('active_tasks', [])
                        if active_tasks:
                            worker_monitor_logger.info(f"\nActive Tasks: {len(active_tasks)}")
                            for task_data in active_tasks[:5]:  # Show first 5
                                if isinstance(task_data, dict):
                                    task_id = task_data.get('task_id', 'unknown')
                                    title = task_data.get('title', 'Unknown')
                                    worker_monitor_logger.info(
                                        f"  {task_id}: {title[:60]}{'...' if len(title) > 60 else ''}"
                                    )
                            if len(active_tasks) > 5:
                                worker_monitor_logger.info(f"  ... and {len(active_tasks) - 5} more tasks")
                        else:
                            worker_monitor_logger.info("\nActive Tasks: None")
                        
                        worker_monitor_logger.info("\n" + "=" * 80 + "\n")
                    
                    # Wait before next update
                    await asyncio.sleep(update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in worker monitor: {e}")
                    worker_monitor_logger.error(f"Error getting worker status: {e}")
                    # Add traceback for better debugging
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    worker_monitor_logger.error(f"Traceback: {traceback.format_exc()}")
                    await asyncio.sleep(update_interval)
                    
        except asyncio.CancelledError:
            self.logger.debug("Worker monitor task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Fatal error in worker monitor: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


async def run_continuous_system(args):
    """Run the continuous AI system with given arguments."""
    # Set up logging
    setup_logging(args.log_level)
    
    # Determine parallel processing setting
    if args.no_parallel:
        enable_parallel = False
    elif args.parallel:
        enable_parallel = True
    else:
        # Default from environment or True
        enable_parallel = os.getenv('ENABLE_PARALLEL', 'true').lower() == 'true'
    
    # Determine research engine setting
    if args.no_research:
        enable_research = False
    else:
        # Default from environment or True
        enable_research = os.getenv('RESEARCH_ENGINE_ENABLED', 'true').lower() == 'true'
    
    # Determine round-robin setting
    if args.round_robin:
        enable_round_robin = True
    else:
        # Default from environment or True
        enable_round_robin = os.getenv('AI_ROUND_ROBIN', 'true').lower() == 'true'
    
    # Determine worker monitor setting
    enable_worker_monitor = args.monitor_workers
    
    # Determine MCP setting
    if args.no_mcp:
        enable_mcp = False
    elif args.mcp:
        enable_mcp = True
    else:
        # Default from environment or True
        enable_mcp = os.getenv('MCP_ENABLED', 'true').lower() == 'true'
    
    # Create and start runner
    runner = ContinuousAIRunner(
        max_workers=args.workers,
        enable_parallel=enable_parallel,
        mode=args.mode,
        enable_research=enable_research,
        enable_round_robin=enable_round_robin,
        enable_worker_monitor=enable_worker_monitor,
        enable_mcp=enable_mcp
    )
    
    # Start the system
    success = await runner.start()
    
    if not success:
        sys.exit(1)


def main():
    """Main entry point for the continuous AI system."""
    parser = argparse.ArgumentParser(
        description="Continuous 24/7 AI System - Never-stopping intelligent worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start in production mode with 3 workers
  python run_continuous_ai.py
  
  # Start in development mode with faster cycles
  python run_continuous_ai.py --mode development
  
  # Start with 5 workers for high-load scenarios
  python run_continuous_ai.py --workers 5
  
  # Start without parallel processing (single worker)
  python run_continuous_ai.py --no-parallel
  
  # Start without research evolution engine
  python run_continuous_ai.py --no-research
  
  # Start with round-robin AI provider selection
  python run_continuous_ai.py --round-robin
  # or use the alias:
  python run_continuous_ai.py --enable-ai-rotation
  
  # Start with automatic worker monitoring
  python run_continuous_ai.py --monitor-workers
  
  # Start with MCP integration disabled
  python run_continuous_ai.py --no-mcp
  
  # Start in test mode for single-cycle testing
  python run_continuous_ai.py --mode test --log-level DEBUG
  
  # Full example with all options
  python run_continuous_ai.py --mode development --workers 10 --no-research --monitor-workers --mcp

Environment Variables:
  ANTHROPIC_API_KEY     - Required for AI operations
  GITHUB_TOKEN          - Required for GitHub integration
  ORCHESTRATOR_MODE     - Default execution mode
  MAX_WORKERS           - Default number of workers
  ENABLE_PARALLEL       - Enable parallel processing (true/false)
  RESEARCH_ENGINE_ENABLED - Enable research evolution engine (true/false)
  AI_ROUND_ROBIN        - Enable round-robin AI provider selection (true/false)
  MCP_ENABLED           - Enable MCP integration (true/false)
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['production', 'development', 'test'],
        default=os.getenv('ORCHESTRATOR_MODE', 'production'),
        help='Execution mode (default: production)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=int(os.getenv('MAX_WORKERS', '10')),
        help='Number of parallel workers (default: 10)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=None,
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        default=None,
        help='Disable parallel processing'
    )
    
    parser.add_argument(
        '--no-research',
        action='store_true',
        default=None,
        help='Disable research evolution engine'
    )
    
    parser.add_argument(
        '--round-robin', '--enable-ai-rotation',
        action='store_true',
        default=None,
        help='Enable round-robin load balancing across AI providers (rotate between Claude, GPT, Gemini, DeepSeek)'
    )
    
    parser.add_argument(
        '--monitor-workers',
        action='store_true',
        default=True,
        help='Enable automatic worker status monitoring with periodic updates'
    )
    
    parser.add_argument(
        '--mcp',
        action='store_true',
        default=None,
        help='Enable MCP (Model Context Protocol) integration for external services'
    )
    
    parser.add_argument(
        '--no-mcp',
        action='store_true',
        default=None,
        help='Disable MCP integration and use direct API calls'
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
        help='Show status and exit'
    )
    
    args = parser.parse_args()
    
    # Handle status request
    if args.status:
        print("Continuous AI System Status Check")
        print("=" * 40)
        
        # Check if system is already running (simplified check)
        import glob
        state_files = glob.glob('continuous_orchestrator_state.json')
        if state_files:
            print("✅ State file found - system may be running")
            try:
                import json
                with open(state_files[0], 'r') as f:
                    state = json.load(f)
                print(f"Last updated: {state.get('last_updated', 'Unknown')}")
                print(f"Metrics: {state.get('metrics', {})}")
            except Exception as e:
                print(f"❌ Error reading state: {e}")
        else:
            print("⚠️  No state file found - system not running")
        
        return
    
    # Validate arguments
    if args.workers < 1:
        print("Error: Number of workers must be at least 1")
        sys.exit(1)
    
    if args.workers > 10:
        print("Warning: More than 10 workers may impact performance")
    
    # Determine if MCP is enabled for banner
    mcp_enabled = True
    if args.no_mcp:
        mcp_enabled = False
    elif args.mcp:
        mcp_enabled = True
    else:
        mcp_enabled = os.getenv('MCP_ENABLED', 'true').lower() == 'true'
    
    # Print startup banner
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 CONTINUOUS 24/7 AI SYSTEM                    ║
    ║                                                               ║
    ║  🚀 Never-stopping intelligent worker                        ║
    ║  ⚡ Parallel processing with smart work discovery            ║
    ║  🔄 Event-driven continuous operation                        ║
    ║  📊 Real-time performance monitoring                         ║
    ║  🔧 MCP Integration: {'ENABLED' if mcp_enabled else 'DISABLED':<20}                    ║
    ║                                                               ║
    ║  Press Ctrl+C to shutdown gracefully                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run the system
    try:
        asyncio.run(run_continuous_system(args))
    except KeyboardInterrupt:
        print("\n👋 Goodbye! System shutdown complete.")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()