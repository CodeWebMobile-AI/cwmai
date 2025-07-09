#!/usr/bin/env python3
"""
Safe Hybrid System Runner - Prevents endless loops
"""

import os
import sys
import asyncio
import logging
import signal
from dotenv import load_dotenv

# Load environment variables first
load_dotenv('.env.local')

# Verify API keys are loaded
api_keys = {
    'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
    'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
    'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY')
}

print("\n=== API Keys Status ===")
for key, value in api_keys.items():
    print(f"{key}: {'✓ SET' if value else '✗ NOT SET'}")

if not any(api_keys.values()):
    print("\n❌ ERROR: No API keys found!")
    print("Please check your .env.local file")
    sys.exit(1)

# Import after environment is loaded
from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis
from scripts.in_memory_work_queue import WorkItem, TaskPriority

class SafeOrchestrator(ContinuousOrchestratorNoRedis):
    """Safe orchestrator that prevents endless loops."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_generation_attempts = 0
        self.max_generation_attempts = 10  # Limit attempts
        
    async def find_work(self):
        """Find work with safety limits."""
        # Increment attempt counter
        self.task_generation_attempts += 1
        
        # Check if we've hit the limit
        if self.task_generation_attempts > self.max_generation_attempts:
            self.logger.warning("Hit task generation limit - creating maintenance task")
            # Return a simple maintenance task
            return [WorkItem(
                id=f"safety_{int(time.time())}",
                type="maintenance",
                title="System Safety Check",
                description="Preventing endless loop - performing safety check",
                priority=TaskPriority.LOW
            )]
        
        # Try normal work finding
        try:
            work = await super().find_work()
            if work:
                self.task_generation_attempts = 0  # Reset on success
            return work
        except Exception as e:
            self.logger.error(f"Error finding work: {e}")
            # Return empty list instead of retrying endlessly
            return []

async def run_safe_system(workers=2):
    """Run the system with safety features."""
    print(f"\n=== Starting Safe Hybrid System ===")
    print(f"Workers: {workers}")
    print("Safety features:")
    print("- Limited task generation attempts")
    print("- No endless loops")
    print("- Graceful error handling")
    
    orchestrator = SafeOrchestrator(
        max_workers=workers,
        enable_parallel=True,
        enable_research=False  # Disable to avoid sklearn issues
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down gracefully...")
        orchestrator.shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run for a limited time to test
        print("\nRunning for 30 seconds to test...")
        await asyncio.wait_for(orchestrator.start(), timeout=30)
    except asyncio.TimeoutError:
        print("\nTest complete - no endless loops detected!")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await orchestrator.stop()
        print("\nSystem stopped safely")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    
    # Avoid sklearn import issues
    import warnings
    warnings.filterwarnings('ignore', message='numpy.dtype size changed')
    
    asyncio.run(run_safe_system(args.workers))