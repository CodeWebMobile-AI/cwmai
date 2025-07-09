#!/usr/bin/env python3
"""
Test script to verify parallel execution fix for hybrid orchestrator.

This will start the orchestrator with multiple workers and verify
that they work on different repositories in parallel.
"""

import asyncio
import os
import logging
from scripts.hybrid_continuous_orchestrator import HybridContinuousOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def monitor_workers(orchestrator, duration=120):
    """Monitor worker activity for a specified duration."""
    start_time = asyncio.get_event_loop().time()
    
    print("\n=== Starting Parallel Execution Test ===")
    print(f"Workers: {orchestrator.max_workers}")
    print(f"Duration: {duration} seconds")
    print("\n--- Worker Activity ---")
    
    while asyncio.get_event_loop().time() - start_time < duration:
        # Check worker status
        active_repos = {}
        idle_count = 0
        
        for worker_id, worker in orchestrator.workers.items():
            if worker.current_work and worker.current_work.repository:
                repo = worker.current_work.repository
                if repo not in active_repos:
                    active_repos[repo] = []
                active_repos[repo].append(worker_id)
            else:
                idle_count += 1
        
        # Print status
        print(f"\nTime: +{int(asyncio.get_event_loop().time() - start_time)}s")
        print(f"Active workers: {len(orchestrator.workers) - idle_count}")
        print(f"Idle workers: {idle_count}")
        
        if active_repos:
            print("Repositories being worked on:")
            for repo, workers in active_repos.items():
                print(f"  - {repo}: {', '.join(workers)}")
        
        # Check for parallel execution
        if len(active_repos) > 1:
            print(f"✅ PARALLEL EXECUTION CONFIRMED: {len(active_repos)} different repositories being worked on simultaneously!")
        elif len(active_repos) == 1 and len(list(active_repos.values())[0]) > 1:
            print("⚠️  Multiple workers on same repository - not ideal for parallelism")
        
        await asyncio.sleep(10)  # Check every 10 seconds
    
    print("\n=== Test Complete ===")

async def main():
    # Ensure we're in local mode
    os.environ['STORAGE_MODE'] = 'local'
    
    # Create orchestrator with multiple workers
    orchestrator = HybridContinuousOrchestrator(
        max_workers=5,  # 5 workers to test parallelism
        enable_parallel=True,
        enable_research=False  # Disable research for faster testing
    )
    
    # Start orchestrator in background
    orchestrator_task = asyncio.create_task(orchestrator.start())
    
    # Monitor for 2 minutes
    await monitor_workers(orchestrator, duration=120)
    
    # Shutdown
    print("\nShutting down orchestrator...")
    orchestrator.shutdown_requested = True
    
    # Wait for orchestrator to finish
    try:
        await asyncio.wait_for(orchestrator_task, timeout=30)
    except asyncio.TimeoutError:
        print("Orchestrator shutdown timed out")

if __name__ == "__main__":
    asyncio.run(main())