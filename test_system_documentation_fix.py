#!/usr/bin/env python3
"""Test if the system still creates documentation tasks for projects with complete documentation."""

import asyncio
import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_documentation_task_creation():
    """Monitor system for documentation task creation."""
    
    print("🧪 Testing if system creates documentation tasks for complete projects...\n")
    
    # First verify all repos have documentation
    from scripts.local_github_simulator import LocalGitHubSimulator
    simulator = LocalGitHubSimulator()
    
    print("1️⃣ Checking documentation files in all repositories:")
    repos = simulator.list_repositories()
    all_complete = True
    
    for repo in repos:
        print(f"\n  {repo.full_name}:")
        has_all = True
        for doc in ['README.md', 'ARCHITECTURE.md', 'MONETIZATION_STRATEGY.md']:
            try:
                content = repo.get_contents(doc)
                print(f"    ✓ {doc} ({content.size:,} bytes)")
            except:
                print(f"    ✗ {doc} MISSING")
                has_all = False
                all_complete = False
        
        if has_all:
            print(f"    ✅ Complete documentation")
        else:
            print(f"    ❌ Incomplete documentation")
    
    if not all_complete:
        print("\n⚠️  Some repositories have missing documentation!")
        return
    
    print("\n✅ All repositories have complete documentation files\n")
    
    # Now run the system for a short time
    print("2️⃣ Running system for 2 minutes to check for documentation tasks...")
    
    # Import after verification
    from scripts.hybrid_orchestrator_enhanced import run_hybrid_orchestrator
    from scripts.ai_brain_factory import AIBrainFactory
    from scripts.state_manager import StateManager
    
    # Start monitoring
    start_time = time.time()
    doc_tasks_found = []
    
    # Create a custom logger to capture work items
    class WorkItemMonitor:
        def __init__(self):
            self.doc_tasks = []
            
        def check_work_item(self, work_item):
            if hasattr(work_item, 'task_type'):
                task_type = work_item.task_type.lower()
                if 'documentation' in task_type or 'architecture' in task_type or 'monetization' in task_type:
                    self.doc_tasks.append({
                        'type': task_type,
                        'repo': work_item.repository,
                        'title': work_item.title,
                        'time': datetime.now().isoformat()
                    })
    
    monitor = WorkItemMonitor()
    
    # Run system with monitoring
    try:
        # Patch the work discovery to monitor tasks
        from scripts import intelligent_work_finder
        original_discover = intelligent_work_finder.IntelligentWorkFinder.discover_work
        
        async def monitored_discover(self, max_items=5, current_workload=0):
            work_items = await original_discover(self, max_items, current_workload)
            for item in work_items:
                monitor.check_work_item(item)
            return work_items
        
        intelligent_work_finder.IntelligentWorkFinder.discover_work = monitored_discover
        
        # Run for 2 minutes
        print("  ⏳ Monitoring work discovery...")
        
        ai_brain = AIBrainFactory.create_for_production()
        state_manager = StateManager()
        
        # Run with timeout
        task = asyncio.create_task(
            run_hybrid_orchestrator(
                workers=3,
                parallel=True,
                round_robin=True,
                monitor_mode=False
            )
        )
        
        await asyncio.sleep(120)  # 2 minutes
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
    except Exception as e:
        print(f"  ⚠️  Error during monitoring: {e}")
    
    # Check results
    print(f"\n3️⃣ Results after {int(time.time() - start_time)} seconds:")
    
    if monitor.doc_tasks:
        print(f"\n❌ FAILED: Found {len(monitor.doc_tasks)} documentation tasks:")
        for task in monitor.doc_tasks:
            print(f"  - [{task['type']}] {task['repo']}: {task['title']}")
        
        print("\n⚠️  System is still creating documentation tasks for projects with complete docs!")
        return False
    else:
        print("\n✅ SUCCESS: No documentation tasks created!")
        print("  The system correctly recognized that all projects have complete documentation.")
        return True

if __name__ == "__main__":
    result = asyncio.run(test_documentation_task_creation())
    sys.exit(0 if result else 1)