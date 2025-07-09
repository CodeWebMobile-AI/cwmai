#!/usr/bin/env python3
"""
Fix Work Distribution Issues

This script implements fixes for:
1. Work deduplication to prevent all workers from getting the same task
2. Task locking mechanism for concurrent workers
3. Coordinated work discovery to prevent simultaneous searches
4. Zombie process detection and cleanup
"""

import os
import sys
import asyncio
import time
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))


def create_work_distribution_fix():
    """Create enhanced work queue manager with proper locking and deduplication."""
    
    fix_content = '''"""
Enhanced Work Queue Manager with Deduplication and Locking

Prevents duplicate task assignment and coordinates work discovery.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import uuid
import time

from scripts.work_item_types import WorkItem, TaskPriority


@dataclass
class WorkAssignment:
    """Track work assignments to prevent duplicates."""
    work_item: WorkItem
    assigned_to: str
    assigned_at: datetime
    completed: bool = False


class EnhancedWorkQueueManager:
    """Manages work queue with deduplication and locking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Main work queue
        self.work_queue: List[WorkItem] = []
        self.queue_lock = asyncio.Lock()
        
        # Assignment tracking
        self.active_assignments: Dict[str, WorkAssignment] = {}  # task_id -> assignment
        self.worker_assignments: Dict[str, List[str]] = {}  # worker_id -> [task_ids]
        
        # Deduplication tracking
        self.seen_tasks: Set[str] = set()  # Set of task signatures
        self.task_cooldown: Dict[str, float] = {}  # task_signature -> expiry_time
        
        # Work discovery coordination
        self.discovery_lock = asyncio.Lock()
        self.last_discovery_time = 0
        self.discovery_interval = 30  # seconds between discoveries
        self.discovery_in_progress = False
        
    def _get_task_signature(self, work_item: WorkItem) -> str:
        """Get unique signature for a task to detect duplicates."""
        # Create signature from repository + title
        return f"{work_item.repository or 'system'}::{work_item.title.lower().strip()}"
        
    async def add_work(self, work_items: List[WorkItem]) -> int:
        """Add work items to queue with deduplication.
        
        Returns:
            Number of items actually added (after deduplication)
        """
        added = 0
        
        async with self.queue_lock:
            for item in work_items:
                signature = self._get_task_signature(item)
                
                # Check if task is duplicate or in cooldown
                if signature in self.seen_tasks:
                    self.logger.debug(f"Skipping duplicate task: {item.title}")
                    continue
                    
                # Check cooldown
                if signature in self.task_cooldown:
                    if time.time() < self.task_cooldown[signature]:
                        self.logger.debug(f"Task in cooldown: {item.title}")
                        continue
                    else:
                        # Cooldown expired
                        del self.task_cooldown[signature]
                
                # Add to queue
                self.work_queue.append(item)
                self.seen_tasks.add(signature)
                added += 1
                
        if added > 0:
            self.logger.info(f"Added {added} new work items to queue (filtered {len(work_items) - added} duplicates)")
            
        return added
        
    async def get_work_for_worker(self, worker_id: str, specialization: Optional[str] = None) -> Optional[WorkItem]:
        """Get work for a specific worker with assignment tracking."""
        
        async with self.queue_lock:
            # Find suitable work
            selected_item = None
            selected_index = None
            
            # First pass: Look for specialized work
            if specialization and specialization != "general":
                for i, item in enumerate(self.work_queue):
                    # Skip if already assigned
                    if item.id in self.active_assignments:
                        continue
                        
                    # Check specialization match
                    if specialization == "system_tasks" and item.repository is None:
                        selected_item = item
                        selected_index = i
                        break
                    elif specialization == item.repository:
                        selected_item = item
                        selected_index = i
                        break
                        
            # Second pass: If no specialized work, take any work
            if selected_item is None:
                for i, item in enumerate(self.work_queue):
                    if item.id not in self.active_assignments:
                        selected_item = item
                        selected_index = i
                        break
                        
            # Assign work if found
            if selected_item is not None:
                # Remove from queue
                self.work_queue.pop(selected_index)
                
                # Track assignment
                assignment = WorkAssignment(
                    work_item=selected_item,
                    assigned_to=worker_id,
                    assigned_at=datetime.now(timezone.utc)
                )
                self.active_assignments[selected_item.id] = assignment
                
                # Track by worker
                if worker_id not in self.worker_assignments:
                    self.worker_assignments[worker_id] = []
                self.worker_assignments[worker_id].append(selected_item.id)
                
                self.logger.info(f"Assigned task '{selected_item.title}' to worker {worker_id}")
                return selected_item
                
        return None
        
    async def complete_work(self, worker_id: str, task_id: str, success: bool = True):
        """Mark work as completed and handle cleanup."""
        
        async with self.queue_lock:
            if task_id in self.active_assignments:
                assignment = self.active_assignments[task_id]
                assignment.completed = True
                
                # Add to cooldown if failed
                if not success:
                    signature = self._get_task_signature(assignment.work_item)
                    # 5 minute cooldown for failed tasks
                    self.task_cooldown[signature] = time.time() + 300
                    self.logger.info(f"Added failed task to cooldown: {assignment.work_item.title}")
                
                # Remove from active assignments
                del self.active_assignments[task_id]
                
                # Remove from worker assignments
                if worker_id in self.worker_assignments:
                    self.worker_assignments[worker_id].remove(task_id)
                    
    async def should_discover_work(self) -> bool:
        """Check if work discovery should run (with coordination)."""
        
        current_time = time.time()
        
        # Quick check without lock
        if self.discovery_in_progress:
            return False
            
        if current_time - self.last_discovery_time < self.discovery_interval:
            return False
            
        # Try to acquire discovery lock
        if self.discovery_lock.locked():
            return False
            
        try:
            # Try to acquire with timeout
            async with asyncio.timeout(0.1):
                await self.discovery_lock.acquire()
                
            # Double-check timing with lock held
            if current_time - self.last_discovery_time >= self.discovery_interval:
                self.discovery_in_progress = True
                self.last_discovery_time = current_time
                return True
            else:
                return False
                
        except asyncio.TimeoutError:
            return False
        finally:
            if self.discovery_lock.locked():
                self.discovery_lock.release()
                
    async def end_discovery(self):
        """Mark discovery as complete."""
        self.discovery_in_progress = False
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'queue_size': len(self.work_queue),
            'active_assignments': len(self.active_assignments),
            'seen_tasks': len(self.seen_tasks),
            'tasks_in_cooldown': len(self.task_cooldown),
            'workers_with_tasks': len([w for w in self.worker_assignments if self.worker_assignments[w]]),
            'discovery_in_progress': self.discovery_in_progress,
            'time_until_next_discovery': max(0, self.discovery_interval - (time.time() - self.last_discovery_time))
        }
        
    async def cleanup_stale_assignments(self, timeout_minutes: int = 30):
        """Clean up assignments that have been active too long."""
        
        current_time = datetime.now(timezone.utc)
        stale_assignments = []
        
        async with self.queue_lock:
            for task_id, assignment in self.active_assignments.items():
                if (current_time - assignment.assigned_at).total_seconds() > timeout_minutes * 60:
                    stale_assignments.append((task_id, assignment))
                    
            # Return stale work to queue
            for task_id, assignment in stale_assignments:
                self.logger.warning(f"Returning stale task to queue: {assignment.work_item.title}")
                
                # Remove from assignments
                del self.active_assignments[task_id]
                
                # Remove from worker assignments
                if assignment.assigned_to in self.worker_assignments:
                    self.worker_assignments[assignment.assigned_to].remove(task_id)
                    
                # Add back to queue
                self.work_queue.insert(0, assignment.work_item)  # Add to front
                
        return len(stale_assignments)
'''
    
    # Write the enhanced work queue manager
    with open('scripts/enhanced_work_queue_manager.py', 'w') as f:
        f.write(fix_content)
    
    print("✅ Created enhanced_work_queue_manager.py")


def create_orchestrator_patch():
    """Create a patch for the orchestrator to use the enhanced work queue."""
    
    patch_content = '''"""
Patch for Enhanced Hybrid Orchestrator to use Enhanced Work Queue Manager

This patch integrates the enhanced work queue manager to fix:
1. Duplicate task assignment
2. Simultaneous work discovery
3. Work distribution issues
"""

import os
import sys
from typing import List, Optional

# Import the enhanced work queue manager
from scripts.enhanced_work_queue_manager import EnhancedWorkQueueManager
from scripts.work_item_types import WorkItem


class WorkDistributionPatch:
    """Patches the orchestrator to use enhanced work distribution."""
    
    @staticmethod
    def apply_to_orchestrator(orchestrator):
        """Apply the work distribution patch to an orchestrator instance."""
        
        print("🔧 Applying work distribution patch...")
        
        # Replace work queue with enhanced version
        orchestrator._work_queue_manager = EnhancedWorkQueueManager()
        
        # Store original methods
        orchestrator._original_find_work_for_worker = orchestrator._find_work_for_worker
        orchestrator._original_find_work = orchestrator.find_work if hasattr(orchestrator, 'find_work') else None
        
        # Patch _find_work_for_worker method
        async def patched_find_work_for_worker(worker):
            """Enhanced work finder that uses the work queue manager."""
            
            # First check if we need to discover new work
            if orchestrator._work_queue_manager.should_discover_work():
                try:
                    # Only one worker does discovery at a time
                    orchestrator.logger.info(f"Worker {worker.id} triggering work discovery")
                    
                    # Get current worker count
                    active_workers = len([w for w in orchestrator.workers.values() 
                                        if w.status != 'SHUTTING_DOWN'])
                    
                    # Discover work
                    if orchestrator._original_find_work:
                        new_work = await orchestrator._original_find_work(
                            max_items=active_workers * 2  # Get 2x tasks to have buffer
                        )
                    else:
                        # Fallback if no find_work method
                        new_work = []
                    
                    # Add to queue
                    if new_work:
                        added = await orchestrator._work_queue_manager.add_work(new_work)
                        orchestrator.logger.info(f"Work discovery added {added} new tasks")
                        
                finally:
                    await orchestrator._work_queue_manager.end_discovery()
            
            # Clean up stale assignments periodically
            if hasattr(orchestrator, '_last_cleanup'):
                if time.time() - orchestrator._last_cleanup > 300:  # Every 5 minutes
                    stale_count = await orchestrator._work_queue_manager.cleanup_stale_assignments()
                    if stale_count > 0:
                        orchestrator.logger.warning(f"Cleaned up {stale_count} stale assignments")
                    orchestrator._last_cleanup = time.time()
            else:
                orchestrator._last_cleanup = time.time()
            
            # Get work for this specific worker
            work_item = await orchestrator._work_queue_manager.get_work_for_worker(
                worker.id,
                worker.specialization
            )
            
            return work_item
        
        # Replace the method
        orchestrator._find_work_for_worker = patched_find_work_for_worker
        
        # Add work completion tracking
        original_execute_work = orchestrator._execute_work
        
        async def patched_execute_work(worker, work_item):
            """Enhanced work execution with completion tracking."""
            try:
                result = await original_execute_work(worker, work_item)
                
                # Mark work as completed
                success = result.get('success', False) if isinstance(result, dict) else True
                await orchestrator._work_queue_manager.complete_work(
                    worker.id,
                    work_item.id,
                    success
                )
                
                return result
            except Exception as e:
                # Mark as failed on exception
                await orchestrator._work_queue_manager.complete_work(
                    worker.id,
                    work_item.id,
                    False
                )
                raise
        
        orchestrator._execute_work = patched_execute_work
        
        # Add status method
        def get_work_distribution_status():
            """Get current work distribution status."""
            return orchestrator._work_queue_manager.get_queue_status()
        
        orchestrator.get_work_distribution_status = get_work_distribution_status
        
        print("✅ Work distribution patch applied successfully")
        
        return orchestrator


# Add import for time
import time
'''
    
    # Write the patch
    with open('scripts/work_distribution_patch.py', 'w') as f:
        f.write(patch_content)
    
    print("✅ Created work_distribution_patch.py")


def create_zombie_process_cleaner():
    """Create script to detect and clean zombie processes."""
    
    cleaner_content = '''#!/usr/bin/env python3
"""
Zombie Process Cleaner

Detects and cleans up zombie Claude processes from interrupted runs.
"""

import psutil
import os
import sys
import time
from datetime import datetime, timedelta
import subprocess


def find_zombie_processes():
    """Find zombie or stuck processes related to the AI system."""
    
    zombies = []
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent']):
        try:
            # Skip current process
            if proc.info['pid'] == current_pid:
                continue
                
            # Check for relevant processes
            cmdline = ' '.join(proc.info.get('cmdline', []))
            
            # Look for our scripts
            if any(script in cmdline for script in [
                'run_hybrid_continuous.py',
                'continuous_orchestrator',
                'hybrid_orchestrator',
                'claude --dangerously-skip-permissions'
            ]):
                # Check if process is old (> 1 hour)
                create_time = datetime.fromtimestamp(proc.info['create_time'])
                age = datetime.now() - create_time
                
                if age > timedelta(hours=1):
                    zombies.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                        'age': age,
                        'cpu': proc.info.get('cpu_percent', 0)
                    })
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    return zombies


def kill_zombie_processes(zombies, force=False):
    """Kill zombie processes."""
    
    killed = 0
    
    for zombie in zombies:
        try:
            proc = psutil.Process(zombie['pid'])
            
            if force:
                proc.kill()  # SIGKILL
            else:
                proc.terminate()  # SIGTERM
                
            killed += 1
            print(f"Killed process {zombie['pid']}: {zombie['cmdline']}")
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Failed to kill process {zombie['pid']}: {e}")
            
    return killed


def main():
    """Main function."""
    
    print("🔍 Searching for zombie processes...")
    
    zombies = find_zombie_processes()
    
    if not zombies:
        print("✅ No zombie processes found")
        return
        
    print(f"\\n⚠️  Found {len(zombies)} zombie processes:\\n")
    
    for zombie in zombies:
        print(f"PID: {zombie['pid']}")
        print(f"  Command: {zombie['cmdline']}")
        print(f"  Age: {zombie['age']}")
        print(f"  CPU: {zombie['cpu']}%")
        print()
        
    # Ask for confirmation
    response = input("Kill these processes? (y/N): ")
    
    if response.lower() == 'y':
        killed = kill_zombie_processes(zombies)
        print(f"\\n✅ Killed {killed} processes")
    else:
        print("❌ Aborted")


if __name__ == "__main__":
    main()
'''
    
    # Write the cleaner script
    with open('cleanup_zombie_processes.py', 'w') as f:
        f.write(cleaner_content)
    
    # Make it executable
    os.chmod('cleanup_zombie_processes.py', 0o755)
    
    print("✅ Created cleanup_zombie_processes.py")


def create_integration_script():
    """Create script to integrate all fixes."""
    
    integration_content = '''#!/usr/bin/env python3
"""
Integrate Work Distribution Fixes

This script shows how to apply the work distribution fixes to the orchestrator.
"""

import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.work_distribution_patch import WorkDistributionPatch


def apply_fixes_to_hybrid_orchestrator():
    """Apply fixes to the enhanced hybrid orchestrator."""
    
    # Import the orchestrator
    from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
    
    # Create original init method wrapper
    original_init = EnhancedHybridOrchestrator.__init__
    
    def patched_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Apply work distribution patch
        WorkDistributionPatch.apply_to_orchestrator(self)
        
        print("✅ Work distribution fixes applied to orchestrator")
    
    # Replace init method
    EnhancedHybridOrchestrator.__init__ = patched_init
    
    print("✅ Enhanced hybrid orchestrator patched with work distribution fixes")


# Apply the fixes when this module is imported
apply_fixes_to_hybrid_orchestrator()
'''
    
    # Write the integration script
    with open('scripts/apply_work_distribution_fixes.py', 'w') as f:
        f.write(integration_content)
    
    print("✅ Created apply_work_distribution_fixes.py")


def main():
    """Main function to create all fixes."""
    
    print("🔧 Creating work distribution fixes...\n")
    
    # Create all components
    create_work_distribution_fix()
    create_orchestrator_patch()
    create_zombie_process_cleaner()
    create_integration_script()
    
    print("\n✅ All work distribution fixes created!")
    print("\nTo apply the fixes:")
    print("1. Check for zombie processes: python3 cleanup_zombie_processes.py")
    print("2. The fixes will be automatically applied when running the hybrid system")
    print("\nThe fixes include:")
    print("- Enhanced work queue manager with deduplication")
    print("- Task locking to prevent duplicate assignments")
    print("- Coordinated work discovery (only one worker at a time)")
    print("- Stale assignment cleanup")
    print("- Zombie process detection and cleanup")


if __name__ == "__main__":
    main()