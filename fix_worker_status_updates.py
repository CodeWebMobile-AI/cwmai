#!/usr/bin/env python3
"""
Fix for Worker Status Update Issues

This script identifies and fixes the following issues:
1. Worker status monitor using wrong Redis keys
2. Orchestrator not updating worker status immediately when work starts
3. Worker status only updating every 30 seconds
"""

import os
import sys

# Issues found
ISSUES = {
    "1. Key Mismatch": {
        "Problem": "Worker monitor looks for 'workers:{id}' but lockfree manager uses 'worker:state:{id}'",
        "Fix": "Update monitor to use get_worker_state() method",
        "File": "scripts/worker_status_monitor.py",
        "Line": 132
    },
    "2. Delayed Status Updates": {
        "Problem": "Worker status only updates every 30 seconds",
        "Fix": "Update status immediately when work starts/completes",
        "File": "scripts/continuous_orchestrator.py",
        "Line": 509
    },
    "3. Wrong Update Method": {
        "Problem": "Orchestrator uses update_state() with 'workers:{id}' key",
        "Fix": "Use update_worker_state() method instead",
        "File": "scripts/continuous_orchestrator.py",
        "Lines": [494, 526]
    },
    "4. Missing Status Update": {
        "Problem": "Worker status not updated in Redis when work starts",
        "Fix": "Add Redis update in _execute_work method",
        "File": "scripts/continuous_orchestrator.py",
        "Line": 665
    }
}

def print_issues():
    """Print all identified issues."""
    print("=" * 80)
    print("WORKER STATUS UPDATE ISSUES")
    print("=" * 80)
    
    for issue, details in ISSUES.items():
        print(f"\n{issue}:")
        print(f"  Problem: {details['Problem']}")
        print(f"  Fix: {details['Fix']}")
        print(f"  File: {details['File']}")
        if isinstance(details.get('Line'), list):
            print(f"  Lines: {', '.join(map(str, details['Line']))}")
        else:
            print(f"  Line: {details.get('Line', 'N/A')}")

def generate_patches():
    """Generate patch files for the fixes."""
    print("\n" + "=" * 80)
    print("GENERATING PATCHES")
    print("=" * 80)
    
    patches = []
    
    # Patch 1: Fix worker monitor key lookup
    patches.append({
        "file": "scripts/worker_status_monitor.py",
        "description": "Fix worker monitor to use correct Redis method",
        "changes": [
            {
                "line": 132,
                "old": "worker_data = await self.redis_state_manager.get_state(f\"workers:{worker_id}\")",
                "new": "worker_data = await self.redis_state_manager.get_worker_state(worker_id)"
            }
        ]
    })
    
    # Patch 2: Fix orchestrator worker state updates
    patches.append({
        "file": "scripts/continuous_orchestrator.py",
        "description": "Fix orchestrator to use correct Redis update methods",
        "changes": [
            {
                "line": 494,
                "old": "await self.redis_state_manager.update_state(\n                    f\"workers:{worker.id}\", \n                    worker_data,\n                    distributed=True\n                )",
                "new": "await self.redis_state_manager.update_worker_state(\n                    worker.id, \n                    worker_data\n                )"
            },
            {
                "line": 526,
                "old": "await self.redis_state_manager.update_state(\n                                f\"workers:{worker.id}\", \n                                worker_data,\n                                distributed=True\n                            )",
                "new": "await self.redis_state_manager.update_worker_state(\n                                worker.id, \n                                worker_data\n                            )"
            }
        ]
    })
    
    # Patch 3: Add immediate status update when work starts
    patches.append({
        "file": "scripts/continuous_orchestrator.py",
        "description": "Add Redis status update when work starts",
        "addition_after_line": 667,
        "code": """
        # Update worker status in Redis immediately
        if self.redis_state_manager:
            try:
                worker_data = {
                    'status': worker.status.value,
                    'specialization': worker.specialization,
                    'last_activity': datetime.now(timezone.utc).isoformat(),
                    'total_completed': worker.total_completed,
                    'total_errors': worker.total_errors,
                    'current_task': {
                        'id': work_item.id,
                        'title': work_item.title,
                        'task_type': work_item.task_type,
                        'repository': work_item.repository,
                        'started_at': work_item.started_at.isoformat() if work_item.started_at else None
                    }
                }
                await self.redis_state_manager.update_worker_state(worker.id, worker_data)
                self._last_state_update[worker.id] = time.time()
            except Exception as e:
                self.logger.error(f"Failed to update worker {worker.id} status in Redis: {e}")
"""
    })
    
    # Patch 4: Update status immediately after work completes
    patches.append({
        "file": "scripts/continuous_orchestrator.py",
        "description": "Add Redis status update when work completes",
        "addition_after_line": 696,
        "code": """
            # Update worker status in Redis after completion
            if self.redis_state_manager:
                try:
                    worker_data = {
                        'status': WorkerStatus.IDLE.value,
                        'specialization': worker.specialization,
                        'last_activity': datetime.now(timezone.utc).isoformat(),
                        'total_completed': worker.total_completed,
                        'total_errors': worker.total_errors,
                        'current_task': None
                    }
                    await self.redis_state_manager.update_worker_state(worker.id, worker_data)
                    self._last_state_update[worker.id] = time.time()
                except Exception as e:
                    self.logger.error(f"Failed to update worker {worker.id} status after completion: {e}")
"""
    })
    
    return patches

def print_fix_summary():
    """Print summary of fixes."""
    print("\n" + "=" * 80)
    print("FIX SUMMARY")
    print("=" * 80)
    print("""
The following changes need to be made:

1. Update worker_status_monitor.py line 132:
   - Change get_state(f"workers:{worker_id}") to get_worker_state(worker_id)

2. Update continuous_orchestrator.py:
   - Replace update_state() calls with update_worker_state() at lines 494 and 526
   - Add immediate Redis updates when work starts (after line 667)
   - Add immediate Redis updates when work completes (after line 696)
   - Remove or reduce the 30-second update interval check

These changes will ensure:
- Worker monitor reads from the correct Redis keys
- Worker status updates immediately when state changes
- Monitor shows real-time worker activity
""")

if __name__ == "__main__":
    print_issues()
    patches = generate_patches()
    print_fix_summary()
    
    print("\nWould you like to apply these fixes? The script can generate the patch files.")