#!/usr/bin/env python3
"""Reset the system to pure local mode - remove all GitHub repositories from state."""

import json
import os
from datetime import datetime, timezone

print("🏠 Resetting system to PURE LOCAL MODE...")

# 1. Clear the system state
state_file = "system_state.json"
if os.path.exists(state_file):
    # Backup existing state
    backup_name = f"{state_file}.backup_github_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(state_file, backup_name)
    print(f"✓ Backed up existing state to {backup_name}")

# Create fresh local-only state
fresh_state = {
    "charter": {
        "primary_goal": "innovation",
        "secondary_goal": "community_engagement", 
        "constraints": ["maintain_quality", "ensure_security"]
    },
    "projects": {},  # Empty - no projects at all
    "system_performance": {
        "total_cycles": 0,
        "successful_actions": 0,
        "failed_actions": 0,
        "learning_metrics": {
            "decision_accuracy": 0.0,
            "resource_efficiency": 0.0,
            "goal_achievement": 0.0
        }
    },
    "task_queue": [],
    "external_context": {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "market_trends": [],
        "security_alerts": [],
        "technology_updates": []
    },
    "version": "1.0.0",
    "last_updated": datetime.now(timezone.utc).isoformat(),
    "storage_mode": "local",
    "local_mode": True,
    "github_disabled": True
}

# Write fresh state
with open(state_file, 'w') as f:
    json.dump(fresh_state, f, indent=2)
    
print("✓ Created fresh local-only state")

# 2. Clear task state
task_state_file = "task_state.json"
if os.path.exists(task_state_file):
    backup_name = f"{task_state_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(task_state_file, backup_name)
    print(f"✓ Backed up task state to {backup_name}")

# Create empty task state
with open(task_state_file, 'w') as f:
    json.dump({
        "completed_tasks": {},
        "active_tasks": {},
        "task_history": [],
        "last_updated": datetime.now(timezone.utc).isoformat()
    }, f, indent=2)
    
print("✓ Created fresh task state")

# 3. Clear any work queue
work_queue_file = "work_queue_backup.json"
if os.path.exists(work_queue_file):
    os.remove(work_queue_file)
    print("✓ Cleared work queue")

# 4. Clear Claude execution queue
claude_queue_file = "claude_execution_queue.json"
if os.path.exists(claude_queue_file):
    backup_name = f"{claude_queue_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(claude_queue_file, backup_name)
    print(f"✓ Backed up Claude queue to {backup_name}")

# Create empty Claude queue
with open(claude_queue_file, 'w') as f:
    json.dump({"queue": [], "processing": {}}, f, indent=2)
    
print("✓ Created fresh Claude queue")

print("\n✅ System reset to PURE LOCAL MODE complete!")
print("\nThe system will now:")
print("- ❌ NEVER access GitHub API")
print("- ❌ NEVER try to fork repositories") 
print("- ✅ Only work with local files")
print("- ✅ Create projects locally only")
print("\nYou can now run the system without any GitHub interference.")