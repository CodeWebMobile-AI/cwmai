#!/usr/bin/env python3
"""Start fresh in local mode - clear GitHub repositories from state."""

import os
import json
from datetime import datetime, timezone

# Clear the existing system state to remove GitHub repositories
state_file = "system_state.json"

# Create a fresh state with no projects
fresh_state = {
    "charter": {
        "primary_goal": "innovation",
        "secondary_goal": "community_engagement", 
        "constraints": ["maintain_quality", "ensure_security"]
    },
    "projects": {},  # Empty projects - no GitHub repos
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
    "storage_mode": "local"  # Explicitly set local mode
}

# Back up existing state
if os.path.exists(state_file):
    backup_name = f"{state_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(state_file, backup_name)
    print(f"✓ Backed up existing state to {backup_name}")

# Write fresh state
with open(state_file, 'w') as f:
    json.dump(fresh_state, f, indent=2)
    
print("✓ Created fresh local state with no GitHub repositories")
print("\nNow you can run:")
print("export $(cat .env.local | grep -v '^#' | xargs) && \\")
print("USE_LOCAL_MODE=true python3 run_hybrid_local_mode.py --workers 10")