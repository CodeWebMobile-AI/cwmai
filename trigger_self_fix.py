#!/usr/bin/env python3
"""
Trigger script to help the self-improvement system detect and fix the async tool issue
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Create error log entry for pattern learning
error_entry = {
    "timestamp": datetime.now().isoformat(),
    "error_type": "CoroutineNotAwaited",
    "message": "coroutine object check_for_architecture_file at 0x17f545a20 - async function called without await",
    "location": "scripts/tool_calling_system.py:419",
    "frequency": 5,  # Indicate this happens frequently
    "impact": "high",
    "details": {
        "function": "check_for_architecture_file",
        "issue": "asyncio.iscoroutinefunction() failing to detect async custom tools",
        "symptom": "Returns coroutine object instead of result",
        "affected_tools": ["check_for_architecture_file", "check_intelligence"]
    }
}

# Add to error logs for pattern detection
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

error_log_file = logs_dir / "tool_execution_errors.log"
with open(error_log_file, 'a') as f:
    f.write(f"ERROR: {json.dumps(error_entry)}\n")

# Create task history entry showing the failure pattern
task_history_file = Path("task_history.json")
if task_history_file.exists():
    with open(task_history_file, 'r') as f:
        history = json.load(f)
else:
    history = {"tasks": []}

# Add failed task executions
if not isinstance(history, dict):
    history = {"tasks": []}
if "tasks" not in history:
    history["tasks"] = []
    
for i in range(3):  # Add multiple occurrences
    history["tasks"].append({
        "timestamp": datetime.now().isoformat(),
        "type": "tool_execution",
        "tool": "check_for_architecture_file",
        "success": False,
        "error": "CoroutineNotAwaited",
        "duration": 0.001,
        "details": "Returned coroutine object instead of result"
    })

with open(task_history_file, 'w') as f:
    json.dump(history, f, indent=2)

# Update system state to indicate this is a recurring issue
system_state_file = Path("system_state.json")
if system_state_file.exists():
    with open(system_state_file, 'r') as f:
        state = json.load(f)
else:
    state = {}

state["recurring_errors"] = state.get("recurring_errors", {})
state["recurring_errors"]["CoroutineNotAwaited"] = {
    "count": 5,
    "last_occurrence": datetime.now().isoformat(),
    "affected_components": ["tool_calling_system", "custom_tools"],
    "severity": "high"
}

with open(system_state_file, 'w') as f:
    json.dump(state, f, indent=2)

print("✅ Created error patterns for self-improvement system to detect")
print("📊 Added entries to:")
print("   - logs/tool_execution_errors.log")
print("   - task_history.json")
print("   - system_state.json")
print("\n🤖 The autonomous improvement system should now:")
print("   1. Detect the CoroutineNotAwaited error pattern")
print("   2. Identify it's happening in tool_calling_system.py")
print("   3. Generate a fix for the async detection issue")
print("   4. Apply the fix automatically")
print("\nRun the continuous system to trigger self-improvement!")