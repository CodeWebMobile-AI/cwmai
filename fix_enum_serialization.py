#!/usr/bin/env python3
"""
Fix for enum serialization issues in Redis work queue.
"""

import sys
sys.path.append('/workspaces/cwmai')

from scripts.work_item_types import TaskPriority

# Test the issue
print("Testing enum serialization issue:")
print(f"TaskPriority.MEDIUM: {TaskPriority.MEDIUM}")
print(f"str(TaskPriority.MEDIUM): {str(TaskPriority.MEDIUM)}")
print(f"TaskPriority.MEDIUM.name: {TaskPriority.MEDIUM.name}")
print(f"TaskPriority.MEDIUM.value: {TaskPriority.MEDIUM.value}")

# Test dictionary key usage
priority_dict = {}
priority_dict[TaskPriority.MEDIUM] = "test"
print(f"\nDictionary with enum key: {priority_dict}")

# The issue is likely in how the error is being formatted
try:
    # Simulate the error
    raise Exception(TaskPriority.MEDIUM)
except Exception as e:
    print(f"\nException with enum: {e}")
    print(f"Type of exception arg: {type(e.args[0])}")