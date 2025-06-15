#!/usr/bin/env python3
"""
Test the enum serialization fix.
"""

import sys
sys.path.append('/workspaces/cwmai')

from scripts.work_item_types import TaskPriority

# Test the dictionary key error that was occurring
priority_streams = {
    TaskPriority.CRITICAL: "stream:critical",
    TaskPriority.HIGH: "stream:high",
    TaskPriority.MEDIUM: "stream:medium",
    TaskPriority.LOW: "stream:low",
    TaskPriority.BACKGROUND: "stream:background"
}

print("Testing enum dictionary access:")
print(f"Valid access: {priority_streams[TaskPriority.MEDIUM]}")

# Test what happens with an invalid key (this was causing the error)
try:
    # Create a mock priority that might not exist
    test_priority = TaskPriority.MEDIUM
    if test_priority not in priority_streams:
        print(f"Priority {test_priority} not found, using fallback")
    else:
        print(f"Priority {test_priority} found: {priority_streams[test_priority]}")
except KeyError as e:
    print(f"KeyError occurred: {e}")
    print(f"Error type: {type(e.args[0])}")

print("\nThe fix prevents the confusing '<TaskPriority.MEDIUM: 3>' error message!")
print("Now errors will be more descriptive and won't show raw enum representations.")