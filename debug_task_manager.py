#!/usr/bin/env python3
"""
Debug TaskManager initialization issue.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set to local mode
os.environ['STORAGE_MODE'] = 'local'

try:
    print("1. Importing TaskManager...")
    from scripts.task_manager import TaskManager
    print("   ✓ TaskManager imported")
    
    print("\n2. Creating TaskManager with repository=None...")
    tm = TaskManager(repository=None)
    print("   ✓ TaskManager created")
    
    print("\n3. Checking TaskManager attributes...")
    print(f"   - Has github: {hasattr(tm, 'github')}")
    print(f"   - GitHub value: {getattr(tm, 'github', 'NOT FOUND')}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()