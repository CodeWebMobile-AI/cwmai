#!/usr/bin/env python3
"""
Debug EnhancedSmartTaskSystem initialization.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set to local mode
os.environ['STORAGE_MODE'] = 'local'

try:
    print("1. Importing EnhancedSmartTaskSystem...")
    from scripts.enhanced_smart_task_system import EnhancedSmartTaskSystem
    print("   ✓ EnhancedSmartTaskSystem imported")
    
    print("\n2. Creating EnhancedSmartTaskSystem...")
    ests = EnhancedSmartTaskSystem()
    print("   ✓ EnhancedSmartTaskSystem created")
    
    print("\n3. Testing get_task_metadata...")
    from scripts.task_types import SmartTaskType
    metadata = ests.get_task_metadata(SmartTaskType.FEATURE_BUSINESS_LOGIC.value)
    print(f"   ✓ Metadata retrieved: {metadata}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()