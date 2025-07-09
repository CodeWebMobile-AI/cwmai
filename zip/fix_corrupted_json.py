#!/usr/bin/env python3
"""Fix corrupted JSON files in the system."""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime

def fix_json_file(filepath):
    """Attempt to fix a corrupted JSON file."""
    print(f"\nChecking: {filepath}")
    
    try:
        # Try to load the file
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✅ {filepath} is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ {filepath} is corrupted: {e}")
        
        # Backup the corrupted file
        backup_path = f"{filepath}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"📋 Backed up to: {backup_path}")
        
        # Determine appropriate empty structure based on filename
        if 'task_history' in filepath or 'completed_tasks' in filepath:
            empty_data = []
        elif 'research' in filepath:
            empty_data = {
                "research_data": {},
                "opportunities": [],
                "timestamp": datetime.now().isoformat()
            }
        else:
            empty_data = {}
        
        # Write clean empty structure
        with open(filepath, 'w') as f:
            json.dump(empty_data, f, indent=2)
        print(f"✅ Created clean empty JSON file")
        return True
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

def main():
    """Find and fix all corrupted JSON files."""
    corrupted_files = [
        "./local_repos/state/task_history.json",
        "./research_knowledge/research_20250701_024323.json",
        "./research_knowledge/research_20250630_224657.json",
        "./research_knowledge/research_20250630_230058.json",
        "./research_knowledge/research_20250701_114852.json",
        "./research_knowledge/research_20250630_225254.json",
        "./research_knowledge/research_20250630_232315.json"
    ]
    
    print("🔧 Fixing corrupted JSON files...")
    
    fixed = 0
    failed = 0
    
    for filepath in corrupted_files:
        if os.path.exists(filepath):
            if fix_json_file(filepath):
                fixed += 1
            else:
                failed += 1
        else:
            print(f"⚠️ File not found: {filepath}")
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Fixed: {fixed} files")
    print(f"  ❌ Failed: {failed} files")
    
    # Also check unified_task_state.json to ensure it's properly initialized
    unified_state_path = "./unified_task_state.json"
    if os.path.exists(unified_state_path):
        try:
            with open(unified_state_path, 'r') as f:
                data = json.load(f)
            
            # Ensure proper structure
            if not data.get('active_tasks'):
                data['active_tasks'] = {}
            if not data.get('completed_tasks'):
                data['completed_tasks'] = {}
            if not data.get('task_history'):
                data['task_history'] = []
            
            with open(unified_state_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n✅ Verified unified_task_state.json structure")
        except Exception as e:
            print(f"\n❌ Error with unified_task_state.json: {e}")

if __name__ == "__main__":
    main()