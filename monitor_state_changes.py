#!/usr/bin/env python3
"""
Monitor system_state.json file changes in real-time to identify what's adding the deleted repos.
"""

import json
import time
import os
import hashlib
from datetime import datetime

def get_file_hash(filepath):
    """Get MD5 hash of file content."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_project_names(filepath):
    """Get project names from state file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return list(data.get('projects', {}).keys())
    except:
        return []

def monitor_state_file(filepath="system_state.json", interval=0.5):
    """Monitor state file for changes."""
    print(f"Monitoring {filepath} for changes...")
    print("Press Ctrl+C to stop\n")
    
    last_hash = ""
    last_projects = []
    
    try:
        while True:
            if os.path.exists(filepath):
                current_hash = get_file_hash(filepath)
                current_projects = get_project_names(filepath)
                
                if current_hash != last_hash:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    # Check what changed
                    if set(current_projects) != set(last_projects):
                        added = set(current_projects) - set(last_projects)
                        removed = set(last_projects) - set(current_projects)
                        
                        print(f"\n[{timestamp}] PROJECTS CHANGED!")
                        if added:
                            print(f"  ADDED: {list(added)}")
                        if removed:
                            print(f"  REMOVED: {list(removed)}")
                        print(f"  Current projects: {current_projects}")
                    else:
                        print(f"[{timestamp}] File updated (no project changes)")
                    
                    # Show last_updated from file
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            print(f"  last_updated: {data.get('last_updated', 'N/A')}")
                    except:
                        pass
                    
                    last_hash = current_hash
                    last_projects = current_projects
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_state_file()