#!/usr/bin/env python3
"""Monitor the continuous system status"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_system_status():
    """Check the status of the continuous system"""
    print("📊 CWMAI Continuous System Status\n")
    
    # Check system state file
    state_file = project_root / "continuous_orchestrator_state.json"
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            print("✅ System State Found")
            print(f"   Last Update: {state.get('last_update', 'Unknown')}")
            print(f"   Status: {state.get('status', 'Unknown')}")
            
            # Show active workers
            workers = state.get('active_workers', {})
            if workers:
                print(f"\n👷 Active Workers: {len(workers)}")
                for worker_id, worker_info in list(workers.items())[:5]:
                    print(f"   - {worker_id}: {worker_info.get('status', 'Unknown')}")
                if len(workers) > 5:
                    print(f"   ... and {len(workers) - 5} more")
            
            # Show work items
            work_items = state.get('work_items_in_progress', [])
            if work_items:
                print(f"\n📋 Work Items in Progress: {len(work_items)}")
                for item in work_items[:3]:
                    print(f"   - {item.get('title', 'Untitled')}")
                if len(work_items) > 3:
                    print(f"   ... and {len(work_items) - 3} more")
                    
        except Exception as e:
            print(f"❌ Error reading state: {e}")
    else:
        print("⚠️  No system state file found")
        print("   The continuous system may not be running")
    
    # Check log file
    log_file = project_root / "continuous_ai.log"
    if log_file.exists():
        print(f"\n📄 Log File: {log_file}")
        # Get last few lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("   Recent activity:")
                    for line in lines[-3:]:
                        print(f"   {line.strip()}")
        except:
            pass
    
    # Check task history
    task_history = project_root / "task_history.json"
    if task_history.exists():
        try:
            with open(task_history, 'r') as f:
                history = json.load(f)
            
            completed = len([t for t in history.get('tasks', []) if t.get('status') == 'completed'])
            total = len(history.get('tasks', []))
            print(f"\n📈 Task History:")
            print(f"   Total Tasks: {total}")
            print(f"   Completed: {completed}")
            if total > 0:
                print(f"   Success Rate: {(completed/total)*100:.1f}%")
        except:
            pass
    
    print("\n💡 To start the continuous system, run:")
    print("   ./start_continuous_system.sh")
    print("\n📖 Options:")
    print("   --basic          Use basic mode (no MCP)")
    print("   --workers N      Set number of workers")
    print("   --no-parallel    Disable parallel processing")
    print("   --no-research    Disable research engine")

if __name__ == "__main__":
    check_system_status()