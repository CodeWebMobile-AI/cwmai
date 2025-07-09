#!/usr/bin/env python3
"""Check the status of the background daemon without starting it."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_daemon_status():
    """Check and display the daemon status."""
    state_file = Path.home() / '.cwmai' / 'background_daemon_state.json'
    
    print("🔍 Background Intelligence Daemon Status")
    print("=" * 50)
    
    if not state_file.exists():
        print("❌ Daemon state file not found")
        print("   The daemon has never been run on this system")
        return
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Check last update time
        last_update = datetime.fromisoformat(state.get('last_updated', '2000-01-01T00:00:00+00:00'))
        current_time = datetime.now(timezone.utc)
        time_since_update = (current_time - last_update).total_seconds()
        
        # Determine if daemon is likely running
        if time_since_update < 120:  # Updated within last 2 minutes
            status = "🟢 RUNNING"
            status_detail = f"Last active {time_since_update:.0f} seconds ago"
        elif time_since_update < 600:  # Updated within last 10 minutes
            status = "🟡 POSSIBLY STOPPED"
            status_detail = f"Last active {time_since_update/60:.1f} minutes ago"
        else:
            status = "🔴 STOPPED"
            hours = time_since_update / 3600
            if hours < 24:
                status_detail = f"Last active {hours:.1f} hours ago"
            else:
                days = hours / 24
                status_detail = f"Last active {days:.1f} days ago"
        
        print(f"\n📊 Status: {status}")
        print(f"   {status_detail}")
        print(f"   Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Show task information
        print("\n📋 Scheduled Tasks:")
        tasks = state.get('tasks', {})
        
        for task_name, task_info in tasks.items():
            interval = task_info.get('interval', 0)
            last_run_str = task_info.get('last_run')
            
            # Format interval
            if interval < 3600:
                interval_str = f"{interval/60:.0f} min"
            else:
                interval_str = f"{interval/3600:.1f} hours"
            
            # Check last run
            if last_run_str:
                last_run = datetime.fromisoformat(last_run_str)
                time_since_run = (current_time - last_run).total_seconds()
                next_run_in = interval - time_since_run
                
                if next_run_in > 0:
                    if next_run_in < 3600:
                        next_str = f"in {next_run_in/60:.0f} min"
                    else:
                        next_str = f"in {next_run_in/3600:.1f} hours"
                else:
                    next_str = "overdue" if time_since_update < 600 else "N/A"
                
                # Format last run time
                if time_since_run < 3600:
                    last_run_str = f"{time_since_run/60:.0f} min ago"
                elif time_since_run < 86400:
                    last_run_str = f"{time_since_run/3600:.1f} hours ago"
                else:
                    last_run_str = f"{time_since_run/86400:.1f} days ago"
            else:
                last_run_str = "Never"
                next_str = "Now" if time_since_update < 600 else "N/A"
            
            print(f"\n   📌 {task_name}:")
            print(f"      Interval: Every {interval_str}")
            print(f"      Last run: {last_run_str}")
            print(f"      Next run: {next_str}")
        
        # Provide recommendations
        print("\n💡 Recommendations:")
        if time_since_update > 600:
            print("   - The daemon appears to be stopped")
            print("   - Run 'python3 start_background_daemon.py' to start it")
        elif time_since_update > 120:
            print("   - The daemon may have stopped unexpectedly")
            print("   - Check logs for errors")
        else:
            print("   - The daemon is running normally")
            print("   - No action needed")
            
    except Exception as e:
        print(f"\n❌ Error reading daemon state: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_daemon_status()