#!/usr/bin/env python3
"""Start the background daemon as a separate process."""

import asyncio
import sys
import os
import signal
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.background_intelligence_daemon import BackgroundIntelligenceDaemon
from scripts.conversational_ai_logger import setup_conversational_ai_logging

# Global daemon instance for signal handling
daemon_instance = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n⚠️  Received signal {signum}, shutting down daemon...")
    if daemon_instance and daemon_instance.running:
        # Schedule daemon stop
        asyncio.create_task(daemon_instance.stop())

async def run_daemon():
    """Run the background daemon continuously."""
    global daemon_instance
    
    setup_conversational_ai_logging()
    
    print("🚀 Starting Background Intelligence Daemon")
    print("=" * 50)
    print(f"📅 Start time: {datetime.now(timezone.utc)}")
    print(f"📁 State file: {Path.home() / '.cwmai' / 'background_daemon_state.json'}")
    print("=" * 50)
    
    # Create daemon instance
    daemon_instance = BackgroundIntelligenceDaemon()
    
    # Check previous state
    if daemon_instance.state_file.exists():
        daemon_instance._load_state()
        print("\n📊 Previous task runs:")
        for task_name, task_info in daemon_instance.tasks.items():
            last_run = task_info.get('last_run')
            if last_run:
                print(f"  - {task_name}: {last_run}")
            else:
                print(f"  - {task_name}: Never run")
    
    print("\n✨ Daemon is starting background tasks...")
    print("Press Ctrl+C to stop the daemon gracefully\n")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the daemon
        await daemon_instance.start()
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received")
    except Exception as e:
        print(f"\n❌ Daemon error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        if daemon_instance.running:
            await daemon_instance.stop()
        print("\n✅ Daemon stopped successfully")

def main():
    """Main entry point."""
    # Check if daemon is already running by looking at recent state updates
    state_file = Path.home() / '.cwmai' / 'background_daemon_state.json'
    
    if state_file.exists():
        import json
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                last_update = datetime.fromisoformat(state.get('last_updated', '2000-01-01T00:00:00+00:00'))
                time_since_update = (datetime.now(timezone.utc) - last_update).total_seconds()
                
                if time_since_update < 60:  # Updated within last minute
                    print("⚠️  Warning: Daemon appears to be already running")
                    print(f"   Last update: {last_update} ({time_since_update:.0f} seconds ago)")
                    response = input("   Continue anyway? (y/N): ")
                    if response.lower() != 'y':
                        print("Aborted.")
                        return
        except:
            pass
    
    # Run the daemon
    try:
        asyncio.run(run_daemon())
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()