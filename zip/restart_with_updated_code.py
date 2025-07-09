#!/usr/bin/env python3
"""
Script to restart the system with the updated gemini_deep_research.py code
that contains the schema complexity fix.
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

def kill_existing_processes():
    """Kill any existing Python processes running the system."""
    print("🔍 Looking for existing system processes...")
    
    # Find processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        killed = 0
        for line in lines:
            # Look for Python processes running our scripts
            if 'python' in line and ('continuous_orchestrator' in line or 
                                     'intelligent_work_finder' in line or
                                     'gemini_deep_research' in line or
                                     'market_discovery' in line):
                parts = line.split()
                if len(parts) > 1:
                    pid = int(parts[1])
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"   ✅ Killed process {pid}")
                        killed += 1
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        print(f"   ⚠️  No permission to kill process {pid}")
        
        if killed > 0:
            print(f"   Killed {killed} processes, waiting 3 seconds...")
            time.sleep(3)
        else:
            print("   No existing processes found")
            
    except Exception as e:
        print(f"   Error checking processes: {e}")

def verify_schema_fix():
    """Verify the schema fix is in place."""
    print("\n🔍 Verifying schema fix in gemini_deep_research.py...")
    
    file_path = Path(__file__).parent / "scripts" / "gemini_deep_research.py"
    if not file_path.exists():
        print("   ❌ gemini_deep_research.py not found!")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for the simplified schema
    if '"search_evidence": {' in content and '"summary": {"type": "string"}' in content:
        print("   ✅ Schema fix is present!")
        return True
    else:
        print("   ❌ Schema fix not found in file!")
        return False

def clear_cache():
    """Clear any Python cache that might contain old code."""
    print("\n🧹 Clearing Python cache...")
    
    # Clear __pycache__ directories
    scripts_dir = Path(__file__).parent / "scripts"
    pycache_dirs = list(scripts_dir.rglob("__pycache__"))
    
    for pycache in pycache_dirs:
        try:
            import shutil
            shutil.rmtree(pycache)
            print(f"   ✅ Removed {pycache}")
        except Exception as e:
            print(f"   ⚠️  Could not remove {pycache}: {e}")
    
    # Clear .pyc files
    pyc_files = list(scripts_dir.rglob("*.pyc"))
    for pyc in pyc_files:
        try:
            pyc.unlink()
            print(f"   ✅ Removed {pyc}")
        except Exception as e:
            print(f"   ⚠️  Could not remove {pyc}: {e}")

def restart_system():
    """Restart the continuous orchestrator with fresh code."""
    print("\n🚀 Starting system with updated code...")
    
    # Export environment variables and start the system
    cmd = [
        'bash', '-c',
        'export $(cat .env.local | grep -v "^#" | xargs) && '
        'PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip '
        'python3 scripts/continuous_orchestrator_no_redis.py'
    ]
    
    print("   Running:", ' '.join(cmd))
    print("\n" + "="*60)
    print("System is starting with the schema fix applied.")
    print("Watch for any schema errors in the output.")
    print("Press Ctrl+C to stop.")
    print("="*60 + "\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✅ System stopped by user")

def main():
    """Main function to restart the system with updated code."""
    print("Schema Fix Restart Script")
    print("=" * 60)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Verify the fix is present
    if not verify_schema_fix():
        print("\n❌ ERROR: Schema fix not found in code!")
        print("The fix should be in scripts/gemini_deep_research.py")
        print("in the _filter_and_rank method around line 890-1040")
        sys.exit(1)
    
    # Clear cache
    clear_cache()
    
    # Restart the system
    restart_system()

if __name__ == "__main__":
    main()