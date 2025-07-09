#!/usr/bin/env python3
"""Monitor and diagnose disk I/O errors in the system."""

import os
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime

def check_sqlite_integrity():
    """Check integrity of SQLite databases."""
    print("\n🔍 Checking SQLite database integrity...")
    
    db_files = list(Path('.').rglob('*.db'))
    
    for db_file in db_files:
        print(f"\nChecking: {db_file}")
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result[0] == 'ok':
                print(f"✅ {db_file} - Integrity OK")
            else:
                print(f"❌ {db_file} - Integrity issues: {result}")
                
            conn.close()
        except Exception as e:
            print(f"❌ Error checking {db_file}: {e}")

def check_file_locks():
    """Check for file locks that might cause I/O errors."""
    print("\n🔍 Checking for file locks...")
    
    # Check common lock patterns
    lock_patterns = ['*.lock', '*.lck', '*.pid', '*-journal', '*-wal']
    
    for pattern in lock_patterns:
        locks = list(Path('.').rglob(pattern))
        if locks:
            print(f"\n⚠️ Found {len(locks)} {pattern} files:")
            for lock in locks[:5]:  # Show first 5
                print(f"  - {lock}")
                # Check if file is stale
                try:
                    age = time.time() - os.path.getmtime(lock)
                    if age > 3600:  # Older than 1 hour
                        print(f"    ⚠️ Stale lock (age: {age/3600:.1f} hours)")
                except:
                    pass

def check_disk_writes():
    """Test disk write capabilities."""
    print("\n🔍 Testing disk write capabilities...")
    
    test_dirs = [
        './local_repos/state',
        './research_knowledge',
        './logs',
        '.'
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            test_file = os.path.join(test_dir, f'.write_test_{int(time.time())}.tmp')
            try:
                # Test write
                with open(test_file, 'w') as f:
                    f.write('test')
                
                # Test read
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Clean up
                os.remove(test_file)
                
                print(f"✅ {test_dir} - Write/Read OK")
            except Exception as e:
                print(f"❌ {test_dir} - Write/Read failed: {e}")

def check_open_files():
    """Check number of open files."""
    print("\n🔍 Checking open file descriptors...")
    
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"File descriptor limits - Soft: {soft}, Hard: {hard}")
        
        # Count open files in /proc (Linux) or lsof (macOS)
        import subprocess
        try:
            # For macOS
            result = subprocess.run(['lsof', '-p', str(os.getpid())], 
                                  capture_output=True, text=True)
            open_files = len(result.stdout.strip().split('\n')) - 1  # Subtract header
            print(f"Open files for this process: {open_files}")
            
            if open_files > soft * 0.8:
                print("⚠️ Warning: Approaching file descriptor limit!")
        except:
            pass
    except Exception as e:
        print(f"Could not check file limits: {e}")

def main():
    """Run all diagnostics."""
    print("🩺 Disk I/O Diagnostics")
    print("=" * 50)
    
    # Run all checks
    check_disk_writes()
    check_file_locks()
    check_sqlite_integrity()
    check_open_files()
    
    print("\n✅ Diagnostics complete!")
    
    # Recommendations
    print("\n📋 Recommendations:")
    print("1. If SQLite integrity issues found: backup and recreate databases")
    print("2. If stale locks found: remove them carefully")
    print("3. If write tests fail: check disk space and permissions")
    print("4. If approaching file limit: increase ulimit or close unused files")

if __name__ == "__main__":
    main()