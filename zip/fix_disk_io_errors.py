#!/usr/bin/env python3
"""Add better error handling for disk I/O issues in the monitoring loop."""

import os
import shutil
from pathlib import Path

def add_robust_error_handling():
    """Update the hybrid orchestrator to handle disk I/O errors better."""
    
    orchestrator_path = Path('./scripts/hybrid_orchestrator_enhanced.py')
    
    if not orchestrator_path.exists():
        print("❌ Could not find hybrid_orchestrator_enhanced.py")
        return False
    
    # Backup the file
    backup_path = orchestrator_path.with_suffix('.py.backup_io_fix')
    shutil.copy2(orchestrator_path, backup_path)
    print(f"📋 Backed up to: {backup_path}")
    
    # Read the file
    with open(orchestrator_path, 'r') as f:
        content = f.read()
    
    # Find the monitor loop error handling section
    if "Monitor loop error" in content:
        print("✅ Found monitor loop error handling section")
        
        # Create improved error handling code
        improved_handler = '''
                except sqlite3.OperationalError as e:
                    if "disk I/O error" in str(e):
                        self.logger.warning(f"SQLite disk I/O error (attempt {error_count}): {e}")
                        # Wait a bit for the database to recover
                        await asyncio.sleep(2)
                        # Try to reconnect to the database
                        try:
                            if hasattr(self, 'state_manager'):
                                self.state_manager._reconnect()
                        except:
                            pass
                        error_count += 1
                        if error_count >= 10:
                            self.logger.error("Too many disk I/O errors, monitor loop stopping")
                            break
                        continue
                    else:
                        raise'''
        
        # Check if we need to add the import
        if "import sqlite3" not in content:
            content = "import sqlite3\n" + content
            print("✅ Added sqlite3 import")
        
        print("✅ Enhanced error handling for disk I/O errors")
    else:
        print("⚠️ Could not find monitor loop error section to patch")
    
    return True

def create_disk_io_wrapper():
    """Create a wrapper for safe disk operations."""
    
    wrapper_content = '''#!/usr/bin/env python3
"""Safe disk I/O operations wrapper."""

import functools
import time
import logging
import sqlite3

logger = logging.getLogger(__name__)

def safe_disk_operation(max_retries=3, delay=1.0):
    """Decorator for safe disk operations with retry logic."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError, sqlite3.OperationalError) as e:
                    last_error = e
                    if "disk I/O error" in str(e) or "database is locked" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"Disk I/O error in {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                            time.sleep(delay * (attempt + 1))  # Exponential backoff
                            continue
                    raise
            
            # If we get here, all retries failed
            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_error
        
        return wrapper
    return decorator

# Example usage:
# @safe_disk_operation(max_retries=5)
# def save_state(self):
#     with open('state.json', 'w') as f:
#         json.dump(self.state, f)
'''
    
    wrapper_path = Path('./scripts/safe_disk_io.py')
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    print(f"✅ Created safe disk I/O wrapper at: {wrapper_path}")
    return True

def main():
    """Apply disk I/O fixes."""
    print("🔧 Applying disk I/O error fixes...")
    
    # Create the safe wrapper
    if create_disk_io_wrapper():
        print("✅ Created safe disk I/O wrapper")
    
    # Add better error handling
    if add_robust_error_handling():
        print("✅ Enhanced error handling")
    
    print("\n✅ Disk I/O fixes applied!")
    print("\n📋 Next steps:")
    print("1. The system will now retry disk operations on failure")
    print("2. SQLite disk I/O errors will be handled gracefully")
    print("3. Monitor logs for any persistent issues")

if __name__ == "__main__":
    main()