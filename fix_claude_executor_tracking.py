#!/usr/bin/env python3
"""Fix the issue with Claude executors not properly tracking their current tasks."""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def fix_executor_tracking():
    """Apply fix to ClaudeExecutor to ensure current_task is properly tracked."""
    
    print("=== FIXING CLAUDE EXECUTOR TRACKING ===")
    
    # Read the current executor file
    executor_file = "scripts/claude_executor.py"
    with open(executor_file, 'r') as f:
        content = f.read()
    
    # Check if the fix is already applied
    if "self.current_task = None  # Clear immediately after completion" in content:
        print("✅ Fix already applied!")
        return
    
    # The issue is in the run() method - we need to ensure current_task is cleared
    # even when the task fails or times out
    
    # Find the section where current_task is cleared
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        # Look for where we set current_task = None after processing
        if line.strip() == "self.current_task = None" and i > 70 and i < 85:
            # This is the correct location, but we need to ensure it's always executed
            # Check if there's proper error handling
            print(f"Found current_task clearing at line {i+1}")
            
            # Look backwards to ensure we're in a finally block or after all error handling
            in_finally = False
            for j in range(max(0, i-10), i):
                if "finally:" in lines[j]:
                    in_finally = True
                    break
            
            if not in_finally:
                print("⚠️ current_task clearing is not in a finally block!")
                # We need to restructure this section
                modified = True
    
    if modified or True:  # Force the fix
        # Apply a more robust fix
        print("Applying comprehensive fix...")
        
        # Replace the entire run method with a more robust version
        new_run_method = '''    async def run(self):
        """Main executor loop."""
        self.logger.info(f"Claude executor {self.executor_id} started")
        
        while not self.shutdown_requested:
            try:
                # Get next task from queue
                task = await self.queue.get_next_task(self.executor_id)
                
                if not task:
                    # No work available, wait a bit
                    await asyncio.sleep(5)
                    continue
                
                # Set current task BEFORE processing
                self.current_task = task
                
                try:
                    # Execute the task
                    result = await self._execute_task(task)
                    
                    if result['success']:
                        await self.queue.complete_task(task.id, result)
                        self.stats['tasks_completed'] += 1
                    else:
                        # Check if we should retry
                        retry_count = result.get('retry_count', 0)
                        if retry_count < self.max_retries:
                            result['retry_count'] = retry_count + 1
                            await self.queue.fail_task(task.id, result.get('error', 'Unknown error'), retry=True)
                        else:
                            await self.queue.fail_task(task.id, result.get('error', 'Max retries exceeded'), retry=False)
                            self.stats['tasks_failed'] += 1
                
                finally:
                    # ALWAYS clear current task, even if an exception occurred
                    self.current_task = None
                    self.logger.debug(f"Cleared current_task for executor {self.executor_id}")
                
            except Exception as e:
                self.logger.error(f"Executor error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Ensure current_task is cleared even on unexpected errors
                self.current_task = None
                await asyncio.sleep(5)
        
        self.logger.info(f"Claude executor {self.executor_id} stopped")'''
        
        # Find the run method and replace it
        import re
        pattern = r'(    async def run\(self\):.*?)(\n    async def [^r]|\n    def |\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # Replace the run method
            content = content[:match.start(1)] + new_run_method + content[match.start(2):]
            
            # Write the fixed content
            with open(executor_file, 'w') as f:
                f.write(content)
            
            print("✅ Fix applied successfully!")
            print("\nChanges made:")
            print("1. Added proper try/finally block to ensure current_task is always cleared")
            print("2. Added debug logging when current_task is cleared")
            print("3. Ensured current_task is set BEFORE processing starts")
            print("\n⚠️ IMPORTANT: Restart the executor pool for changes to take effect!")
        else:
            print("❌ Could not find run method to patch!")
    
    # Also check the monitoring logic in dynamic pool
    print("\nChecking dynamic pool monitoring logic...")
    pool_file = "scripts/dynamic_claude_executor_pool.py"
    
    with open(pool_file, 'r') as f:
        pool_content = f.read()
    
    # The issue might also be in how the pool checks executor status
    if "if executor.current_task:" in pool_content:
        print("✅ Pool is checking current_task correctly")
    else:
        print("⚠️ Pool might not be checking executor status correctly")

if __name__ == "__main__":
    fix_executor_tracking()