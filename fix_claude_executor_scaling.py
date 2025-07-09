#!/usr/bin/env python3
"""
Fix Claude executor scaling logic to check available repositories before scaling up.
"""

import re

def fix_executor_scaling():
    """Add repository availability check to scaling logic."""
    
    # Read the file
    with open('scripts/dynamic_claude_executor_pool.py', 'r') as f:
        content = f.read()
    
    # Find the _should_scale_up method
    scale_up_pattern = r'(def _should_scale_up\(self\) -> bool:.*?)(return scale_up)'
    
    # New method to check available repositories
    new_method = '''
    def _get_available_repositories_count(self) -> int:
        """Get count of repositories without active tasks."""
        if not hasattr(self, 'local_simulator'):
            return 0
            
        try:
            repos = self.local_simulator.list_repositories()
            available_count = 0
            
            for repo_data in repos:
                repo_name = repo_data.get('name', '')
                full_name = f"CodeWebMobile-AI/{repo_name}"
                
                # Check if repository has active task
                if not self.claude_queue.has_active_task_for_repository(full_name):
                    available_count += 1
                    
            return available_count
        except Exception as e:
            self.logger.error(f"Error checking available repositories: {e}")
            return 0
'''
    
    # Insert the new method before _should_scale_up
    scale_up_match = re.search(r'def _should_scale_up\(self\)', content)
    if scale_up_match:
        insert_pos = scale_up_match.start()
        content = content[:insert_pos] + new_method + '\n    ' + content[insert_pos:]
        print("✅ Added _get_available_repositories_count method")
    
    # Now update the scaling logic
    old_scale_logic = '''        # Check various conditions
        queue_size = self.claude_queue.get_queue_size()
        processing_count = len(self.claude_queue.get_processing_tasks())
        
        # Scale up if queue is building up
        if queue_size > self.current_executors * 2:
            self.logger.info(f"Scaling up due to: queue size ({queue_size} > {self.current_executors * 2})")
            return True'''
    
    new_scale_logic = '''        # Check various conditions
        queue_size = self.claude_queue.get_queue_size()
        processing_count = len(self.claude_queue.get_processing_tasks())
        available_repos = self._get_available_repositories_count()
        
        # Don't scale if no repositories available
        if available_repos == 0:
            self.logger.debug("No scaling needed - all repositories have active tasks")
            return False
        
        # Scale up if queue is building up AND repositories are available
        if queue_size > self.current_executors * 2 and available_repos > 0:
            self.logger.info(f"Scaling up due to: queue size ({queue_size} > {self.current_executors * 2}) and {available_repos} repos available")
            return True'''
    
    content = content.replace(old_scale_logic, new_scale_logic)
    print("✅ Updated scaling logic to check repository availability")
    
    # Also update the wait time scaling logic
    old_wait_logic = '''        # Scale up if tasks are waiting too long
        if oldest_wait_time > 300:  # 5 minutes
            self.logger.info(f"Scaling up due to: long wait times ({oldest_wait_time:.0f}s)")
            return True'''
    
    new_wait_logic = '''        # Scale up if tasks are waiting too long AND repositories are available
        if oldest_wait_time > 300 and available_repos > 0:  # 5 minutes
            self.logger.info(f"Scaling up due to: long wait times ({oldest_wait_time:.0f}s) and {available_repos} repos available")
            return True'''
    
    content = content.replace(old_wait_logic, new_wait_logic)
    print("✅ Updated wait time scaling to check repository availability")
    
    # Import local simulator if not already imported
    if 'from scripts.local_github_simulator import LocalGitHubSimulator' not in content:
        # Add import at the top with other imports
        import_pattern = r'(from scripts\..*? import .*?\n)'
        import_match = re.findall(import_pattern, content)
        if import_match:
            last_import_pos = content.rfind(import_match[-1]) + len(import_match[-1])
            new_import = 'from scripts.local_github_simulator import LocalGitHubSimulator\n'
            content = content[:last_import_pos] + new_import + content[last_import_pos:]
            print("✅ Added LocalGitHubSimulator import")
    
    # Initialize local_simulator in __init__ if not present
    if 'self.local_simulator' not in content:
        init_pattern = r'(def __init__\(self.*?\):.*?)(self\.running = False)'
        init_match = re.search(init_pattern, content, re.DOTALL)
        if init_match:
            before_running = init_match.start(2)
            simulator_init = '        self.local_simulator = LocalGitHubSimulator()\n        '
            content = content[:before_running] + simulator_init + content[before_running:]
            print("✅ Added local_simulator initialization")
    
    # Save the file
    with open('scripts/dynamic_claude_executor_pool.py', 'w') as f:
        f.write(content)
    
    print("\n✅ Successfully fixed Claude executor scaling!")
    print("\nThe executor pool will now:")
    print("1. Check how many repositories don't have active tasks")
    print("2. Only scale up if there are available repositories")
    print("3. Prevent creating executors that can't find work")

if __name__ == "__main__":
    fix_executor_scaling()