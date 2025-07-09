#!/usr/bin/env python3
"""Fix project creator to use local templates instead of GitHub forking."""

import os
import re

# Read the current project creator
with open('scripts/project_creator.py', 'r') as f:
    content = f.read()

# Replace GitHub forking with local template copying
new_fork_method = '''async def _fork_starter_kit(self, project_name: str, tech_stack: str = None):
        """Copy from local template instead of forking from GitHub.
        
        Args:
            project_name: Name for the new project
            tech_stack: Technology stack to use (ignored - always uses Laravel React)
            
        Returns:
            Dictionary with local repo info or None
        """
        try:
            import shutil
            
            # Always use Laravel React starter kit from local templates
            template_path = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/templates/react-starter-kit"
            
            # Check if template exists
            if not os.path.exists(template_path):
                self.logger.error(f"Template not found at {template_path}")
                return None
                
            # Create destination path
            dest_path = f"/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/{project_name}"
            
            # Copy template to new location
            self.logger.info(f"Copying template from {template_path} to {dest_path}")
            shutil.copytree(template_path, dest_path)
            
            # Create a mock repo object for compatibility
            class LocalRepo:
                def __init__(self, name, path):
                    self.name = name
                    self.full_name = f"local/{name}"
                    self.html_url = f"file://{path}"
                    self.clone_url = path
                    self.path = path
                    self.owner = type('obj', (object,), {'login': 'local'})
                    
                def get_contents(self, path):
                    """Mock get_contents for local files."""
                    full_path = os.path.join(self.path, path)
                    if os.path.exists(full_path):
                        class Content:
                            def __init__(self, path):
                                self.path = path
                                with open(path, 'rb') as f:
                                    self.content = f.read()
                                    self.decoded_content = self.content
                        return Content(full_path)
                    return None
                    
                def create_file(self, path, message, content, branch="main"):
                    """Mock create_file for local files."""
                    full_path = os.path.join(self.path, path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w') as f:
                        f.write(content)
                    return {"content": {"path": path}}
                    
                def update_file(self, path, message, content, sha, branch="main"):
                    """Mock update_file for local files."""
                    return self.create_file(path, message, content, branch)
            
            return LocalRepo(project_name, dest_path)
            
        except Exception as e:
            self.logger.error(f"Failed to copy template: {e}")
            return None'''

# Find the _fork_starter_kit method and replace it
pattern = r'async def _fork_starter_kit\(self.*?\n(?:.*?\n)*?return None'
match = re.search(pattern, content, re.DOTALL)

if match:
    # Replace the method
    content = content[:match.start()] + new_fork_method + content[match.end():]
    
    # Also update the __init__ to not require github_token
    init_pattern = r'def __init__\(self, github_token: str, ai_brain=None\):'
    init_replacement = 'def __init__(self, github_token: str = None, ai_brain=None):'
    content = re.sub(init_pattern, init_replacement, content)
    
    # Update Github initialization to be optional
    github_init_pattern = r'self\.github = Github\(github_token\)'
    github_init_replacement = 'self.github = Github(github_token) if github_token else None'
    content = re.sub(github_init_pattern, github_init_replacement, content)
    
    # Update task manager initialization
    task_mgr_pattern = r'self\.task_manager = TaskManager\(github_token\)'
    task_mgr_replacement = 'self.task_manager = TaskManager(github_token) if github_token else None'
    content = re.sub(task_mgr_pattern, task_mgr_replacement, content)
    
    # Save the updated file
    with open('scripts/project_creator.py', 'w') as f:
        f.write(content)
        
    print("✅ Updated project_creator.py to use local templates!")
    print("✅ GitHub token is now optional")
    print("✅ Projects will be created in local_repos/ directory")
else:
    print("❌ Could not find _fork_starter_kit method to replace")