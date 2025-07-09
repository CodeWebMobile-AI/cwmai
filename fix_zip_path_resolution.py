#!/usr/bin/env python3
"""
Fix path resolution in all Python files to ensure they work correctly
when run from within the zip folder.
"""

import os
import re
from pathlib import Path


def fix_state_manager():
    """Fix StateManager to use script-relative paths instead of cwd."""
    state_manager_path = Path("scripts/state_manager.py")
    
    if not state_manager_path.exists():
        print(f"Warning: {state_manager_path} not found")
        return
        
    content = state_manager_path.read_text()
    
    # Fix the discover_local_repositories method
    old_pattern = r"local_repos_path = os\.path\.join\(os\.getcwd\(\), 'local_repos', 'repositories'\)"
    new_code = """# Use script location to find local_repos, not current working directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_repos_path = os.path.join(script_dir, 'local_repos', 'repositories')"""
    
    if old_pattern in content:
        content = re.sub(old_pattern, new_code, content)
        print("✓ Fixed path resolution in StateManager.discover_local_repositories")
    
    # Fix the state file path to be relative to script location
    old_init = r'def __init__\(self, local_path: str = "system_state\.json"'
    if re.search(old_init, content):
        # Insert code to make paths relative to script directory
        fix = '''def __init__(self, local_path: str = "system_state.json", repo_path: str = "system_state.json"):
        """Initialize the StateManager.
        
        Args:
            local_path: Local file path for state storage
            repo_path: Repository path for remote state storage
        """
        # Make paths relative to script directory, not working directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(local_path):
            local_path = os.path.join(script_dir, local_path)
        self.local_path = local_path'''
        
        # Replace the init method start
        pattern = r'def __init__\(self,[^}]+?\n\s+self\.local_path = local_path'
        content = re.sub(pattern, fix, content, flags=re.DOTALL)
        print("✓ Fixed state file path resolution in StateManager.__init__")
    
    state_manager_path.write_text(content)
    print(f"✓ Updated {state_manager_path}")


def fix_local_github_simulator():
    """Fix LocalGitHubSimulator to use script-relative paths."""
    simulator_path = Path("scripts/local_github_simulator.py")
    
    if not simulator_path.exists():
        print(f"Warning: {simulator_path} not found")
        return
        
    content = simulator_path.read_text()
    
    # Check if already using Path.cwd()
    if "Path.cwd()" in content:
        # Replace with script-relative path
        old_pattern = r"self\.base_path = Path\.cwd\(\) / \"local_repos\""
        new_code = """# Use script location instead of working directory
            script_dir = Path(__file__).parent.parent
            self.base_path = script_dir / "local_repos\""""
        
        content = re.sub(old_pattern, new_code, content)
        print("✓ Fixed path resolution in LocalGitHubSimulator")
        
        simulator_path.write_text(content)
        print(f"✓ Updated {simulator_path}")


def fix_repository_tools():
    """Fix all repository analysis tools to use script-relative paths."""
    tools_dir = Path("scripts/custom_tools/repository")
    
    if not tools_dir.exists():
        print(f"Warning: {tools_dir} not found")
        return
    
    fixed_count = 0
    for tool_path in tools_dir.glob("*.py"):
        content = tool_path.read_text()
        
        if "Path.cwd()" in content:
            # Replace Path.cwd() with script-relative path
            old_pattern = r"Path\.cwd\(\)(\s*/\s*[\"']local_repos[\"'])"
            new_code = r'Path(__file__).parent.parent.parent.parent / "local_repos"'
            
            new_content = re.sub(old_pattern, new_code, content)
            
            if new_content != content:
                tool_path.write_text(new_content)
                fixed_count += 1
                print(f"✓ Fixed {tool_path.name}")
    
    if fixed_count > 0:
        print(f"✓ Updated {fixed_count} repository tools")


def create_zip_specific_config():
    """Create a configuration file for zip-specific settings."""
    config_path = Path("zip_config.py")
    
    config_content = '''"""
Zip-specific configuration to ensure proper path resolution.
"""

import os
from pathlib import Path

# Get the zip directory (where this config file is located)
ZIP_DIR = Path(__file__).parent.absolute()

# Override commonly used paths to be relative to zip directory
LOCAL_REPOS_PATH = ZIP_DIR / "local_repos"
REPOSITORIES_PATH = LOCAL_REPOS_PATH / "repositories"
SYSTEM_STATE_PATH = ZIP_DIR / "system_state.json"
TASK_STATE_PATH = ZIP_DIR / "task_state.json"

# Ensure directories exist
LOCAL_REPOS_PATH.mkdir(exist_ok=True)
REPOSITORIES_PATH.mkdir(parents=True, exist_ok=True)

# Function to get absolute path relative to zip
def get_zip_path(relative_path):
    """Get absolute path relative to zip directory."""
    return ZIP_DIR / relative_path

# Export for other modules
__all__ = [
    'ZIP_DIR',
    'LOCAL_REPOS_PATH', 
    'REPOSITORIES_PATH',
    'SYSTEM_STATE_PATH',
    'TASK_STATE_PATH',
    'get_zip_path'
]
'''
    
    config_path.write_text(config_content)
    print(f"✓ Created {config_path}")


def main():
    print("Fixing path resolution for zip folder execution...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("scripts").exists():
        print("Error: Must run this script from the cwmai directory")
        return
    
    # Apply fixes
    fix_state_manager()
    fix_local_github_simulator()
    fix_repository_tools()
    create_zip_specific_config()
    
    print("\n" + "=" * 50)
    print("✓ Path resolution fixes applied!")
    print("\nTo run the system from the zip folder, use:")
    print("  ./run_from_zip.sh run_hybrid_continuous.py --workers 10")
    print("\nOr if you're already in the zip folder:")
    print("  python3 run_hybrid_continuous.py --workers 10")


if __name__ == "__main__":
    main()