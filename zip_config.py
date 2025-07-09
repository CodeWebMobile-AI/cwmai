"""
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
