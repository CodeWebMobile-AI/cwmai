#!/usr/bin/env python3
"""Test that state manager uses local repository discovery when USE_LOCAL_MODE is set."""

import os
import sys

# Set the environment variable BEFORE importing modules
os.environ['USE_LOCAL_MODE'] = 'true'

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.state_manager import StateManager

def test_local_mode():
    """Test that state manager discovers local repositories when USE_LOCAL_MODE=true."""
    print("Testing state manager in local mode...")
    print(f"USE_LOCAL_MODE = {os.environ.get('USE_LOCAL_MODE')}")
    
    # Create state manager
    state_manager = StateManager()
    
    # Load state with repository discovery
    print("\nLoading state with repository discovery...")
    state = state_manager.load_state_with_repository_discovery()
    
    # Check what repositories were discovered
    projects = state.get('projects', {})
    print(f"\nDiscovered {len(projects)} repositories")
    
    # List the repositories
    if projects:
        print("\nRepositories found:")
        for proj_id, proj_data in projects.items():
            print(f"  - {proj_data.get('full_name', proj_id)}: {proj_data.get('url', 'No URL')}")
    else:
        print("\nNo repositories found.")
        
    # Check if they're local repositories
    if projects:
        first_proj = next(iter(projects.values()))
        if first_proj.get('url', '').startswith('local://'):
            print("\n✅ SUCCESS: Using local repositories!")
        else:
            print("\n❌ ERROR: Still using GitHub repositories!")
            print(f"   URL: {first_proj.get('url')}")

if __name__ == "__main__":
    test_local_mode()