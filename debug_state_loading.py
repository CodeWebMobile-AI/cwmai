#!/usr/bin/env python3
"""
Debug script to understand why the state is not loading correctly.
"""

import os
import sys
import json

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from state_manager import StateManager


def main():
    """Debug state loading."""
    print("Debugging state loading issue...")
    print("-" * 50)
    
    # Check the file directly
    state_file = "system_state.json"
    print(f"\n1. Checking {state_file} directly:")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            file_state = json.load(f)
            projects = file_state.get('projects', {})
            print(f"   File contains {len(projects)} projects")
            if projects:
                print(f"   Project names: {list(projects.keys())[:5]}...")
    else:
        print(f"   File {state_file} does not exist!")
    
    # Create StateManager and check its behavior
    print("\n2. Testing StateManager:")
    state_manager = StateManager()
    print(f"   Local path: {state_manager.local_path}")
    print(f"   State cache: {state_manager.state}")
    
    # Load state
    print("\n3. Loading state:")
    state = state_manager.load_state()
    print(f"   Loaded state has {len(state.get('projects', {}))} projects")
    print(f"   State cache after load: {state_manager.state is not None}")
    
    # Force reload
    print("\n4. Force reloading state:")
    if hasattr(state_manager, 'force_reload_state'):
        state = state_manager.force_reload_state()
        print(f"   Force reloaded state has {len(state.get('projects', {}))} projects")
    else:
        print("   force_reload_state method not available")
    
    # Check if state is being modified somewhere
    print("\n5. Checking state object identity:")
    state1 = state_manager.get_state()
    state2 = state_manager.get_state()
    print(f"   Same object? {state1 is state2}")
    print(f"   Projects in state1: {len(state1.get('projects', {}))}")
    print(f"   Projects in state2: {len(state2.get('projects', {}))}")
    
    # Try creating a new StateManager instance
    print("\n6. Creating new StateManager instance:")
    new_state_manager = StateManager()
    new_state = new_state_manager.load_state()
    print(f"   New instance loaded {len(new_state.get('projects', {}))} projects")


if __name__ == "__main__":
    main()