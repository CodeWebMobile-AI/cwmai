#!/usr/bin/env python3
"""
Fix the state caching issue in the StateManager to ensure it loads fresh data.
"""

import os
import sys

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))


def fix_state_manager():
    """Add a method to force reload state from disk."""
    state_manager_path = os.path.join('scripts', 'state_manager.py')
    
    print("Fixing StateManager caching issue...")
    
    # Read the current file
    with open(state_manager_path, 'r') as f:
        content = f.read()
    
    # Check if the force_reload_state method already exists
    if 'def force_reload_state' in content:
        print("✓ force_reload_state method already exists")
        return
    
    # Find where to insert the new method (after the load_state method)
    insert_position = content.find('    def save_state_locally(')
    
    if insert_position == -1:
        print("❌ Could not find insertion point")
        return
    
    # Create the new method
    new_method = '''
    def force_reload_state(self) -> Dict[str, Any]:
        """Force reload state from disk, bypassing cache."""
        self.state = None  # Clear cached state
        return self.load_state()
    '''
    
    # Insert the new method
    content = content[:insert_position] + new_method + '\n' + content[insert_position:]
    
    # Write the updated file
    with open(state_manager_path, 'w') as f:
        f.write(content)
    
    print("✓ Added force_reload_state method to StateManager")
    
    # Now fix the tool_calling_system to use force_reload_state
    tool_calling_path = os.path.join('scripts', 'tool_calling_system.py')
    
    with open(tool_calling_path, 'r') as f:
        tool_content = f.read()
    
    # Replace load_state with force_reload_state in the _count_repositories method
    old_line = '        self.state_manager.load_state()'
    new_line = '        # Force reload to get fresh state\n        if hasattr(self.state_manager, "force_reload_state"):\n            self.state_manager.force_reload_state()\n        else:\n            self.state_manager.load_state()'
    
    tool_content = tool_content.replace(old_line, new_line)
    
    with open(tool_calling_path, 'w') as f:
        f.write(tool_content)
    
    print("✓ Updated tool_calling_system to use force_reload_state")
    print("\n✅ State caching issue fixed!")


if __name__ == "__main__":
    fix_state_manager()