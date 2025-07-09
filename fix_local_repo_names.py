#!/usr/bin/env python3
"""
Fix incorrect repository names with 'local/' prefix in system_state.json
They should use 'CodeWebMobile-AI/' prefix instead
"""

import json

# Load system state
with open('system_state.json', 'r') as f:
    state = json.load(f)

# Fix repository names
fixed_count = 0
for project_id, project in state.get('projects', {}).items():
    if project.get('full_name', '').startswith('local/'):
        old_name = project['full_name']
        # Extract repo name after 'local/'
        repo_name = old_name.replace('local/', '')
        new_name = f'CodeWebMobile-AI/{repo_name}'
        project['full_name'] = new_name
        fixed_count += 1
        print(f"Fixed: {old_name} -> {new_name}")

# Save updated state
with open('system_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print(f"\n✅ Fixed {fixed_count} repository names")