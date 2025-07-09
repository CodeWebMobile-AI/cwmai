#!/usr/bin/env python3
"""
Fix missing full_name fields in system_state.json projects.

This script adds full_name fields to projects that are missing them,
using the project ID as the repository name (assuming local projects).
"""

import json
import os
from datetime import datetime

def fix_missing_full_names():
    """Fix projects with missing full_name fields."""
    
    # Load system state
    state_file = 'system_state.json'
    if not os.path.exists(state_file):
        print(f"Error: {state_file} not found")
        return
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    projects = state.get('projects', {})
    fixed_count = 0
    
    print(f"Checking {len(projects)} projects...")
    
    for project_id, project_data in projects.items():
        if 'full_name' not in project_data or not project_data.get('full_name'):
            # For local projects, use CodeWebMobile-AI organization prefix
            # This matches the actual local repository naming convention
            full_name = f"CodeWebMobile-AI/{project_id}"
            
            print(f"\nFixing project: {project_id}")
            print(f"  Adding full_name: {full_name}")
            
            # Add missing fields
            project_data['full_name'] = full_name
            project_data['id'] = project_id
            
            # Add other fields if missing
            if 'url' not in project_data:
                project_data['url'] = f"local://repos/{project_id}"
            if 'clone_url' not in project_data:
                project_data['clone_url'] = f"local://repos/{project_id}.git"
            if 'language' not in project_data:
                project_data['language'] = "Unknown"
            if 'updated_at' not in project_data:
                project_data['updated_at'] = project_data.get('created_at', datetime.now().isoformat())
            if 'topics' not in project_data:
                project_data['topics'] = []
            if 'open_issues' not in project_data:
                project_data['open_issues'] = 0
            if 'health_score' not in project_data:
                project_data['health_score'] = 70
            if 'metrics' not in project_data:
                project_data['metrics'] = {
                    'stars': 0,
                    'forks': 0,
                    'issues_open': 0,
                    'size': 0
                }
            
            fixed_count += 1
    
    if fixed_count > 0:
        # Backup original file
        backup_file = f"{state_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(state_file, backup_file)
        print(f"\nBacked up original to: {backup_file}")
        
        # Save fixed state
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"\nFixed {fixed_count} projects with missing full_name fields")
        print(f"Updated {state_file}")
    else:
        print("\nNo projects with missing full_name fields found")

if __name__ == '__main__':
    fix_missing_full_names()