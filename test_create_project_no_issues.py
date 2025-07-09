#!/usr/bin/env python3
"""
Create a test project with zero issues to test Initial Project Setup
"""
import os
import sys
import json

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

from scripts.state_manager import StateManager
from scripts.local_github_simulator import LocalGitHubSimulator

def create_test_project():
    print("Creating test project with zero issues...")
    
    # Initialize components
    state_manager = StateManager()
    simulator = LocalGitHubSimulator()
    
    # Create test project
    test_project = {
        'name': 'test-zero-issues',
        'full_name': 'fortressarigo/test-zero-issues',
        'description': 'Test project to verify Initial Project Setup creation',
        'owner': 'fortressarigo',
        'private': False,
        'created_at': '2025-06-23T00:00:00Z',
        'updated_at': '2025-06-23T00:00:00Z',
        'pushed_at': '2025-06-23T00:00:00Z',
        'language': 'JavaScript',
        'size': 100,
        'stargazers_count': 0,
        'watchers_count': 0,
        'forks_count': 0,
        'has_issues': True,
        'has_projects': True,
        'has_downloads': True,
        'has_wiki': True,
        'has_pages': False,
        'archived': False,
        'disabled': False,
        'open_issues_count': 0,
        'license': None,
        'forks': 0,
        'open_issues': 0,
        'watchers': 0,
        'default_branch': 'main',
        'lifecycle_stage': 'inception'
    }
    
    # Add to simulator storage
    simulator.repositories.append(test_project)
    
    # Create repository data file with no issues
    repo_file = os.path.join(simulator.storage_dir, 'fortressarigo_test-zero-issues.json')
    repo_data = {
        'repository': test_project,
        'issues': []  # No issues
    }
    
    with open(repo_file, 'w') as f:
        json.dump(repo_data, f, indent=2)
    
    print(f"✅ Created test project: {test_project['full_name']}")
    print(f"   - Lifecycle stage: {test_project['lifecycle_stage']}")
    print(f"   - Issues: 0")
    print(f"   - Storage file: {repo_file}")
    
    # Update system state
    state = state_manager.load_state()
    if 'repositories' not in state:
        state['repositories'] = {}
    
    state['repositories'][test_project['full_name']] = test_project
    state_manager.state = state
    state_manager.save_state()
    
    print("\n✅ Updated system state with test project")

if __name__ == "__main__":
    create_test_project()