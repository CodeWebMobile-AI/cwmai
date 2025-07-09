#!/usr/bin/env python3
"""
Simple test to verify project creation fix without complex AI generation.
"""

import os
import sys
from pathlib import Path
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.storage_mode import StorageMode

def test_simple_creation():
    """Test creating multiple simple projects."""
    # Set local mode
    os.environ['STORAGE_MODE'] = StorageMode.LOCAL
    
    # Initialize simulator
    simulator = LocalGitHubSimulator()
    
    repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    print("Starting simple project creation test...")
    print(f"Repository path: {repos_path}")
    
    # Check initial state
    if repos_path.exists():
        initial_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
        print(f"Initial projects: {initial_projects}")
        print(f"Initial count: {len(initial_projects)}")
    else:
        initial_projects = []
        print("No initial projects")
    
    # Test projects
    test_projects = [
        ("test-project-1", "Test Project 1 Description"),
        ("test-project-2", "Test Project 2 Description"),
        ("test-project-3", "Test Project 3 Description")
    ]
    
    created = []
    
    # Create projects
    for name, desc in test_projects:
        print(f"\n{'='*60}")
        print(f"Creating project: {name}")
        
        try:
            repo = simulator.create_repository(name=name, description=desc)
            created.append(name)
            print(f"✅ Successfully created: {name}")
            
            # Check current state
            if repos_path.exists():
                current_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
                print(f"Current projects: {current_projects}")
                print(f"Total projects: {len(current_projects)}")
        except Exception as e:
            print(f"❌ Error creating {name}: {e}")
    
    # Final verification
    print(f"\n{'='*60}")
    print("Final verification:")
    
    if repos_path.exists():
        final_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
        print(f"Final projects: {final_projects}")
        print(f"Total projects: {len(final_projects)}")
        
        # Check if all created projects exist
        for project in created:
            project_path = repos_path / project
            if project_path.exists():
                print(f"✅ {project} exists")
            else:
                print(f"❌ {project} is missing!")
        
        # Success criteria
        expected_count = len(initial_projects) + len(created)
        if len(final_projects) == expected_count:
            print(f"\n✅ SUCCESS: All projects were created and preserved!")
            print(f"   Expected {expected_count} projects, found {len(final_projects)}")
        else:
            print(f"\n❌ FAILURE: Expected {expected_count} projects, found {len(final_projects)}")
            
            # Check state file
            state_file = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state/repositories.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                print(f"\nState file shows {len(state['repositories'])} repositories")
    else:
        print("❌ Repository path doesn't exist!")

if __name__ == "__main__":
    test_simple_creation()