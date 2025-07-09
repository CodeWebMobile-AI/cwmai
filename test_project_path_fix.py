#!/usr/bin/env python3
"""
Test script to verify the project path fix.
Tests that projects are created in the correct directory without overwriting each other.
"""

import asyncio
import os
import sys
from pathlib import Path
import shutil
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import AIBrain
from scripts.storage_mode import StorageMode

async def test_project_path_fix():
    """Test that projects are created in correct paths without overwriting."""
    # Set local mode
    os.environ['STORAGE_MODE'] = StorageMode.LOCAL
    
    # Initialize AI brain
    ai_brain = AIBrain()
    
    # Initialize project creator
    creator = ProjectCreator(ai_brain=ai_brain, storage_mode=StorageMode.LOCAL)
    
    # Clear any existing test projects
    test_projects = ['test-path-fix-1', 'test-path-fix-2', 'test-path-fix-3']
    repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    print("Cleaning up any existing test projects...")
    for project in test_projects:
        project_path = repos_path / project
        if project_path.exists():
            shutil.rmtree(project_path)
            print(f"Removed existing project: {project}")
    
    # Also clean from state file
    state_file = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state/repositories.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Filter out test projects
        state['repositories'] = [
            repo for repo in state['repositories'] 
            if repo['name'] not in test_projects
        ]
        state['total_count'] = len(state['repositories'])
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    print("\nStarting project creation test...")
    print(f"Repository path: {repos_path}")
    
    # Create test projects
    created_projects = []
    
    for i, project_name in enumerate(test_projects):
        print(f"\n{'='*60}")
        print(f"Creating project {i+1}: {project_name}")
        
        task = {
            'title': f'Test Project Path Fix {i+1}',
            'type': 'NEW_PROJECT',
            'metadata': {}
        }
        
        # Force specific project name for testing
        creator._generate_project_details = async_mock_project_details(project_name, i+1)
        
        try:
            result = await creator.create_project(task)
            if result.get('success'):
                created_projects.append(project_name)
                print(f"✅ Successfully created: {project_name}")
                
                # Verify the project exists at the correct path
                expected_path = repos_path / project_name
                if expected_path.exists():
                    print(f"✅ Project exists at correct path: {expected_path}")
                else:
                    print(f"❌ Project NOT found at expected path: {expected_path}")
                    
                    # Check if it was created elsewhere
                    for p in repos_path.parent.rglob(project_name):
                        print(f"   Found at: {p}")
            else:
                print(f"❌ Failed to create project: {result.get('error')}")
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            import traceback
            traceback.print_exc()
        
        # Check current state after each creation
        if repos_path.exists():
            current_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
            print(f"Current projects in directory: {current_projects}")
            print(f"Total projects: {len(current_projects)}")
        
        # Small delay between creations
        if i < len(test_projects) - 1:
            await asyncio.sleep(1)
    
    # Final verification
    print(f"\n{'='*60}")
    print("Final verification:")
    
    if repos_path.exists():
        final_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
        print(f"Final projects in directory: {final_projects}")
        
        # Check each test project
        all_exist = True
        for project in test_projects:
            project_path = repos_path / project
            if project_path.exists():
                print(f"✅ {project} exists")
                # Check if it has the starter template files
                if (project_path / "package.json").exists():
                    print(f"   ✅ Has package.json")
                if (project_path / "composer.json").exists():
                    print(f"   ✅ Has composer.json")
            else:
                print(f"❌ {project} is missing!")
                all_exist = False
        
        if all_exist:
            print("\n✅ SUCCESS: All test projects were created without overwriting!")
        else:
            print("\n❌ FAILURE: Some projects are missing!")
    else:
        print("❌ Repository path doesn't exist!")

def async_mock_project_details(name, idx):
    """Create a mock function that returns specific project details."""
    async def mock_fn(task):
        return {
            'name': name,
            'display_name': f'Test Path Fix {idx}',
            'description': f'Test project {idx} to verify path fix',
            'initial_features': ['Basic feature'],
            'project_type': 'Web Application'
        }
    return mock_fn

if __name__ == "__main__":
    asyncio.run(test_project_path_fix())