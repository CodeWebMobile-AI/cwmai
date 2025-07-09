#!/usr/bin/env python3
"""
Test script to verify project creation fix.
Tests that multiple projects can be created without overwriting each other.
Also tests the specific fixes for:
1. "local variable 'creation_record' referenced before assignment" error
2. BackgroundTaskPreprocessor import error
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import AIBrain
from scripts.storage_mode import StorageMode

async def test_imports():
    """Test that all imports work correctly after fixes."""
    print("Testing imports...")
    
    try:
        # Test BackgroundTaskPreprocessor import
        from scripts.background_task_preprocessor import BackgroundTaskPreprocessor
        print("✅ BackgroundTaskPreprocessor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import BackgroundTaskPreprocessor: {e}")
        return False
    
    return True

async def test_multiple_project_creation():
    """Test creating multiple projects in sequence."""
    # First test imports
    if not await test_imports():
        print("Import tests failed, aborting...")
        return
    
    # Set local mode
    os.environ['STORAGE_MODE'] = StorageMode.LOCAL
    
    # Initialize AI brain
    ai_brain = AIBrain()
    
    # Initialize project creator
    creator = ProjectCreator(ai_brain=ai_brain, storage_mode=StorageMode.LOCAL)
    
    # Test projects to create
    test_projects = [
        {
            'title': 'Test Project 1',
            'type': 'NEW_PROJECT',
            'metadata': {}
        },
        {
            'title': 'Test Project 2', 
            'type': 'NEW_PROJECT',
            'metadata': {}
        },
        {
            'title': 'Test Project 3',
            'type': 'NEW_PROJECT',
            'metadata': {}
        }
    ]
    
    created_projects = []
    repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    print("Starting project creation test...")
    print(f"Repository path: {repos_path}")
    
    # Check initial state
    if repos_path.exists():
        initial_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
        print(f"Initial projects: {initial_projects}")
    else:
        initial_projects = []
        print("No initial projects")
    
    # Create projects
    for i, task in enumerate(test_projects):
        print(f"\n{'='*60}")
        print(f"Creating project {i+1} of {len(test_projects)}")
        
        try:
            result = await creator.create_project(task)
            if result.get('success'):
                created_projects.append(result['repo_name'])
                print(f"✅ Successfully created: {result['repo_name']}")
            else:
                print(f"❌ Failed to create project: {result.get('error')}")
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            import traceback
            traceback.print_exc()
        
        # Check current state
        if repos_path.exists():
            current_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
            print(f"Current projects: {current_projects}")
            print(f"Total projects: {len(current_projects)}")
        
        # Add delay between creations
        if i < len(test_projects) - 1:
            print("Waiting before next project...")
            await asyncio.sleep(2)
    
    # Final verification
    print(f"\n{'='*60}")
    print("Final verification:")
    
    if repos_path.exists():
        final_projects = [p.name for p in repos_path.iterdir() if p.is_dir()]
        print(f"Final projects: {final_projects}")
        print(f"Total projects: {len(final_projects)}")
        
        # Check if all created projects exist
        for project in created_projects:
            project_path = repos_path / project
            if project_path.exists():
                print(f"✅ {project} exists")
            else:
                print(f"❌ {project} is missing!")
        
        # Success criteria
        if len(final_projects) >= len(initial_projects) + len(created_projects):
            print("\n✅ SUCCESS: All projects were created and preserved!")
        else:
            print(f"\n❌ FAILURE: Expected at least {len(initial_projects) + len(created_projects)} projects, found {len(final_projects)}")
    else:
        print("❌ Repository path doesn't exist!")

if __name__ == "__main__":
    asyncio.run(test_multiple_project_creation())