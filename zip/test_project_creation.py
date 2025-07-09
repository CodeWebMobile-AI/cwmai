#!/usr/bin/env python3
"""Test project creation to see if templates are copied"""

import asyncio
import os
import sys

# Add the zip folder to Python path
sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai/zip')

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import IntelligentAIBrain
from scripts.local_github_simulator import LocalGitHubSimulator

async def test_project_creation():
    """Test creating a project in local mode"""
    
    # Set environment variables
    os.environ['STORAGE_MODE'] = 'local'
    
    # Create AI brain
    ai_brain = IntelligentAIBrain()
    
    # Create project creator in local mode
    creator = ProjectCreator(github_token=None, ai_brain=ai_brain)
    
    # Create a test task
    task = {
        'title': 'Test Template Project',
        'description': 'A test project to verify template copying works',
        'metadata': {
            'project_type': 'web',
            'starter_kit': 'laravel-react'
        }
    }
    
    print("Creating project...")
    try:
        result = await creator.create_project(task)
        print(f"Result: {result}")
        
        if result.get('success'):
            print(f"✅ Project created successfully!")
            
            # Check if template files exist
            repo_name = result.get('repository_name')
            if repo_name:
                repo_path = f"/Users/fortressarigo/PhpstormProjects/cwmai/zip/local_repos/repositories/CodeWebMobile-AI/{repo_name}"
                
                print(f"\nChecking template files in: {repo_path}")
                
                essential_files = ['package.json', 'composer.json', 'vite.config.ts', 'artisan']
                for file in essential_files:
                    file_path = os.path.join(repo_path, file)
                    if os.path.exists(file_path):
                        print(f"✅ {file} exists")
                    else:
                        print(f"❌ {file} missing")
            else:
                print("❌ No repository name in result")
        else:
            print(f"❌ Project creation failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_project_creation())