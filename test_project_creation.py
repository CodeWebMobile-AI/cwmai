#!/usr/bin/env python3
"""Test project creation to see error messages"""

import asyncio
import os
import sys

# Add the zip folder to Python path
sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai/zip')

from scripts.project_creator import ProjectCreator
from scripts.ai_brain_factory import AIBrainFactory

async def test_project_creation():
    """Test creating a project in local mode"""
    
    # Create AI brain
    ai_brain = AIBrainFactory.create_brain()
    
    # Create project creator in local mode
    creator = ProjectCreator(github_token=None, ai_brain=ai_brain)
    
    # Create a test task
    task = {
        'title': 'Test Project',
        'description': 'A test project to check error handling',
        'metadata': {
            'project_type': 'web',
            'starter_kit': 'laravel-react'
        }
    }
    
    print("Creating project...")
    try:
        result = await creator.create_project(task)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    # Set environment variables
    os.environ['STORAGE_MODE'] = 'local'
    
    # Run the test
    asyncio.run(test_project_creation())