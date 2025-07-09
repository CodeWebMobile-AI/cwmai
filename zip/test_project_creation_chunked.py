#!/usr/bin/env python3
"""Test project creation with chunked generation."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import IntelligentAIBrain
from scripts.state_manager import StateManager

async def test_project_creation():
    """Test creating a project with chunked generation."""
    print("Testing project creation with chunked generation...")
    
    # Create AI brain
    state_manager = StateManager()
    state = state_manager.load_state()
    ai_brain = IntelligentAIBrain(state, {})
    
    # Create project creator with AI brain
    creator = ProjectCreator(None, ai_brain)  # None for local mode
    
    # Test task
    test_task = {
        'id': 'test_chunked_001',
        'task_type': 'NEW_PROJECT',
        'title': 'AI Recipe Optimizer',
        'description': 'An AI-powered recipe optimization platform that adapts recipes based on dietary restrictions, available ingredients, and nutritional goals. Uses machine learning to suggest ingredient substitutions and cooking modifications.',
        'metadata': {
            'project_type': 'AI-Powered Recipe Platform',
            'starter_kit': 'laravel/react-starter-kit',
            'detailed_description': 'Many home cooks struggle to adapt recipes for dietary restrictions or missing ingredients. This platform uses AI to intelligently modify recipes while maintaining taste and nutritional balance.'
        }
    }
    
    try:
        print("\nCreating project with chunked generation...")
        result = await creator.create_project(test_task)
        
        if result.get('success'):
            print(f"\n✅ Project created successfully!")
            print(f"Repository: {result.get('repository', 'Unknown')}")
            print(f"Documentation created: {result.get('documentation_created', {})}")
            
            # Check if chunked generation was used
            if result.get('documentation_created', {}).get('architecture'):
                print("\n✓ Architecture document created using chunked generation")
            if result.get('documentation_created', {}).get('monetization'):
                print("✓ Monetization strategy created using chunked generation")
            if result.get('documentation_created', {}).get('readme'):
                print("✓ README created")
                
            # Check metadata
            repo_name = result.get('repository', '').split('/')[-1]
            metadata_path = Path(f"local_repos/repositories/CodeWebMobile-AI/{repo_name}/metadata.json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                print(f"\n✓ Metadata saved with {len(metadata.get('documentation', {}))} documentation sections")
        else:
            print(f"\n❌ Project creation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Set up environment
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    
    # Run test
    success = asyncio.run(test_project_creation())
    sys.exit(0 if success else 1)