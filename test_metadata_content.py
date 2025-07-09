#!/usr/bin/env python3
"""
Test script to verify that metadata.json includes full documentation content
for new projects created in local mode.
"""

import json
import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from scripts.project_creator import ProjectCreator
from scripts.ai_brain import AIBrain

async def test_metadata_content():
    """Test that metadata.json includes full documentation content."""
    
    print("Testing metadata.json content inclusion...")
    
    # Initialize components
    ai_brain = AIBrain()
    creator = ProjectCreator(ai_brain=ai_brain, storage_mode='local')
    
    # Create a test task
    test_task = {
        'type': 'NEW_PROJECT',
        'title': 'Test Project for Metadata Content',
        'metadata': {
            'market_opportunity': {
                'problem_description': 'Test problem for metadata verification',
                'affected_audience': 'Test users',
                'opportunity': 'Test opportunity'
            }
        }
    }
    
    try:
        # Create the project
        print("\nCreating test project...")
        result = await creator.create_project(test_task)
        
        if result.get('success'):
            print(f"✓ Project created successfully: {result['project_name']}")
            
            # Check the metadata.json file
            repo_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI") / result['project_name']
            metadata_path = repo_path / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print("\n✓ metadata.json exists")
                
                # Check for documentation content fields
                has_readme = 'readme_content' in metadata and metadata['readme_content']
                has_architecture = 'architecture_content' in metadata and metadata['architecture_content']
                has_monetization = 'monetization_content' in metadata and metadata['monetization_content']
                
                print(f"\nDocumentation content in metadata.json:")
                print(f"  - README content: {'✓ Present' if has_readme else '✗ Missing'}")
                print(f"  - ARCHITECTURE content: {'✓ Present' if has_architecture else '✗ Missing'}")
                print(f"  - MONETIZATION content: {'✓ Present' if has_monetization else '✗ Missing'}")
                
                if has_readme:
                    print(f"  - README content length: {len(metadata['readme_content'])} characters")
                if has_architecture:
                    print(f"  - ARCHITECTURE content length: {len(metadata['architecture_content'])} characters")
                if has_monetization:
                    print(f"  - MONETIZATION content length: {len(metadata['monetization_content'])} characters")
                
                # Verify content matches actual files
                if has_readme:
                    readme_path = repo_path / "README.md"
                    if readme_path.exists():
                        with open(readme_path, 'r') as f:
                            actual_readme = f.read()
                        matches = metadata['readme_content'] == actual_readme
                        print(f"\n  - README content matches file: {'✓ Yes' if matches else '✗ No'}")
                
                # Show a sample of the content
                if has_readme:
                    print(f"\nSample of README content in metadata:")
                    print(metadata['readme_content'][:200] + "...")
                
                # Success summary
                if has_readme and has_architecture and has_monetization:
                    print("\n✅ SUCCESS: All documentation content is included in metadata.json!")
                else:
                    print("\n❌ FAILED: Some documentation content is missing from metadata.json")
                    
            else:
                print(f"✗ metadata.json not found at {metadata_path}")
                
        else:
            print(f"✗ Project creation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_metadata_content())