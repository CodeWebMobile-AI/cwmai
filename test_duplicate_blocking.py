#!/usr/bin/env python3
"""
Test Duplicate Project Blocking

Simple test to verify that duplicate projects are actually blocked.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import shutil

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import IntelligentAIBrain
from scripts.storage_mode import StorageMode


async def test_duplicate_blocking():
    """Test that duplicate projects are blocked."""
    print("Testing Duplicate Project Blocking")
    print("=" * 60)
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    # Set up local mode
    os.environ['STORAGE_MODE'] = StorageMode.LOCAL
    
    # Use the actual local repos path
    repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    # Check what projects already exist
    print("\nExisting projects:")
    if repos_path.exists():
        for project_dir in sorted(repos_path.iterdir()):
            if project_dir.is_dir():
                print(f"  - {project_dir.name}")
    
    # Initialize project creator
    project_creator = ProjectCreator(ai_brain=ai_brain, storage_mode=StorageMode.LOCAL)
    
    # Test cases - try to create duplicates of existing projects
    test_cases = [
        {
            "name": "sub-wise",
            "task": {
                "id": "test-sub-wise",
                "task_type": "NEW_PROJECT",
                "metadata": {
                    "name": "sub-wise",
                    "description": "Duplicate subscription tracker",
                    "problem_statement": "Track subscriptions",
                    "target_market": "Consumers",
                    "initial_features": ["subscription tracking"]
                }
            }
        },
        {
            "name": "subscrip-ai",
            "task": {
                "id": "test-subscrip-ai",
                "task_type": "NEW_PROJECT",
                "metadata": {
                    "name": "subscrip-ai",
                    "description": "Another subscription app",
                    "problem_statement": "Manage subscriptions",
                    "target_market": "Consumers",
                    "initial_features": ["subscription management"]
                }
            }
        }
    ]
    
    print("\nTesting duplicate creation attempts:")
    print("-" * 60)
    
    results = []
    
    for test in test_cases:
        print(f"\nAttempting to create: {test['name']}")
        
        try:
            result = await project_creator.create_project(test['task'])
            
            success = result.get('success', False)
            message = result.get('message', 'No message')
            
            print(f"  Result: success={success}")
            print(f"  Message: {message}")
            
            if not success and ('duplicate' in message.lower() or 'already exists' in message.lower()):
                print("  ✅ Duplicate correctly blocked!")
                results.append(True)
            else:
                print("  ❌ Duplicate was NOT blocked!")
                results.append(False)
                
        except Exception as e:
            # If it fails because the project exists, that's what we want
            if 'already exists' in str(e):
                print(f"  ✅ Exception correctly thrown: {e}")
                results.append(True)
            else:
                print(f"  ❌ Unexpected error: {e}")
                results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All duplicate blocking tests PASSED!")
        print("The system IS preventing duplicate project creation.")
    else:
        print("\n❌ Some tests FAILED!")
        print("The system may not be properly blocking duplicates.")
    
    return passed == total


async def main():
    """Run the test."""
    try:
        success = await test_duplicate_blocking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    asyncio.run(main())