#!/usr/bin/env python3
"""
Real Scenario Test for Duplicate Project Prevention

Tests the actual project creation flow with duplicate detection.
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


async def test_real_duplicate_prevention():
    """Test duplicate prevention in a real scenario."""
    print("\n=== Testing Real Duplicate Project Prevention ===")
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    # Set up local mode
    os.environ['STORAGE_MODE'] = StorageMode.LOCAL
    
    # Create test directory structure
    base_path = Path("/tmp/test_duplicate_real")
    repos_path = base_path / "repositories" / "CodeWebMobile-AI"
    repos_path.mkdir(parents=True, exist_ok=True)
    
    # Create existing projects
    existing_projects = [
        {
            "name": "sub-wise",
            "description": "AI-powered subscription tracking and management app",
            "problem_domain": "People lose track of their subscriptions and overspend",
            "key_features": ["subscription tracking", "payment reminders", "spending analytics"]
        },
        {
            "name": "fitness-tracker",
            "description": "Comprehensive fitness and health tracking application",
            "problem_domain": "People want to monitor their fitness journey",
            "key_features": ["workout tracking", "progress charts", "goal setting"]
        }
    ]
    
    # Create metadata files for existing projects
    for project in existing_projects:
        project_path = repos_path / project["name"]
        project_path.mkdir(exist_ok=True)
        
        metadata = {
            "repository_name": project["name"],
            "display_name": project["name"],
            "description": project["description"],
            "problem_domain": project["problem_domain"],
            "tech_stack": "laravel-react",
            "project_type": "web-app",
            "key_features": project["key_features"],
            "created_at": "2024-01-01T00:00:00Z",
            "created_by": "test"
        }
        
        with open(project_path / "PROJECT_METADATA.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created existing project: {project['name']}")
    
    # Initialize project creator with the test path
    project_creator = ProjectCreator(ai_brain=ai_brain, storage_mode=StorageMode.LOCAL)
    # Override the local paths
    project_creator.local_repos_path = repos_path
    project_creator.duplicate_detector.local_repos_path = repos_path
    
    # Test cases
    test_cases = [
        {
            "name": "Test 1: Exact Duplicate Name",
            "task": {
                "id": "test-1",
                "task_type": "NEW_PROJECT",
                "metadata": {
                    "name": "sub-wise",  # Exact match
                    "description": "Another subscription tracker",
                    "problem_statement": "Track subscriptions",
                    "target_market": "Consumers",
                    "initial_features": ["subscription tracking"]
                }
            },
            "expected_result": "duplicate_blocked"
        },
        {
            "name": "Test 2: Similar Project",
            "task": {
                "id": "test-2",
                "task_type": "NEW_PROJECT",
                "metadata": {
                    "name": "subscrip-ai",
                    "description": "Smart subscription management with AI",
                    "problem_statement": "Help users manage their recurring payments",
                    "target_market": "Consumers",
                    "initial_features": ["subscription tracking", "AI recommendations"]
                }
            },
            "expected_result": "duplicate_blocked"
        },
        {
            "name": "Test 3: Different Domain Project",
            "task": {
                "id": "test-3",
                "task_type": "NEW_PROJECT",
                "metadata": {
                    "name": "recipe-manager",
                    "description": "Organize and share cooking recipes",
                    "problem_statement": "Home cooks need to organize recipes",
                    "target_market": "Home cooks",
                    "initial_features": ["recipe storage", "meal planning"]
                }
            },
            "expected_result": "project_created"
        }
    ]
    
    # Run test cases
    results = []
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 40)
        
        try:
            # Force reload metadata to ensure fresh check
            project_creator.duplicate_detector.load_all_metadata()
            
            result = await project_creator.create_project(test_case['task'])
            
            success = result.get('success', False)
            message = result.get('message', '')
            
            print(f"Result: success={success}")
            print(f"Message: {message}")
            
            # Check if result matches expectation
            if test_case['expected_result'] == 'duplicate_blocked':
                if not success and 'duplicate' in message.lower():
                    print("✅ Correctly blocked duplicate")
                    results.append((test_case['name'], True))
                else:
                    print("❌ Failed to block duplicate")
                    results.append((test_case['name'], False))
            elif test_case['expected_result'] == 'project_created':
                if success:
                    print("✅ Correctly created unique project")
                    results.append((test_case['name'], True))
                else:
                    print("❌ Failed to create unique project")
                    results.append((test_case['name'], False))
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((test_case['name'], False))
    
    # Summary
    print("\n" + "=" * 60)
    print("REAL SCENARIO TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")
    
    # Cleanup
    shutil.rmtree(base_path, ignore_errors=True)
    
    return passed == total


async def main():
    """Run the real scenario test."""
    print("Testing Duplicate Project Prevention - Real Scenario")
    print("=" * 60)
    
    try:
        success = await test_real_duplicate_prevention()
        
        if success:
            print("\n✅ All real scenario tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
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