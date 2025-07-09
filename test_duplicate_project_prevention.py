#!/usr/bin/env python3
"""
Test Duplicate Project Prevention

Tests that the system correctly prevents creation of duplicate projects
in both local and GitHub modes.
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
from scripts.enhanced_duplicate_detector import EnhancedDuplicateDetector
from scripts.storage_mode import StorageMode


async def test_duplicate_project_prevention():
    """Test that duplicate projects are prevented."""
    print("\n=== Testing Duplicate Project Prevention ===")
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    # Test projects that should be detected as duplicates
    test_cases = [
        {
            "existing": {
                "name": "sub-wise",
                "description": "Subscription tracking and management app",
                "problem_statement": "People lose track of their subscriptions",
            },
            "new_attempts": [
                {
                    "name": "subscrip-ai",
                    "description": "AI-powered subscription manager",
                    "problem_statement": "Help users manage their subscriptions",
                    "expected": "duplicate"
                },
                {
                    "name": "subtrack-ai", 
                    "description": "Track and manage subscriptions intelligently",
                    "problem_statement": "Users need to track recurring payments",
                    "expected": "duplicate"
                },
                {
                    "name": "subscription-tracker",
                    "description": "Simple subscription tracking tool",
                    "problem_statement": "Keep track of monthly subscriptions",
                    "expected": "duplicate"
                }
            ]
        },
        {
            "existing": {
                "name": "fitness-tracker",
                "description": "Track workouts and fitness progress",
                "problem_statement": "People want to monitor their fitness journey",
            },
            "new_attempts": [
                {
                    "name": "fit-track-pro",
                    "description": "Professional fitness tracking application", 
                    "problem_statement": "Athletes need to track their performance",
                    "expected": "duplicate"
                },
                {
                    "name": "recipe-manager",
                    "description": "Manage and organize cooking recipes",
                    "problem_statement": "Home cooks need to organize their recipes",
                    "expected": "not_duplicate"  # Different domain
                }
            ]
        }
    ]
    
    # Test in local mode
    print("\n1. Testing in LOCAL mode...")
    os.environ['STORAGE_MODE'] = StorageMode.LOCAL
    
    # Create local repos directory
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    local_repos_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize project creator
    project_creator = ProjectCreator(ai_brain=ai_brain, storage_mode=StorageMode.LOCAL)
    
    for test_case in test_cases:
        existing = test_case["existing"]
        
        # Simulate existing project by creating metadata
        existing_path = local_repos_path / existing["name"]
        existing_path.mkdir(exist_ok=True)
        
        metadata_path = existing_path / "PROJECT_METADATA.json"
        metadata = {
            "repository_name": existing["name"],
            "display_name": existing["name"],
            "description": existing["description"],
            "problem_domain": existing["problem_statement"],
            "created_at": "2024-01-01T00:00:00Z",
            "tech_stack": "laravel-react",
            "project_type": "web-app"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  Created existing project: {existing['name']}")
        
        # Test duplicate detection
        for attempt in test_case["new_attempts"]:
            print(f"\n  Testing: {attempt['name']} (expected: {attempt['expected']})")
            
            # Create project task
            task = {
                "task_type": "NEW_PROJECT",
                "metadata": {
                    "name": attempt["name"],
                    "description": attempt["description"],
                    "problem_statement": attempt["problem_statement"],
                    "target_market": "General users",
                    "initial_features": ["Core functionality"]
                }
            }
            
            # Check for duplicates using the enhanced detector
            detector = project_creator.duplicate_detector
            result = await detector.check_duplicate_with_enhanced_matching(task["metadata"])
            
            is_duplicate = result.get("is_duplicate", False)
            confidence = result.get("duplicate_confidence", 0)
            
            print(f"    - Duplicate detected: {is_duplicate}")
            print(f"    - Confidence: {confidence:.2f}")
            print(f"    - AI Analysis: {result.get('ai_analysis', 'N/A')[:100]}...")
            
            # Verify expectation
            if attempt["expected"] == "duplicate":
                if is_duplicate:
                    print("    ✅ Correctly identified as duplicate")
                else:
                    print("    ❌ FAILED: Should have been detected as duplicate")
            else:
                if not is_duplicate:
                    print("    ✅ Correctly identified as unique")
                else:
                    print("    ❌ FAILED: Should not have been detected as duplicate")
    
    # Test actual project creation attempt
    print("\n\n2. Testing actual project creation with duplicate...")
    
    duplicate_task = {
        "id": "test-duplicate-task",
        "task_type": "NEW_PROJECT", 
        "metadata": {
            "name": "sub-wise-clone",
            "description": "Another subscription tracking app",
            "problem_statement": "Track subscriptions easily",
            "target_market": "Consumers",
            "initial_features": ["Subscription tracking", "Notifications"]
        }
    }
    
    try:
        result = await project_creator.create_project(duplicate_task)
        
        if result.get("success") == False and "duplicate" in result.get("message", "").lower():
            print("✅ Project creation correctly blocked due to duplicate detection")
            print(f"   Message: {result.get('message')}")
        else:
            print("❌ FAILED: Duplicate project was created!")
            print(f"   Result: {result}")
            
    except Exception as e:
        print(f"❌ Error during project creation: {e}")
    
    # Cleanup
    print("\n\n3. Cleaning up test data...")
    for test_case in test_cases:
        test_path = local_repos_path / test_case["existing"]["name"]
        if test_path.exists():
            shutil.rmtree(test_path)
            print(f"   Removed: {test_path}")


async def test_github_mode_simulation():
    """Simulate GitHub mode duplicate detection."""
    print("\n\n=== Testing GitHub Mode Simulation ===")
    
    # This would normally interact with GitHub API
    # For testing, we'll verify the detector can handle GitHub mode
    
    ai_brain = IntelligentAIBrain()
    github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
    
    if not github_token:
        print("⚠️  No GitHub token found - skipping GitHub mode test")
        return
    
    # Test that enhanced detector works in GitHub mode
    detector = EnhancedDuplicateDetector(
        Path("/tmp/test"),
        ai_brain,
        github_token=github_token
    )
    
    print("✅ Enhanced duplicate detector initialized for GitHub mode")
    print(f"   - Has GitHub client: {detector.github_client is not None}")
    print(f"   - Can load GitHub repos: {hasattr(detector, '_load_github_repositories')}")


async def main():
    """Run all duplicate prevention tests."""
    print("Testing Duplicate Project Prevention System")
    print("=" * 60)
    
    try:
        await test_duplicate_project_prevention()
        await test_github_mode_simulation()
        
        print("\n" + "=" * 60)
        print("✅ Duplicate project prevention tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    asyncio.run(main())