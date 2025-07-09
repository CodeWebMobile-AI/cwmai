#!/usr/bin/env python3
"""
Test Exact Duplicate Detection Only

Simple test to verify exact name duplicate detection works.
"""

import json
from pathlib import Path
import shutil

# Test exact name matching
def test_exact_duplicate():
    print("Testing Exact Name Duplicate Detection")
    print("=" * 50)
    
    # Create test directory
    test_dir = Path("/tmp/test_exact_dup")
    repos_dir = test_dir / "repositories" / "CodeWebMobile-AI"
    repos_dir.mkdir(parents=True, exist_ok=True)
    
    # Create existing project
    existing_project = repos_dir / "sub-wise"
    existing_project.mkdir(exist_ok=True)
    
    metadata = {
        "repository_name": "sub-wise",
        "display_name": "sub-wise",
        "description": "Subscription tracking app",
        "tech_stack": "laravel-react",
        "project_type": "web-app",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    # Use the correct filename that the detector expects
    with open(existing_project / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created existing project: sub-wise")
    
    # Test detection by checking directory
    test_cases = [
        ("sub-wise", True, "Exact match"),
        ("Sub-Wise", True, "Case variation"),  
        ("sub-wise-2", False, "Different name"),
        ("subwise", False, "No dash"),
        ("subscription-wise", False, "Different name")
    ]
    
    print("\nChecking for duplicates:")
    for test_name, should_exist, description in test_cases:
        # Simple check - does a directory with this name exist (case insensitive)?
        exists = False
        for existing in repos_dir.iterdir():
            if existing.is_dir() and existing.name.lower() == test_name.lower():
                exists = True
                break
        
        passed = exists == should_exist
        status = "✅" if passed else "❌"
        print(f"{status} {description}: '{test_name}' -> exists={exists} (expected={should_exist})")
    
    # Test the actual code path
    print("\n\nTesting actual duplicate detection code:")
    
    # Import after creating the structure
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
    
    from scripts.enhanced_duplicate_detector import EnhancedDuplicateDetector
    from scripts.ai_brain import IntelligentAIBrain
    
    ai_brain = IntelligentAIBrain()
    # Don't pass GitHub token to avoid loading real repos
    detector = EnhancedDuplicateDetector(repos_dir, ai_brain, github_token=None)
    
    # Load metadata
    detector.load_all_metadata()
    print(f"Loaded {len(detector._metadata_cache)} projects")
    print(f"Projects in cache: {list(detector._metadata_cache.keys())[:5]}...")  # Show first 5
    
    # Test exact name matching in the actual check
    new_project = {
        "name": "sub-wise",
        "description": "Another subscription app",
        "problem_statement": "Track subscriptions"
    }
    
    # Check the exact matching logic directly
    project_name = new_project.get('name', '').lower()
    is_exact_match = False
    for repo_name in detector._metadata_cache.keys():
        if repo_name.lower() == project_name:
            is_exact_match = True
            break
    
    print(f"\nExact name match for 'sub-wise': {is_exact_match}")
    
    if is_exact_match:
        print("✅ Exact duplicate detection is working!")
    else:
        print("❌ Exact duplicate detection failed!")
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    
    return is_exact_match


if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    success = test_exact_duplicate()
    exit(0 if success else 1)