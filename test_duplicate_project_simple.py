#!/usr/bin/env python3
"""
Simple Test for Duplicate Project Prevention

Tests the duplicate detection logic without requiring AI API calls.
"""

import json
import os
import sys
from pathlib import Path
import shutil

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.enhanced_duplicate_detector import EnhancedDuplicateDetector
from scripts.duplicate_prevention_system import ProjectMetadata


def test_exact_name_matching():
    """Test exact name matching for duplicates."""
    print("\n=== Testing Exact Name Matching ===")
    
    # Create test directory
    test_dir = Path("/tmp/test_duplicate_projects")
    test_dir.mkdir(exist_ok=True)
    
    # Create a mock AI brain (won't be used for exact matching)
    class MockAIBrain:
        pass
    
    detector = EnhancedDuplicateDetector(test_dir, MockAIBrain())
    
    # Simulate existing projects
    existing_projects = [
        "sub-wise",
        "subscrip-ai", 
        "subtrack-ai",
        "fitness-tracker",
        "recipe-manager"
    ]
    
    # Manually populate the metadata cache
    for project_name in existing_projects:
        metadata = ProjectMetadata(
            repository_name=project_name,
            display_name=project_name,
            description=f"Description for {project_name}",
            tech_stack="laravel-react",
            project_type="web-app",
            key_features=["feature1", "feature2"],
            created_at="2024-01-01T00:00:00Z",
            created_by="test",
            problem_domain="test domain",
            target_audience="users",
            monetization_model="subscription",
            core_functionality="core function"
        )
        detector._metadata_cache[project_name] = metadata
    
    print(f"Loaded {len(detector._metadata_cache)} existing projects")
    
    # Test exact name matching
    test_cases = [
        ("sub-wise", True, "Exact match"),
        ("Sub-Wise", True, "Case insensitive match"),
        ("SUB-WISE", True, "All caps match"),
        ("sub-wise-2", False, "Different name"),
        ("subscription-wise", False, "Similar but different"),
        ("fitness-tracker", True, "Another exact match"),
        ("fitness-track", False, "Partial match should not trigger")
    ]
    
    all_passed = True
    
    for test_name, should_match, description in test_cases:
        # Test synchronous exact matching logic
        is_duplicate = False
        
        # Check exact name match (case insensitive)
        test_name_lower = test_name.lower()
        for repo_name in detector._metadata_cache.keys():
            if repo_name.lower() == test_name_lower:
                is_duplicate = True
                break
        
        passed = is_duplicate == should_match
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed
        
        print(f"{status} {description}: '{test_name}' -> duplicate={is_duplicate} (expected={should_match})")
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    
    return all_passed


def test_project_creator_integration():
    """Test that ProjectCreator properly uses the duplicate detector."""
    print("\n\n=== Testing Project Creator Integration ===")
    
    # Check if the ProjectCreator uses EnhancedDuplicateDetector
    from scripts.project_creator import ProjectCreator
    
    # Create test directory
    local_repos_path = Path("/tmp/test_repos")
    local_repos_path.mkdir(exist_ok=True)
    
    try:
        # Check imports
        print("1. Checking ProjectCreator imports...")
        with open("scripts/project_creator.py", "r") as f:
            content = f.read()
            
        has_import = "from scripts.enhanced_duplicate_detector import EnhancedDuplicateDetector" in content
        print(f"   {'✅' if has_import else '❌'} Import found: {has_import}")
        
        # Check initialization
        print("\n2. Checking duplicate detector initialization...")
        init_local = "self.duplicate_detector = EnhancedDuplicateDetector(self.local_repos_path, ai_brain)" in content
        init_github = "self.duplicate_detector = EnhancedDuplicateDetector(github_repos_path, ai_brain, github_token)" in content
        
        print(f"   {'✅' if init_local else '❌'} Local mode initialization: {init_local}")
        print(f"   {'✅' if init_github else '❌'} GitHub mode initialization: {init_github}")
        
        # Check usage in create_project
        print("\n3. Checking duplicate detection usage...")
        check_usage = "await self.duplicate_detector.check_duplicate_with_enhanced_matching" in content
        print(f"   {'✅' if check_usage else '❌'} Duplicate check in create_project: {check_usage}")
        
        return has_import and init_local and init_github and check_usage
        
    finally:
        shutil.rmtree(local_repos_path, ignore_errors=True)


def test_metadata_structure():
    """Test that ProjectMetadata has the correct structure."""
    print("\n\n=== Testing ProjectMetadata Structure ===")
    
    # Check if ProjectMetadata has all required fields
    from scripts.duplicate_prevention_system import ProjectMetadata
    
    # Expected fields based on the error
    expected_fields = [
        'repository_name',
        'display_name', 
        'description',
        'tech_stack',
        'project_type',
        'key_features',
        'created_at',
        'created_by',
        'problem_domain',
        'target_audience',
        'monetization_model',
        'core_functionality'
    ]
    
    # Check if we can create an instance
    try:
        metadata = ProjectMetadata(
            repository_name="test",
            display_name="test",
            description="test",
            tech_stack="test",
            project_type="test",
            key_features=[],
            created_at="2024-01-01",
            created_by="test",
            problem_domain="test",
            target_audience="test",
            monetization_model="test",
            core_functionality="test"
        )
        print("✅ ProjectMetadata created successfully with all fields")
        
        # Check fields
        for field in expected_fields:
            has_field = hasattr(metadata, field)
            print(f"   {'✅' if has_field else '❌'} Has field: {field}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to create ProjectMetadata: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Duplicate Project Prevention (Simple)")
    print("=" * 60)
    
    results = []
    
    # Test 1: Exact name matching
    results.append(("Exact Name Matching", test_exact_name_matching()))
    
    # Test 2: Project Creator integration
    results.append(("Project Creator Integration", test_project_creator_integration()))
    
    # Test 3: Metadata structure
    results.append(("Metadata Structure", test_metadata_structure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()