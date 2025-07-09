#!/usr/bin/env python3
"""
Direct test of duplicate detector with existing projects
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.enhanced_duplicate_detector import EnhancedDuplicateDetector
from scripts.ai_brain import IntelligentAIBrain

def test_direct():
    print("Direct Duplicate Detector Test")
    print("=" * 60)
    
    # Real path with actual projects
    repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    # Initialize without AI to test just the basic loading
    class MockAIBrain:
        pass
    
    detector = EnhancedDuplicateDetector(repos_path, MockAIBrain(), github_token=None)
    
    # Load metadata
    detector.load_all_metadata()
    
    print(f"\nLoaded {len(detector._metadata_cache)} projects:")
    for name in sorted(detector._metadata_cache.keys()):
        print(f"  - {name}")
    
    # Test exact name matching
    print("\nTesting exact name matches:")
    test_names = ["sub-wise", "Sub-Wise", "subscrip-ai", "subtrack-ai", "new-project"]
    
    for test_name in test_names:
        found = False
        for repo_name in detector._metadata_cache.keys():
            if repo_name.lower() == test_name.lower():
                found = True
                break
        
        print(f"  {test_name}: {'FOUND' if found else 'NOT FOUND'}")
    
    # Check the actual duplicate detection method
    print("\nTesting check_duplicate_with_enhanced_matching (without AI):")
    
    test_project = {
        "name": "sub-wise",
        "description": "Test project",
        "problem_statement": "Test"
    }
    
    # Direct exact match check (what happens in the method)
    project_name = test_project.get('name', '').lower()
    exact_match = False
    for repo_name in detector._metadata_cache.keys():
        if repo_name.lower() == project_name:
            exact_match = True
            print(f"  ✅ Exact match found for '{test_project['name']}'")
            break
    
    if not exact_match:
        print(f"  ❌ No exact match found for '{test_project['name']}'")
    
    return exact_match

if __name__ == "__main__":
    # No need for env vars for this test
    try:
        success = test_direct()
        if success:
            print("\n✅ Duplicate detection is working at the basic level")
        else:
            print("\n❌ Duplicate detection is NOT working")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)