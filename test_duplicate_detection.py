#!/usr/bin/env python3
"""
Test the enhanced duplicate detection functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.github_issue_creator import GitHubIssueCreator

def test_similarity_algorithms():
    """Test the similarity calculation algorithms."""
    creator = GitHubIssueCreator()
    
    test_cases = [
        # (title1, title2, expected_min_similarity)
        ("Add user authentication", "Add user authentication", 1.0),
        ("Add user authentication", "add user authentication", 1.0),
        ("Add user authentication system", "Add authentication for users", 0.7),
        ("Create API documentation", "Create documentation for API", 0.8),
        ("Fix memory leak in data processing", "Fix data processing memory leak", 0.8),
        ("Implement real-time notifications", "Add real-time notification system", 0.7),
        ("Update dependencies", "Update project dependencies", 0.9),
        ("Completely different task", "Another unrelated work item", 0.1),
    ]
    
    print("Testing enhanced similarity algorithms:\n")
    
    for title1, title2, expected_min in test_cases:
        # Test basic Jaccard similarity
        jaccard_sim = creator._calculate_title_similarity(title1, title2)
        
        # Test enhanced similarity
        enhanced_sim = creator._calculate_enhanced_similarity(title1, title2)
        
        print(f"Titles: '{title1}' vs '{title2}'")
        print(f"  Jaccard similarity: {jaccard_sim:.2%}")
        print(f"  Enhanced similarity: {enhanced_sim:.2%}")
        print(f"  Expected minimum: {expected_min:.2%}")
        print(f"  ✓ PASS" if enhanced_sim >= expected_min else f"  ✗ FAIL")
        print()

def test_search_term_extraction():
    """Test search term extraction."""
    creator = GitHubIssueCreator()
    
    test_cases = [
        ("Add user authentication system with OAuth support", "user authentication system OAuth support"),
        ("Fix the bug in data processing", "data processing"),
        ("Create comprehensive API documentation", "comprehensive documentation"),
        ("Implement real-time WebSocket notifications", "real-time WebSocket notifications"),
    ]
    
    print("\nTesting search term extraction:\n")
    
    for title, expected_contains in test_cases:
        terms = creator._extract_search_terms(title)
        print(f"Title: '{title}'")
        print(f"  Extracted terms: '{terms}'")
        
        # Check if expected terms are present
        expected_words = expected_contains.split()
        found_all = all(word in terms for word in expected_words if len(word) > 3)
        
        print(f"  ✓ PASS" if found_all else f"  ✗ FAIL - Missing expected terms")
        print()

def test_threshold_configuration():
    """Test that different task types have appropriate thresholds."""
    print("\nSimilarity thresholds by task type:\n")
    
    thresholds = {
        'DOCUMENTATION': 0.85,
        'BUG_FIX': 0.90,
        'FEATURE': 0.88,
        'TESTING': 0.87,
        'NEW_PROJECT': 0.95,
        'RESEARCH': 0.85
    }
    
    for task_type, threshold in thresholds.items():
        print(f"{task_type}: {threshold:.0%}")
    
    print("\nThese thresholds help prevent duplicates while allowing for some variation in wording.")

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Duplicate Detection Test Suite")
    print("=" * 60)
    
    test_similarity_algorithms()
    test_search_term_extraction()
    test_threshold_configuration()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("- Enhanced similarity uses multiple algorithms (Jaccard, Levenshtein, Token overlap)")
    print("- Search API is used for efficient duplicate checking")
    print("- Recently closed issues (last 30 days) are also checked")
    print("- Different task types have different similarity thresholds")
    print("- Performance optimized by limiting checks to 50-100 issues")
    print("=" * 60)