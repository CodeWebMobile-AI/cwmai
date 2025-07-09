#!/usr/bin/env python3
"""
Test Semantic Duplicate Detection with Fixed Schema

Tests that AI can detect semantically similar projects.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.enhanced_duplicate_detector import EnhancedDuplicateDetector
from scripts.ai_brain import IntelligentAIBrain


async def test_semantic_similarity():
    """Test semantic similarity detection."""
    print("Testing Semantic Duplicate Detection")
    print("=" * 60)
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    # Real path with actual projects
    repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    # Initialize detector
    detector = EnhancedDuplicateDetector(repos_path, ai_brain, github_token=None)
    
    # Load existing projects
    detector.load_all_metadata()
    print(f"\nLoaded {len(detector._metadata_cache)} projects")
    
    # Test cases - projects that should be detected as similar
    test_cases = [
        {
            "name": "Test 1: Very Similar to sub-wise",
            "project": {
                "name": "subscription-manager-pro",
                "description": "Professional subscription management and tracking tool",
                "problem_statement": "Users lose money on forgotten subscriptions",
                "target_market": "Consumers who want to track subscriptions",
                "initial_features": ["subscription tracking", "payment alerts", "spending analytics"]
            },
            "expected_similar_to": "sub-wise"
        },
        {
            "name": "Test 2: Similar concept different name",
            "project": {
                "name": "recurpay-tracker",
                "description": "Track all your recurring payments in one place",
                "problem_statement": "People don't know how much they spend on subscriptions",
                "target_market": "Budget-conscious consumers",
                "initial_features": ["recurring payment tracking", "budget alerts"]
            },
            "expected_similar_to": "sub-wise or subscrip-ai"
        },
        {
            "name": "Test 3: Different domain",
            "project": {
                "name": "cookbook-organizer",
                "description": "Digital cookbook and recipe management",
                "problem_statement": "Home cooks need to organize their recipes",
                "target_market": "Home cooks and food enthusiasts",
                "initial_features": ["recipe storage", "meal planning", "shopping lists"]
            },
            "expected_similar_to": "None (unique project)"
        }
    ]
    
    # Run tests
    print("\nRunning semantic similarity tests...")
    print("-" * 60)
    
    results = []
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"Project: {test['project']['name']}")
        print(f"Description: {test['project']['description']}")
        
        try:
            # Use the enhanced matching
            result = await detector.check_duplicate_with_enhanced_matching(test['project'])
            
            is_duplicate = result.get('is_duplicate', False)
            confidence = result.get('duplicate_confidence', 0)
            ai_analysis = result.get('ai_analysis', 'No analysis')
            recommendation = result.get('recommendation', 'unknown')
            
            print(f"\nResults:")
            print(f"  Is Duplicate: {is_duplicate}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Recommendation: {recommendation}")
            print(f"  AI Analysis: {ai_analysis[:200]}..." if len(ai_analysis) > 200 else f"  AI Analysis: {ai_analysis}")
            
            # Check if it matches expectations
            expected = test['expected_similar_to']
            if "None" in expected and not is_duplicate:
                print("  ✅ Correctly identified as unique")
                results.append(True)
            elif "None" not in expected and is_duplicate:
                print(f"  ✅ Correctly identified as similar (expected: {expected})")
                results.append(True)
            else:
                print(f"  ❌ Incorrect result (expected: {expected})")
                results.append(False)
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SEMANTIC SIMILARITY TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All semantic similarity tests PASSED!")
        print("The AI can successfully detect similar projects.")
    else:
        print("\n❌ Some tests FAILED!")
        print("The AI may not be properly detecting similar projects.")
    
    return passed == total


async def main():
    """Run the semantic similarity test."""
    try:
        # Export environment properly
        print("Note: Make sure to run with proper environment:")
        print("export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 test_semantic_duplicate_detection.py")
        print()
        
        success = await test_semantic_similarity()
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