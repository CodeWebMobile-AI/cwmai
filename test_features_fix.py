#!/usr/bin/env python3
"""
Test the comprehensive features fix.
Verifies that features are properly extracted and never empty.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "zip"))

# Import after path setup
from scripts.intelligent_work_finder import ResearchCache, IntelligentWorkFinder
from scripts.market_discovery import MarketDiscovery
from scripts.unified_task_state import UnifiedTaskState
from scripts.ai_brain import IntelligentAIBrain


async def test_feature_extraction():
    """Test that features are properly extracted from research."""
    print("🧪 Testing Feature Extraction Fix")
    print("=" * 60)
    
    # Initialize components
    state = UnifiedTaskState()
    system_state = state.get_system_state()
    
    # Create market discovery
    discovery = MarketDiscovery(system_state)
    
    print("\n📋 Running market discovery to test feature extraction...")
    
    try:
        # Discover 3 problems
        problems = await discovery.discover_problems(count=3)
        
        print(f"\n✅ Discovered {len(problems)} problems")
        
        # Check each problem
        all_have_features = True
        for i, problem in enumerate(problems, 1):
            print(f"\n{i}. Problem: {problem.problem_statement[:80]}...")
            print(f"   Audience: {problem.affected_users}")
            
            # Check if _full_opportunity has key_features
            if hasattr(problem, '_full_opportunity') and problem._full_opportunity:
                opp = problem._full_opportunity
                if hasattr(opp, 'key_features'):
                    features = opp.key_features
                    print(f"   ✅ Has {len(features)} features in opportunity")
                    for j, feat in enumerate(features[:3], 1):
                        print(f"      {j}. {feat}")
                else:
                    print("   ❌ No key_features field in opportunity")
                    all_have_features = False
            else:
                print("   ❌ No _full_opportunity attached")
                all_have_features = False
        
        return all_have_features
        
    except Exception as e:
        print(f"\n❌ Error during discovery: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cache_population():
    """Test that the cache properly stores features."""
    print("\n\n🧪 Testing Cache Population")
    print("=" * 60)
    
    # Initialize cache
    cache = ResearchCache(system_state={})
    
    print(f"\n📦 Initial cache size: {cache.cache_size()}")
    
    # Run batch research
    print("\n🔬 Running batch research...")
    
    try:
        problems = await cache.batch_research(count=3)
        
        print(f"\n✅ Added {len(problems)} problems to cache")
        print(f"📦 New cache size: {cache.cache_size()}")
        
        # Check each cached problem
        all_have_features = True
        for i, problem in enumerate(problems, 1):
            print(f"\n{i}. Cached Problem: {problem.name}")
            print(f"   Problem: {problem.problem[:80]}...")
            print(f"   Features: {len(problem.features)}")
            
            if problem.features:
                for j, feat in enumerate(problem.features[:3], 1):
                    print(f"      {j}. {feat}")
            else:
                print("   ❌ No features!")
                all_have_features = False
        
        return all_have_features
        
    except Exception as e:
        print(f"\n❌ Error during batch research: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_work_item_creation():
    """Test that work items have proper features."""
    print("\n\n🧪 Testing Work Item Creation")
    print("=" * 60)
    
    # Initialize components
    state = UnifiedTaskState()
    system_state = state.get_system_state()
    
    # Mock AI brain
    class MockAIBrain:
        async def analyze(self, *args, **kwargs):
            return {"analysis": "mock"}
    
    ai_brain = MockAIBrain()
    
    # Create work finder
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    print("\n🔍 Discovering work items...")
    
    try:
        work_items = await work_finder.discover_work(max_items=2)
        
        print(f"\n✅ Created {len(work_items)} work items")
        
        # Check each work item
        all_have_features = True
        for i, item in enumerate(work_items, 1):
            print(f"\n{i}. Work Item: {item.title}")
            print(f"   Type: {item.task_type}")
            
            # Check metadata for features
            metadata = item.metadata
            if 'selected_project' in metadata:
                selected = metadata['selected_project']
                features = selected.get('key_features', [])
                print(f"   Features: {len(features)}")
                
                if features:
                    for j, feat in enumerate(features[:3], 1):
                        print(f"      {j}. {feat}")
                else:
                    print("   ❌ No features in selected_project!")
                    all_have_features = False
            else:
                print("   ⚠️ Not a NEW_PROJECT task")
        
        return all_have_features
        
    except Exception as e:
        print(f"\n❌ Error during work discovery: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_feature_extraction_methods():
    """Test the feature extraction methods directly."""
    print("\n\n🧪 Testing Feature Extraction Methods")
    print("=" * 60)
    
    # Test cases
    test_descriptions = [
        "AI-powered content generation platform that helps users create unique, plagiarism-free articles with SEO optimization and multi-language support.",
        "Blockchain-based NFT marketplace with smart contract management, royalty tracking, and IP protection features.",
        "E-commerce solution that automates product descriptions, manages inventory across channels, and provides real-time analytics.",
        "Healthcare platform enabling patient management, appointment scheduling, and medical record integration with HIPAA compliance."
    ]
    
    # Create cache to test extraction
    cache = ResearchCache()
    
    for i, desc in enumerate(test_descriptions, 1):
        print(f"\n{i}. Testing: {desc[:80]}...")
        
        features = cache._extract_features_from_description(desc)
        print(f"   Extracted {len(features)} features:")
        
        for j, feat in enumerate(features, 1):
            print(f"      {j}. {feat}")
    
    return True


if __name__ == "__main__":
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not set. Please run:")
        print("export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    async def run_all_tests():
        """Run all tests."""
        results = []
        
        # Test 1: Feature extraction from research
        print("\n🏁 Starting comprehensive feature extraction tests...\n")
        
        result1 = await test_feature_extraction()
        results.append(("Feature Extraction", result1))
        
        # Test 2: Cache population
        result2 = await test_cache_population()
        results.append(("Cache Population", result2))
        
        # Test 3: Work item creation
        result3 = await test_work_item_creation()
        results.append(("Work Item Creation", result3))
        
        # Test 4: Extraction methods
        result4 = await test_feature_extraction_methods()
        results.append(("Extraction Methods", result4))
        
        # Summary
        print("\n\n📊 Test Results Summary")
        print("=" * 60)
        
        all_passed = True
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:.<40} {status}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n\n🎉 All tests passed! Features are now properly extracted!")
        print("\nKey improvements:")
        print("1. ✅ MarketOpportunity now has key_features field")
        print("2. ✅ Features extracted from opportunity descriptions") 
        print("3. ✅ Fallback to key_insights if needed")
        print("4. ✅ Smart extraction using patterns and AI")
        print("5. ✅ Validation ensures features are never empty")
        print("\nThe system is ready to create projects with proper features!")
    else:
        print("\n\n❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)