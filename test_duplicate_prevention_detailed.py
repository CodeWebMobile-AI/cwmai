#!/usr/bin/env python3
"""
Detailed test of duplicate prevention with real scenarios
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.duplicate_prevention_system import DuplicatePreventionSystem
from scripts.ai_brain import IntelligentAIBrain

async def test_real_duplicates():
    """Test with scenarios that should trigger duplicate detection."""
    
    # Initialize system
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    ai_brain = IntelligentAIBrain()
    
    duplicate_system = DuplicatePreventionSystem(local_repos_path, ai_brain)
    
    print("🔍 Advanced Duplicate Prevention Tests")
    print("=" * 60)
    
    # Load existing metadata
    print("\n📂 Loading existing project metadata...")
    existing_metadata = duplicate_system.load_all_metadata()
    
    # Show existing projects with more detail
    print(f"\nExisting {len(existing_metadata)} projects:")
    for repo_name, metadata in existing_metadata.items():
        print(f"\n📦 {metadata.display_name}")
        print(f"   Repo: {repo_name}")
        print(f"   Description: {metadata.description}")
        print(f"   Features: {', '.join(metadata.key_features[:3])}...")
    
    # Test Case 1: Very similar to ai-text-summarizer
    print("\n\n🧪 Test 1: Similar to AI Text Summarizer")
    print("-" * 50)
    
    test_summarizer = {
        'name': 'Smart Content Digest',
        'description': 'AI tool that summarizes articles, videos, and podcasts for busy professionals',
        'problem_statement': 'Information overload - people dont have time to consume all content',
        'target_market': 'Busy professionals and students',
        'initial_features': [
            'Text summarization',
            'Audio transcription and summary',
            'Video summary generation',
            'Key points extraction',
            'Multi-language support'
        ],
        'monetization_model': 'Freemium with premium features'
    }
    
    result = duplicate_system.check_duplicate(test_summarizer)
    print(f"\n✅ Is Duplicate: {result['is_duplicate']}")
    print(f"📋 Recommendation: {result['recommendation']}")
    print(f"\n📊 Similar Projects ({len(result['similar_projects'])}):")
    
    for proj in result['similar_projects']:
        print(f"\n   🔹 {proj['display_name']} ({proj['similarity_score']:.0%} similar)")
        print(f"      Repository: {proj['repository_name']}")
        print(f"      Reasons: {', '.join(proj['reasons'])}")
    
    if result['suggested_differentiators']:
        print(f"\n💡 Suggested Differentiators:")
        for i, diff in enumerate(result['suggested_differentiators'], 1):
            print(f"   {i}. {diff}")
    
    # Test Case 2: Similar to micronutrient-optimizer
    print("\n\n🧪 Test 2: Similar to Micronutrient Optimizer")
    print("-" * 50)
    
    test_nutrition = {
        'name': 'NutriTrack Pro',
        'description': 'Personal nutrition tracking app with AI-powered recommendations',
        'problem_statement': 'People struggle to track vitamins and minerals in their diet',
        'target_market': 'Health-conscious individuals',
        'initial_features': [
            'Food logging',
            'Nutrient tracking',
            'AI recommendations',
            'Meal planning',
            'Health reports'
        ],
        'monetization_model': 'Subscription-based'
    }
    
    result2 = duplicate_system.check_duplicate(test_nutrition)
    print(f"\n✅ Is Duplicate: {result2['is_duplicate']}")
    print(f"📋 Recommendation: {result2['recommendation']}")
    
    # Test Case 3: Similar to subtrack-ai
    print("\n\n🧪 Test 3: Similar to SubTrack AI")
    print("-" * 50)
    
    test_subscriptions = {
        'name': 'SubscriptionGuard',
        'description': 'Manage and optimize your recurring subscriptions with AI insights',
        'problem_statement': 'People lose money on forgotten subscriptions',
        'target_market': 'Consumers with multiple subscriptions',
        'initial_features': [
            'Subscription tracking',
            'Spending analysis',
            'Cancellation reminders',
            'AI optimization suggestions',
            'Budget alerts'
        ],
        'monetization_model': 'Freemium model'
    }
    
    result3 = duplicate_system.check_duplicate(test_subscriptions)
    print(f"\n✅ Is Duplicate: {result3['is_duplicate']}")
    print(f"📋 Recommendation: {result3['recommendation']}")
    print(f"🔍 Similarity Reasons: {result3['similarity_reasons']}")
    
    # Test Case 4: Combination of existing features
    print("\n\n🧪 Test 4: Combination Project")
    print("-" * 50)
    
    test_combo = {
        'name': 'LifeOptimizer AI',
        'description': 'All-in-one AI platform for health tracking, content summarization, and subscription management',
        'problem_statement': 'People need multiple apps to manage different aspects of their life',
        'target_market': 'Busy professionals seeking life optimization',
        'initial_features': [
            'Nutrition tracking',
            'Content summarization',
            'Subscription management',
            'Health insights',
            'Unified dashboard'
        ],
        'monetization_model': 'Premium all-in-one subscription'
    }
    
    result4 = duplicate_system.check_duplicate(test_combo)
    print(f"\n✅ Is Duplicate: {result4['is_duplicate']}")
    print(f"📋 Recommendation: {result4['recommendation']}")
    
    if result4['similar_projects']:
        print(f"\n📊 Overlaps with {len(result4['similar_projects'])} projects:")
        for proj in result4['similar_projects']:
            print(f"   - {proj['display_name']}: {proj['similarity_score']:.0%}")
    
    # Test AI Validation on the most similar case
    if result3['similar_projects']:
        print("\n\n🤖 AI Validation Test")
        print("-" * 50)
        print("Getting AI opinion on the subscription management duplicate...")
        
        ai_result = await duplicate_system.validate_with_ai(
            test_subscriptions,
            result3['similar_projects'][:2]
        )
        
        if ai_result and ai_result.get('uniqueness_score') is not None:
            print(f"\n🎯 AI Uniqueness Score: {ai_result.get('uniqueness_score', 0):.0%}")
            print(f"📋 AI Recommendation: {ai_result.get('recommendation', 'N/A')}")
            
            if ai_result.get('unique_aspects'):
                print("\n✨ Unique Aspects:")
                for aspect in ai_result['unique_aspects']:
                    print(f"   ✓ {aspect}")
            
            if ai_result.get('overlap_concerns'):
                print("\n⚠️ Overlap Concerns:")
                for concern in ai_result['overlap_concerns']:
                    print(f"   - {concern}")
                    
            if ai_result.get('pivot_suggestions'):
                print("\n💡 Pivot Suggestions:")
                for suggestion in ai_result['pivot_suggestions']:
                    print(f"   → {suggestion}")
    
    print("\n\n✅ All tests completed!")


def main():
    """Run the tests."""
    try:
        asyncio.run(test_real_duplicates())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()