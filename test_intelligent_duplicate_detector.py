#!/usr/bin/env python3
"""
Test the AI-powered intelligent duplicate detector
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.intelligent_duplicate_detector import IntelligentDuplicateDetector
from scripts.ai_brain import IntelligentAIBrain

async def test_intelligent_duplicate_detection():
    """Test the AI-powered duplicate detection system."""
    
    # Initialize system
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    ai_brain = IntelligentAIBrain()
    
    detector = IntelligentDuplicateDetector(local_repos_path, ai_brain)
    
    print("🤖 Intelligent Duplicate Detection Test")
    print("=" * 60)
    
    # Load existing metadata
    print("\n📂 Loading existing projects...")
    existing_metadata = detector.load_all_metadata()
    print(f"Found {len(existing_metadata)} existing projects")
    
    # Test Case 1: Very similar to ai-text-summarizer
    print("\n\n🧪 Test 1: Similar to AI Text Summarizer")
    print("-" * 50)
    
    test_summarizer = {
        'name': 'QuickDigest AI',
        'description': 'AI-powered content summarization tool that helps busy people quickly understand articles, videos, and audio content',
        'problem_statement': 'People are overwhelmed with content and dont have time to consume everything they need to stay informed',
        'target_market': 'Busy professionals, students, and researchers',
        'initial_features': [
            'Summarize long articles and documents',
            'Extract key points from videos',
            'Convert audio to text summaries',
            'Highlight important insights',
            'Export summaries in various formats'
        ],
        'monetization_model': 'Freemium - free tier with limits, paid plans for heavy users'
    }
    
    print(f"\n📋 Testing: {test_summarizer['name']}")
    print(f"Description: {test_summarizer['description']}")
    
    result = await detector.check_duplicate_with_ai(test_summarizer)
    
    print(f"\n🎯 Results:")
    print(f"Is Duplicate: {result['is_duplicate']} (Confidence: {result.get('duplicate_confidence', 0):.0%})")
    print(f"AI Analysis: {result['ai_analysis']}")
    print(f"Recommendation: {result['recommendation']}")
    
    if result['similar_projects']:
        print(f"\n📊 Similar Projects Found:")
        for proj in result['similar_projects']:
            print(f"\n  • {proj['display_name']} ({proj['similarity_score']:.0%} similar)")
            print(f"    Type: {proj['similarity_type']}")
            if proj['reasons']:
                print(f"    Overlaps: {', '.join(proj['reasons'])}")
            if proj.get('could_merge'):
                print(f"    💡 Could merge with this project")
    
    if result.get('unique_aspects'):
        print(f"\n✨ Unique Aspects:")
        for aspect in result['unique_aspects']:
            print(f"  • {aspect}")
    
    # Test Case 2: Similar to subscription manager
    print("\n\n🧪 Test 2: Similar to Subscription Manager")
    print("-" * 50)
    
    test_subscription = {
        'name': 'BillGuardian',
        'description': 'Smart subscription tracking app that monitors all your recurring payments and helps you save money',
        'problem_statement': 'People lose hundreds of dollars yearly on forgotten subscriptions and dont realize when prices increase',
        'target_market': 'Individual consumers with multiple subscriptions',
        'initial_features': [
            'Auto-detect subscriptions from bank',
            'Track spending over time',
            'Alert on price changes',
            'One-click cancellation help',
            'Spending analytics dashboard'
        ],
        'monetization_model': 'Paid app with one-time purchase'
    }
    
    print(f"\n📋 Testing: {test_subscription['name']}")
    result2 = await detector.check_duplicate_with_ai(test_subscription)
    
    print(f"\n🎯 Results:")
    print(f"Is Duplicate: {result2['is_duplicate']} (Confidence: {result2.get('duplicate_confidence', 0):.0%})")
    print(f"AI Analysis: {result2['ai_analysis']}")
    
    # Test Case 3: Completely unique project
    print("\n\n🧪 Test 3: Unique Project")
    print("-" * 50)
    
    test_unique = {
        'name': 'LocalHarvest Connect',
        'description': 'Platform connecting local farmers directly with consumers for fresh produce delivery',
        'problem_statement': 'Consumers want fresh, local produce but dont know where to find it, while small farmers struggle to reach customers',
        'target_market': 'Health-conscious consumers and small local farmers',
        'initial_features': [
            'Farmer marketplace listings',
            'Seasonal produce calendar',
            'Direct ordering system',
            'Delivery route optimization',
            'Farm visit scheduling'
        ],
        'monetization_model': 'Commission on sales + farmer subscription fees'
    }
    
    print(f"\n📋 Testing: {test_unique['name']}")
    result3 = await detector.check_duplicate_with_ai(test_unique)
    
    print(f"\n🎯 Results:")
    print(f"Is Duplicate: {result3['is_duplicate']}")
    print(f"Innovation Score: {result3.get('innovation_score', 0):.0%}")
    
    if result3.get('market_gaps'):
        print(f"\n🎯 Market Gaps Filled:")
        for gap in result3['market_gaps']:
            print(f"  • {gap}")
    
    # Test Portfolio Analysis
    print("\n\n🧪 Test 4: Portfolio Fit Analysis")
    print("-" * 50)
    
    portfolio_analysis = await detector.analyze_portfolio_overlap(test_unique)
    
    print(f"\nPortfolio Fit Score: {portfolio_analysis.get('portfolio_fit_score', 0):.0%}")
    print(f"Strategic Value: {portfolio_analysis.get('strategic_value', 'unknown')}")
    print(f"Recommendation: {portfolio_analysis.get('recommendation', 'No recommendation')}")
    
    if portfolio_analysis.get('synergies'):
        print(f"\n🤝 Potential Synergies:")
        for synergy in portfolio_analysis['synergies']:
            print(f"  • With {synergy['project']}: {synergy['description']}")
    
    print("\n\n✅ All tests completed!")


def main():
    """Run the test."""
    try:
        asyncio.run(test_intelligent_duplicate_detection())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()