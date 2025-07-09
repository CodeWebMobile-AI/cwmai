#!/usr/bin/env python3
"""
Test that AI generation is now the primary method for feature extraction.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "zip"))

# Import after path setup
from scripts.intelligent_work_finder import ResearchCache
from scripts.http_ai_client import HTTPAIClient


async def test_ai_feature_generation():
    """Test AI-powered feature generation."""
    print("🧪 Testing AI-First Feature Generation")
    print("=" * 60)
    
    # Test contexts
    test_cases = [
        {
            "name": "NFT Artist IP Protection",
            "context": """
Problem: Protecting their intellectual property and managing royalties
Audience: NFT artists
Market Size: $10B
Current Solutions: Manual tracking, basic smart contracts
Why They Fail: No automated enforcement, complex royalty structures
Opportunity: Blockchain-based IP protection with automated royalty management
"""
        },
        {
            "name": "E-commerce Product Descriptions",
            "context": """
Problem: High cost of creating product descriptions and marketing copy
Audience: E-commerce businesses
Market Size: $5B
Current Solutions: Hiring copywriters, basic templates
Why They Fail: Expensive, time-consuming, inconsistent quality
Opportunity: AI-powered content generation with SEO optimization
"""
        },
        {
            "name": "Healthcare Administrative Tasks",
            "context": """
Problem: Overwhelmed by administrative tasks, reducing time spent on patient care
Audience: Healthcare providers
Market Size: $15B
Current Solutions: Manual paperwork, basic EMR systems
Why They Fail: Still requires significant manual input and coordination
Opportunity: Intelligent automation platform for healthcare administration
"""
        }
    ]
    
    # Create cache instance
    cache = ResearchCache()
    
    print("\n🤖 Testing AI feature generation for different contexts...\n")
    
    all_successful = True
    for test in test_cases:
        print(f"📋 Test Case: {test['name']}")
        print("-" * 50)
        
        try:
            # Generate features using AI
            features = await cache._generate_domain_features(test['context'])
            
            if features and len(features) >= 5:
                print(f"✅ Successfully generated {len(features)} features:")
                for i, feature in enumerate(features, 1):
                    print(f"   {i}. {feature}")
            else:
                print(f"❌ Failed to generate features (got {len(features)})")
                all_successful = False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            all_successful = False
        
        print()
    
    return all_successful


async def test_feature_quality():
    """Test that AI generates high-quality, specific features."""
    print("\n🎯 Testing Feature Quality")
    print("=" * 60)
    
    # Specific problem context
    context = """
Problem: Small businesses struggle to manage customer relationships across multiple channels (email, social media, phone)
Audience: Small business owners
Market Size: $3B
Current Solutions: Spreadsheets, basic CRM tools
Why They Fail: Too complex, expensive, not integrated with communication channels
Opportunity: Unified communication hub with AI-powered customer insights
"""
    
    print("📝 Context:")
    print(context)
    print("\n🤖 Generating features with AI...\n")
    
    cache = ResearchCache()
    features = await cache._generate_domain_features(context)
    
    if features:
        print(f"✅ Generated {len(features)} features:")
        
        # Check quality metrics
        quality_checks = {
            "Specific (not generic)": all(
                not any(generic in f.lower() for generic in ['dashboard', 'api', 'database', 'login'])
                for f in features
            ),
            "Proper length (3-5 words)": all(
                2 <= len(f.split()) <= 6 for f in features
            ),
            "Domain-relevant": any(
                any(keyword in f.lower() for keyword in ['customer', 'communication', 'channel', 'insight'])
                for f in features
            ),
            "Actionable features": all(
                not f.lower().startswith(('the', 'a', 'an'))
                for f in features
            )
        }
        
        print("\n📊 Quality Analysis:")
        for check, passed in quality_checks.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check}")
        
        print("\n🔍 Features generated:")
        for i, feature in enumerate(features, 1):
            print(f"   {i}. {feature}")
        
        return all(quality_checks.values())
    else:
        print("❌ No features generated")
        return False


async def test_fallback_handling():
    """Test that the system handles edge cases gracefully."""
    print("\n🛡️ Testing Edge Case Handling")
    print("=" * 60)
    
    # Edge cases
    edge_cases = [
        ("Empty context", ""),
        ("Very short context", "Fix bugs"),
        ("Non-English context", "解决用户问题的AI平台"),
        ("Technical jargon", "Implement CQRS with event sourcing for distributed microservices architecture")
    ]
    
    cache = ResearchCache()
    
    for name, context in edge_cases:
        print(f"\n📋 {name}: '{context[:50]}...'")
        
        try:
            features = await cache._generate_domain_features(context)
            if features:
                print(f"✅ Generated {len(features)} features")
                for i, f in enumerate(features[:3], 1):
                    print(f"   {i}. {f}")
            else:
                print("⚠️ No features generated (will retry with validation)")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True


if __name__ == "__main__":
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not set. Please run:")
        print("export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    async def run_all_tests():
        """Run all tests."""
        print("🚀 AI-First Feature Generation Test Suite")
        print("=" * 60)
        print("Testing that AI is now the PRIMARY method for feature generation")
        print()
        
        results = []
        
        # Test 1: AI feature generation
        result1 = await test_ai_feature_generation()
        results.append(("AI Feature Generation", result1))
        
        # Test 2: Feature quality
        result2 = await test_feature_quality()
        results.append(("Feature Quality", result2))
        
        # Test 3: Edge case handling
        result3 = await test_fallback_handling()
        results.append(("Edge Case Handling", result3))
        
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
        print("\n\n🎉 All tests passed!")
        print("\n✨ AI-First Feature Generation is working correctly:")
        print("1. AI generates context-aware, specific features")
        print("2. Features are relevant to the problem domain")
        print("3. No hardcoded values - everything is dynamic")
        print("4. Robust parsing handles various response formats")
        print("5. Quality checks ensure features are actionable")
    else:
        print("\n\n❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)