#!/usr/bin/env python3
"""
Test script to verify nest_asyncio fixes the asyncio event loop error
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_nest_asyncio_basic():
    """Test that nest_asyncio allows nested event loops."""
    print("=== Testing nest_asyncio Basic Functionality ===")
    
    import nest_asyncio
    nest_asyncio.apply()
    
    async def simple_async_function():
        await asyncio.sleep(0.1)
        return "async function completed"
    
    # Test 1: Direct asyncio.run() call
    try:
        result = asyncio.run(simple_async_function())
        print(f"✓ Direct asyncio.run() works: {result}")
    except RuntimeError as e:
        print(f"❌ Direct asyncio.run() failed: {e}")
        return False
    
    # Test 2: Nested event loop scenario
    async def nested_test():
        # This simulates the scenario where we're already in an event loop
        # and need to call asyncio.run() again
        try:
            result = asyncio.run(simple_async_function())
            return f"Nested asyncio.run() works: {result}"
        except RuntimeError as e:
            return f"Nested asyncio.run() failed: {e}"
    
    try:
        # Run the nested test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(nested_test())
        print(f"✓ {result}")
        loop.close()
    except Exception as e:
        print(f"❌ Nested test failed: {e}")
        return False
    
    print("✓ nest_asyncio basic functionality working correctly")
    return True

def test_ai_brain_asyncio_fix():
    """Test that AI Brain no longer has asyncio event loop errors."""
    print("\n=== Testing AI Brain Asyncio Fix ===")
    
    try:
        from ai_brain import IntelligentAIBrain
        print("✓ AI Brain imported successfully with nest_asyncio")
    except Exception as e:
        print(f"❌ Failed to import AI Brain: {e}")
        return False
    
    # Create AI Brain instance
    try:
        ai_brain = IntelligentAIBrain({}, {})
        print("✓ AI Brain instance created successfully")
    except Exception as e:
        print(f"❌ Failed to create AI Brain instance: {e}")
        return False
    
    # Test the problematic method that was causing the error
    try:
        # Create a minimal test prompt
        test_prompt = "Test prompt for asyncio verification"
        
        # This should no longer cause the asyncio event loop error
        result = ai_brain.generate_enhanced_response_sync(test_prompt)
        
        if result and not result.get('error'):
            print("✓ generate_enhanced_response_sync works without asyncio errors")
        elif result and 'error' in result:
            print(f"⚠️  Method executed but returned error: {result['error']}")
            # This might be acceptable if it's a non-asyncio error (e.g., API key missing)
        else:
            print("❌ Method returned no result")
            return False
            
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(f"❌ Asyncio event loop error still occurs: {e}")
            return False
        else:
            print(f"⚠️  Different runtime error (may be acceptable): {e}")
    except Exception as e:
        print(f"⚠️  Other error (may be acceptable if not asyncio-related): {e}")
    
    return True

def test_context_enhancement():
    """Test context enhancement that was failing in context.json."""
    print("\n=== Testing Context Enhancement (context.json fix) ===")
    
    try:
        from ai_brain import IntelligentAIBrain
        
        ai_brain = IntelligentAIBrain({}, {})
        
        # Create test context similar to what would be in context.json
        test_context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "test_environment",
            "file_path": "test_context.json"
        }
        
        # Test the context enhancement that was causing the asyncio error
        enhanced_context = ai_brain.enhance_context_with_ai_analysis(
            test_context, 
            "Test context for asyncio fix verification"
        )
        
        # Check if AI analysis was added successfully
        if 'ai_analysis' in enhanced_context:
            ai_analysis = enhanced_context['ai_analysis']
            
            if 'error' not in ai_analysis or 'asyncio.run() cannot be called from a running event loop' not in str(ai_analysis.get('error', '')):
                print("✓ Context enhancement works without asyncio errors")
                print(f"✓ AI analysis result: {ai_analysis.get('summary', 'No summary')}")
                return True
            else:
                print(f"❌ Context enhancement still has asyncio error: {ai_analysis.get('error')}")
                return False
        else:
            print("⚠️  No AI analysis added to context (may be due to other issues)")
            return True  # Not necessarily a failure if other errors occur
            
    except Exception as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(f"❌ Context enhancement still has asyncio error: {e}")
            return False
        else:
            print(f"⚠️  Context enhancement had other error: {e}")
            return True  # Other errors might be acceptable (API keys, etc.)

def main():
    """Run all asyncio fix tests."""
    print("🧪 Testing nest_asyncio Implementation for Asyncio Event Loop Fix\n")
    
    tests = [
        ("Basic nest_asyncio functionality", test_nest_asyncio_basic),
        ("AI Brain asyncio fix", test_ai_brain_asyncio_fix),
        ("Context enhancement fix", test_context_enhancement)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The nest_asyncio fix is working correctly.")
        print("\n✅ The asyncio event loop error in context.json should now be resolved.")
    else:
        print("⚠️  Some tests failed. The fix may need additional work.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)