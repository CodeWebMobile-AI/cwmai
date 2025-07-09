#!/usr/bin/env python3
"""
Test Gemini Pro model access.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.http_ai_client import HTTPAIClient


async def test_gemini_models():
    """Test different Gemini model configurations."""
    print("=" * 80)
    print("🧪 TESTING GEMINI MODEL ACCESS")
    print("=" * 80)
    
    # Initialize client
    client = HTTPAIClient()
    
    # Test prompt
    test_prompt = "What are the top 3 benefits of using AI for market research? Be concise."
    
    print("\n1️⃣ Testing Gemini Flash (default)")
    print("-" * 40)
    try:
        response = await client.generate_enhanced_response(test_prompt, model='gemini')
        print(f"✅ Model: {response.get('model', 'unknown')}")
        print(f"Response: {response.get('content', 'No response')[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n2️⃣ Testing Gemini Pro (gemini-2.5-pro)")
    print("-" * 40)
    try:
        response = await client.generate_enhanced_response(test_prompt, model='gemini-2.5-pro')
        print(f"✅ Model: {response.get('model', 'unknown')}")
        print(f"Response: {response.get('content', 'No response')[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n3️⃣ Available Gemini Models")
    print("-" * 40)
    print("Currently supported:")
    print("  • 'gemini' → gemini-2.0-flash (fast, general purpose)")
    print("  • 'gemini-2.5-pro' → gemini-pro (advanced reasoning)")
    
    print("\n📝 Note: The actual model names in Google's API are:")
    print("  • gemini-2.0-flash-001")
    print("  • gemini-pro")
    print("  • gemini-pro-vision (for multimodal)")
    print("  • gemini-1.5-pro (if available)")


async def main():
    """Run tests."""
    await test_gemini_models()


if __name__ == "__main__":
    print("🚀 Gemini Model Test")
    print("This will test access to different Gemini models\n")
    
    asyncio.run(main())