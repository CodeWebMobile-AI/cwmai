#!/usr/bin/env python3
"""
Test AI Fallback Mechanism

This script tests that the AI client properly falls back to other providers
when Anthropic API has insufficient credits.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.http_ai_client import HTTPAIClient


async def test_fallback():
    """Test the AI fallback mechanism."""
    print("Testing AI Fallback Mechanism")
    print("=" * 50)
    
    # Show available providers
    client = HTTPAIClient(enable_round_robin=True)
    print("\nAvailable providers:")
    for provider, available in client.providers_available.items():
        status = "✓" if available else "✗"
        print(f"  {status} {provider}")
    
    print(f"\nRound-robin order: {client.available_providers}")
    
    # Test 1: Direct call without model specification (should use round-robin)
    print("\n\nTest 1: Round-robin selection")
    print("-" * 30)
    
    for i in range(3):
        response = await client.generate_enhanced_response(
            f"Say 'Hello from test {i+1}' and nothing else"
        )
        print(f"Attempt {i+1}:")
        print(f"  Provider: {response.get('provider', 'unknown')}")
        print(f"  Content: {response.get('content', 'No content')[:100]}")
        print(f"  Error: {response.get('error', 'None')}")
    
    # Test 2: Specific model request
    print("\n\nTest 2: Specific model requests")
    print("-" * 30)
    
    models = ['claude', 'gpt', 'gemini', 'deepseek']
    for model in models:
        if model == 'claude' and not client.providers_available['anthropic']:
            print(f"\nSkipping {model} (no API key)")
            continue
            
        print(f"\nTesting {model}:")
        response = await client.generate_enhanced_response(
            f"Say 'Hello from {model}' and nothing else",
            model=model
        )
        print(f"  Provider: {response.get('provider', 'unknown')}")
        print(f"  Success: {'error' not in response}")
        if 'error' in response:
            print(f"  Error: {response['error'][:100]}")
    
    # Test 3: Auto mode (should fallback on errors)
    print("\n\nTest 3: Auto-fallback mode")
    print("-" * 30)
    
    # Temporarily disable round-robin for this test
    client2 = HTTPAIClient(enable_round_robin=False)
    response = await client2.generate_enhanced_response(
        "What is 2+2? Answer with just the number."
    )
    print(f"Auto mode result:")
    print(f"  Provider: {response.get('provider', 'unknown')}")
    print(f"  Content: {response.get('content', 'No content')}")
    print(f"  Success: {'error' not in response}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    load_dotenv('.env')
    
    asyncio.run(test_fallback())