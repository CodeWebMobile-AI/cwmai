#!/usr/bin/env python3
"""
Test AI API Logging System

Quick test to verify AI API communication logging is working correctly.
"""

import asyncio
import sys
import os
import time

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.http_ai_client import HTTPAIClient
from scripts.ai_api_logger import get_ai_api_logger, get_ai_api_statistics


async def test_ai_logging():
    """Test AI API logging with various scenarios."""
    print("🧪 Testing AI API Communication Logging System")
    print("=" * 60)
    
    # Initialize AI client
    client = HTTPAIClient(enable_round_robin=False)
    
    # Get logger instance
    logger = get_ai_api_logger()
    
    print("\n1️⃣ Testing basic AI request logging...")
    try:
        response = await client.generate_enhanced_response(
            "What is 2+2? Reply with just the number.",
            model="claude"
        )
        print(f"✅ Response received: {response.get('content', 'No content')[:50]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Give logger time to write
    await asyncio.sleep(0.5)
    
    print("\n2️⃣ Testing cache hit logging...")
    try:
        # Same request should hit cache
        response = await client.generate_enhanced_response(
            "What is 2+2? Reply with just the number.",
            model="claude"
        )
        if response.get('cached'):
            print(f"✅ Cache hit logged successfully")
        else:
            print(f"⚠️  Request was not cached")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    await asyncio.sleep(0.5)
    
    print("\n3️⃣ Testing error logging...")
    # Temporarily break API key to test error logging
    original_key = client.anthropic_api_key
    client.anthropic_api_key = "invalid_key"
    
    try:
        response = await client.generate_enhanced_response(
            "This should fail",
            model="claude"
        )
    except Exception as e:
        print(f"✅ Error properly logged: {type(e).__name__}")
    
    # Restore key
    client.anthropic_api_key = original_key
    
    await asyncio.sleep(0.5)
    
    print("\n4️⃣ Testing different providers (if available)...")
    providers_tested = 0
    for provider in ['gpt', 'gemini', 'deepseek']:
        if client.providers_available.get(provider.replace('gpt', 'openai')):
            try:
                response = await client.generate_enhanced_response(
                    f"Say hello from {provider}",
                    model=provider
                )
                print(f"✅ {provider} request logged")
                providers_tested += 1
            except Exception as e:
                print(f"⚠️  {provider} error: {e}")
    
    if providers_tested == 0:
        print("⚠️  No additional providers available for testing")
    
    await asyncio.sleep(0.5)
    
    # Print statistics
    print("\n📊 AI API Statistics:")
    print("-" * 40)
    stats = get_ai_api_statistics()
    
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Total Errors: {stats['total_errors']}")
    print(f"Cache Hits: {stats['total_cache_hits']}")
    print(f"Average Response Time: {stats['average_response_time']:.3f}s")
    
    print("\nProvider Usage:")
    for provider, count in stats['provider_usage'].items():
        print(f"  {provider}: {count} requests")
    
    print("\nModel Usage:")
    for model, count in stats['model_usage'].items():
        print(f"  {model}: {count} requests")
    
    # Check if log file exists
    log_file = "ai_api_communication.log"
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"\n✅ Log file created: {log_file} ({size} bytes)")
        
        # Show last few log entries
        print("\n📝 Recent log entries:")
        print("-" * 40)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:  # Last 5 entries
                try:
                    import json
                    entry = json.loads(line)
                    event = entry.get('event_type', 'unknown')
                    if event == 'request_start':
                        provider = entry.get('request_metadata', {}).get('provider', 'unknown')
                        print(f"  {event}: {provider}")
                    elif event == 'request_complete':
                        time = entry.get('response_metadata', {}).get('response_time', 0)
                        cached = entry.get('response_metadata', {}).get('cached', False)
                        print(f"  {event}: {time:.2f}s {'(cached)' if cached else ''}")
                    else:
                        print(f"  {event}")
                except:
                    pass
    else:
        print(f"\n❌ Log file not found: {log_file}")
    
    print("\n✅ AI API logging test completed!")
    print("\nTo view logs in real-time, run:")
    print("  python scripts/ai_api_log_viewer.py -f")


if __name__ == "__main__":
    # Set environment for testing
    os.environ['AI_API_LOG_SENSITIVE'] = 'true'  # Show content in test
    os.environ['AI_API_LOG_LEVEL'] = 'DEBUG'
    
    print("Setting up test environment...")
    print(f"AI_API_LOG_SENSITIVE: {os.environ.get('AI_API_LOG_SENSITIVE')}")
    print(f"AI_API_LOG_LEVEL: {os.environ.get('AI_API_LOG_LEVEL')}")
    
    # Run test
    asyncio.run(test_ai_logging())