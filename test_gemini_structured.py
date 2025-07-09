#!/usr/bin/env python3
"""
Debug script for Gemini structured output.
"""

import aiohttp
import asyncio
import json
import os
import ssl


async def test_gemini_direct():
    """Test Gemini API directly with structured output."""
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not set")
        return
    
    # Simple schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    
    prompt = "Extract: John is 25 years old."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent?key={api_key}"
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": schema
        }
    }
    
    print("Request data:")
    print(json.dumps(data, indent=2))
    
    # Create SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, json=data) as response:
                print(f"\nStatus: {response.status}")
                result = await response.json()
                
                if response.status == 200:
                    print("\nSuccess! Response:")
                    print(json.dumps(result, indent=2))
                    
                    # Extract content
                    if result.get("candidates"):
                        content = result["candidates"][0]["content"]["parts"][0]["text"]
                        print(f"\nExtracted content: {content}")
                        
                        # Parse JSON
                        parsed = json.loads(content)
                        print(f"\nParsed JSON: {parsed}")
                else:
                    print(f"\nError: {result}")
                    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing Gemini Structured Output Directly")
    print("=" * 50)
    asyncio.run(test_gemini_direct())