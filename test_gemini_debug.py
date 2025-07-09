#!/usr/bin/env python3
"""
Debug Gemini structured output issue.
"""

import asyncio
import json
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from scripts.http_ai_client import HTTPAIClient


async def test_gemini_structured():
    """Test Gemini with structured output."""
    
    client = HTTPAIClient()
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    
    prompt = "Extract: John is 25 years old."
    
    print("Testing Gemini with structured output...")
    print("Schema:", json.dumps(schema, indent=2))
    print("Prompt:", prompt)
    print("-" * 50)
    
    response = await client.generate_enhanced_response(
        prompt,
        model='gemini',
        response_schema=schema
    )
    
    print("\nResponse:")
    print(f"Provider: {response.get('provider')}")
    print(f"Structured Output: {response.get('structured_output', False)}")
    print(f"Content: {response.get('content')}")
    
    if 'parsed_content' in response:
        print(f"Parsed Content: {response['parsed_content']}")


if __name__ == "__main__":
    asyncio.run(test_gemini_structured())