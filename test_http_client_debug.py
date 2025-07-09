#!/usr/bin/env python3
"""
Debug HTTP client structured output.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.http_ai_client import HTTPAIClient
from scripts.structured_output_handler import StructuredOutputHandler


async def test_direct():
    """Test the actual method calls."""
    
    client = HTTPAIClient()
    handler = StructuredOutputHandler()
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    
    # Test what the handler returns
    config = handler.prepare_structured_output('gemini', "Extract: John is 25", schema)
    print("Handler config:")
    print(json.dumps(config, indent=2))
    
    # Test what we're sending to Gemini
    print("\n" + "="*50)
    
    # Manually call the Gemini method with schema
    response = await client._call_gemini_http(
        "Extract: John is 25",
        "test_req",
        None,  # model_override
        schema  # response_schema
    )
    
    print("\nDirect call response:")
    print(f"Content: {response.get('content')}")
    print(f"Structured output detected: {'parsed_content' in response}")


if __name__ == "__main__":
    asyncio.run(test_direct())