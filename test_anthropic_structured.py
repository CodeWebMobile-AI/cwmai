#!/usr/bin/env python3
"""
Test Anthropic structured output implementation.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.ai_brain import IntelligentAIBrain


async def test_anthropic_structured():
    """Test Anthropic with various structured output scenarios."""
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    print("=" * 80)
    print("ANTHROPIC STRUCTURED OUTPUT TEST")
    print("=" * 80)
    
    # Test 1: Simple schema
    print("\n\n--- Test 1: Simple Schema ---")
    simple_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
    }
    
    prompt1 = "Extract: Sarah Johnson is 32 years old and lives in Boston."
    
    response1 = await ai_brain.generate_enhanced_response(
        prompt1,
        model='claude',
        response_schema=simple_schema
    )
    
    print(f"Provider: {response1.get('provider')}")
    print(f"Model: {response1.get('model')}")
    print(f"Structured Output: {response1.get('structured_output', False)}")
    print(f"Content: {response1.get('content')}")
    
    if 'parsed_content' in response1:
        print(f"Parsed Content: {json.dumps(response1['parsed_content'], indent=2)}")
    
    # Test 2: Complex nested schema
    print("\n\n--- Test 2: Complex Nested Schema ---")
    complex_schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "firstName": {"type": "string"},
                    "lastName": {"type": "string"},
                    "contacts": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"}
                        }
                    }
                },
                "required": ["firstName", "lastName"]
            },
            "preferences": {
                "type": "array",
                "items": {"type": "string"}
            },
            "active": {"type": "boolean"}
        },
        "required": ["user", "active"]
    }
    
    prompt2 = """Extract user information:
    
    User: Michael Chen
    Email: mchen@example.com
    Phone: 555-0123
    Status: Active account
    Preferences: dark mode, email notifications, weekly digest
    """
    
    response2 = await ai_brain.generate_enhanced_response(
        prompt2,
        model='claude',
        response_schema=complex_schema
    )
    
    print(f"Provider: {response2.get('provider')}")
    print(f"Structured Output: {response2.get('structured_output', False)}")
    
    content = response2.get('content', '')
    if content:
        try:
            parsed = json.loads(content)
            print("✅ Successfully parsed complex JSON:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Raw content: {content[:200]}...")
    
    # Test 3: Without custom prefill (let handler manage it)
    print("\n\n--- Test 3: Standard Structured Output ---")
    
    response3 = await ai_brain.generate_enhanced_response(
        "Extract the product name and price: iPhone 15 Pro costs $999",
        model='claude',
        response_schema={
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "price": {"type": "number"}
            },
            "required": ["product", "price"]
        }
        # Don't provide custom prefill - let the handler manage it
    )
    
    print(f"Provider: {response3.get('provider')}")
    print(f"Structured Output: {response3.get('structured_output', False)}")
    print(f"Content: {response3.get('content')}")
    
    if 'parsed_content' in response3:
        print(f"Parsed Content: {json.dumps(response3['parsed_content'], indent=2)}")
    
    # Test 4: Array response
    print("\n\n--- Test 4: Array Schema ---")
    
    array_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "quantity": {"type": "integer"}
            },
            "required": ["item", "quantity"]
        }
    }
    
    prompt4 = "Shopping list: 3 apples, 2 bananas, 5 oranges"
    
    response4 = await ai_brain.generate_enhanced_response(
        prompt4,
        model='claude',
        response_schema=array_schema
    )
    
    print(f"Provider: {response4.get('provider')}")
    print(f"Array Content: {response4.get('content')}")
    
    print("\n\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_anthropic_structured())