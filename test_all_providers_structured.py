#!/usr/bin/env python3
"""
Test all providers with structured output.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.ai_brain import IntelligentAIBrain


async def test_all_providers():
    """Test structured output with all providers."""
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    # Simple test schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
    }
    
    prompt = "Extract: John Doe is 30 years old and lives in New York."
    
    providers = [
        ('gemini', 'Gemini'),
        ('gpt', 'OpenAI'),
        ('deepseek', 'DeepSeek'),
        ('claude', 'Anthropic')
    ]
    
    print("=" * 80)
    print("STRUCTURED OUTPUT TEST - ALL PROVIDERS")
    print("=" * 80)
    print(f"\nPrompt: {prompt}")
    print(f"Schema: {json.dumps(schema, indent=2)}")
    print("\n" + "=" * 80)
    
    results = {}
    
    for model, provider_name in providers:
        print(f"\n\n--- Testing {provider_name} ({model}) ---")
        
        try:
            response = await ai_brain.generate_enhanced_response(
                prompt,
                model=model,
                response_schema=schema
            )
            
            results[provider_name] = {
                'success': True,
                'structured_output': response.get('structured_output', False),
                'content': response.get('content'),
                'parsed_content': response.get('parsed_content'),
                'provider': response.get('provider'),
                'model': response.get('model')
            }
            
            print(f"✅ Success!")
            print(f"Provider: {response.get('provider')}")
            print(f"Model: {response.get('model')}")
            print(f"Structured Output: {response.get('structured_output', False)}")
            
            if 'parsed_content' in response:
                print(f"Parsed Content: {json.dumps(response['parsed_content'], indent=2)}")
            else:
                print(f"Raw Content: {response.get('content')}")
                
        except Exception as e:
            results[provider_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ Error: {e}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for provider_name, result in results.items():
        if result['success']:
            status = "✅" if result.get('structured_output') else "⚠️"
            print(f"{status} {provider_name}: {'Structured' if result.get('structured_output') else 'Regular'} output")
        else:
            print(f"❌ {provider_name}: Failed - {result['error']}")
    
    # Check if all providers support structured output
    all_structured = all(
        r.get('structured_output', False) 
        for r in results.values() 
        if r['success']
    )
    
    print(f"\n{'✅' if all_structured else '⚠️'} All providers using structured output: {all_structured}")


if __name__ == "__main__":
    asyncio.run(test_all_providers())