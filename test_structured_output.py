#!/usr/bin/env python3
"""
Test script for structured output implementation across all AI providers.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.ai_brain import IntelligentAIBrain


async def test_structured_output():
    """Test structured output with all providers."""
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    # Define a test schema
    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "employed": {"type": "boolean"}
        },
        "required": ["name", "age", "city", "skills", "employed"]
    }
    
    # Test prompt
    prompt = """Extract the person's information from this text:
    
    John Doe is a 28-year-old software engineer living in San Francisco. 
    He is currently employed and has skills in Python, JavaScript, and Docker.
    """
    
    # Test each provider
    providers = ['gemini', 'openai', 'deepseek', 'anthropic']
    
    print("=" * 80)
    print("STRUCTURED OUTPUT TEST - All Providers")
    print("=" * 80)
    
    for provider in providers:
        print(f"\n\n{'='*40}")
        print(f"Testing {provider.upper()}")
        print(f"{'='*40}")
        
        try:
            # Use provider name as model hint
            model = None
            if provider == 'anthropic':
                model = 'claude'
            elif provider == 'openai':
                model = 'gpt'
            elif provider == 'deepseek':
                model = 'deepseek'
            elif provider == 'gemini':
                model = 'gemini'
            
            # Make request with structured output
            response = await ai_brain.generate_enhanced_response(
                prompt, 
                model=model,
                response_schema=test_schema
            )
            
            print(f"\nProvider: {response.get('provider')}")
            print(f"Model: {response.get('model')}")
            print(f"Structured Output: {response.get('structured_output', False)}")
            print(f"Confidence: {response.get('confidence')}")
            
            if response.get('error'):
                print(f"Error: {response['error']}")
            
            # Check if we got JSON
            content = response.get('content', '')
            if content:
                try:
                    parsed = json.loads(content)
                    print(f"\n✅ Successfully parsed JSON:")
                    print(json.dumps(parsed, indent=2))
                    
                    # Validate against schema
                    print("\nValidation:")
                    for field in test_schema['properties'].keys():
                        if field in parsed:
                            print(f"  ✓ {field}: {type(parsed[field]).__name__}")
                        else:
                            print(f"  ✗ {field}: MISSING")
                            
                except json.JSONDecodeError as e:
                    print(f"\n❌ Failed to parse JSON: {e}")
                    print(f"Raw content: {content[:200]}...")
            else:
                print("\n❌ No content in response")
                
            # Also check parsed_content if available
            if 'parsed_content' in response:
                print("\n📦 Pre-parsed content available:")
                print(json.dumps(response['parsed_content'], indent=2))
                
        except Exception as e:
            print(f"\n❌ Error testing {provider}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


async def test_complex_schema():
    """Test a more complex nested schema."""
    
    ai_brain = IntelligentAIBrain()
    
    # Complex schema with nested objects
    complex_schema = {
        "type": "object",
        "properties": {
            "company": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "founded": {"type": "integer"},
                    "employees": {"type": "integer"}
                },
                "required": ["name", "founded"]
            },
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "features": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "price"]
                }
            },
            "revenue": {"type": "number"},
            "profitable": {"type": "boolean"}
        },
        "required": ["company", "products"]
    }
    
    prompt = """Extract company information from this text:
    
    TechCorp was founded in 2018 and now has 150 employees. They offer two main products:
    1. CloudSync ($49.99/month) - features include real-time sync, encryption, and collaboration
    2. DataVault ($99.99/month) - features include backup, recovery, and compliance tools
    
    The company reported $5.2 million in revenue last year and is now profitable.
    """
    
    print("\n\n" + "="*80)
    print("COMPLEX SCHEMA TEST - Gemini Only")
    print("="*80)
    
    try:
        response = await ai_brain.generate_enhanced_response(
            prompt,
            model='gemini',
            response_schema=complex_schema
        )
        
        print(f"\nProvider: {response.get('provider')}")
        print(f"Structured Output: {response.get('structured_output', False)}")
        
        content = response.get('content', '')
        if content:
            print(f"\nRaw content ({len(content)} chars):")
            print(content[:500] + "..." if len(content) > 500 else content)
            
            try:
                parsed = json.loads(content)
                print("\n✅ Successfully parsed complex JSON:")
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError as e:
                print(f"\n❌ Failed to parse JSON: {e}")
                
            # Check if parsed_content is available
            if 'parsed_content' in response:
                print("\n📦 Pre-parsed content available:")
                print(json.dumps(response['parsed_content'], indent=2))
        else:
            print("\n❌ No content in response")
            
    except Exception as e:
        print(f"\n❌ Error in complex schema test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run both tests
    asyncio.run(test_structured_output())
    asyncio.run(test_complex_schema())