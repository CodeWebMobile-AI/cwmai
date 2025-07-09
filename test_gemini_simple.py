#!/usr/bin/env python3
"""
Simple test of Gemini JSON response.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.ai_brain import IntelligentAIBrain


async def test_simple():
    """Test simple Gemini request."""
    print("Testing simple Gemini request...")
    
    ai_brain = IntelligentAIBrain()
    
    prompt = """
    List 3 business problems as a JSON array:
    [
        {
            "problem": "description",
            "audience": "who it affects"
        }
    ]
    
    Return ONLY the JSON array, no other text.
    """
    
    try:
        response = await ai_brain.generate_enhanced_response(prompt, model='gemini-pro')
        
        if response and 'content' in response:
            print(f"\nResponse:\n{response['content']}")
        else:
            print("No content in response")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple())