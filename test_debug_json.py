#!/usr/bin/env python3
"""
Debug JSON parsing in Gemini Deep Research.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.ai_brain import IntelligentAIBrain


async def test_json_parsing():
    """Test JSON parsing from Gemini responses."""
    ai_brain = IntelligentAIBrain()
    
    # Simple test prompt
    prompt = """
    List 3 market opportunities in JSON format:
    [
        {
            "problem_description": "Small businesses struggle with inventory management",
            "affected_audience": "Small retail businesses",
            "market_size": "Large",
            "competition_level": "Medium"
        }
    ]
    
    Provide exactly 3 opportunities as a JSON array.
    """
    
    print("Testing JSON response parsing...")
    response = await ai_brain.generate_enhanced_response(prompt, model='gemini')
    
    if response and 'content' in response:
        content = response['content']
        print(f"\nRaw response:\n{content}\n")
        
        import re
        import json
        
        # Try to find JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                print(f"✅ Successfully parsed {len(data)} items")
                for item in data:
                    print(f"  - {item.get('problem_description', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"Attempted to parse: {json_match.group()[:200]}...")
        else:
            print("❌ No JSON array found in response")


asyncio.run(test_json_parsing())