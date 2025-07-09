#!/usr/bin/env python3
"""Debug what AI returns for code generation"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.http_ai_client import HTTPAIClient
from scripts.evolutionary_code_generator_fix import RealCodeEvolutionFix

async def debug_ai_response():
    """Debug AI code generation response."""
    
    client = HTTPAIClient()
    fixer = RealCodeEvolutionFix()
    
    improvement = {
        'name': 'smart_cache_optimizer',
        'type': 'cache',
        'description': 'Intelligent cache with LRU eviction and statistics'
    }
    
    prompt = fixer.create_implementation_prompt(improvement)
    
    print("PROMPT SENT TO AI:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    response = await client.generate_enhanced_response(prompt)
    
    print("\nAI RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    # Try to extract code
    if isinstance(response, dict) and 'content' in response:
        content = response['content']
    else:
        content = str(response)
        
    print("\nEXTRACTED CONTENT:")
    print("=" * 80)
    print(content[:1000] if len(content) > 1000 else content)
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(debug_ai_response())