#!/usr/bin/env python3
"""Debug the list error in enhanced conversational assistant."""

import asyncio
import sys
import os
import traceback

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def debug_error():
    """Debug the specific error."""
    try:
        from scripts.enhanced_conversational_assistant import EnhancedConversationalAssistant
        
        print("Creating assistant instance...")
        assistant = EnhancedConversationalAssistant()
        
        print("Initializing assistant...")
        await assistant.initialize()
        
        print("Handling conversation...")
        # Add more debugging
        original_process_with_ai = assistant._process_with_ai
        
        async def debug_process_with_ai(*args, **kwargs):
            try:
                print(f"Debug: _process_with_ai called with args: {len(args)}")
                result = await original_process_with_ai(*args, **kwargs)
                print(f"Debug: _process_with_ai returned type: {type(result)}")
                return result
            except Exception as e:
                print(f"Debug: Error in _process_with_ai: {e}")
                traceback.print_exc()
                raise
        
        assistant._process_with_ai = debug_process_with_ai
        
        response = await assistant.handle_conversation('test')
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_error())