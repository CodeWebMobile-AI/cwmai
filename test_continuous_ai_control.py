#!/usr/bin/env python3
"""
Test script for the continuous AI control capabilities in the conversational assistant.
"""

import asyncio
import logging
from scripts.conversational_ai_assistant import ConversationalAIAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_continuous_ai_control():
    """Test the continuous AI control features."""
    # Create assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    print("=== Testing Continuous AI Control ===\n")
    
    # Test commands
    test_inputs = [
        "Is the continuous AI system running?",
        "Start the continuous AI system with 3 workers",
        "Check the continuous AI status",
        "Monitor the continuous AI health",
        "Stop the continuous AI system",
        "What's the status of everything?"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        print("-" * 50)
        
        response = await assistant.handle_conversation(user_input)
        print(f"🤖 Assistant: {response}")
        
        # Wait a bit between commands
        await asyncio.sleep(2)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_continuous_ai_control())