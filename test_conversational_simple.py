#!/usr/bin/env python3
"""Simple test of conversational AI assistant."""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.conversational_ai_assistant import ConversationalAIAssistant

async def test_assistant():
    """Test the conversational assistant."""
    print("Creating conversational AI assistant...")
    assistant = ConversationalAIAssistant()
    
    print("Initializing assistant...")
    await assistant.initialize()
    
    print("\nTesting conversation...")
    
    # Test 1: Greeting
    response = await assistant.handle_conversation("Hi there!")
    print(f"\nUser: Hi there!")
    print(f"Assistant: {response}")
    
    # Test 2: Question about capabilities
    response = await assistant.handle_conversation("What can you help me with?")
    print(f"\nUser: What can you help me with?")
    print(f"Assistant: {response[:200]}...")  # Truncate for display
    
    # Test 3: Simple command
    response = await assistant.handle_conversation("Show me the system status")
    print(f"\nUser: Show me the system status")
    print(f"Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(test_assistant())