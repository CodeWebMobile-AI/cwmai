#!/usr/bin/env python3
"""Test the enhanced conversational assistant with a question."""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_with_question():
    """Test the assistant with the original question."""
    from scripts.enhanced_conversational_assistant import EnhancedConversationalAssistant
    
    print("Testing Enhanced Conversational Assistant with question...")
    
    # Create assistant instance
    assistant = EnhancedConversationalAssistant()
    
    # The original question
    question = "whats up with our system? how many projects are we managing?"
    
    print(f"\nUser: {question}")
    
    # Initialize the assistant
    await assistant.initialize()
    
    # Process the question
    try:
        response = await assistant.handle_conversation(question)
        print(f"\nAssistant: {response}")
    except Exception as e:
        print(f"\nError processing question: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_with_question())