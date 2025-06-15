#!/usr/bin/env python3
"""
Test the conversational AI's ability to count repositories.
"""

import asyncio
import sys
import os

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from conversational_ai_assistant import ConversationalAIAssistant


async def main():
    """Test conversational AI repository counting."""
    print("Testing Conversational AI repository counting...")
    print("-" * 50)
    
    # Initialize the assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Test queries
    queries = [
        "How many repositories are you managing?",
        "Count the repositories",
        "Show me a breakdown of repositories by language"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        response = await assistant.handle_conversation(query)
        print("Response:")
        print(response)
        
        # Since handle_conversation returns a string, we can't check tool calls directly
        # But we can verify the response mentions the correct number
        if "12" in response or "twelve" in response.lower():
            print("\n✅ Response correctly mentions 12 repositories!")
        elif "0" in response or "zero" in response.lower() or "no repo" in response.lower():
            print("\n❌ Response incorrectly says 0 repositories!")
        else:
            print("\n⚠️  Could not determine repository count from response")


if __name__ == "__main__":
    asyncio.run(main())