#!/usr/bin/env python3
"""
Test the Smart Conversational Assistant with Dynamic Context

This script tests that the assistant can answer questions with real system data.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant, ResponseStyle


async def test_assistant():
    """Test the conversational assistant with various queries."""
    print("Testing Smart Conversational Assistant")
    print("=" * 50)
    
    # Create assistant
    assistant = ConversationalAIAssistant(style=ResponseStyle.FRIENDLY_PROFESSIONAL)
    
    # Initialize
    print("\nInitializing assistant...")
    await assistant.initialize()
    print("✓ Assistant ready")
    
    # Test queries that should use real context
    test_queries = [
        "What is 2+2?",  # Simple test
        "Is the system currently running?",  # Should check actual status
        "Show me the repositories in our system",  # Should list actual repos
        "What can you do?",  # Should list actual capabilities
        "List active tasks",  # Should show real tasks
        "What's the system status?",  # Should show real status
        "How many repositories do we have?",  # Should count actual repos
        "Is the continuous AI running?",  # Should check continuous AI
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("-" * 50)
        
        try:
            response = await assistant.handle_conversation(query)
            print(f"Response: {response[:500]}...")  # Limit output length
            
            # Check if response uses generic language vs specific data
            generic_phrases = ["typically", "usually", "would", "should", "might"]
            specific_phrases = ["currently", "actual", "found", "running", "active"]
            
            generic_count = sum(1 for phrase in generic_phrases if phrase in response.lower())
            specific_count = sum(1 for phrase in specific_phrases if phrase in response.lower())
            
            print(f"\nAnalysis: Generic phrases: {generic_count}, Specific phrases: {specific_count}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("Test complete!")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    load_dotenv('.env')
    
    # Run test
    asyncio.run(test_assistant())