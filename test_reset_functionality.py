#!/usr/bin/env python3
"""Test the smart reset functionality of the conversational AI assistant."""

import asyncio
import logging
from scripts.conversational_ai_assistant import ConversationalAIAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_reset_conversations():
    """Test various reset-related conversations."""
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Test conversations
    test_inputs = [
        # Direct reset requests
        "Reset the system",
        "Clean up everything but keep the knowledge",
        "Clear all the logs and state files",
        
        # System problem scenarios
        "The system seems broken, can you fix it?",
        "Everything is stuck and not working",
        
        # Specific reset types
        "Do a full system reset",
        "Clear only the log files",
        "Reset the state but keep the logs",
        "Clear the cache",
        
        # Emergency scenarios
        "The system crashed, need emergency reset",
        
        # Analysis request
        "Should I reset the system?"
    ]
    
    print("=== Testing Conversational AI Reset Functionality ===\n")
    
    for test_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"USER: {test_input}")
        print(f"{'='*60}\n")
        
        response = await assistant.handle_conversation(test_input)
        print(f"ASSISTANT: {response}")
        
        # If there's a pending confirmation, simulate a "yes" response
        if "yes/no" in response.lower() or "would you like" in response.lower():
            print(f"\n{'='*60}")
            print(f"USER: yes")
            print(f"{'='*60}\n")
            
            response = await assistant.handle_conversation("yes")
            print(f"ASSISTANT: {response}")
        
        # Add a small delay between tests
        await asyncio.sleep(1)
    
    # Test the analysis function directly
    print(f"\n{'='*60}")
    print("=== Direct System Analysis ===")
    print(f"{'='*60}\n")
    
    analysis = await assistant.analyze_reset_need()
    print(f"Needs Reset: {analysis['needs_reset']}")
    print(f"Urgency: {analysis['urgency']}")
    print(f"Reasons: {analysis['reasons']}")
    print(f"Recommended Type: {analysis['recommended_type']}")
    print(f"Health Indicators: {analysis['health_indicators']}")

if __name__ == "__main__":
    asyncio.run(test_reset_conversations())