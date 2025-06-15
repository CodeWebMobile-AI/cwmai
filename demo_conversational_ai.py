#!/usr/bin/env python3
"""
Demo script for CWMAI Conversational AI Assistant

Shows various conversation examples and capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant

async def demo_conversation():
    """Run a demo conversation."""
    print("=== CWMAI Conversational AI Demo ===\n")
    
    # Create assistant
    assistant = ConversationalAIAssistant()
    
    # Initialize (suppress warnings)
    try:
        await assistant.initialize()
    except:
        pass
    
    # Demo conversations
    conversations = [
        ("Hello! What can you do for me?", "Greeting and capabilities"),
        ("Can you show me the system status?", "Status request"),
        ("Create an issue for the auth-api about improving error messages", "Issue creation"),
        ("Search for Python testing frameworks", "Search request"),
        ("Thanks, that was helpful!", "Feedback"),
    ]
    
    for user_input, description in conversations:
        print(f"\n--- {description} ---")
        print(f"👤 User: {user_input}")
        
        response = await assistant.handle_conversation(user_input)
        
        # Truncate long responses for demo
        if len(response) > 300:
            response = response[:300] + "..."
        
        print(f"🤖 Assistant: {response}")
        print("-" * 50)
    
    # Show conversation summary
    print("\n=== Conversation Summary ===")
    print(assistant.get_conversation_summary())

if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    asyncio.run(demo_conversation())