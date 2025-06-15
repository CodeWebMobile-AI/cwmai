#!/usr/bin/env python3
"""
Run the CWMAI Conversational AI Assistant

A simple launcher that ensures all environment variables are loaded
and the conversational AI assistant runs properly.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging before any other imports
from scripts.conversational_ai_logger import setup_conversational_ai_logging
setup_conversational_ai_logging()

def load_env_file(env_path: Path):
    """Load environment variables from a .env file."""
    if not env_path.exists():
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                value = value.strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                os.environ[key.strip()] = value

async def main():
    """Main entry point."""
    # Load environment variables
    load_env_file(Path('.env.local'))
    load_env_file(Path('.env'))
    
    # Check for required variables
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Please set it in .env.local or .env file")
        return 1
    
    # Import and run the assistant
    from scripts.conversational_ai_assistant import ConversationalAIAssistant
    
    print("🤖 CWMAI Conversational AI Assistant")
    print("=" * 40)
    print("Chat naturally with me! I can help you:")
    print("• Start/stop the AI system")
    print("• Check how things are running")
    print("• Reset when things go wrong")
    print("• Create issues and tasks")
    print("• Search for repositories")
    print("• And much more!")
    print("\nJust talk naturally - I understand phrases like:")
    print('"Fire up the system" or "How are things going?"')
    print("\nType 'exit' or 'quit' to leave.")
    print("=" * 40)
    print()
    
    # Create assistant
    assistant = ConversationalAIAssistant()
    
    # Initialize
    try:
        await assistant.initialize()
    except Exception as e:
        print(f"⚠️  Warning during initialization: {e}")
        print("Continuing with limited functionality...\n")
    
    # Run interactive loop
    while True:
        try:
            # Get user input
            user_input = input("You > ")
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n👋 Goodbye! Thanks for using CWMAI.")
                break
            
            if not user_input.strip():
                continue
            
            # Process with assistant
            print("Assistant > ", end='', flush=True)
            response = await assistant.handle_conversation(user_input)
            print(response)
            print()  # Extra line for readability
            
        except KeyboardInterrupt:
            print("\n\nUse 'exit' to quit properly.")
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or rephrase your request.\n")

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)