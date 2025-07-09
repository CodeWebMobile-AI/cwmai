#!/usr/bin/env python3
"""
Proper CWMAI implementation that separates concerns between
interactive mode and background/silent mode.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def setup_interactive_environment():
    """Set up environment for interactive use - no silent logging."""
    # Don't redirect stdout/stderr
    # Use normal console logging
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(message)s'
    )
    
    # Suppress noisy loggers but keep console output
    for logger_name in ['urllib3', 'asyncio', 'redis']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

async def run_interactive():
    """Run in interactive mode with proper terminal support."""
    from scripts.conversational_ai_assistant import ConversationalAIAssistant, ResponseStyle
    from rich.console import Console
    from rich.markdown import Markdown
    
    console = Console()
    
    # Create assistant
    assistant = ConversationalAIAssistant(style=ResponseStyle.FRIENDLY_PROFESSIONAL)
    
    # Assistant is ready after creation
    print("AI assistant ready!\n")
    
    # Simple input loop without prompt_toolkit
    while True:
        try:
            # Use simple input
            user_input = input("You > ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
            
            # Get response
            print("Assistant > Thinking...", end='\r')
            response = await assistant.handle_conversation(user_input)
            
            # Clear the "Thinking..." message and display response
            print(" " * 30, end='\r')  # Clear line
            console.print("[blue]Assistant >[/blue]", Markdown(response))
            print()  # Extra line for readability
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

async def run_background(command):
    """Run in background mode with silent logging."""
    from scripts.silent_logger import setup_silent_logging
    from scripts.conversational_ai_assistant import ConversationalAIAssistant, ResponseStyle
    
    # Set up silent logging for background operation
    setup_silent_logging()
    
    # Create assistant
    assistant = ConversationalAIAssistant(style=ResponseStyle.CONCISE)
    
    # Process command and return
    response = await assistant.handle_conversation(command)
    
    # Output only the response
    print(response)

def main():
    """Main entry point with proper mode separation."""
    parser = argparse.ArgumentParser(description="CWMAI - Conversational AI Assistant")
    parser.add_argument('command', nargs='*', help='Command to execute (interactive if omitted)')
    parser.add_argument('--background', action='store_true', help='Run in background mode')
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    load_dotenv('.env')
    
    # Check for required API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found in environment")
        return 1
    
    try:
        if args.command:
            # Single command mode
            command = ' '.join(args.command)
            if args.background:
                asyncio.run(run_background(command))
            else:
                # Run command with minimal output
                setup_interactive_environment()
                asyncio.run(run_background(command))
        else:
            # Interactive mode
            setup_interactive_environment()
            asyncio.run(run_interactive())
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())