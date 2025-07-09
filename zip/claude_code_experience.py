#!/usr/bin/env python3
"""
Claude Code Experience Demo

Demonstrates the full Claude Code-like experience with:
- Streaming character-by-character output
- Concurrent input acceptance
- Visual state indicators
- Queue management
"""

import asyncio
import sys
import os
from pathlib import Path
import threading
import queue
import termios
import tty
import select

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from enhanced_conversational_assistant import EnhancedConversationalAssistant
from streaming_response_generator import StreamingResponseGenerator, VisualStateIndicator


class InteractiveInputHandler:
    """Handles non-blocking input while allowing streaming output."""
    
    def __init__(self):
        self.input_queue = queue.Queue()
        self.input_thread = None
        self.stop_flag = threading.Event()
        
    def start(self):
        """Start the input handler thread."""
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
    def stop(self):
        """Stop the input handler."""
        self.stop_flag.set()
        
    def _input_loop(self):
        """Background thread for capturing input."""
        while not self.stop_flag.is_set():
            try:
                # Check if input is available
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        self.input_queue.put(line)
            except:
                pass
                
    def get_input(self, timeout=0.1):
        """Get input if available, non-blocking."""
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class ClaudeCodeExperience:
    """Full Claude Code-like experience."""
    
    def __init__(self):
        self.assistant = None
        self.streamer = StreamingResponseGenerator(typing_speed=0.02)
        self.indicator = VisualStateIndicator()
        self.input_handler = InteractiveInputHandler()
        self.is_streaming = False
        self.pending_inputs = []
        
    async def initialize(self):
        """Initialize the system."""
        print("🚀 Initializing Claude Code Experience...")
        
        self.assistant = EnhancedConversationalAssistant()
        await self.assistant.initialize()
        
        # Enable streaming
        self.assistant.enable_streaming = True
        
        print("✅ System ready with full streaming support!\n")
        
    async def run(self):
        """Run the interactive experience."""
        await self.initialize()
        
        # Show instructions
        self._show_instructions()
        
        # Start input handler
        self.input_handler.start()
        
        try:
            while True:
                # Check for input
                user_input = self.input_handler.get_input()
                
                if user_input:
                    if user_input.lower() in ['exit', 'quit']:
                        print("\n👋 Goodbye!")
                        break
                        
                    # If currently streaming, queue the input
                    if self.is_streaming:
                        self.pending_inputs.append(user_input)
                        print(f"\n📋 Command queued: '{user_input[:50]}...'")
                        await self.indicator.show_queue_status(
                            len(self.pending_inputs), 
                            1 if self.is_streaming else 0
                        )
                    else:
                        # Process immediately
                        await self.process_input(user_input)
                
                # Process any pending inputs
                if not self.is_streaming and self.pending_inputs:
                    next_input = self.pending_inputs.pop(0)
                    await self.process_input(next_input)
                    
                await asyncio.sleep(0.05)  # Small delay for responsiveness
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        finally:
            self.input_handler.stop()
            if self.assistant.todo_processor_pool:
                await self.assistant.todo_processor_pool.stop()
    
    async def process_input(self, user_input: str):
        """Process user input with streaming."""
        self.is_streaming = True
        
        # Show input
        print(f"\n> {user_input}")
        
        # Show processing indicator
        spinner_task = asyncio.create_task(
            self.indicator.show_spinner("Processing your request")
        )
        
        try:
            # Get response from assistant
            response = await self.assistant.handle_conversation(user_input, stream_output=False)
            
            # Stop spinner
            self.indicator.stop_spinner()
            await asyncio.sleep(0.1)  # Brief pause
            
            # Stream the response
            if response:
                await self.streamer.stream_response(response, indicator="🤖")
            
            # Show completion indicator
            await self.indicator.show_state("completed", "")
            
        except Exception as e:
            self.indicator.stop_spinner()
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            await self.streamer.stream_response(error_msg, indicator="❌")
            
        finally:
            self.is_streaming = False
            
            # Show queue status if there are pending items
            if self.pending_inputs:
                print(f"\n📋 {len(self.pending_inputs)} command(s) queued")
    
    def _show_instructions(self):
        """Show usage instructions."""
        print("=" * 60)
        print("🤖 Claude Code Experience - Natural Language Todo System")
        print("=" * 60)
        print("\n✨ Features:")
        print("  • Character-by-character streaming output")
        print("  • Type while I'm responding - your input will be queued")
        print("  • Visual indicators show processing state")
        print("  • Natural language understanding - no syntax required")
        
        print("\n📝 Example commands:")
        print("  • 'Add a task to review the documentation'")
        print("  • 'I need to call the client tomorrow at 3pm'")
        print("  • 'Show me my tasks'")
        print("  • 'Mark the documentation task as complete'")
        print("  • 'What's on my plate for today?'")
        
        print("\n💡 Try typing while I'm responding to see concurrent input!")
        print("\nType 'exit' to quit.\n")


async def demo_concurrent_processing():
    """Demonstrate concurrent processing capabilities."""
    print("🎭 Concurrent Processing Demo")
    print("=" * 40)
    
    # Create components
    streamer = StreamingResponseGenerator(typing_speed=0.03)
    indicator = VisualStateIndicator()
    
    print("\n1️⃣ Starting to stream a response...")
    
    # Start streaming
    stream_task = asyncio.create_task(
        streamer.stream_response(
            "I'm processing your request. This is a longer response to demonstrate "
            "that you can type while I'm still outputting text. The system will queue "
            "your input and process it as soon as I'm done with this message.",
            indicator="🤖"
        )
    )
    
    # Simulate user typing while streaming
    await asyncio.sleep(1.5)
    print("\n> add urgent task: fix the bug")
    print("📋 Command queued (1 pending)")
    
    await asyncio.sleep(1.0)
    print("\n> show me all tasks")
    print("📋 Command queued (2 pending)")
    
    # Wait for streaming to complete
    await stream_task
    
    print("\n✅ First response complete. Processing queued commands...")
    
    # Process queued commands
    await asyncio.sleep(0.5)
    await streamer.stream_response("Adding urgent task: fix the bug...", indicator="🤖")
    
    await asyncio.sleep(0.5)
    await streamer.stream_response(
        "Here are your tasks:\n1. 🔴 [urgent] Fix the bug\n2. 🟢 [medium] Review documentation",
        indicator="🤖"
    )
    
    print("\n\n✅ Demo complete! This is how Claude Code handles concurrent input.")


async def main():
    """Run the experience or demo."""
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        await demo_concurrent_processing()
    else:
        experience = ClaudeCodeExperience()
        await experience.run()


if __name__ == "__main__":
    asyncio.run(main())