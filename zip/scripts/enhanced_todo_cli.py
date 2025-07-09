#!/usr/bin/env python3
"""
Enhanced Natural Language Todo CLI with Streaming Support

Provides Claude Code-like experience with:
- Character-by-character streaming output
- Concurrent input handling
- Visual state indicators
- Queue status display
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from natural_language_todo_queue import ConcurrentTodoQueue, MessagePriority
from todo_processor_pool import DynamicProcessorPool
from todo_nlp_processor import TodoNLPProcessor, TodoIntent
from natural_language_todo_manager import NaturalLanguageTodoManager
from streaming_response_generator import (
    StreamingResponseGenerator, 
    VisualStateIndicator,
    InterruptHandler
)


class EnhancedTodoCLI:
    """Enhanced CLI with streaming and concurrent input support."""
    
    def __init__(self):
        """Initialize the enhanced CLI."""
        self.queue = ConcurrentTodoQueue()
        self.nlp = TodoNLPProcessor(use_ai=False)
        self.manager = NaturalLanguageTodoManager()
        self.streamer = StreamingResponseGenerator()
        self.indicator = VisualStateIndicator()
        self.interrupt_handler = InterruptHandler()
        
        # Processing pool
        self.processor_pool = None
        
        # State tracking
        self.is_processing = False
        self.pending_inputs = []
        
        # Response templates
        self.responses = {
            TodoIntent.ADD: [
                "Adding that to your todo list...",
                "I'll add that task for you...",
                "Creating a new todo item...",
            ],
            TodoIntent.LIST: [
                "Here are your current tasks:",
                "Let me show you what's on your list:",
                "Your todo items:",
            ],
            TodoIntent.COMPLETE: [
                "Marking that task as complete...",
                "Great job! I'll mark that as done...",
                "Completing that task for you...",
            ],
            TodoIntent.DELETE: [
                "Removing that task...",
                "I'll delete that for you...",
                "Deleting the task...",
            ],
            TodoIntent.UPDATE: [
                "Updating that task...",
                "I'll modify that for you...",
                "Making those changes...",
            ],
            TodoIntent.SEARCH: [
                "Searching for matching tasks...",
                "Let me find that for you...",
                "Looking through your tasks...",
            ]
        }
    
    async def start(self):
        """Start the enhanced CLI."""
        # Clear screen and show welcome
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🤖 Natural Language Todo System")
        print("=" * 50)
        print("Type naturally to manage your todos.")
        print("You can type while I'm responding!")
        print("Type 'exit' or 'quit' to leave.\n")
        
        # Start processor pool
        self.processor_pool = DynamicProcessorPool(
            self.queue, 
            self.process_todo_command,
            min_processors=1,
            max_processors=3
        )
        await self.processor_pool.start()
        
        # Start input loop
        await self.run_input_loop()
    
    async def run_input_loop(self):
        """Main input loop with concurrent handling."""
        input_task = None
        
        try:
            while True:
                # Show queue status if there are pending items
                status = self.queue.get_status()
                if status['pending'] > 0:
                    await self.indicator.show_queue_status(
                        status['pending'], 
                        status['processing']
                    )
                
                # Get input
                if not self.is_processing:
                    sys.stdout.write("> ")
                    sys.stdout.flush()
                
                # Create input task if needed
                if input_task is None or input_task.done():
                    input_task = asyncio.create_task(self.get_user_input())
                
                # Check for input with timeout
                try:
                    user_input = await asyncio.wait_for(input_task, timeout=0.1)
                    
                    if user_input.lower() in ['exit', 'quit']:
                        print("\n👋 Goodbye!")
                        break
                    
                    # Queue the input
                    await self.queue_todo_input(user_input)
                    
                except asyncio.TimeoutError:
                    # No input yet, continue loop
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        finally:
            if self.processor_pool:
                await self.processor_pool.stop()
    
    async def get_user_input(self) -> str:
        """Get user input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input)
    
    async def queue_todo_input(self, user_input: str):
        """Queue a todo command for processing."""
        # Parse intent first
        command = await self.nlp.process(user_input)
        
        # Determine priority
        priority = MessagePriority.NORMAL
        if command.entities.get('priority') == 'urgent':
            priority = MessagePriority.URGENT
        elif command.entities.get('priority') == 'high':
            priority = MessagePriority.HIGH
        
        # Queue the message
        message = self.queue.enqueue(user_input, priority)
        
        # Show queued indicator if we're processing something else
        if self.is_processing:
            print(f"\n📋 Command queued (ID: {message.id[:8]})")
    
    async def process_todo_command(self, message):
        """Process a todo command with streaming output."""
        self.is_processing = True
        
        try:
            # Parse the command
            command = await self.nlp.process(message.content)
            
            # Get appropriate response template
            responses = self.responses.get(command.intent, ["Processing your request..."])
            response_intro = responses[0]
            
            # Stream the intro
            await self.streamer.stream_response(response_intro)
            
            # Process based on intent
            result = await self.execute_command(command, message.content)
            
            # Stream the result
            if result:
                await self.streamer.stream_response(result, indicator="✅")
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            await self.streamer.stream_response(error_msg, indicator="❌")
        
        finally:
            self.is_processing = False
            
            # Check queue status after processing
            status = self.queue.get_status()
            if status['pending'] > 0:
                print(f"📋 {status['pending']} more command(s) in queue")
    
    async def execute_command(self, command, original_text: str) -> Optional[str]:
        """Execute the parsed command."""
        intent = command.intent
        entities = command.entities
        
        if intent == TodoIntent.ADD:
            # Extract task description
            task = entities.get('task', original_text)
            priority = entities.get('priority', 'medium')
            due_date = entities.get('due_date')
            
            todo = self.manager.add_todo(
                description=task,
                priority=priority,
                due_date=due_date
            )
            
            return f"Added: [{todo.priority}] {todo.description}"
        
        elif intent == TodoIntent.LIST:
            todos = self.manager.list_todos()
            if not todos:
                return "Your todo list is empty!"
            
            # Format todos
            lines = []
            for i, todo in enumerate(todos, 1):
                status_icon = "✅" if todo.status.value == "completed" else "⭕"
                lines.append(f"{i}. {status_icon} [{todo.priority}] {todo.description}")
            
            return "\n".join(lines)
        
        elif intent == TodoIntent.COMPLETE:
            reference = entities.get('task_reference', original_text)
            completed = self.manager.complete_todo(reference)
            
            if completed:
                return f"Completed: {completed.description}"
            else:
                return f"Couldn't find a task matching '{reference}'"
        
        elif intent == TodoIntent.DELETE:
            reference = entities.get('task_reference', original_text)
            deleted = self.manager.delete_todo(reference)
            
            if deleted:
                return f"Deleted: {deleted.description}"
            else:
                return f"Couldn't find a task matching '{reference}'"
        
        elif intent == TodoIntent.UPDATE:
            # This would need more sophisticated parsing
            return "Update functionality coming soon!"
        
        elif intent == TodoIntent.SEARCH:
            query = entities.get('query', original_text)
            results = self.manager.search_todos(query)
            
            if not results:
                return f"No tasks found matching '{query}'"
            
            lines = [f"Found {len(results)} matching task(s):"]
            for todo in results:
                status_icon = "✅" if todo.status.value == "completed" else "⭕"
                lines.append(f"- {status_icon} [{todo.priority}] {todo.description}")
            
            return "\n".join(lines)
        
        else:
            return "I'm not sure what you want me to do. Try 'add task', 'show tasks', or 'complete task'."


async def demo_concurrent_input():
    """Demonstrate concurrent input handling."""
    print("=== Concurrent Input Demo ===\n")
    
    cli = EnhancedTodoCLI()
    
    # Simulate streaming a response
    print("> add a task to review documentation")
    response_task = asyncio.create_task(
        cli.streamer.stream_response(
            "I'm adding that task to your todo list. This might take a moment while I process the details and update your task database...",
            indicator="🤖"
        )
    )
    
    # Simulate user typing while streaming
    await asyncio.sleep(1.5)  # User starts typing while response is streaming
    print("\n> show me my tasks")
    print("📋 Command queued")
    
    # Wait for first response to complete
    await response_task
    
    # Process queued command
    await cli.streamer.stream_response(
        "Here are your current tasks:\n1. ⭕ [medium] Review documentation",
        indicator="🤖"
    )


async def main():
    """Run the enhanced CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        await demo_concurrent_input()
    else:
        cli = EnhancedTodoCLI()
        await cli.start()


if __name__ == "__main__":
    asyncio.run(main())