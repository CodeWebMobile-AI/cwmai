"""
Natural Language Todo CLI

Interactive command-line interface that mimics Claude Code's natural conversation style
for todo management. Allows concurrent message processing and natural language input.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import readline  # For command history
import signal

# Auto-install rich if needed
from module_installer import module_installer
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.columns import Columns
except ImportError:
    if module_installer.try_install('rich'):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.columns import Columns
    else:
        # Fallback to basic console if rich can't be installed
        print("Warning: Could not install 'rich' module. Using basic console output.")
        Console = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.natural_language_todo_queue import (
    ConcurrentTodoQueue, TodoMessage, MessagePriority, MessageStatus
)
from scripts.todo_processor_pool import DynamicProcessorPool
from scripts.todo_nlp_processor import TodoNLPProcessor, TodoIntent, ProcessedTodoCommand
from scripts.natural_language_todo_manager import (
    NaturalLanguageTodoManager, TodoItem, TodoStatus
)


class TodoConversationContext:
    """Maintains conversation context for natural interactions."""
    
    def __init__(self):
        self.last_command: Optional[ProcessedTodoCommand] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_listed_todos: List[TodoItem] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.clarification_pending: bool = False
        
    def add_turn(self, user_input: str, assistant_response: str, command: Optional[ProcessedTodoCommand] = None):
        """Add a conversation turn."""
        self.conversation_history.append({
            'timestamp': datetime.now(timezone.utc),
            'user': user_input,
            'assistant': assistant_response,
            'command': command
        })
        
        # Keep last 20 turns
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)


class NaturalLanguageTodoCLI:
    """Interactive CLI for natural language todo management."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()
        self.running = False
        
        # Core components
        self.queue = ConcurrentTodoQueue()
        self.nlp_processor = TodoNLPProcessor(use_ai=False)  # Start without AI
        self.todo_manager = NaturalLanguageTodoManager()
        self.context = TodoConversationContext()
        
        # Processor pool
        self.processor_pool = DynamicProcessorPool(
            self.queue,
            self._process_todo_message,
            min_processors=2,
            max_processors=5
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Input handling
        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        
    async def start(self):
        """Start the CLI."""
        self.running = True
        
        # Welcome message
        self._show_welcome()
        
        # Start processor pool
        await self.processor_pool.start()
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._input_handler()),
            asyncio.create_task(self._response_handler()),
            asyncio.create_task(self._status_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop the CLI."""
        self.running = False
        await self.processor_pool.stop()
        
    def _show_welcome(self):
        """Show welcome message."""
        welcome_text = """
[bold cyan]Natural Language Todo System[/bold cyan]
        
Just type what you want to do with your todos in natural language:
• "Add a task to review the documentation tomorrow"
• "Show me what I need to do today"
• "Mark the API task as done"
• "What's my highest priority task?"

Type 'help' for more examples or 'quit' to exit.
        """
        
        self.console.print(Panel(welcome_text, title="Welcome", border_style="cyan"))
        
    async def _input_handler(self):
        """Handle user input concurrently."""
        # Use a thread to handle blocking input
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Show prompt
                self.console.print("\n[bold green]>[/bold green] ", end="")
                
                # Get input in thread to not block
                user_input = await loop.run_in_executor(None, input)
                
                if not user_input.strip():
                    continue
                    
                # Check for quit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self.running = False
                    break
                    
                # Process the input
                await self._handle_user_input(user_input)
                
            except EOFError:
                # Handle Ctrl+D
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Input handler error: {e}")
                
    async def _handle_user_input(self, user_input: str):
        """Process user input and queue for processing."""
        # Show processing indicator
        self.console.print("[dim]Processing...[/dim]")
        
        # Add to queue with appropriate priority
        priority = MessagePriority.NORMAL
        
        # Boost priority for certain keywords
        if any(word in user_input.lower() for word in ['urgent', 'asap', 'immediately']):
            priority = MessagePriority.URGENT
        elif any(word in user_input.lower() for word in ['help', 'stuck', 'error']):
            priority = MessagePriority.HIGH
            
        # Queue the message
        message = self.queue.enqueue(user_input, priority)
        
        # Track for response handling
        await self.response_queue.put(message.id)
        
    async def _response_handler(self):
        """Handle responses as they complete."""
        while self.running:
            try:
                # Wait for a message ID to handle
                message_id = await asyncio.wait_for(
                    self.response_queue.get(),
                    timeout=1.0
                )
                
                # Poll for completion
                while True:
                    # Check if message is completed
                    if message_id in self.queue._completed:
                        message = self.queue._completed[message_id]
                        
                        # Display result
                        if message.status == MessageStatus.COMPLETED:
                            await self._display_result(message.result)
                        else:
                            self.console.print(f"[red]Error: {message.error}[/red]")
                            
                        break
                        
                    await asyncio.sleep(0.1)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Response handler error: {e}")
                
    async def _process_todo_message(self, message: TodoMessage) -> Dict[str, Any]:
        """Process a single todo message."""
        try:
            # Parse with NLP
            command = await self.nlp_processor.process(message.content)
            
            # Handle based on intent
            result = await self._execute_command(command)
            
            # Update context
            self.context.last_command = command
            self.context.last_result = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': "I encountered an error processing your request."
            }
            
    async def _execute_command(self, command: ProcessedTodoCommand) -> Dict[str, Any]:
        """Execute a parsed command."""
        intent = command.intent
        entities = command.entities
        
        # Check if clarification needed
        if command.needs_clarification:
            return {
                'success': False,
                'needs_clarification': True,
                'message': command.clarification_prompt,
                'suggestions': command.suggested_actions
            }
            
        # Execute based on intent
        if intent == TodoIntent.ADD:
            return await self._handle_add(entities)
            
        elif intent == TodoIntent.LIST:
            return await self._handle_list(entities)
            
        elif intent == TodoIntent.COMPLETE:
            return await self._handle_complete(entities)
            
        elif intent == TodoIntent.DELETE:
            return await self._handle_delete(entities)
            
        elif intent == TodoIntent.UPDATE:
            return await self._handle_update(entities)
            
        elif intent == TodoIntent.SEARCH:
            return await self._handle_search(entities)
            
        elif intent == TodoIntent.HELP:
            return await self._handle_help()
            
        else:
            return {
                'success': False,
                'message': "I'm not sure how to help with that. Try 'help' for examples."
            }
            
    async def _handle_add(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add todo command."""
        description = entities.get('task', 'Untitled task')
        priority = entities.get('priority', 'medium')
        due_date = None
        tags = entities.get('tags', [])
        
        # Parse due date if provided
        if 'due_date' in entities:
            try:
                due_date = datetime.fromisoformat(entities['due_date'])
            except:
                pass
                
        # Add the todo
        todo = self.todo_manager.add_todo(
            description=description,
            priority=priority,
            due_date=due_date,
            tags=tags
        )
        
        return {
            'success': True,
            'action': 'added',
            'todo': todo,
            'message': f"Added: \"{description}\"" + 
                      (f" with {priority} priority" if priority != 'medium' else "") +
                      (f" due {entities.get('date_string', '')}" if due_date else "")
        }
        
    async def _handle_list(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list todos command."""
        # Determine filters
        status = TodoStatus.PENDING  # Default to pending
        
        # Check for time-based listing
        if 'due_date' in entities:
            # List for specific date
            try:
                date = datetime.fromisoformat(entities['due_date'])
                todos = self.todo_manager.get_todos_for_date(date)
                time_context = entities.get('date_string', 'that date')
            except:
                todos = self.todo_manager.list_todos(status=status)
                time_context = None
        else:
            todos = self.todo_manager.list_todos(status=status)
            time_context = None
            
        # Update context
        self.context.last_listed_todos = todos
        
        return {
            'success': True,
            'action': 'list',
            'todos': todos,
            'count': len(todos),
            'time_context': time_context
        }
        
    async def _handle_complete(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complete todo command."""
        task_ref = entities.get('task_reference', '')
        
        # Try to find and complete
        todo = self.todo_manager.complete_todo(task_ref)
        
        if todo:
            return {
                'success': True,
                'action': 'completed',
                'todo': todo,
                'message': f"Completed: \"{todo.description}\""
            }
        else:
            # Check last listed todos for index
            if task_ref.isdigit() and self.context.last_listed_todos:
                index = int(task_ref) - 1
                if 0 <= index < len(self.context.last_listed_todos):
                    todo = self.context.last_listed_todos[index]
                    completed = self.todo_manager.complete_todo(todo.id)
                    if completed:
                        return {
                            'success': True,
                            'action': 'completed',
                            'todo': completed,
                            'message': f"Completed: \"{completed.description}\""
                        }
                        
            return {
                'success': False,
                'message': f"Couldn't find task: \"{task_ref}\". Try being more specific."
            }
            
    async def _handle_delete(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete todo command."""
        task_ref = entities.get('task_reference', '')
        
        todo = self.todo_manager.delete_todo(task_ref)
        
        if todo:
            return {
                'success': True,
                'action': 'deleted',
                'todo': todo,
                'message': f"Deleted: \"{todo.description}\""
            }
        else:
            return {
                'success': False,
                'message': f"Couldn't find task: \"{task_ref}\""
            }
            
    async def _handle_update(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update todo command."""
        task_ref = entities.get('task_reference', '')
        
        # Determine what to update
        updates = {}
        if 'priority' in entities:
            updates['priority'] = entities['priority']
        if 'due_date' in entities:
            updates['due_date'] = datetime.fromisoformat(entities['due_date'])
        if 'task' in entities:
            updates['description'] = entities['task']
            
        if not updates:
            return {
                'success': False,
                'message': "Not sure what to update. Please be more specific."
            }
            
        # Find and update
        todo = self.todo_manager.find_todo(task_ref)
        if todo:
            updated = self.todo_manager.update_todo(todo.id, **updates)
            return {
                'success': True,
                'action': 'updated',
                'todo': updated,
                'message': f"Updated: \"{updated.description}\""
            }
        else:
            return {
                'success': False,
                'message': f"Couldn't find task: \"{task_ref}\""
            }
            
    async def _handle_search(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search command."""
        query = entities.get('query', '')
        
        todos = self.todo_manager.list_todos(search_query=query)
        
        return {
            'success': True,
            'action': 'search',
            'todos': todos,
            'query': query,
            'count': len(todos)
        }
        
    async def _handle_help(self) -> Dict[str, Any]:
        """Handle help command."""
        return {
            'success': True,
            'action': 'help',
            'help_text': """
Here are some things you can say:

[bold]Adding tasks:[/bold]
• "Add a task to review the documentation"
• "Remind me to call John tomorrow at 3pm"
• "Create a high priority task for the API update"

[bold]Viewing tasks:[/bold]
• "Show me my tasks"
• "What do I need to do today?"
• "List high priority tasks"

[bold]Completing tasks:[/bold]
• "Mark the documentation task as done"
• "I finished the API review"
• "Complete task 1" (from the list)

[bold]Managing tasks:[/bold]
• "Delete the meeting task"
• "Update the deadline to next Friday"
• "Make the report task high priority"

[bold]Other:[/bold]
• "Search for tasks about documentation"
• "Show statistics"
• "Undo" (undo last action)
            """
        }
        
    async def _display_result(self, result: Dict[str, Any]):
        """Display command result."""
        if not result.get('success'):
            # Error or clarification needed
            self.console.print(f"[yellow]{result.get('message', 'Error')}[/yellow]")
            
            if result.get('suggestions'):
                self.console.print("\n[dim]Suggestions:[/dim]")
                for suggestion in result['suggestions']:
                    self.console.print(f"  • {suggestion}")
                    
        else:
            # Success - display based on action
            action = result.get('action')
            
            if action == 'list':
                await self._display_todo_list(result['todos'], result.get('time_context'))
                
            elif action == 'search':
                await self._display_search_results(
                    result['todos'],
                    result['query'],
                    result['count']
                )
                
            elif action == 'help':
                self.console.print(result['help_text'])
                
            else:
                # Simple confirmation message
                self.console.print(f"[green]✓[/green] {result.get('message', 'Done')}")
                
    async def _display_todo_list(self, todos: List[TodoItem], time_context: Optional[str] = None):
        """Display a list of todos."""
        if not todos:
            message = "No tasks"
            if time_context:
                message += f" for {time_context}"
            self.console.print(f"[dim]{message}[/dim]")
            return
            
        # Create table
        table = Table(title=f"Tasks{' for ' + time_context if time_context else ''}")
        table.add_column("#", style="dim", width=4)
        table.add_column("Priority", width=8)
        table.add_column("Task", no_wrap=False)
        table.add_column("Due", width=15)
        table.add_column("Tags", width=15)
        
        # Priority colors
        priority_colors = {
            "urgent": "red",
            "high": "yellow",
            "medium": "blue",
            "low": "dim"
        }
        
        for i, todo in enumerate(todos, 1):
            priority_style = priority_colors.get(todo.priority, "white")
            
            # Format due date
            if todo.due_date:
                due = self._format_relative_time(todo.due_date)
            else:
                due = "-"
                
            # Format tags
            tags = ", ".join(todo.tags) if todo.tags else "-"
            
            table.add_row(
                str(i),
                f"[{priority_style}]{todo.priority}[/{priority_style}]",
                todo.description,
                due,
                tags
            )
            
        self.console.print(table)
        
        # Show summary
        overdue = self.todo_manager.get_overdue_todos()
        if overdue:
            self.console.print(f"\n[red]⚠ {len(overdue)} overdue task(s)[/red]")
            
    async def _display_search_results(self, todos: List[TodoItem], query: str, count: int):
        """Display search results."""
        if not todos:
            self.console.print(f"[dim]No tasks found matching \"{query}\"[/dim]")
            return
            
        self.console.print(f"\n[bold]Found {count} task(s) matching \"{query}\":[/bold]")
        await self._display_todo_list(todos)
        
    def _format_relative_time(self, dt: datetime) -> str:
        """Format datetime as relative time."""
        now = datetime.now(timezone.utc)
        diff = dt - now
        
        if diff.days < 0:
            return f"[red]{-diff.days} days ago[/red]"
        elif diff.days == 0:
            hours = diff.seconds // 3600
            if hours == 0:
                return "[yellow]< 1 hour[/yellow]"
            else:
                return f"[yellow]{hours} hours[/yellow]"
        elif diff.days == 1:
            return "Tomorrow"
        elif diff.days < 7:
            return f"{diff.days} days"
        else:
            return dt.strftime("%b %d")
            
    async def _status_monitor(self):
        """Monitor and display system status."""
        while self.running:
            try:
                # Get queue status
                queue_status = self.queue.get_status()
                
                # Only show if there's activity
                if queue_status['processing'] > 0:
                    status = f"[dim]Processing {queue_status['processing']} message(s)...[/dim]"
                    self.console.print(status, end='\r')
                    
            except Exception as e:
                self.logger.error(f"Status monitor error: {e}")
                
            await asyncio.sleep(1)


def main():
    """Run the CLI."""
    cli = NaturalLanguageTodoCLI()
    
    try:
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        

if __name__ == "__main__":
    main()