"""
CWMAI Command Line Interface

An interactive REPL for natural language interaction with the CWMAI system.
"""

import asyncio
import readline
import sys
import os
from datetime import datetime
from typing import Optional, List
import json
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.natural_language_interface import NaturalLanguageInterface


class CWMAICommandLine:
    """Interactive command-line interface for CWMAI."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.nli = NaturalLanguageInterface()
        self.running = False
        self.history = []
        self.prompt = f"{Fore.GREEN}CWMAI>{Style.RESET_ALL} "
        
        # Configure readline for better interaction
        readline.parse_and_bind('tab: complete')
        readline.set_completer(self.completer)
        
        # Command shortcuts
        self.shortcuts = {
            'q': 'quit',
            'exit': 'quit',
            '?': 'help',
            'h': 'help',
            'st': 'show status',
            'lt': 'list tasks',
            'ap': 'analyze performance'
        }
        
    def completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for commands."""
        commands = [
            'create issue', 'search repositories', 'create architecture',
            'show status', 'list tasks', 'analyze performance', 
            'create task', 'help', 'quit', 'clear', 'history'
        ]
        
        # Filter commands that start with the current text
        matches = [cmd for cmd in commands if cmd.startswith(text.lower())]
        
        try:
            return matches[state]
        except IndexError:
            return None
            
    def print_banner(self):
        """Print the welcome banner."""
        print(f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════════╗
║                    CWMAI - Natural Language               ║
║              Continuous Work Management AI                ║
╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}

Welcome to CWMAI! Type 'help' for available commands or 'quit' to exit.
Use natural language to interact with the system.
        """)
        
    def format_result(self, result: dict):
        """Format and display command results."""
        status = result.get('status', 'unknown')
        message = result.get('message', '')
        data = result.get('data', {})
        
        # Status indicator
        if status == 'success':
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {message}")
        elif status == 'error':
            print(f"{Fore.RED}✗{Style.RESET_ALL} {message}")
        else:
            print(f"{Fore.YELLOW}?{Style.RESET_ALL} {message}")
            
        # Format data based on type
        if data:
            if isinstance(data, list):
                self._format_list(data)
            elif isinstance(data, dict):
                self._format_dict(data)
            else:
                print(f"  {data}")
                
    def _format_list(self, items: List):
        """Format a list of items."""
        for i, item in enumerate(items[:10], 1):  # Limit to 10 items
            if isinstance(item, dict):
                # Format based on item type
                if 'name' in item:  # Repository
                    print(f"  {i}. {Fore.YELLOW}{item['name']}{Style.RESET_ALL}")
                    if 'description' in item:
                        print(f"     {item['description'][:60]}...")
                elif 'title' in item:  # Task or issue
                    status = item.get('status', '')
                    status_color = Fore.GREEN if status == 'completed' else Fore.YELLOW
                    print(f"  {i}. [{status_color}{status}{Style.RESET_ALL}] {item['title']}")
                else:
                    print(f"  {i}. {item}")
            else:
                print(f"  {i}. {item}")
                
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
            
    def _format_dict(self, data: dict):
        """Format a dictionary of data."""
        # Special formatting for known data types
        if 'architecture' in data:
            print(f"\n{Fore.CYAN}=== System Architecture ==={Style.RESET_ALL}")
            print(data['architecture'])
        elif 'commands' in data:
            print(f"\n{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
            for cmd in data['commands']:
                print(f"  {cmd}")
        elif 'help' in data:
            print(f"  {data['help']}")
        else:
            # Generic formatting
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  {Fore.CYAN}{key}:{Style.RESET_ALL} {len(value)} items")
                else:
                    print(f"  {Fore.CYAN}{key}:{Style.RESET_ALL} {value}")
                    
    async def process_command(self, command: str):
        """Process a single command."""
        # Handle special commands
        if command.lower() in ['quit', 'q', 'exit']:
            self.running = False
            print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            return
            
        if command.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            return
            
        if command.lower() == 'history':
            print(f"\n{Fore.CYAN}Command History:{Style.RESET_ALL}")
            for i, cmd in enumerate(self.history[-10:], 1):
                print(f"  {i}. {cmd}")
            return
            
        # Apply shortcuts
        command = self.shortcuts.get(command.lower(), command)
        
        # Process with NLI
        print(f"{Fore.BLUE}Processing...{Style.RESET_ALL}")
        try:
            result = await self.nli.process_natural_language(command)
            self.format_result(result)
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            
    async def run_interactive(self):
        """Run the interactive CLI loop."""
        self.print_banner()
        
        # Initialize NLI
        print(f"{Fore.BLUE}Initializing CWMAI...{Style.RESET_ALL}")
        initialized = await self.nli.initialize()
        
        if initialized:
            print(f"{Fore.GREEN}✓ CWMAI initialized successfully{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ CWMAI initialized with limited functionality{Style.RESET_ALL}")
            
        self.running = True
        
        # Main loop
        while self.running:
            try:
                # Get user input
                command = input(self.prompt).strip()
                
                if command:
                    # Add to history
                    self.history.append(command)
                    
                    # Process command
                    await self.process_command(command)
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'quit' to exit{Style.RESET_ALL}")
            except EOFError:
                self.running = False
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
                
        # Cleanup
        await self.nli.close()
        
    async def process_single_command(self, command: str):
        """Process a single command and exit."""
        # Initialize NLI
        await self.nli.initialize()
        
        # Process command
        result = await self.nli.process_natural_language(command)
        self.format_result(result)
        
        # Cleanup
        await self.nli.close()


def main():
    """Main entry point for the CLI."""
    cli = CWMAICommandLine()
    
    # Check if a command was provided as argument
    if len(sys.argv) > 1:
        command = ' '.join(sys.argv[1:])
        asyncio.run(cli.process_single_command(command))
    else:
        # Run interactive mode
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()