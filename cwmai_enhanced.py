#!/usr/bin/env python3
"""
CWMAI Enhanced - Claude Code-like Terminal Interface

A sophisticated conversational AI assistant with rich terminal features:
- Command history with arrow keys
- Bracketed paste mode (paste without auto-execution)
- Syntax highlighting and structured output
- Multi-line editing support
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

# Import prompt_toolkit for rich terminal features
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import get_app

# Rich for beautiful output formatting
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

# Initialize rich console for structured output
console = Console()

# Custom style for prompt_toolkit
custom_style = Style.from_dict({
    'prompt': '#00aa00 bold',
    'continuation': '#888888',
    'command': '#0088ff',
    'error': '#ff0000 bold',
    'info': '#0088ff',
    'success': '#00aa00',
})

# Common commands for auto-completion
COMMON_COMMANDS = [
    'list tools', 'list all tools', 'create tool', 'execute workflow',
    'show status', 'help', 'exit', 'quit', 'clear',
    'create issue', 'analyze repository', 'generate report'
]

def load_env_file(env_path: Path) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    env_vars = {}
    
    if not env_path.exists():
        return env_vars
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE format - handle export prefix
            if line.startswith('export '):
                line = line[7:]  # Remove 'export ' prefix
            
            if '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                env_vars[key.strip()] = value
    
    return env_vars

def setup_environment():
    """Set up environment variables from .env files."""
    # Suppress NLTK download messages
    os.environ['NLTK_DATA'] = os.path.expanduser('~/nltk_data')
    
    # Priority order: .env.local > .env
    env_files = [
        Path('.env.local'),
        Path('.env')
    ]
    
    loaded_vars = {}
    for env_file in env_files:
        if env_file.exists():
            file_vars = load_env_file(env_file)
            # Don't override already loaded vars (priority order)
            for key, value in file_vars.items():
                if key not in loaded_vars:
                    loaded_vars[key] = value
                    os.environ[key] = value
    
    return loaded_vars

def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    # Add project root to path first
    project_root = Path(__file__).parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Also add current working directory if different
    cwd = Path.cwd()
    if str(cwd) not in sys.path and cwd != project_root:
        sys.path.insert(0, str(cwd))
    
    try:
        # Try to import just the module without initializing it
        import importlib.util
        spec = importlib.util.find_spec("scripts.conversational_ai_assistant")
        return spec is not None
    except Exception:
        # Silently fail - the import check might fail due to missing dependencies
        # but the actual import might work when we try it
        return True

def print_banner():
    """Print a beautiful welcome banner using rich."""
    banner = Panel.fit(
        "[bold cyan]🤖 CWMAI - Your AI Development Assistant[/bold cyan]\n\n"
        "[green]Chat naturally with Claude-like intelligence[/green]\n"
        "[yellow]Execute commands, manage projects, and more![/yellow]",
        border_style="cyan",
        box=box.DOUBLE_EDGE
    )
    console.print(banner)

def create_key_bindings():
    """Create custom key bindings for the prompt."""
    kb = KeyBindings()
    
    @kb.add('c-c')
    def _(event):
        """Handle Ctrl-C gracefully."""
        console.print("\n[yellow]Use 'exit' or 'quit' to leave properly.[/yellow]")
        event.app.current_buffer.reset()
    
    @kb.add('c-d')
    def _(event):
        """Handle Ctrl-D (EOF) gracefully."""
        event.app.exit()
    
    return kb

async def run_conversational_assistant(args):
    """Run the conversational AI assistant with enhanced terminal interface."""
    # Suppress macOS malloc warnings
    os.environ['MallocNanoZone'] = '0'
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Suppress NLTK SSL errors during imports
        import ssl
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Temporarily redirect stderr to suppress NLTK download messages
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            from scripts.conversational_ai_assistant import ConversationalAIAssistant
            from scripts.conversational_ai_assistant import ResponseStyle
            from scripts.silent_logger import setup_silent_logging, temporary_stdout
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
        
        # Set up completely silent logging - everything goes to file
        log_file = setup_silent_logging()
        
        # Determine response style
        style = ResponseStyle.FRIENDLY_PROFESSIONAL
        if args.concise:
            style = ResponseStyle.CONCISE
        elif args.detailed:
            style = ResponseStyle.DETAILED
        elif args.technical:
            style = ResponseStyle.TECHNICAL
        elif args.casual:
            style = ResponseStyle.CASUAL
        
        # Create assistant
        assistant = ConversationalAIAssistant(style=style)
        
        # Initialize
        with console.status("[cyan]Initializing AI assistant...[/cyan]"):
            await assistant.initialize()
        console.print("[green]✅ AI assistant ready![/green]\n")
        
        if args.command:
            # Single command mode
            response = await assistant.handle_conversation(' '.join(args.command))
            # Format response as markdown
            md = Markdown(response)
            console.print(md)
        else:
            # Interactive mode with enhanced terminal
            
            # Create history file
            history_file = Path.home() / '.cwmai_history'
            history = FileHistory(str(history_file))
            
            # Create completer with common commands
            completer = WordCompleter(COMMON_COMMANDS, ignore_case=True)
            
            # Create prompt session with all features
            session = PromptSession(
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                complete_while_typing=True,
                enable_history_search=True,
                mouse_support=True,
                style=custom_style,
                key_bindings=create_key_bindings(),
                # Enable bracketed paste mode to prevent auto-execution
                enable_suspend=True,
                vi_mode=False,  # Use emacs mode by default
                multiline=False,  # Single line by default
            )
            
            console.print("[cyan]Type your message or question. Use 'exit' or 'quit' to leave.[/cyan]")
            console.print("[dim]Tip: Use arrow keys for history, Tab for completion[/dim]\n")
            
            while True:
                try:
                    # Get user input with rich prompt
                    user_input = await session.prompt_async(
                        HTML('<ansigreen><b>You > </b></ansigreen>'),
                        # Enable bracketed paste - prevents immediate execution on paste
                        enable_open_in_editor=True,
                    )
                    
                    if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                        console.print("\n[yellow]👋 Goodbye! Thanks for using CWMAI.[/yellow]")
                        break
                    
                    if not user_input.strip():
                        continue
                    
                    # Clear screen command
                    if user_input.lower() == 'clear':
                        console.clear()
                        continue
                    
                    # Process with assistant
                    console.print("[blue]Assistant > [/blue]", end='')
                    
                    # Show thinking indicator
                    with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                        response = await assistant.handle_conversation(user_input)
                    
                    # Display response with proper formatting
                    # Check if response contains code blocks
                    if "```" in response:
                        # Parse and display code blocks with syntax highlighting
                        parts = response.split("```")
                        for i, part in enumerate(parts):
                            if i % 2 == 0:
                                # Regular text
                                if part.strip():
                                    console.print(Markdown(part))
                            else:
                                # Code block
                                lines = part.split('\n')
                                lang = lines[0] if lines else 'text'
                                code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                                syntax = Syntax(code, lang, theme="monokai", line_numbers=True)
                                console.print(syntax)
                    else:
                        # Regular markdown response
                        md = Markdown(response)
                        console.print(md)
                    
                    console.print()  # Extra line for readability
                    
                except KeyboardInterrupt:
                    # This is handled by key bindings
                    continue
                except EOFError:
                    console.print("\n[yellow]👋 Goodbye![/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]❌ Error: {e}[/red]")
                    
    except ImportError as e:
        console.print(f"[red]❌ Failed to import conversational assistant: {e}[/red]")
        console.print("[yellow]ℹ️  Make sure you're running from the CWMAI project directory.[/yellow]")
        return 1
    except Exception as e:
        import traceback
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        return 1
    
    return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CWMAI Enhanced - Claude Code-like Terminal Interface",
        epilog="Examples:\n"
               "  cwmai                    # Start interactive mode\n"
               "  cwmai create an issue    # Execute single command\n"
               "  cwmai --concise          # Use concise responses\n"
               "  cwmai --no-env           # Skip .env loading",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        nargs='*',
        help='Command to execute (interactive mode if omitted)'
    )
    
    parser.add_argument(
        '--no-env',
        action='store_true',
        help='Skip loading .env files'
    )
    
    parser.add_argument(
        '--concise',
        action='store_true',
        help='Use concise response style'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Use detailed response style'
    )
    
    parser.add_argument(
        '--technical',
        action='store_true',
        help='Use technical response style'
    )
    
    parser.add_argument(
        '--casual',
        action='store_true',
        help='Use casual response style'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Skip the welcome banner'
    )
    
    args = parser.parse_args()
    
    # Show banner unless disabled or in single command mode
    if not args.no_banner and not args.command:
        print_banner()
    
    # Load environment variables unless disabled
    if not args.no_env:
        loaded_vars = setup_environment()
        if loaded_vars:
            console.print(f"[green]✅ Loaded {len(loaded_vars)} environment variables[/green]")
    
    # Check for required environment variables
    required_vars = ['ANTHROPIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        console.print(f"[red]❌ Missing required environment variables: {', '.join(missing_vars)}[/red]")
        console.print("[yellow]ℹ️  Please set them in .env.local or .env file[/yellow]")
        console.print("[dim]Example .env.local:[/dim]")
        for var in missing_vars:
            console.print(f"  {var}=your_key_here")
        return 1
    
    # Check if we can import the assistant
    if not check_dependencies():
        console.print("[red]❌ Cannot find CWMAI scripts. Make sure you're in the CWMAI directory.[/red]")
        return 1
    
    # Run the assistant
    try:
        return asyncio.run(run_conversational_assistant(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
        return 0
    except Exception as e:
        console.print(f"[red]❌ Failed to run assistant: {e}[/red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())