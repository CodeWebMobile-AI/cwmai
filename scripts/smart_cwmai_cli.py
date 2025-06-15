#!/usr/bin/env python3
"""
Smart CWMAI CLI - Advanced Natural Language Interface

An intelligent command-line interface that understands natural language,
learns from usage patterns, and provides contextual assistance.
"""

import asyncio
import sys
import os
import argparse
import json
import readline
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Color support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback for no colorama
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from scripts.smart_natural_language_interface import SmartNaturalLanguageInterface
from scripts.ai_brain import IntelligentAIBrain


class SmartCLI:
    """Smart command-line interface for CWMAI."""
    
    def __init__(self, enable_learning: bool = True, enable_multi_model: bool = True):
        """Initialize the smart CLI.
        
        Args:
            enable_learning: Enable learning from user patterns
            enable_multi_model: Enable multi-model consensus
        """
        self.interface: Optional[SmartNaturalLanguageInterface] = None
        self.enable_learning = enable_learning
        self.enable_multi_model = enable_multi_model
        self.history_file = Path.home() / ".cwmai" / "command_history.json"
        self.shortcuts = self._load_shortcuts()
        
        # Set up readline for better input
        self._setup_readline()
        
    def _setup_readline(self):
        """Set up readline for command history and completion."""
        # Set up history file
        histfile = Path.home() / ".cwmai" / "readline_history"
        histfile.parent.mkdir(exist_ok=True)
        
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        
        readline.set_history_length(1000)
        
        # Save history on exit
        import atexit
        atexit.register(readline.write_history_file, histfile)
        
        # Set up tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer)
        
        # Set up custom key bindings
        readline.parse_and_bind('"\e[A": history-search-backward')  # Up arrow
        readline.parse_and_bind('"\e[B": history-search-forward')   # Down arrow
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion function."""
        # Common commands and phrases
        options = [
            "create issue", "search for", "generate architecture",
            "show status", "analyze market", "create task",
            "list tasks", "help", "quit", "examples",
            "find repos", "make project", "what can you do",
            "explain", "suggest", "analyze"
        ]
        
        # Add learned patterns
        if self.interface and self.interface.context.history:
            recent_commands = [h['input'] for h in self.interface.context.history[-10:]]
            options.extend(recent_commands)
        
        # Filter matches
        matches = [opt for opt in options if opt.startswith(text.lower())]
        
        try:
            return matches[state]
        except IndexError:
            return None
    
    def _load_shortcuts(self) -> Dict[str, str]:
        """Load command shortcuts."""
        return {
            'q': 'quit',
            'h': 'help',
            'st': 'show status',
            's': 'search for',
            'ci': 'create issue',
            'ga': 'generate architecture',
            'am': 'analyze market',
            'ct': 'create task',
            'lt': 'list tasks',
            'ex': 'examples'
        }
    
    async def initialize(self):
        """Initialize the interface and components."""
        print(f"{Fore.CYAN}🧠 Initializing Smart CWMAI Interface...{Style.RESET_ALL}")
        
        # Create AI brain
        ai_brain = IntelligentAIBrain(enable_round_robin=True)
        
        # Create interface
        self.interface = SmartNaturalLanguageInterface(
            ai_brain=ai_brain,
            enable_learning=self.enable_learning,
            enable_multi_model=self.enable_multi_model
        )
        
        # Initialize async components
        await self.interface.initialize()
        
        # Show initialization status
        if self.interface.mcp_hub:
            print(f"{Fore.GREEN}✅ MCP Integration ready{Style.RESET_ALL}")
        
        if self.interface.brave_search:
            print(f"{Fore.GREEN}✅ Brave Search ready{Style.RESET_ALL}")
        
        if self.interface.ai_models:
            print(f"{Fore.GREEN}✅ Multi-model AI ready ({len(self.interface.ai_models)} models){Style.RESET_ALL}")
        
        if self.enable_learning:
            print(f"{Fore.GREEN}✅ Learning mode enabled{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}🚀 Smart interface ready!{Style.RESET_ALL}\n")
    
    def print_banner(self):
        """Print welcome banner."""
        banner = f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════════════╗
║                    {Fore.WHITE}🧠 Smart CWMAI CLI 🧠{Fore.CYAN}                     ║
║                                                               ║
║  {Fore.GREEN}Natural language interface with AI-powered understanding{Fore.CYAN}     ║
║  {Fore.YELLOW}Learning: {'ON' if self.enable_learning else 'OFF':<3} | Multi-Model: {'ON' if self.enable_multi_model else 'OFF':<3}{Fore.CYAN}                        ║
║                                                               ║
║  {Fore.WHITE}Examples:{Fore.CYAN}                                                    ║
║  • "Create an issue for auth-api about adding OAuth"         ║
║  • "Search for AI development tools"                         ║
║  • "Generate architecture for a chat application"            ║
║  • "Find all repos with security issues and fix them"        ║
║                                                               ║
║  {Fore.MAGENTA}Type 'help' for commands or 'examples' for more{Fore.CYAN}             ║
╚═══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""
        print(banner)
    
    def print_help(self):
        """Print help information."""
        help_text = f"""
{Fore.CYAN}🤖 Smart CWMAI CLI - Natural Language Commands{Style.RESET_ALL}

{Fore.GREEN}You can use natural language to interact with the system. Here are some examples:{Style.RESET_ALL}

{Fore.YELLOW}📝 Issue Management:{Style.RESET_ALL}
  • Create an issue for [repo] about [topic]
  • Make a bug report for [project] saying [description]
  • File a feature request in [repo] to add [feature]

{Fore.YELLOW}🔍 Search & Discovery:{Style.RESET_ALL}
  • Search for [topic]
  • Find repositories about [technology]
  • Look for projects related to [keyword]

{Fore.YELLOW}🏗️ Architecture & Design:{Style.RESET_ALL}
  • Generate architecture for [project type]
  • Design a system for [use case]
  • Create a blueprint for [application]

{Fore.YELLOW}📊 Analysis & Insights:{Style.RESET_ALL}
  • Analyze market for [technology/idea]
  • Show system status
  • What's trending in [field]

{Fore.YELLOW}📋 Task Management:{Style.RESET_ALL}
  • Create a task to [action]
  • List active tasks
  • What needs to be done

{Fore.YELLOW}🔗 Complex Operations:{Style.RESET_ALL}
  • Find all repos with [issue] and then create tasks to fix them
  • Search for [tech] and create architecture for the best one
  • Analyze market for [idea] and then create project plan

{Fore.MAGENTA}Shortcuts:{Style.RESET_ALL}
  q/quit - Exit | h/help - This help | st - Status | s - Search
  ci - Create issue | ga - Generate architecture | ex - Examples

{Fore.CYAN}Tips:{Style.RESET_ALL}
  • Use Tab for command completion
  • Up/Down arrows search command history
  • The system learns from your patterns
  • Be specific for better results
"""
        print(help_text)
    
    def print_examples(self):
        """Print extended examples."""
        examples = f"""
{Fore.CYAN}📚 Extended Examples - Smart Commands{Style.RESET_ALL}

{Fore.GREEN}1. Smart Issue Creation:{Style.RESET_ALL}
   {Fore.WHITE}"Create an issue for the auth service about users reporting slow login times"{Style.RESET_ALL}
   → Creates enhanced issue with performance label and detailed description

{Fore.GREEN}2. Intelligent Search:{Style.RESET_ALL}
   {Fore.WHITE}"Find AI tools that help with code review"{Style.RESET_ALL}
   → Searches GitHub, web, and local projects with relevance ranking

{Fore.GREEN}3. Market-Aware Architecture:{Style.RESET_ALL}
   {Fore.WHITE}"Design architecture for a SaaS metrics dashboard that competes with Datadog"{Style.RESET_ALL}
   → Generates architecture with market analysis and competitive insights

{Fore.GREEN}4. Complex Multi-Step Operations:{Style.RESET_ALL}
   {Fore.WHITE}"Search for JavaScript testing frameworks and then create a comparison task"{Style.RESET_ALL}
   → Executes search, analyzes results, creates detailed comparison task

{Fore.GREEN}5. Context-Aware Commands:{Style.RESET_ALL}
   {Fore.WHITE}"Create another issue about performance"{Style.RESET_ALL}
   → Uses context from previous commands to determine repository

{Fore.GREEN}6. Learning from Patterns:{Style.RESET_ALL}
   {Fore.WHITE}"Do the usual morning check"{Style.RESET_ALL}
   → Executes your common morning routine based on learned patterns

{Fore.GREEN}7. Natural Conversations:{Style.RESET_ALL}
   {Fore.WHITE}"What's the status of my projects?"{Style.RESET_ALL}
   {Fore.WHITE}"Are there any critical issues?"{Style.RESET_ALL}
   {Fore.WHITE}"What should I work on next?"{Style.RESET_ALL}

{Fore.YELLOW}The system understands context and can handle typos, variations, and complex requests!{Style.RESET_ALL}
"""
        print(examples)
    
    async def process_command(self, command: str) -> bool:
        """Process a single command.
        
        Args:
            command: User command
            
        Returns:
            True to continue, False to exit
        """
        # Check for shortcuts
        if command.lower() in self.shortcuts:
            command = self.shortcuts[command.lower()]
        
        # Handle special commands
        if command.lower() in ['quit', 'exit', 'q']:
            print(f"{Fore.YELLOW}👋 Goodbye! Thanks for using Smart CWMAI.{Style.RESET_ALL}")
            return False
        
        if command.lower() in ['help', 'h', '?']:
            self.print_help()
            return True
        
        if command.lower() in ['examples', 'ex']:
            self.print_examples()
            return True
        
        # Process with smart interface
        try:
            print(f"{Fore.CYAN}🤔 Processing: {command}{Style.RESET_ALL}")
            
            result = await self.interface.process_input(command)
            
            # Display results
            self._display_result(result)
            
            # Save to history
            self._save_to_history(command, result)
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Try rephrasing or use 'help' for guidance{Style.RESET_ALL}")
        
        return True
    
    def _display_result(self, result: Dict[str, Any]):
        """Display command result with formatting."""
        # Check if confirmation needed
        if result.get('needs_confirmation'):
            print(f"\n{Fore.YELLOW}❓ Confirmation needed:{Style.RESET_ALL}")
            print(f"Interpreted as: {result['interpreted_as']['action']}")
            print(f"Details: {json.dumps(result['interpreted_as']['details'], indent=2)}")
            print(f"Confidence: {result['confidence']}")
            return
        
        # Show success/error status
        if result.get('success', False):
            print(f"\n{Fore.GREEN}✅ Success!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}❌ Failed{Style.RESET_ALL}")
            if result.get('error'):
                print(f"Error: {result['error']}")
        
        # Show main result
        action = result.get('action', 'unknown')
        
        if action == 'issue_created':
            issue = result.get('issue', {})
            print(f"Created issue #{issue.get('issue_number', '?')} in {result.get('repository')}")
            print(f"URL: {issue.get('html_url', 'pending')}")
            if result.get('enhancements'):
                print(f"Enhancements: {', '.join(result['enhancements'])}")
        
        elif action == 'search_completed':
            print(f"Found {result.get('total_found', 0)} results for '{result.get('query')}'")
            print(f"Sources: {', '.join(result.get('sources', []))}")
            
            # Show top results
            for i, item in enumerate(result.get('results', [])[:5], 1):
                source = item.get('source', 'unknown')
                data = item.get('item', {})
                
                if source == 'github':
                    print(f"\n{i}. {Fore.BLUE}[GitHub]{Style.RESET_ALL} {data.get('full_name', 'Unknown')}")
                    print(f"   {data.get('description', 'No description')[:80]}...")
                    print(f"   ⭐ {data.get('stargazers_count', 0)} | 🍴 {data.get('forks_count', 0)}")
                elif source == 'web':
                    print(f"\n{i}. {Fore.GREEN}[Web]{Style.RESET_ALL} {data.get('title', 'Unknown')}")
                    print(f"   {data.get('description', '')[:80]}...")
                    print(f"   {Fore.CYAN}{data.get('url', '')}{Style.RESET_ALL}")
                elif source == 'local':
                    print(f"\n{i}. {Fore.YELLOW}[Local]{Style.RESET_ALL} {data.get('name', 'Unknown')}")
        
        elif action == 'architecture_generated':
            print(f"Generated architecture for '{result.get('project')}'")
            print(f"Saved to: {result.get('saved_to')}")
            if result.get('market_insights'):
                print(f"Included {len(result['market_insights'])} market insights")
        
        elif action == 'market_analysis_completed':
            print(f"Market analysis for '{result.get('topic')}':")
            print(result.get('analysis', 'No analysis available'))
        
        elif action == 'status_displayed':
            stats = result.get('stats', {})
            print(f"\n{Fore.CYAN}System Status:{Style.RESET_ALL}")
            print(f"Projects: {stats.get('total_projects', 0)}")
            print(f"Active Tasks: {stats.get('active_tasks', 0)}")
            print(f"Completed Tasks: {stats.get('completed_tasks', 0)}")
            print(f"Success Rate: {stats.get('success_rate', 0):.1%}")
            print(f"\n{result.get('summary', '')}")
        
        elif action == 'complex_operation_completed':
            print(f"Completed {result.get('steps', 0)} steps")
            print(result.get('summary', ''))
        
        elif action.startswith('plugin_'):
            plugin_name = result.get('plugin_name', 'unknown')
            plugin_data = result.get('plugin_data', {})
            
            print(f"\n{Fore.MAGENTA}🔌 {plugin_name.title()} Plugin Result:{Style.RESET_ALL}")
            
            # Handle different plugin types
            if plugin_name == 'automation':
                workflow = plugin_data.get('workflow', {})
                print(f"Created {workflow.get('type', 'unknown')} workflow")
                print(f"Description: {workflow.get('description', 'N/A')}")
                if workflow.get('schedule'):
                    print(f"Schedule: {workflow['schedule'].get('type')} at {workflow['schedule'].get('value', 'N/A')}")
                print(f"Actions: {len(workflow.get('actions', []))}")
                
            elif plugin_name == 'visualization':
                viz_type = plugin_data.get('type', 'unknown')
                print(f"Generated {viz_type} visualization")
                print(f"Data source: {plugin_data.get('source', 'N/A')}")
                if result.get('visualizations'):
                    print(f"📊 Visualization ready (would display chart here)")
                    
            elif plugin_name == 'explanation':
                subject = plugin_data.get('subject', 'unknown')
                explanation = plugin_data.get('explanation', {})
                print(f"Explanation for: {subject}")
                print(f"\n{Fore.CYAN}Overview:{Style.RESET_ALL}")
                print(explanation.get('overview', 'N/A'))
                if explanation.get('benefits'):
                    print(f"\n{Fore.GREEN}Benefits:{Style.RESET_ALL}")
                    print(explanation.get('benefits', 'N/A'))
        
        # Show explanation
        if result.get('explanation'):
            print(f"\n{Fore.CYAN}ℹ️ {result['explanation']}{Style.RESET_ALL}")
        
        # Show suggestions
        if result.get('suggestions'):
            print(f"\n{Fore.MAGENTA}💡 Suggestions:{Style.RESET_ALL}")
            for i, suggestion in enumerate(result['suggestions'], 1):
                print(f"  {i}. {suggestion}")
        
        # Show recovery suggestions for errors
        if result.get('recovery_suggestions'):
            print(f"\n{Fore.YELLOW}🔧 Try these:{Style.RESET_ALL}")
            for suggestion in result['recovery_suggestions']:
                print(f"  • {suggestion}")
        
        # Show clarification questions
        if result.get('clarification_needed'):
            print(f"\n{Fore.YELLOW}❓ I need more information:{Style.RESET_ALL}")
            for question in result.get('questions', []):
                print(f"  • {question}")
            
            if result.get('possible_interpretations'):
                print(f"\n{Fore.CYAN}Did you mean:{Style.RESET_ALL}")
                for interp in result['possible_interpretations']:
                    print(f"  • {interp}")
    
    def _save_to_history(self, command: str, result: Dict[str, Any]):
        """Save command and result to history."""
        try:
            history = []
            
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            
            history.append({
                'timestamp': datetime.now().isoformat(),
                'command': command,
                'success': result.get('success', False),
                'action': result.get('action', 'unknown')
            })
            
            # Keep last 1000 commands
            history = history[-1000:]
            
            self.history_file.parent.mkdir(exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            # Don't fail on history save errors
            pass
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        self.print_banner()
        
        while True:
            try:
                # Show smart prompt with context
                prompt = self._get_smart_prompt()
                command = input(prompt).strip()
                
                if not command:
                    continue
                
                # Process command
                should_continue = await self.process_command(command)
                
                if not should_continue:
                    break
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'quit' to exit properly{Style.RESET_ALL}")
            except EOFError:
                print()
                break
            except Exception as e:
                print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
    
    def _get_smart_prompt(self) -> str:
        """Get context-aware prompt."""
        base_prompt = "cwmai"
        
        if self.interface and self.interface.context.current_project:
            base_prompt = f"cwmai:{self.interface.context.current_project}"
        
        # Add learning indicator
        if self.enable_learning and self.interface:
            patterns = len(self.interface.context.command_patterns)
            if patterns > 10:
                base_prompt += f"[🧠{patterns}]"
        
        return f"{Fore.GREEN}{base_prompt}> {Style.RESET_ALL}"


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smart CWMAI CLI - Natural Language Interface with AI"
    )
    
    parser.add_argument(
        'command',
        nargs='*',
        help='Command to execute (interactive mode if omitted)'
    )
    
    parser.add_argument(
        '--no-learning',
        action='store_true',
        help='Disable learning from user patterns'
    )
    
    parser.add_argument(
        '--single-model',
        action='store_true',
        help='Use single AI model instead of multi-model consensus'
    )
    
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip environment variable check'
    )
    
    args = parser.parse_args()
    
    # Check environment
    if not args.skip_check:
        required_vars = ['ANTHROPIC_API_KEY']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            print(f"{Fore.RED}❌ Missing environment variables: {', '.join(missing)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Set them in .env.local or export them{Style.RESET_ALL}")
            sys.exit(1)
    
    # Create CLI
    cli = SmartCLI(
        enable_learning=not args.no_learning,
        enable_multi_model=not args.single_model
    )
    
    # Initialize
    await cli.initialize()
    
    # Run command or interactive mode
    if args.command:
        command = ' '.join(args.command)
        await cli.process_command(command)
    else:
        await cli.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())