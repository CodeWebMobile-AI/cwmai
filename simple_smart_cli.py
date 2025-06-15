#!/usr/bin/env python3
"""
Simple Smart CLI - Basic natural language interface

A simplified version that works without all dependencies.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.ai_brain import IntelligentAIBrain
from scripts.task_manager import TaskManager
from scripts.state_manager import StateManager


class SimpleSmartCLI:
    """Simple smart CLI with basic natural language understanding."""
    
    def __init__(self):
        self.ai_brain = IntelligentAIBrain(enable_round_robin=True)
        self.task_manager = TaskManager()
        self.state_manager = StateManager()
        
    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process a natural language command."""
        command_lower = command.lower()
        
        # Simple pattern matching
        if "status" in command_lower or "how are" in command_lower:
            return await self.show_status()
        elif "create issue" in command_lower:
            return await self.create_issue(command)
        elif "search" in command_lower:
            return await self.search(command)
        elif "help" in command_lower:
            return self.show_help()
        else:
            return await self.ai_interpret(command)
    
    async def show_status(self) -> Dict[str, Any]:
        """Show system status."""
        state = self.state_manager.load_state()
        
        stats = {
            'total_projects': len(state.get('projects', {})),
            'active_tasks': len([t for t in state.get('tasks', {}).values() if t.get('status') == 'active']),
            'completed_tasks': len([t for t in state.get('tasks', {}).values() if t.get('status') == 'completed']),
            'total_operations': state.get('metrics', {}).get('total_operations', 0)
        }
        
        # Generate AI summary
        prompt = f"Summarize this system status in a friendly way: {json.dumps(stats, indent=2)}"
        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        return {
            'success': True,
            'action': 'status_displayed',
            'stats': stats,
            'summary': response.get('result', 'System is operational')
        }
    
    async def create_issue(self, command: str) -> Dict[str, Any]:
        """Create an issue from natural language."""
        # Extract repo and description using AI
        prompt = f"""Extract from this command:
Command: {command}

Extract:
1. Repository name (or null if not specified)
2. Issue title
3. Issue description

Format as JSON with keys: repository, title, description"""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        try:
            extracted = json.loads(response.get('result', '{}'))
            
            # Create task as a placeholder for issue
            task = self.task_manager.create_task(
                type='github_issue',
                title=extracted.get('title', 'New Issue'),
                description=extracted.get('description', 'Details to be added'),
                metadata={'repository': extracted.get('repository', 'cwmai')}
            )
            
            return {
                'success': True,
                'action': 'issue_created',
                'repository': extracted.get('repository', 'cwmai'),
                'issue': {'issue_number': task['id'], 'title': task['title']}
            }
        except:
            return {
                'success': False,
                'error': 'Could not parse issue details'
            }
    
    async def search(self, command: str) -> Dict[str, Any]:
        """Search based on natural language."""
        # Extract search query
        query = command.lower().replace('search for', '').replace('search', '').strip()
        
        # Search in local projects
        state = self.state_manager.load_state()
        projects = state.get('projects', {})
        
        results = [
            {'name': name, 'match': 'local'}
            for name in projects.keys()
            if query in name.lower()
        ]
        
        return {
            'success': True,
            'action': 'search_completed',
            'query': query,
            'results': results,
            'total_found': len(results)
        }
    
    def show_help(self) -> Dict[str, Any]:
        """Show help information."""
        return {
            'success': True,
            'action': 'help_displayed',
            'help_text': """
Available commands:
- Show status / How are things?
- Create issue for [repo] about [topic]
- Search for [query]
- Help

You can use natural language - I'll try to understand!
"""
        }
    
    async def ai_interpret(self, command: str) -> Dict[str, Any]:
        """Use AI to interpret unclear commands."""
        prompt = f"""Interpret this command for a development task system:
Command: {command}

Determine what the user wants to do and suggest how to help them.
Available actions: show status, create issue, search, help"""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        return {
            'success': False,
            'action': 'interpretation',
            'suggestion': response.get('result', 'Please try rephrasing your command')
        }


def print_result(result: Dict[str, Any]):
    """Print result in a nice format."""
    if result.get('success'):
        print("✅ Success!")
        
        action = result.get('action')
        if action == 'status_displayed':
            stats = result.get('stats', {})
            print(f"Projects: {stats.get('total_projects', 0)}")
            print(f"Active Tasks: {stats.get('active_tasks', 0)}")
            print(f"Completed Tasks: {stats.get('completed_tasks', 0)}")
            print(f"\nSummary: {result.get('summary', '')}")
        
        elif action == 'issue_created':
            print(f"Created issue in {result.get('repository')}")
            issue = result.get('issue', {})
            print(f"Issue #{issue.get('issue_number')}: {issue.get('title')}")
        
        elif action == 'search_completed':
            print(f"Found {result.get('total_found', 0)} results for '{result.get('query')}'")
            for r in result.get('results', [])[:5]:
                print(f"  - {r.get('name')}")
        
        elif action == 'help_displayed':
            print(result.get('help_text', ''))
    
    else:
        print("❌ Could not complete request")
        if result.get('error'):
            print(f"Error: {result['error']}")
        if result.get('suggestion'):
            print(f"Suggestion: {result['suggestion']}")


async def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              Simple Smart CWMAI CLI                           ║
║                                                               ║
║  Natural language interface for task management               ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Create CLI
    cli = SimpleSmartCLI()
    
    # Interactive loop
    print("Type 'help' for commands or 'quit' to exit\n")
    
    while True:
        try:
            command = input("cwmai> ").strip()
            
            if not command:
                continue
                
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Process command
            result = await cli.process_command(command)
            print_result(result)
            print()  # Empty line
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())