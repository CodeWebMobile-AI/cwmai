#!/usr/bin/env python3
"""
Demo script to showcase enhanced terminal features
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.syntax import Syntax

console = Console()

def main():
    """Demonstrate the enhanced terminal features."""
    
    # Welcome panel
    console.print(Panel.fit(
        "[bold cyan]CWMAI Enhanced Terminal Demo[/bold cyan]\n\n"
        "Showing off the new Claude Code-like features!",
        border_style="cyan",
        box=box.DOUBLE_EDGE
    ))
    
    # Feature table
    table = Table(title="New Terminal Features", box=box.ROUNDED)
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Status", justify="center", style="green")
    
    features = [
        ("Command History", "Use ↑/↓ arrow keys to navigate previous commands", "✅"),
        ("Bracketed Paste", "Paste without auto-execution - review before Enter", "✅"),
        ("Tab Completion", "Press Tab to autocomplete common commands", "✅"),
        ("Syntax Highlighting", "Code blocks are highlighted with proper syntax", "✅"),
        ("Rich Formatting", "Beautiful panels, tables, and markdown rendering", "✅"),
        ("Ctrl+C Handling", "Graceful interrupt without crashing", "✅"),
        ("Clear Command", "Type 'clear' to clear the screen", "✅"),
        ("History File", "Commands saved to ~/.cwmai_history", "✅"),
    ]
    
    for feature, desc, status in features:
        table.add_row(feature, desc, status)
    
    console.print("\n")
    console.print(table)
    console.print("\n")
    
    # Example code highlighting
    console.print(Panel("Example: Syntax Highlighting", border_style="yellow"))
    
    code = '''def calculate_project_statistics(**kwargs):
    """Calculate statistics for a repository."""
    repo_name = kwargs.get('repo_name')
    if not repo_name:
        # Extract from context
        context = kwargs.get('context', {})
        user_input = context.get('user_input', '')
        # Pattern: "repo-name 'tool_name'"
        match = re.search(r'^["\']?([a-zA-Z0-9-_]+)', user_input)
        if match:
            repo_name = match.group(1)
    
    return {"repo": repo_name, "stats": {...}}'''
    
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)
    
    console.print("\n")
    
    # Usage instructions
    console.print(Panel.fit(
        "[yellow]Try it yourself![/yellow]\n\n"
        "1. Run [bold]./cwmai[/bold] to start the enhanced terminal\n"
        "2. Type some commands and use arrow keys to recall them\n"
        "3. Copy and paste multiple lines - they won't execute until you press Enter\n"
        "4. Press Tab to see command suggestions\n"
        "5. Type 'clear' to clear the screen\n\n"
        "[dim]Your command history is saved in ~/.cwmai_history[/dim]",
        border_style="yellow"
    ))

if __name__ == "__main__":
    main()