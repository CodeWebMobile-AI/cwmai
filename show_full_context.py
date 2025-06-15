#!/usr/bin/env python3
"""
Show the full context available to the tool generation system
"""

import json
from scripts.tool_generation_templates import ToolGenerationTemplates


def main():
    """Show the complete context available."""
    templates = ToolGenerationTemplates()
    
    print("CWMAI Tool Generation Context Discovery")
    print("=" * 80)
    
    # Show summary
    print("\nModule Discovery Summary:")
    print(templates.get_available_modules_summary())
    
    # Show custom tools specifically
    print("\n" + "=" * 80)
    print("Custom Tools Available:")
    custom_tools = []
    for module_name, info in templates.discovered_scripts.items():
        if info.get('category') == 'custom_tools':
            tool_name = module_name.split('.')[-1]
            desc = info['docstring'][:60] + "..." if len(info['docstring']) > 60 else info['docstring']
            custom_tools.append((tool_name, desc))
    
    for i, (tool, desc) in enumerate(sorted(custom_tools), 1):
        print(f"{i}. {tool}: {desc}")
    
    # Show categories with counts
    print("\n" + "=" * 80)
    print("All Module Categories:")
    categories = {}
    for module_name, info in templates.discovered_scripts.items():
        cat = info.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(module_name)
    
    for cat, modules in sorted(categories.items()):
        print(f"\n{cat.upper()} ({len(modules)} modules):")
        # Show first 5 modules in each category
        for module in sorted(modules)[:5]:
            info = templates.discovered_scripts[module]
            desc = info['docstring'][:50] + "..." if len(info['docstring']) > 50 else info['docstring']
            print(f"  - {module}: {desc}")
        if len(modules) > 5:
            print(f"  ... and {len(modules) - 5} more")
    
    print("\n" + "=" * 80)
    print("Context Integration Complete!")
    print("\nThe tool generation system now has full awareness of:")
    print(f"✓ {len(templates.discovered_scripts)} total modules")
    print(f"✓ {len(custom_tools)} custom tools ready to use")
    print("✓ All project modules organized by category")
    print("✓ Dynamic import suggestions based on tool requirements")
    print("\nWhen generating new tools, the AI will have access to all these")
    print("modules and can intelligently choose which ones to import based")
    print("on the tool's requirements.")


if __name__ == "__main__":
    main()