#!/usr/bin/env python3
"""
Reorganize tools into category-based structure
"""

import os
import shutil
from pathlib import Path

# Tool categorization
TOOL_CATEGORIES = {
    'core': [
        'universal_counter.py',
        'repository_manager.py',
        'architecture_analyzer.py',
        'code_analyzer.py',
        'file_manager.py',
        'system_controller.py',
        'smart_creator.py',
        'system_monitor.py'
    ],
    'monetization': [
        'analyze_project_monetization.py',
        'discover_business_opportunities.py',
        'evaluate_project_opportunities.py',
        'generate_monetization_strategy.py',
        'create_market_analysis.py',
        'generate_business_architecture.py',
        'portfolio_monetization_report.py',
        'create_monetized_project.py'
    ],
    'repository': [
        'manage_local_repository.py',
        'repository_file_manager.py',
        'git_repository_operations.py',
        'code_generator_for_repos.py',
        'dependency_manager.py',
        'project_builder.py',
        'development_server_manager.py',
        'database_operations.py',
        'environment_config_manager.py'
    ]
}

def reorganize_tools():
    """Reorganize tools into category directories."""
    tools_dir = Path("scripts/custom_tools")
    
    # Create category directories
    for category in TOOL_CATEGORIES:
        category_dir = tools_dir / category
        category_dir.mkdir(exist_ok=True)
        print(f"Created directory: {category_dir}")
    
    # Move tools to their categories
    moved_count = 0
    for category, tools in TOOL_CATEGORIES.items():
        category_dir = tools_dir / category
        
        for tool_name in tools:
            tool_path = tools_dir / tool_name
            if tool_path.exists():
                new_path = category_dir / tool_name
                shutil.move(str(tool_path), str(new_path))
                print(f"Moved {tool_name} to {category}/")
                moved_count += 1
            else:
                print(f"Warning: {tool_name} not found")
    
    # Create __init__.py files for each category
    for category in TOOL_CATEGORIES:
        init_file = tools_dir / category / "__init__.py"
        init_file.write_text('"""Tool category: {}"""'.format(category))
    
    print(f"\n✅ Reorganization complete!")
    print(f"Moved {moved_count} tools into {len(TOOL_CATEGORIES)} categories")
    
    # List remaining tools in root
    remaining = list(tools_dir.glob("*.py"))
    if remaining:
        print(f"\nTools remaining in root directory ({len(remaining)}):")
        for tool in remaining:
            print(f"  - {tool.name}")

if __name__ == "__main__":
    reorganize_tools()