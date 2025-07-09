#!/usr/bin/env python3
"""Fix all relative imports in scripts directory."""

import os
import re

def fix_imports_in_file(filepath):
    """Fix relative imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match imports from local modules (not standard library or third-party)
    # This will match lines like: from module_name import something
    # But not: from scripts.module_name import something
    pattern = r'^from ([a-z_]+) import'
    
    # Common local modules in the scripts directory
    local_modules = {
        'state_manager', 'ai_brain', 'task_manager', 'repository_exclusion',
        'capability_extractor', 'external_agent_discoverer', 'external_knowledge_integrator',
        'capability_synthesizer', 'research_scheduler', 'ai_brain_factory',
        'github_issue_creator', 'mcp_integration', 'mcp_client', 'redis_integration',
        'work_item_types', 'continuous_orchestrator_no_redis', 'hybrid_state_manager',
        'dynamic_charter', 'outcome_learning', 'intelligent_task_generator', 'dynamic_validator',
        'safe_self_improver', 'task_coordinator', 'dynamic_research_trigger', 'portfolio_intelligence',
        'research_evolution_engine', 'redis_state_adapter', 'error_analyzer', 'github_issue_queue',
        'intelligence_integration', 'multi_agent_system', 'project_creator', 'alternative_task_generator'
    }
    
    modified = False
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            module_name = match.group(1)
            if module_name in local_modules:
                # Fix the import
                new_line = line.replace(f'from {module_name}', f'from scripts.{module_name}')
                new_lines.append(new_line)
                if new_line != line:
                    modified = True
                    print(f"Fixed in {filepath}: {line} -> {new_line}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))
    
    return modified

def main():
    """Fix imports in all Python files in scripts directory."""
    scripts_dir = 'scripts'
    fixed_count = 0
    
    for filename in os.listdir(scripts_dir):
        if filename.endswith('.py'):
            filepath = os.path.join(scripts_dir, filename)
            if fix_imports_in_file(filepath):
                fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == '__main__':
    main()