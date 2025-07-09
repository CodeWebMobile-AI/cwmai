#!/usr/bin/env python3
"""
Script to remove old tools after consolidation
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# List of old tools to remove
OLD_TOOLS = [
    # Counting tools (18)
    "count_files_in_directory.py",
    "count_files.py",
    "count_how_many_python.py",
    "count_lines_by_language.py",
    "count_markdown_files.py",
    "count_python_files_in_directory.py",
    "count_python_files.py",
    "count_repositories_evolved.py",
    "calculate_project_statistics.py",
    "calculate_the_average_file.py",
    "find_all_functions_that.py",
    "find_async_functions.py",
    "find_todo_comments_in_python_files.py",
    "tool_task_1.py",
    "tool_task_2.py",
    "tool_task_3.py",
    "tool_task_4.py",
    "tool_task_5.py",
    
    # Repository tools (14)
    "get_repositories_evolved.py",
    "get_repositories.py",
    "list_repositories.py",
    "get_repository_issues.py",
    "get_repository_issue_content.py",
    "get_repository_creation_date.py",
    "analyze_repository_health.py",
    "ensure_claude_workflow.py",
    "fix_all_repositories.py",
    "generate_weekly_report.py",
    "tool_task_6.py",
    "tool_task_8.py",
    "download_all_github_issues.py",
    "download_github_issues.py",
    
    # Architecture tools (8)
    "check_architecture_files.py",
    "check_architecture_clean.py",
    "check_architecture_files_final.py",
    "check_architecture_simple.py",
    "check_for_architecture_file.py",
    "ensure_architecture_docs.py",
    "update_architecture_docs.py",
    "check_and_ensure_architecture_files.py",
    
    # Analysis tools (16)
    "analyze_code_complexity.py",
    "analyze_file_types_and_size.py",
    "analyze_file_types.py",
    "analyze_json_files.py",
    "analyze_tool_success.py",
    "advanced_code_analysis.py",
    "code_complexity_analyzer.py",
    "security_vulnerability_scanner.py",
    "test_coverage_analyzer.py",
    "performance_analyzer.py",
    "dependency_checker.py",
    "analyze_create_tool_like_claude.py",
    "deep_code_analyzer.py",
    "code_documentation_generator.py",
    "check_laravel_react_completeness.py",
    "setup_laravel_react_starter.py",
    
    # File tools (12)
    "list_files.py",
    "read_file.py",
    "grep_files.py",
    "search_codebase.py",
    "search_repos.py",
    "show_me_files_changed_today.py",
    "access_public_api_tool.py",
    "get_me.py",
    "list_tasks.py",
    "api_endpoint_tester.py",
    "smart_research.py",
    
    # System tools (6)
    "start_system.py",
    "stop_system.py",
    "restart_system.py",
    "check_system_status.py",
    "control_system.py",
    "system_performance_monitor.py",
    
    # Creation tools (2)
    "create_tool_from_search.py",
    "create_tool_like_claude.py",
    "create_tool_from_search_simple.py",
    
    # Other tools
    "monitor_worker_activity.py",
    "how_many_active_workers_do_we_.py",
    "orchestrate_workflows.py",
    
    # Test/Demo tools
    "demo_calculator.py",
    "simple_test_tool.py",
    "test_pandas_tool.py",
    "test_tool_final.py",
    "create_spaced_tool.py",
    "whats_the_disk_usage_of_log_fi.py",
    
    # Additional tools found
    "add_missing_configs.py",
    "check_intelligence.py",
    "delete_all_issues.py",
    "get_project_info.py",
    "get_repositories_simple.py",
    "list_tools.py",
    "test_generator.py"
]

def main():
    """Remove old tools and create backup"""
    tools_dir = Path("scripts/custom_tools")
    backup_dir = Path(f"old_tools_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create backup directory
    backup_dir.mkdir(exist_ok=True)
    
    removed_count = 0
    backed_up = []
    not_found = []
    
    print("Removing old tools after consolidation...")
    print(f"Backup directory: {backup_dir}")
    print("=" * 60)
    
    for tool_name in OLD_TOOLS:
        tool_path = tools_dir / tool_name
        
        if tool_path.exists():
            # Backup the file
            backup_path = backup_dir / tool_name
            shutil.copy2(tool_path, backup_path)
            backed_up.append(tool_name)
            
            # Remove the file
            tool_path.unlink()
            removed_count += 1
            print(f"✓ Removed: {tool_name}")
        else:
            not_found.append(tool_name)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tools removed: {removed_count}")
    print(f"Tools backed up: {len(backed_up)}")
    print(f"Tools not found: {len(not_found)}")
    
    if not_found:
        print(f"\nTools not found ({len(not_found)}):")
        for tool in not_found[:10]:  # Show first 10
            print(f"  - {tool}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")
    
    # List remaining tools
    remaining_tools = list(tools_dir.glob("*.py"))
    consolidated_tools = [
        "universal_counter.py",
        "repository_manager.py", 
        "architecture_analyzer.py",
        "code_analyzer.py",
        "file_manager.py",
        "system_controller.py",
        "smart_creator.py",
        "system_monitor.py"
    ]
    
    other_tools = [t.name for t in remaining_tools if t.name not in consolidated_tools]
    
    print(f"\nConsolidated tools remaining ({len(consolidated_tools)}):")
    for tool in consolidated_tools:
        if (tools_dir / tool).exists():
            print(f"  ✓ {tool}")
    
    if other_tools:
        print(f"\nOther tools still present ({len(other_tools)}):")
        for tool in other_tools[:20]:  # Show first 20
            print(f"  - {tool}")
        if len(other_tools) > 20:
            print(f"  ... and {len(other_tools) - 20} more")
    
    print(f"\n✅ Old tools have been backed up to: {backup_dir}")
    print("🎉 Tool consolidation complete!")

if __name__ == "__main__":
    main()