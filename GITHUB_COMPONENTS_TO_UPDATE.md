# GitHub Components to Update for Local Mode

## Core GitHub Integration Files

### Primary GitHub Classes
1. **scripts/github_issue_creator.py** - GitHubIssueCreator class
2. **scripts/project_creator.py** - ProjectCreator class  
3. **scripts/github_retry_manager.py** - GitHub API retry logic
4. **scripts/repository_analyzer.py** - Repository analysis functionality

### Local Mode Implementations (Already Created)
1. **scripts/local_issue_creator.py** - LocalIssueCreator
2. **scripts/local_project_creator.py** - LocalProjectCreator
3. **scripts/local_github_simulator.py** - Local GitHub simulator
4. **scripts/storage_factory.py** - Factory for switching between local/GitHub

## Files Importing GitHubIssueCreator (40 files)
- scripts/continuous_orchestrator.py
- scripts/continuous_orchestrator_no_redis.py
- scripts/mcp_github_issue_creator.py
- scripts/enhanced_continuous_orchestrator.py
- scripts/github_issue_queue.py
- scripts/github_issue_queue_no_redis.py
- scripts/github_issue_queue_with_retry.py
- scripts/conversational_ai_assistant.py
- scripts/tool_calling_system.py
- scripts/intelligent_tool_generator.py
- scripts/fix_tool_generation_system.py
- scripts/smart_tool_generator_improvements.py
- scripts/auto_wrap_creator.py
- scripts/intelligent_method_proxy.py
- scripts/mcp_continuous_orchestrator.py

### Custom Tools Using GitHubIssueCreator
- scripts/custom_tools/get_repositories.py
- scripts/custom_tools/check_intelligence.py
- scripts/custom_tools/check_architecture_clean.py
- scripts/custom_tools/check_architecture_simple.py
- scripts/custom_tools/check_and_ensure_architecture_files.py
- scripts/custom_tools/check_architecture_files_final.py
- scripts/custom_tools/delete_all_issues.py
- scripts/custom_tools/create_tool_from_search.py
- scripts/custom_tools/tool_task_8.py
- scripts/custom_tools/search_codebase.py
- scripts/custom_tools/fix_all_repositories.py
- scripts/custom_tools/ensure_claude_workflow.py
- scripts/custom_tools/security_vulnerability_scanner.py
- scripts/custom_tools/analyze_tool_success.py
- scripts/custom_tools/calculate_project_statistics.py
- scripts/custom_tools/get_repository_issues.py
- scripts/custom_tools/get_repository_issue_content.py
- scripts/custom_tools/control_system.py

## Files Importing ProjectCreator (12 files)
- scripts/continuous_orchestrator.py
- scripts/continuous_orchestrator_no_redis.py
- scripts/hybrid_orchestrator_enhanced.py
- scripts/enhanced_hybrid_orchestrator.py
- scripts/dynamic_god_mode_controller.py
- scripts/hybrid_continuous_orchestrator.py
- scripts/architecture_generator.py

## Files Using GitHub API Directly (33 files)
### State Management
- scripts/state_manager.py
- scripts/state_manager_fixed.py
- scripts/task_manager.py
- scripts/update_task_dashboard.py

### Repository Management
- scripts/download_all_github_issues.py
- scripts/download_github_issues.py
- scripts/multi_repo_coordinator.py
- scripts/external_agent_discoverer.py
- scripts/repository_analyzer.py
- scripts/clean_deleted_repos.py
- scripts/task_analyzer.py

### Architecture Management
- scripts/architecture_generator.py
- simple_architecture_fix.py
- update_all_architectures.py
- quick_architecture_fix.py
- fix_repository_discovery.py
- fix_repository_customizations.py
- fix_remote_state.py
- fix_generic_readmes.py
- fix_architecture_properly.py
- cleanup_deleted_repos.py
- batch_update_architectures.py

### Migration Scripts
- migrate_github_to_local.py
- clone_all_repos.py

## Files with Repository Operations (53 files)
### Issue Creation
- Functions: create_issue(), create_github_issue()
- Used in: continuous_orchestrator.py, github_issue_creator.py, task_manager.py, etc.

### Repository Creation  
- Functions: create_repository()
- Used in: project_creator.py, architecture_generator.py

### Repository Listing
- Functions: list_repositories(), get_repositories(), fetch_repositories()
- Used in: state_manager.py, repository_analyzer.py, custom tools

### Repository Queries
- Functions: get_repo(), get_issues()
- Used in: github_issue_creator.py, repository_analyzer.py

## Custom Tools for Repository Operations
1. **scripts/custom_tools/get_repositories_evolved.py** - Advanced repository listing
2. **scripts/custom_tools/get_repositories.py** - Basic repository listing
3. **scripts/custom_tools/get_repositories_simple.py** - Simple repository listing
4. **scripts/custom_tools/count_repositories_evolved.py** - Repository counting
5. **scripts/custom_tools/get_repository_creation_date.py** - Get repo creation date
6. **scripts/custom_tools/get_repository_issues.py** - Get repository issues
7. **scripts/custom_tools/get_repository_issue_content.py** - Get issue content
8. **scripts/custom_tools/get_me.py** - Get current user info
9. **scripts/custom_tools/generate_weekly_report.py** - Weekly report generation

## Workflow System Files
- scripts/system_workflows.py - Uses repository operations
- scripts/workflows/response_cache_optimizer.py

## Conversational AI Assistants
- scripts/conversational_ai_assistant.py
- scripts/conversational_ai_assistant_v4.py
- scripts/conversational_ai_assistant_v5.py
- scripts/conversational_ai_assistant_v5_optimized.py

## Natural Language Interfaces
- scripts/natural_language_interface.py
- scripts/smart_natural_language_interface.py

## Test Files Using GitHub
- test_local_mode.py
- test_local_mode_simple.py
- test_ultimate_local.py
- verify_local_mode.py
- test_github_retry.py
- test_get_repositories_evolved.py

## Update Strategy

### Phase 1: Core Infrastructure
1. Ensure storage_factory.py properly switches between local/GitHub modes
2. Update all imports to use storage_factory instead of direct imports
3. Add environment variable checks for LOCAL_MODE

### Phase 2: Custom Tools
1. Update all custom tools to use storage_factory
2. Ensure tools handle both local and GitHub responses
3. Add error handling for mode switching

### Phase 3: Orchestrators and Managers
1. Update continuous_orchestrator.py and variants
2. Update state_manager.py to support local storage
3. Update task_manager.py for local task storage

### Phase 4: Testing
1. Create comprehensive tests for both modes
2. Ensure seamless switching between modes
3. Validate all functionality works in both modes