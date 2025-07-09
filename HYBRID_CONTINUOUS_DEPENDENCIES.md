# Hybrid Continuous System Dependencies

This document lists all files associated with `run_hybrid_continuous.py` and its dependencies.

## Main Entry Point
- `run_hybrid_continuous.py` - Main entry point for the hybrid continuous system

## Core Orchestrator
- `scripts/hybrid_orchestrator_enhanced.py` - Enhanced hybrid orchestrator with dynamic work generation
- `scripts/continuous_orchestrator_no_redis.py` - Base continuous orchestrator (Redis-free version)

## Work Management
- `scripts/work_item_types.py` - Work item data structures and enums
- `scripts/dynamic_work_generator.py` - Dynamic work generation based on context
- `scripts/intelligent_work_finder.py` - Intelligent work discovery system
- `scripts/work_distribution_patch.py` - Work distribution fixes
- `scripts/in_memory_work_queue.py` - In-memory work queue implementation

## State Management
- `scripts/state_manager.py` - Main state management system
- `scripts/hybrid_state_manager.py` - Hybrid state manager for orchestrator
- `scripts/sqlite_state_manager.py` - SQLite-based state persistence
- `scripts/storage_factory.py` - Factory for creating storage adapters
- `scripts/storage_mode.py` - Storage mode configurations
- `scripts/unified_task_state.py` - Unified task state management
- `scripts/task_persistence.py` - Task persistence layer

## AI Components
- `scripts/ai_brain.py` - Base AI brain interface
- `scripts/ai_brain_factory.py` - Factory for creating AI brain instances
- `scripts/http_ai_client.py` - HTTP-based AI client
- `scripts/conversational_ai_assistant.py` - Conversational AI interface

## Dynamic Components
- `scripts/dynamic_charter.py` - Dynamic charter system for adaptive behavior
- `scripts/outcome_learning.py` - Outcome-based learning system
- `scripts/intelligent_task_generator.py` - Intelligent task generation
- `scripts/dynamic_validator.py` - Dynamic task validation
- `scripts/dynamic_task_selector.py` - Dynamic task selection
- `scripts/dynamic_task_loader.py` - Dynamic task loading

## Repository Analysis
- `scripts/repository_analyzer.py` - Repository analysis and caching
- `scripts/repository_exclusion.py` - Repository exclusion logic
- `scripts/repository_analysis_cache.py` - Repository analysis caching
- `scripts/repository_data_provider.py` - Repository data provider

## Project Management
- `scripts/project_creator.py` - Project creation functionality
- `scripts/project_context_model.py` - Project context data models
- `scripts/portfolio_intelligence.py` - Portfolio analysis and intelligence
- `scripts/multi_repo_coordinator.py` - Multi-repository coordination
- `scripts/market_research_engine.py` - Market research functionality

## Local Mode Support
- `scripts/local_github_simulator.py` - Local GitHub simulator for offline mode
- `scripts/local_issue_creator.py` - Local issue creation
- `scripts/sync_local_repos.py` - Local repository synchronization

## Claude Integration
- `scripts/claude_executor.py` - Claude Code CLI executor
- `scripts/claude_execution_queue.py` - Claude execution queue management
- `scripts/dynamic_claude_executor_pool.py` - Dynamic Claude executor pool

## Monitoring & Analytics
- `scripts/monitoring_integration.py` - Task monitoring integration
- `scripts/task_monitoring_system.py` - Task monitoring system
- `scripts/worker_status_monitor_no_redis.py` - Worker status monitoring

## Tool System
- `scripts/tool_calling_system.py` - Tool calling system
- `scripts/tool_executor.py` - Tool execution engine
- `scripts/tool_loader.py` - Tool loading system

## Configuration Files
- `.env.local` - Environment variables and API keys
- `mcp_config.json` - MCP configuration
- `production_context.json` - Production context configuration
- `evolutionary_config.json` - Evolutionary system configuration

## State Files (Runtime)
- `system_state.json` - System-wide state data
- `continuous_orchestrator_state.json` - Orchestrator state and metrics
- `task_state.json` - Task tracking data
- `task_history.json` - Historical task data
- `claude_execution_queue.json` - Claude execution queue state
- `in_progress_tasks.json` - Currently executing tasks
- `work_queue_backup.json` - Work queue backup
- `dynamic_claude_pool_stats.json` - Claude pool statistics
- `processed_issues.json` - Processed issues tracking

## Log Files
- `hybrid_continuous.log` - Main hybrid system log
- `ai_api_communication.log` - AI API communication log
- `claude_executor.log` - Claude executor log

## Database Files
- `cwmai_state.db` - SQLite state database
- `redis_fallback.db` - Redis fallback database
- `ultimate_orchestrator.db` - Ultimate orchestrator database

## Backup Files
- Various `.backup_*` files for state recovery

## Scripts Directory Structure
The system also depends on these subdirectories in scripts/:
- `scripts/custom_tools/` - Generated custom tools
- `scripts/workflows/` - Workflow definitions
- `scripts/improvements/` - Improvement modules
- `scripts/task_type_configs/` - Task type configurations
- `scripts/tool_health_reports/` - Tool health reports
- `scripts/redis_integration/` - Redis integration modules (legacy)
- `scripts/specialized_agents/` - Specialized agent implementations
- `scripts/evolutionary_plugins/` - Evolutionary plugin system

## Key Dependencies Summary
The hybrid continuous system is built on:
1. **State Management**: SQLite and in-memory state management
2. **Work Discovery**: Intelligent work finder and dynamic work generator
3. **Task Execution**: Claude executor pool and tool calling system
4. **Repository Management**: Local and GitHub repository handling
5. **AI Integration**: Multiple AI providers with fallback support
6. **Monitoring**: Comprehensive task and worker monitoring
7. **Learning Systems**: Outcome-based and pattern learning

## Usage
To run the hybrid continuous system:
```bash
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai python3 run_hybrid_continuous.py
```

## Important Notes
- The system uses SQLite instead of Redis for persistence
- Supports both local and GitHub storage modes
- Includes dynamic Claude executor pool for scaling
- Features intelligent work discovery and generation
- Implements comprehensive monitoring and analytics