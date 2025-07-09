# CLAUDE.md - Important System Instructions
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai python3 
## Environment Variables for API Keys

**CRITICAL**: Before running any command that requires API keys (AI clients, GitHub integration, etc.), you MUST export the environment variables:

```bash
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3
```

### Examples:

```bash
# Testing tool creation
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3
python3 test_tool_creation_simple.py

# Running the main system
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3
./cwmai

# Testing AI client
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3
python3 scripts/test_ai_client.py
```

## Performance Optimizations

### Work Discovery Caching
The system implements caching at multiple levels to reduce API calls:

#### 1. Lifecycle Analysis Caching
Reduces repeated analysis when multiple workers check the same repositories:
- **Cache Duration**: 15 minutes by default
- **Environment Variable**: `LIFECYCLE_CACHE_MINUTES=30` (customize cache duration)
- **Cache Stats**: Available via `ProjectLifecycleAnalyzer.get_cache_stats()`
- **Manual Clear**: `ProjectLifecycleAnalyzer.clear_cache(repo_name=None)`

#### 2. Portfolio Intelligence Caching
Prevents excessive portfolio analysis during work discovery cycles:
- **Cache Duration**: 5 minutes by default
- **Environment Variable**: `PORTFOLIO_CACHE_MINUTES=10` (customize cache duration)
- **Cache Stats**: Available via `PortfolioIntelligence.get_cache_stats()`
- **Manual Clear**: `PortfolioIntelligence.clear_cache()`

These caches significantly reduce API calls when the system searches for work every 30 seconds.

## System Architecture

### AI Provider Configuration
The system is configured to use specific AI providers for different purposes:

#### Conversational AI Assistant
- **Exclusive Provider**: Google Gemini
- **Configuration**: All chat assistant requests use Gemini exclusively
- **Fallback**: If Gemini is unavailable, falls back to OpenAI → DeepSeek → Anthropic

#### Tool Generation and System Operations
- **Primary Provider**: Google Gemini (default when no model specified)
- **Priority Order**: Gemini → OpenAI → DeepSeek → Anthropic
- **Round-Robin**: Disabled for consistent provider usage

#### Automatic Provider Fallback
When any provider returns an error:
1. System logs the error and attempts the next provider in priority order
2. For "insufficient credits" errors, provider is removed from session
3. No manual intervention required - automatic failover

### Key Components:
1. **Tool Error Monitor** - Proactively detects and fixes tool errors
2. **Module Discovery** - Maps classes to modules for import resolution
3. **Enhanced Tool Regenerator** - Intelligently regenerates tools with correct imports
4. **Smart Tool Generator** - Creates tools with proper dependency management
5. **Tool Health Checker** - Validates tools before they fail

### Common Module Mappings:
- `scripts.tool_executor` → `scripts.tool_calling_system`
- `scripts.github_api` → `scripts.github_issue_creator`
- `scripts.error_reporter` → `scripts.error_handler`
- `scripts.config_manager` → `scripts.mcp_config`

### StateManager Usage:
```python
# Correct usage - save_state() takes NO arguments
state_manager = StateManager()
state_manager.state = new_data
state_manager.save_state()
```

## Testing Commands

Always export environment variables first:

```bash
# Test tool creation
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3
python3 test_improved_tool_creation.py

# Test simple tool
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3
python3 test_tool_creation_simple.py

# Run main system
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3
./cwmai
```

## Workflow System

The system includes 10 comprehensive workflows:
1. System Health Check & Auto-Fix
2. Repository Analysis & Issue Creation
3. Daily Maintenance
4. Research & Knowledge Update
5. Code Quality Pipeline
6. Multi-Tool Data Processing
7. Intelligent Task Distribution
8. System Performance Monitoring
9. Documentation Auto-Generation
10. Emergency Response

Access workflows through the AI assistant:
- CREATE_WORKFLOW: Create new workflows
- EXECUTE_WORKFLOW: Run existing workflows
- LIST_WORKFLOWS: See available workflows

## Error Handling

The system automatically fixes common errors:
- Import errors (module not found)
- Attribute errors (missing methods)
- Argument errors (incorrect function signatures)
- NoneType errors (null pointer exceptions)

## Claude Code CLI Integration

The system uses Claude Code CLI for autonomous code implementation:

### Execution Configuration
- **Input Method**: Prompts sent via stdin (not command-line arguments)
- **Permission Mode**: Uses `--dangerously-skip-permissions` flag
- **Timeout**: 5 minutes per execution
- **Working Directory**: Set to repository root

### Debugging Claude Execution
If Claude isn't making file changes:
1. Check logs for "I need permission" messages
2. Verify `--dangerously-skip-permissions` flag is present
3. Run test: `export $(cat .env.local | grep -v '^#' | xargs) && python3 test_claude_execution.py`
4. Check git status shows modified files (not just untracked)

### Performance Expectations
- Simple documentation tasks: 20-40 seconds
- Feature implementation: 60-180 seconds
- Complex refactoring: 180-300 seconds

## Conversation History

The AI assistant now maintains full conversation history including:
- User inputs and assistant responses
- Tool calls and their results
- Execution times and success status

### Key Features
- Last 10 conversation turns included in AI context
- Assistant can answer "what tools did you use?"
- New tool: `tool_usage_history` for querying past tool usage
- Full audit trail of all interactions

### Testing Conversation Memory
```bash
export $(cat .env.local | grep -v '^#' | xargs) && python3 test_conversation_history_fix.py
```

## Important Files

- `.env.local` - Contains all API keys and configuration
- `task_state.json` - Task tracking data
- `continuous_orchestrator_state.json` - Orchestrator state and metrics
- `system_state.json` - System-wide state data
- `scripts/custom_tools/` - Generated tools directory
- `scripts/workflows/` - Workflow definitions
- `CLAUDE_CODE_EXECUTION_FIX.md` - Claude Code CLI execution fix documentation
- `CONVERSATION_HISTORY_FIX.md` - Conversation history tracking fix documentation

## Claude Process Management

### Monitoring Claude Execution
- Claude now runs without timeouts (can take several minutes per task)
- Progress is logged every 30 seconds during execution
- Worker status is updated every minute in the main log
- Warnings are logged for executions taking longer than 5 minutes

### Cleaning Up Orphaned Claude Processes
If Claude processes accumulate from interrupted runs:

```bash
# Check what would be cleaned up (dry run)
python3 cleanup_claude_processes.py

# Actually clean up old processes
python3 cleanup_claude_processes.py --kill
```

The cleanup script will:
- Keep recent Claude processes (from today)
- Remove old processes from previous days
- Show CPU usage to identify stuck processes

## Dynamic Claude Executor Pool

The system now supports dynamic scaling of Claude executors based on workload:

### Configuration
Environment variables for dynamic scaling:
- `CLAUDE_MIN_EXECUTORS=1` - Minimum number of executors (default: 1)
- `CLAUDE_MAX_EXECUTORS=10` - Maximum number of executors (default: 10)
- `CLAUDE_SCALE_UP_THRESHOLD=5.0` - Queue items per executor to trigger scale up (default: 5.0)
- `CLAUDE_SCALE_DOWN_THRESHOLD=0.5` - Utilization threshold to trigger scale down (default: 0.5)
- `CLAUDE_SCALE_CHECK_INTERVAL=30` - Seconds between scaling checks (default: 30)
- `CLAUDE_IDLE_TIMEOUT=300` - Seconds before scaling down idle executors (default: 300)

### Scaling Behavior
The pool automatically:
- **Scales UP** when:
  - Queue depth per executor exceeds threshold (e.g., > 5 tasks per executor)
  - Average wait time is high (> 5 minutes)
  - Utilization is high (> 90%) with pending tasks
  - Expected wait time for new tasks is too long (> 10 minutes)

- **Scales DOWN** when:
  - Utilization is low (< 50%) with no pending tasks
  - Executors have been idle beyond timeout
  - Queue depth is very low

### Monitoring
Monitor the dynamic pool in real-time:

```bash
# Monitor pool metrics and scaling events
python3 monitor_dynamic_claude_pool.py

# Show detailed queue information
python3 monitor_dynamic_claude_pool.py --details

# Custom update interval
python3 monitor_dynamic_claude_pool.py --interval 5
```

### Testing Dynamic Scaling
Test the dynamic scaling behavior:

```bash
# Simulate different workload patterns
python3 test_dynamic_claude_pool.py

# Test with real Claude tasks (requires Claude CLI)
python3 test_dynamic_claude_pool.py --real
```

### Metrics Tracked
- **Queue Metrics**: Depth history, wait times, completion rates
- **Executor Metrics**: Utilization, idle time, task counts
- **Scaling Events**: All scale up/down events with reasons
- **Performance**: Average execution times, success/failure rates