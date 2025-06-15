# CWMAI 50 Essential Commands with Self-Creation Capabilities

## Overview
CWMAI now features an intelligent conversational AI system that can:
- Execute existing commands/tools
- **Automatically create new tools when they don't exist**
- Learn from usage patterns
- Provide real system data (not generic responses)

## How It Works

### When a Command Exists:
```
User > how many repositories are we managing?
Assistant > Managing 15 repositories (12 active, 3 archived)
```

### When a Command Doesn't Exist:
```
User > count Docker containers
Assistant > I notice we don't have a 'count_docker_containers' tool yet. Let me create it for you...

Creating new tool: count_docker_containers
✓ Tool created successfully!

You have 8 running Docker containers.
The tool has been saved and will be available for future use.
```

## Current Available Commands (22 built-in + infinite AI-created)

### Repository Management
1. **count_repositories** - Count total managed repositories with breakdown
2. **get_repositories** - List repositories with filtering options
3. **analyze_repository** - Deep analysis of code quality and issues
4. **repository_health_check** - Check health status of all repos
5. **search_repositories** (AI creates on demand)
6. **clone_repository** (AI creates on demand)
7. **archive_repository** (AI creates on demand)
8. **compare_repositories** (AI creates on demand)
9. **repository_insights** (AI creates on demand)
10. **bulk_update_repositories** (AI creates on demand)

### Task & Issue Management
11. **get_tasks** - Get current tasks from queue
12. **create_issue** - Create GitHub issue with templates
13. **count_tasks** - Count tasks by status/type/priority
14. **create_task** (AI creates on demand)
15. **prioritize_tasks** (AI creates on demand)
16. **assign_task** (AI creates on demand)
17. **complete_task** (AI creates on demand)
18. **bulk_create_issues** (AI creates on demand)
19. **track_issue_progress** (AI creates on demand)
20. **schedule_recurring_task** (AI creates on demand)

### AI & System Control
21. **start_continuous_ai** - Start AI workers with config
22. **stop_continuous_ai** - Gracefully stop AI system
23. **get_continuous_ai_status** - Get AI system status
24. **ai_health_dashboard** - Comprehensive AI health metrics
25. **scale_ai_workers** (AI creates on demand)
26. **monitor_ai_performance** (AI creates on demand)
27. **configure_ai_behavior** (AI creates on demand)
28. **train_ai_on_codebase** (AI creates on demand)
29. **create_ai_agent** (AI creates on demand)
30. **orchestrate_ai_swarm** (AI creates on demand)

### Code & Development
31. **search_code** - Search for patterns across repos
32. **analyze_code_quality** (AI creates on demand)
33. **suggest_refactoring** (AI creates on demand)
34. **generate_tests** (AI creates on demand)
35. **find_security_issues** (AI creates on demand)
36. **optimize_performance** (AI creates on demand)
37. **generate_documentation** (AI creates on demand)
38. **code_review** (AI creates on demand)
39. **fix_linting_issues** (AI creates on demand)
40. **dependency_audit** (AI creates on demand)

### System Management
41. **get_system_status** - Get comprehensive system status
42. **clear_logs** - Clear old log files
43. **reset_system** - Reset system state
44. **research_topic** - Research technical topics
45. **execute_command** - Execute natural language commands

### Meta & Self-Improvement
46. **create_new_tool** - Manually create new tools
47. **enhance_tool** - Enhance existing tools
48. **get_tool_usage_stats** - Get tool usage statistics
49. **list_available_commands** - List all available commands
50. **learn_from_usage** (AI creates on demand)

## Usage Examples

### Direct Commands
```bash
# Start conversational mode
export $(cat .env.local | grep -v '^#' | xargs) && ./cwmai

# Single command
./cwmai "how many repositories?"
./cwmai "create an issue about login bug"
./cwmai "show system status"
```

### Natural Language Understanding
The AI understands various phrasings:
- "how many repos?" → executes `count_repositories()`
- "list my tasks" → executes `get_tasks()`
- "check system health" → executes `ai_health_dashboard()`
- "what can you do?" → executes `list_available_commands()`

### Tool Creation Examples
```
User > analyze Docker performance
Assistant > I notice we don't have an 'analyze_docker_performance' tool. Creating it now...
[Creates tool that analyzes Docker container metrics]

User > generate weekly report
Assistant > Creating 'generate_weekly_report' tool...
[Creates tool that generates comprehensive weekly reports]
```

## Key Features

### 1. **Intelligent Tool Detection**
- Understands user intent from natural language
- Maps requests to appropriate tools
- Suggests similar tools if exact match not found

### 2. **Automatic Tool Creation**
- Analyzes tool name and user intent
- Generates appropriate tool code using AI
- Saves tools for future use
- Tools integrate with existing system components

### 3. **Self-Improvement**
- Tracks tool usage statistics
- Learns from successful patterns
- Can enhance existing tools based on usage
- Suggests new tools based on common requests

### 4. **Real System Integration**
- All tools work with actual system data
- No hardcoded or generic responses
- Full access to state manager, task manager, etc.
- Proper error handling and validation

## Advanced Usage

### Creating Custom Tools Manually
```
User > create a tool to monitor Redis performance
Assistant > I'll create a Redis performance monitoring tool for you...
[Generates complete tool with metrics collection, analysis, and reporting]
```

### Enhancing Existing Tools
```
User > enhance the repository analyzer to check for security vulnerabilities
Assistant > I'll add security scanning capabilities to the repository analyzer...
[Updates tool with new security features]
```

### Batch Operations
```
User > create issues for all repos missing documentation
Assistant > I'll analyze all repositories and create documentation issues...
[Creates 'batch_create_doc_issues' tool and executes it]
```

## Configuration

### Environment Variables
```bash
# Required for API fallback
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
```

### Tool Storage
- Built-in tools: `/scripts/tool_calling_system.py`
- AI-created tools: `/scripts/custom_tools/`
- Tool metadata preserved for future sessions

## Benefits

1. **No More "Command Not Found"**
   - System creates missing functionality on demand
   - Natural language understanding prevents confusion

2. **Continuous Evolution**
   - System becomes smarter with use
   - Tools improve based on patterns
   - New capabilities emerge organically

3. **Developer Friendly**
   - Write requests naturally
   - No need to memorize exact command names
   - System guides you to the right tool

4. **Fully Integrated**
   - All tools work with real system components
   - Consistent interface and error handling
   - Automatic parameter validation

## Future Enhancements

The system will continue to evolve with:
- Tool composition (combining multiple tools)
- Workflow automation
- Predictive tool creation
- Cross-tool optimization
- Community tool sharing