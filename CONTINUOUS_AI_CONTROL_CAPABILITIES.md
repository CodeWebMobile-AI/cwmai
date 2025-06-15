# Continuous AI Control Capabilities

The conversational AI assistant now includes comprehensive control capabilities for managing the continuous AI system. These features allow you to start, stop, check status, and monitor the health of the continuous AI system through natural language commands.

## Features

### 1. Check System Status
Ask the assistant if the continuous AI system is running:
- "Is the continuous AI system running?"
- "Check continuous AI status"
- "Is the continuous AI active?"

The assistant will tell you:
- Whether the system is running
- Process ID (PID)
- Uptime
- Number of active workers
- Queued and completed tasks
- System health percentage

### 2. Start the System
Start the continuous AI system with various options:
- "Start the continuous AI system"
- "Launch the continuous AI with 5 workers"
- "Start continuous AI in development mode"
- "Run the continuous AI without research engine"
- "Start the continuous AI system with no MCP"

Options you can specify:
- **Workers**: Number of parallel workers (e.g., "with 5 workers")
- **Mode**: Execution mode (production/development/test)
- **Research**: Enable/disable research engine ("without research")
- **Monitor**: Enable/disable worker monitoring ("without monitor")
- **MCP**: Enable/disable MCP integration ("with no MCP")

### 3. Stop the System
Stop the continuous AI system gracefully or forcefully:
- "Stop the continuous AI system"
- "Shutdown the continuous AI"
- "Force stop the continuous AI system"
- "Kill the continuous AI"

The system will:
- Attempt graceful shutdown by default (SIGTERM)
- Wait up to 30 seconds for graceful shutdown
- Force kill if requested or if graceful shutdown times out

### 4. Monitor System Health
Get detailed health information about the running system:
- "Monitor the continuous AI health"
- "Show continuous AI system health"
- "How healthy is the continuous AI?"

Health report includes:
- **Health Score**: 0-100 with status (healthy/degraded/unhealthy)
- **Performance Metrics**: Worker efficiency, active workers, task throughput
- **Resource Usage**: CPU percentage, memory usage, thread count
- **Issues Detected**: Any problems found
- **Recommendations**: Suggestions for improving system health

## Implementation Details

### Status Checking
The assistant checks for the continuous AI process by:
1. Checking if it started the process itself
2. Scanning running processes for `run_continuous_ai.py`
3. Reading the state file (`continuous_orchestrator_state.json`)
4. Parsing recent log entries from `continuous_ai.log`

### Health Monitoring
Health score calculation considers:
- System health metric from state file
- Worker activity vs queued tasks
- Task completion rate
- State file freshness
- Resource usage (CPU and memory)

### Process Management
- Uses `subprocess.Popen` to start the system
- Sends SIGTERM for graceful shutdown
- Falls back to SIGKILL if needed
- Tracks process state accurately

## Usage Examples

### Basic Status Check
```
User: Is the continuous AI running?
Assistant: The continuous AI system is not currently running.
           Would you like me to start it? Just say 'start the continuous AI system'.
```

### Starting with Options
```
User: Start the continuous AI with 5 workers in development mode
Assistant: 🚀 Continuous AI system started successfully (PID: 12345)

The system is now running and will:
• Continuously discover and execute tasks
• Monitor all repositories
• Generate improvements automatically

Started with options: workers=5, mode=development

You can check its status anytime by asking 'how is the continuous AI doing?'
```

### Health Monitoring
```
User: Monitor the continuous AI health
Assistant: 📊 Continuous AI System Health Report
Generated: 2024-01-15T10:30:00Z

💚 Health Score: 85/100 (healthy)

📈 Performance Metrics:
• Worker efficiency: 66.7%
• Active workers: 2
• Tasks queued: 5
• Tasks completed: 127

💻 Resource Usage:
• CPU: 15.2%
• Memory: 245.8 MB
• Threads: 12

✓ No issues detected - system is healthy!
```

## Integration with Conversational Flow

The continuous AI control is seamlessly integrated into the conversational assistant:
- Natural language understanding detects continuous AI-related commands
- Status information is included in general status queries
- Contextual suggestions guide users to available commands
- Error handling provides helpful feedback and alternatives

## Error Handling

The assistant handles errors gracefully:
- Clear error messages when operations fail
- Suggestions for alternative approaches
- Automatic fallback options (e.g., force kill if graceful shutdown fails)
- Helpful recommendations based on system state

## Future Enhancements

Potential improvements could include:
- Historical performance tracking
- Automatic restart on failure
- Performance tuning recommendations
- Integration with system notifications
- Scheduled start/stop capabilities