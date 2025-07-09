# CWMAI Continuous System Guide

## Overview
The CWMAI Continuous System is an autonomous AI orchestration platform that runs 24/7, continuously discovering and executing work items across your GitHub repositories.

## Quick Start

### 1. Start the System
```bash
./start_continuous_system.sh
```

### 2. Monitor Status
```bash
python monitor_continuous_system.py
```

### 3. Stop the System
```bash
# Press Ctrl+C in the running terminal, or:
pkill -f "run_continuous_ai.py"
```

## System Modes

### Enhanced Mode (Default)
- Full MCP (Model Context Protocol) integration
- Redis-based state management
- GitHub issue creation via MCP
- Advanced work discovery
- Parallel processing with multiple workers

```bash
./start_continuous_system.sh
```

### Basic Mode
- Direct API integrations only
- File-based state management
- Simpler operation without Redis

```bash
./start_continuous_system.sh --basic
```

## Configuration Options

### Worker Configuration
```bash
# Set number of parallel workers (default: 5)
./start_continuous_system.sh --workers 10

# Disable parallel processing
./start_continuous_system.sh --no-parallel
```

### Feature Toggles
```bash
# Disable research evolution engine
./start_continuous_system.sh --no-research

# Combine options
./start_continuous_system.sh --basic --workers 3 --no-research
```

## What the System Does

### 1. Repository Discovery
- Scans your GitHub repositories
- Identifies projects needing work
- Prioritizes based on activity and issues

### 2. Work Item Generation
- Creates tasks for:
  - Bug fixes
  - Feature implementations
  - Documentation updates
  - Code improvements
  - Architecture documentation

### 3. Autonomous Execution
- Assigns work to AI workers
- Executes tasks in parallel
- Creates GitHub issues for tracking
- Updates system state continuously

### 4. Self-Improvement
- Learns from completed tasks
- Evolves work generation strategies
- Optimizes resource allocation

## System Architecture

```
┌─────────────────────────────────────────┐
│         Continuous Orchestrator         │
├─────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐  │
│  │ Work Queue  │───▶│  AI Workers  │  │
│  └─────────────┘    └──────────────┘  │
│         │                    │         │
│         ▼                    ▼         │
│  ┌─────────────┐    ┌──────────────┐  │
│  │   GitHub    │    │   Redis/MCP  │  │
│  │ Integration │    │    State     │  │
│  └─────────────┘    └──────────────┘  │
└─────────────────────────────────────────┘
```

## Monitoring & Logs

### Real-time Logs
```bash
tail -f continuous_ai.log
```

### System State
```bash
cat continuous_orchestrator_state.json | jq
```

### Task History
```bash
cat task_history.json | jq '.tasks[-10:]'
```

## Troubleshooting

### Redis Connection Issues
```bash
# Check Redis status
redis-cli ping

# Start Redis if needed
redis-server --daemonize yes
```

### Architecture Issues
```bash
# Force ARM64 mode (for Apple Silicon)
arch -arm64 ./start_continuous_system.sh
```

### Reset System State
```bash
# Backup current state
cp continuous_orchestrator_state.json continuous_orchestrator_state.backup.json

# Reset state
echo '{"status": "ready", "work_items_in_progress": [], "active_workers": {}}' > continuous_orchestrator_state.json
```

## Environment Variables

Required in `.env.local`:
- `GITHUB_TOKEN` - GitHub personal access token
- `ANTHROPIC_API_KEY` - Claude API key
- `REDIS_URL` - Redis connection URL (for enhanced mode)

Optional:
- `OPENAI_API_KEY` - For GPT models
- `GEMINI_API_KEY` - For Gemini models
- `MCP_ENABLED` - Enable/disable MCP (default: true)

## Safety Features

1. **Duplicate Prevention** - Won't create duplicate issues
2. **Rate Limiting** - Respects API limits
3. **Error Recovery** - Automatic retry with backoff
4. **State Persistence** - Survives restarts
5. **Repository Exclusions** - Configurable blacklist

## Best Practices

1. **Start Small** - Begin with 2-3 workers
2. **Monitor Initially** - Watch logs for first hour
3. **Review Generated Work** - Check created issues
4. **Adjust Configuration** - Tune based on needs
5. **Regular Maintenance** - Clean old logs weekly

## Advanced Usage

### Custom Work Generation
Edit `scripts/continuous_orchestrator.py` to customize work discovery logic.

### Integration with CI/CD
The system can trigger builds and deployments through GitHub Actions.

### Multi-Repository Management
Configure repository priorities in `system_state.json`.

## Support

- Check logs: `continuous_ai.log`
- System state: `continuous_orchestrator_state.json`
- Monitor script: `python monitor_continuous_system.py`
- Force stop: `pkill -f continuous_ai`