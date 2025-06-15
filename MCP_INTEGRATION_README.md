# MCP Integration for CWMAI

## Overview

CWMAI now supports Model Context Protocol (MCP) servers for enhanced external integrations. MCPs provide standardized, reliable interfaces to external services with built-in retry logic, rate limiting, and consistent error handling.

## Quick Start

1. **Install MCP servers:**
   ```bash
   ./setup_mcp.sh
   ```

2. **Set environment variables:**
   ```bash
   export GITHUB_TOKEN="your-github-token"
   export MCP_ENABLED=true
   ```

3. **Test MCP integration:**
   ```bash
   python test_mcp_integration.py
   ```

## Available MCPs

### 1. GitHub MCP
- **Purpose**: GitHub API operations
- **Features**: Issue/PR creation, repository search, label management
- **Configuration**: Requires `GITHUB_TOKEN`

### 2. Filesystem MCP
- **Purpose**: Safe file operations
- **Features**: Read/write files, directory operations
- **Configuration**: Allowed directories in `mcp_config.json`

### 3. Memory MCP
- **Purpose**: Persistent context storage
- **Features**: Cross-session state, fast caching
- **Configuration**: No additional config needed

### 4. MySQL MCP
- **Purpose**: Database operations
- **Features**: Task persistence, analytics, metrics
- **Configuration**: Set MySQL credentials in environment

### 5. Git MCP
- **Purpose**: Repository operations
- **Features**: Commits, branches, history
- **Configuration**: Repository path in config

### 6. Fetch MCP
- **Purpose**: HTTP operations
- **Features**: API calls, web scraping
- **Configuration**: Max redirects in config

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   CWMAI Core                    │
├─────────────────────────────────────────────────┤
│           MCP Integration Layer                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ GitHub   │  │Filesystem│  │ Memory   │     │
│  │   MCP    │  │   MCP    │  │   MCP    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  MySQL   │  │   Git    │  │  Fetch   │     │
│  │   MCP    │  │   MCP    │  │   MCP    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
```

## Key Components

### 1. MCP Client (`scripts/mcp_client.py`)
Low-level client handling MCP protocol communication via stdio/websocket.

### 2. MCP Integration Hub (`scripts/mcp_integration.py`)
High-level interface providing typed methods for each MCP server.

### 3. Enhanced Components
- `mcp_github_issue_creator.py` - GitHub issue creation via MCP
- `enhanced_continuous_orchestrator.py` - Orchestrator with MCP support
- `mcp_market_research_engine.py` - Market research using Fetch MCP
- `mcp_task_persistence.py` - MySQL-based task storage

### 4. Monitoring (`scripts/mcp_monitor.py`)
Health checks, performance metrics, and automatic recovery.

## Usage Examples

### Basic Usage
```python
from scripts.mcp_integration import MCPIntegrationHub

async with MCPIntegrationHub() as mcp:
    # GitHub operations
    issue = await mcp.github.create_issue(
        repo="owner/repo",
        title="New feature",
        body="Description",
        labels=["enhancement"]
    )
    
    # File operations
    content = await mcp.filesystem.read_file("/path/to/file")
    
    # Memory storage
    await mcp.memory.store_context("key", {"data": "value"})
```

### Enhanced Orchestrator
```python
from scripts.enhanced_continuous_orchestrator import EnhancedContinuousOrchestrator

orchestrator = EnhancedContinuousOrchestrator(enable_mcp=True)
await orchestrator.initialize()
await orchestrator.run()
```

## Configuration

### Environment Variables
```bash
# Core
MCP_ENABLED=true
MCP_CONFIG_PATH=mcp_config.json
MCP_LOG_LEVEL=INFO

# GitHub
GITHUB_TOKEN=your-token

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=cwmai
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=cwmai
```

### Configuration File (`mcp_config.json`)
```json
{
  "servers": {
    "github": {
      "name": "github",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  },
  "default_timeout": 30,
  "max_retries": 3
}
```

## Monitoring

### Health Checks
```python
from scripts.mcp_monitor import check_mcp_health

health = await check_mcp_health(mcp_hub)
print(f"Healthy servers: {health['healthy']}/{health['total_servers']}")
```

### Continuous Monitoring
```python
from scripts.mcp_monitor import MCPMonitor

monitor = MCPMonitor(mcp_hub, check_interval=60)
await monitor.start_monitoring()
```

## GitHub Actions Integration

The enhanced workflow (`continuous-ai-mcp.yml`) includes:
- Automatic MCP server setup
- Health monitoring
- Graceful fallback to direct APIs
- Comprehensive reporting

Enable MCP in workflows:
```yaml
- name: Run with MCP
  env:
    MCP_ENABLED: true
```

## Migration Guide

See `MCP_MIGRATION_GUIDE.md` for detailed migration instructions.

### Quick Migration
1. Replace `PyGithub` → `mcp.github`
2. Replace `requests` → `mcp.fetch`
3. Replace file I/O → `mcp.filesystem`
4. Add MySQL persistence → `mcp.mysql`

## Troubleshooting

### MCP Server Not Starting
```bash
# Check Node.js
node --version  # Should be 16+

# Install servers manually
npm install -g @modelcontextprotocol/server-github
```

### Authentication Issues
```bash
# Verify token
echo $GITHUB_TOKEN

# Test GitHub MCP
npx @modelcontextprotocol/server-github --test
```

### Connection Problems
```python
# Enable debug logging
import logging
logging.getLogger('scripts.mcp_client').setLevel(logging.DEBUG)
```

## Benefits

1. **Reliability**: Built-in retry and error handling
2. **Performance**: Connection pooling and caching
3. **Consistency**: Uniform interface across services
4. **Monitoring**: Health checks and metrics
5. **Flexibility**: Easy to add new MCP servers

## Future Enhancements

- [ ] Slack MCP for notifications
- [ ] PostgreSQL MCP alternative
- [ ] Kubernetes MCP for deployments
- [ ] AWS/GCP/Azure MCPs for cloud operations
- [ ] Custom MCP servers for specialized tasks

## Support

For issues or questions:
1. Check logs in `mcp_*.log` files
2. Run `python test_mcp_integration.py`
3. Review `MCP_MIGRATION_GUIDE.md`
4. Open an issue with MCP health report