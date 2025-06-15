# MCP Migration Guide for CWMAI

This guide explains how to migrate CWMAI components to use Model Context Protocol (MCP) servers for external integrations.

## Overview

MCPs provide standardized, reliable interfaces to external services. By migrating to MCPs, you get:
- Automatic retry logic and rate limiting
- Consistent error handling
- Simplified API interactions
- Better monitoring and debugging

## MCP Components Available

### 1. GitHub MCP
Replaces direct PyGithub usage for:
- Creating issues and PRs
- Searching repositories
- Managing labels and milestones

### 2. Filesystem MCP
Replaces direct file I/O for:
- Reading/writing files
- Directory operations
- File management in sandboxed environments

### 3. Memory MCP
Provides persistent context storage for:
- Cross-session state
- Learning data
- Temporary caches

### 4. MySQL MCP
Replaces direct database connections for:
- Task persistence
- Analytics data
- Long-term storage

### 5. Git MCP
Replaces GitPython for:
- Repository operations
- Commit management
- Branch operations

### 6. Fetch MCP
Replaces requests library for:
- HTTP API calls
- Web scraping
- External service integration

## Migration Examples

### Example 1: Migrating GitHub Issue Creation

**Before (Direct PyGithub):**
```python
from github import Github

g = Github(github_token)
repo = g.get_repo(repo_name)
issue = repo.create_issue(
    title=title,
    body=body,
    labels=labels
)
```

**After (MCP):**
```python
from scripts.mcp_integration import MCPIntegrationHub

async with MCPIntegrationHub() as mcp:
    result = await mcp.github.create_issue(
        repo=repo_name,
        title=title,
        body=body,
        labels=labels
    )
```

### Example 2: Migrating File Operations

**Before (Direct file I/O):**
```python
with open(file_path, 'w') as f:
    f.write(content)
```

**After (MCP):**
```python
async with MCPIntegrationHub() as mcp:
    success = await mcp.filesystem.write_file(
        path=file_path,
        content=content
    )
```

### Example 3: Migrating HTTP Requests

**Before (requests library):**
```python
import requests

response = requests.get(url)
data = response.json()
```

**After (MCP):**
```python
async with MCPIntegrationHub() as mcp:
    data = await mcp.fetch.fetch_json(url)
```

## Component-Specific Migration

### 1. GitHub Issue Creator
- Already migrated! Use `mcp_github_issue_creator.py` instead of `github_issue_creator.py`
- Provides same interface but uses MCP under the hood

### 2. Market Research Engine
```python
# Update scripts/market_research_engine.py
# Replace requests calls with MCP fetch
async with MCPIntegrationHub() as mcp:
    # Search tech news
    news_data = await mcp.fetch.fetch_json(news_api_url)
    
    # Store insights in memory
    await mcp.memory.store_context(
        key=f"market_insight_{timestamp}",
        value=insight_data
    )
```

### 3. Multi-Repository Coordinator
```python
# Update scripts/multi_repo_coordinator.py
# Replace GitPython with MCP git operations
async with MCPIntegrationHub() as mcp:
    # Get repository status
    status = await mcp.git.status(repo_path)
    
    # Create branches
    await mcp.git.branch(name="feature/new-feature")
    
    # Search GitHub for repositories
    repos = await mcp.github.search_repositories(
        query="machine learning python"
    )
```

### 4. Task Persistence
```python
# Replace Redis/file storage with MySQL MCP
async with MCPIntegrationHub() as mcp:
    # Create tasks table if needed
    await mcp.mysql.create_table("tasks", {
        "id": "VARCHAR(255) PRIMARY KEY",
        "title": "TEXT",
        "status": "VARCHAR(50)",
        "created_at": "TIMESTAMP"
    })
    
    # Insert task
    task_id = await mcp.mysql.insert_record("tasks", {
        "id": task.id,
        "title": task.title,
        "status": "pending",
        "created_at": datetime.now()
    })
```

## Configuration

1. **Environment Variables:**
```bash
# GitHub MCP
export GITHUB_TOKEN="your-github-token"

# MySQL MCP
export MYSQL_HOST="localhost"
export MYSQL_PORT="3306"
export MYSQL_USER="cwmai"
export MYSQL_PASSWORD="your-password"
export MYSQL_DATABASE="cwmai"
```

2. **MCP Configuration File:**
The system automatically creates `mcp_config.json` with default settings. Customize as needed:

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
  }
}
```

## Best Practices

1. **Use Context Managers:**
   Always use `async with` to ensure proper cleanup:
   ```python
   async with MCPIntegrationHub() as mcp:
       # Your code here
   ```

2. **Handle MCP Unavailability:**
   ```python
   if not mcp.github:
       # Fallback to direct API or skip
       logger.warning("GitHub MCP not available")
   ```

3. **Batch Operations:**
   MCPs handle rate limiting, but still batch when possible:
   ```python
   # Good: Single search
   repos = await mcp.github.search_repositories("AI", limit=50)
   
   # Avoid: Multiple searches in tight loop
   for term in terms:
       await mcp.github.search_repositories(term)
   ```

4. **Error Handling:**
   ```python
   try:
       result = await mcp.github.create_issue(...)
   except Exception as e:
       logger.error(f"MCP operation failed: {e}")
       # Handle gracefully
   ```

## Testing

Run the test suite to verify MCP integration:

```bash
python test_mcp_integration.py
```

This will test all configured MCPs and show available tools.

## Gradual Migration

You don't need to migrate everything at once:

1. Start with new features using MCPs
2. Migrate high-value components first (GitHub integration)
3. Keep fallbacks during transition
4. Remove old code once MCP version is stable

## Troubleshooting

1. **MCP Server Not Starting:**
   - Check if Node.js is installed: `node --version`
   - Install MCP server: `npm install -g @modelcontextprotocol/server-github`

2. **Authentication Errors:**
   - Verify environment variables are set
   - Check token permissions

3. **Connection Issues:**
   - Check firewall settings
   - Verify MCP server is running
   - Check logs in `mcp_client.py`

## Next Steps

1. Update `continuous_orchestrator.py` to use MCP GitHub integration
2. Migrate `market_research_engine.py` to use Fetch MCP
3. Convert task storage to MySQL MCP
4. Add MCP-based monitoring and analytics