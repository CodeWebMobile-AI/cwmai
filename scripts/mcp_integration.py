"""
MCP Integration Layer for CWMAI
Provides high-level interfaces for using MCPs in the CWMAI system
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from scripts.mcp_client import MCPClient, MCPTool
from scripts.mcp_config import MCPConfigManager

logger = logging.getLogger(__name__)


class MCPGitHubIntegration:
    """GitHub operations through MCP"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.server_name = "github"
        
    async def create_issue(self, repo: str, title: str, body: str, labels: Optional[List[str]] = None) -> Dict:
        """Create a GitHub issue"""
        arguments = {
            "repository": repo,
            "title": title,
            "body": body
        }
        
        if labels:
            arguments["labels"] = labels
            
        result = await self.client.call_tool("github:create_issue", arguments)
        return result[0] if result else {}
        
    async def list_issues(self, repo: str, state: str = "open", labels: Optional[List[str]] = None) -> List[Dict]:
        """List GitHub issues"""
        arguments = {
            "repository": repo,
            "state": state
        }
        
        if labels:
            arguments["labels"] = labels
            
        result = await self.client.call_tool("github:list_issues", arguments)
        return result[0].get("issues", []) if result else []
        
    async def create_pull_request(self, repo: str, title: str, body: str, base: str, head: str) -> Dict:
        """Create a pull request"""
        arguments = {
            "repository": repo,
            "title": title,
            "body": body,
            "base": base,
            "head": head
        }
        
        result = await self.client.call_tool("github:create_pull_request", arguments)
        return result[0] if result else {}
        
    async def search_repositories(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for repositories"""
        arguments = {
            "query": query,
            "limit": limit
        }
        
        result = await self.client.call_tool("github:search_repositories", arguments)
        return result[0].get("repositories", []) if result else []
        
    async def get_repository_info(self, repo: str) -> Dict:
        """Get repository information"""
        result = await self.client.call_tool("github:get_repository", {"repository": repo})
        return result[0] if result else {}


class MCPFileSystemIntegration:
    """File system operations through MCP"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.server_name = "filesystem"
        
    async def read_file(self, path: str) -> str:
        """Read a file"""
        result = await self.client.call_tool("filesystem:read_file", {"path": path})
        return result[0].get("content", "") if result else ""
        
    async def write_file(self, path: str, content: str) -> bool:
        """Write to a file"""
        result = await self.client.call_tool("filesystem:write_file", {
            "path": path,
            "content": content
        })
        return bool(result)
        
    async def list_directory(self, path: str, recursive: bool = False) -> List[Dict]:
        """List directory contents"""
        result = await self.client.call_tool("filesystem:list_directory", {
            "path": path,
            "recursive": recursive
        })
        return result[0].get("entries", []) if result else []
        
    async def create_directory(self, path: str) -> bool:
        """Create a directory"""
        result = await self.client.call_tool("filesystem:create_directory", {"path": path})
        return bool(result)
        
    async def delete_file(self, path: str) -> bool:
        """Delete a file"""
        result = await self.client.call_tool("filesystem:delete", {"path": path})
        return bool(result)
        
    async def move_file(self, src: str, dest: str) -> bool:
        """Move/rename a file"""
        result = await self.client.call_tool("filesystem:move", {
            "source": src,
            "destination": dest
        })
        return bool(result)


class MCPMemoryIntegration:
    """Memory/context operations through MCP"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.server_name = "memory"
        
    async def store_context(self, key: str, value: Any, metadata: Optional[Dict] = None) -> bool:
        """Store context in memory"""
        result = await self.client.call_tool("memory:store", {
            "key": key,
            "value": json.dumps(value) if not isinstance(value, str) else value,
            "metadata": metadata or {}
        })
        return bool(result)
        
    async def retrieve_context(self, key: str) -> Optional[Any]:
        """Retrieve context from memory"""
        result = await self.client.call_tool("memory:retrieve", {"key": key})
        if result:
            value = result[0].get("value")
            try:
                return json.loads(value) if value else None
            except:
                return value
        return None
        
    async def search_context(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for context entries"""
        result = await self.client.call_tool("memory:search", {
            "query": query,
            "limit": limit
        })
        return result[0].get("results", []) if result else []
        
    async def list_contexts(self, prefix: Optional[str] = None) -> List[str]:
        """List all context keys"""
        arguments = {}
        if prefix:
            arguments["prefix"] = prefix
            
        result = await self.client.call_tool("memory:list", arguments)
        return result[0].get("keys", []) if result else []
        
    async def delete_context(self, key: str) -> bool:
        """Delete a context entry"""
        result = await self.client.call_tool("memory:delete", {"key": key})
        return bool(result)


class MCPMySQLIntegration:
    """MySQL database operations through MCP"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.server_name = "mysql"
        
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict]:
        """Execute a MySQL query"""
        arguments = {"query": query}
        if params:
            arguments["params"] = params
            
        result = await self.client.call_tool("mysql:execute", arguments)
        return result[0].get("rows", []) if result else []
        
    async def create_table(self, table_name: str, schema: Dict) -> bool:
        """Create a table with the given schema"""
        result = await self.client.call_tool("mysql:create_table", {
            "table": table_name,
            "schema": schema
        })
        return bool(result)
        
    async def insert_record(self, table: str, data: Dict) -> Optional[int]:
        """Insert a record and return the ID"""
        result = await self.client.call_tool("mysql:insert", {
            "table": table,
            "data": data
        })
        return result[0].get("id") if result else None
        
    async def update_record(self, table: str, id: int, data: Dict) -> bool:
        """Update a record"""
        result = await self.client.call_tool("mysql:update", {
            "table": table,
            "id": id,
            "data": data
        })
        return bool(result)
        
    async def delete_record(self, table: str, id: int) -> bool:
        """Delete a record"""
        result = await self.client.call_tool("mysql:delete", {
            "table": table,
            "id": id
        })
        return bool(result)


class MCPGitIntegration:
    """Git operations through MCP"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.server_name = "git"
        
    async def status(self, repo_path: Optional[str] = None) -> Dict:
        """Get git status"""
        arguments = {}
        if repo_path:
            arguments["repository"] = repo_path
            
        result = await self.client.call_tool("git:status", arguments)
        return result[0] if result else {}
        
    async def commit(self, message: str, files: Optional[List[str]] = None) -> Dict:
        """Create a git commit"""
        arguments = {"message": message}
        if files:
            arguments["files"] = files
            
        result = await self.client.call_tool("git:commit", arguments)
        return result[0] if result else {}
        
    async def branch(self, name: str, checkout: bool = True) -> bool:
        """Create and optionally checkout a branch"""
        result = await self.client.call_tool("git:branch", {
            "name": name,
            "checkout": checkout
        })
        return bool(result)
        
    async def log(self, limit: int = 10) -> List[Dict]:
        """Get git log"""
        result = await self.client.call_tool("git:log", {"limit": limit})
        return result[0].get("commits", []) if result else []


class MCPFetchIntegration:
    """HTTP fetch operations through MCP"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.server_name = "fetch"
        
    async def fetch(self, url: str, method: str = "GET", headers: Optional[Dict] = None, body: Optional[str] = None) -> Dict:
        """Fetch a URL"""
        arguments = {
            "url": url,
            "method": method
        }
        
        if headers:
            arguments["headers"] = headers
        if body:
            arguments["body"] = body
            
        result = await self.client.call_tool("fetch:request", arguments)
        return result[0] if result else {}
        
    async def fetch_json(self, url: str) -> Optional[Dict]:
        """Fetch JSON from a URL"""
        response = await self.fetch(url)
        if response and response.get("status") == 200:
            try:
                return json.loads(response.get("body", "{}"))
            except:
                return None
        return None


class MCPIntegrationHub:
    """Central hub for all MCP integrations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = MCPConfigManager(config_path)
        self.client = MCPClient()
        self.github: Optional[MCPGitHubIntegration] = None
        self.filesystem: Optional[MCPFileSystemIntegration] = None
        self.memory: Optional[MCPMemoryIntegration] = None
        self.mysql: Optional[MCPMySQLIntegration] = None
        self.git: Optional[MCPGitIntegration] = None
        self.fetch: Optional[MCPFetchIntegration] = None
        
    async def initialize(self, servers: Optional[List[str]] = None):
        """Initialize MCP connections"""
        config = self.config_manager.load_config()
        
        # Add configured servers to client
        for name, server in config.servers.items():
            if servers and name not in servers:
                continue
            self.client.add_server(server)
            
        # Connect to all servers
        await self.client.connect_all()
        
        # Initialize integrations
        if "github" in self.client.servers:
            self.github = MCPGitHubIntegration(self.client)
            
        if "filesystem" in self.client.servers:
            self.filesystem = MCPFileSystemIntegration(self.client)
            
        if "memory" in self.client.servers:
            self.memory = MCPMemoryIntegration(self.client)
            
        if "mysql" in self.client.servers:
            self.mysql = MCPMySQLIntegration(self.client)
            
        if "git" in self.client.servers:
            self.git = MCPGitIntegration(self.client)
            
        if "fetch" in self.client.servers:
            self.fetch = MCPFetchIntegration(self.client)
            
        logger.info(f"Initialized MCP integrations: {list(self.client.servers.keys())}")
        
    async def close(self):
        """Close all MCP connections"""
        await self.client.close()
        
    def list_available_tools(self) -> List[str]:
        """List all available MCP tools"""
        return [f"{tool.server}:{tool.name}" for tool in self.client.list_tools()]
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()