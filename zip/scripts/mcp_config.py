"""
MCP Configuration Management
Handles loading and managing MCP server configurations
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from scripts.mcp_client import MCPServer

logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    """Complete MCP configuration"""
    servers: Dict[str, MCPServer]
    default_timeout: int = 30
    max_retries: int = 3
    

class MCPConfigManager:
    """Manages MCP configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "mcp_config.json")
        self.config: Optional[MCPConfig] = None
        
    def load_default_config(self) -> MCPConfig:
        """Load default MCP configuration for CWMAI"""
        # Find npx executable
        npx_path = shutil.which("npx") or "npx"
        
        return MCPConfig(
            servers={
                "github": MCPServer(
                    name="github",
                    command=[npx_path, "-y", "@modelcontextprotocol/server-github"],
                    env={
                        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "")
                    }
                ),
                "filesystem": MCPServer(
                    name="filesystem",
                    command=[npx_path, "-y", "@modelcontextprotocol/server-filesystem"],
                    args=["--allowed-directories", "/workspaces/cwmai,/tmp"]
                ),
                "memory": MCPServer(
                    name="memory",
                    command=[npx_path, "-y", "@modelcontextprotocol/server-memory"]
                ),
                "mysql": MCPServer(
                    name="mysql",
                    command=[npx_path, "-y", "@modelcontextprotocol/server-mysql"],
                    env={
                        "MYSQL_HOST": os.getenv("MYSQL_HOST", "localhost"),
                        "MYSQL_PORT": os.getenv("MYSQL_PORT", "3306"),
                        "MYSQL_USER": os.getenv("MYSQL_USER", "root"),
                        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD", ""),
                        "MYSQL_DATABASE": os.getenv("MYSQL_DATABASE", "cwmai")
                    }
                ),
                "postgres": MCPServer(
                    name="postgres",
                    command=[npx_path, "-y", "@modelcontextprotocol/server-postgres"],
                    env={
                        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
                        "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
                        "POSTGRES_USER": os.getenv("POSTGRES_USER", "postgres"),
                        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
                        "POSTGRES_DATABASE": os.getenv("POSTGRES_DATABASE", "cwmai")
                    }
                ),
                "brave_search": MCPServer(
                    name="brave_search",
                    command=[npx_path, "-y", "@modelcontextprotocol/server-brave-search"],
                    env={
                        "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")
                    }
                ),
                "redis": MCPServer(
                    name="redis",
                    command=[npx_path, "-y", "@modelcontextprotocol/server-redis"],
                    env={
                        "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379")
                    }
                )
            }
        )
        
    def load_config(self) -> MCPConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                servers = {}
                for name, server_data in data.get("servers", {}).items():
                    servers[name] = MCPServer(**server_data)
                    
                self.config = MCPConfig(
                    servers=servers,
                    default_timeout=data.get("default_timeout", 30),
                    max_retries=data.get("max_retries", 3)
                )
                logger.info(f"Loaded MCP config from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.config = self.load_default_config()
        else:
            logger.info("No config file found, using defaults")
            self.config = self.load_default_config()
            self.save_config()
            
        return self.config
        
    def save_config(self):
        """Save current configuration to file"""
        if not self.config:
            return
            
        data = {
            "servers": {
                name: asdict(server) 
                for name, server in self.config.servers.items()
            },
            "default_timeout": self.config.default_timeout,
            "max_retries": self.config.max_retries
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved MCP config to {self.config_path}")
        
    def add_server(self, server: MCPServer):
        """Add a new server to configuration"""
        if not self.config:
            self.config = self.load_config()
            
        self.config.servers[server.name] = server
        self.save_config()
        
    def remove_server(self, name: str):
        """Remove a server from configuration"""
        if not self.config:
            self.config = self.load_config()
            
        if name in self.config.servers:
            del self.config.servers[name]
            self.save_config()
            
    def get_server(self, name: str) -> Optional[MCPServer]:
        """Get a specific server configuration"""
        if not self.config:
            self.config = self.load_config()
            
        return self.config.servers.get(name)
        
    def list_servers(self) -> List[str]:
        """List all configured server names"""
        if not self.config:
            self.config = self.load_config()
            
        return list(self.config.servers.keys())
        
    def update_server_env(self, name: str, env_vars: Dict[str, str]):
        """Update environment variables for a server"""
        if not self.config:
            self.config = self.load_config()
            
        server = self.config.servers.get(name)
        if server:
            if not server.env:
                server.env = {}
            server.env.update(env_vars)
            self.save_config()
            
    def validate_config(self) -> List[str]:
        """Validate the configuration and return any issues"""
        issues = []
        
        if not self.config:
            self.config = self.load_config()
            
        # Check for required environment variables
        github_server = self.config.servers.get("github")
        if github_server:
            token = github_server.env.get("GITHUB_PERSONAL_ACCESS_TOKEN") if github_server.env else None
            if not token:
                issues.append("GitHub server requires GITHUB_TOKEN environment variable")
                
        # Check for executable commands
        for name, server in self.config.servers.items():
            if server.transport == "stdio" and not server.command:
                issues.append(f"Server '{name}' requires a command")
                
        return issues