"""
MCP (Model Context Protocol) Client Implementation
Provides a unified interface for interacting with MCP servers
"""

import json
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import aiohttp
import websockets
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Configuration for an MCP server"""
    name: str
    command: List[str]
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    transport: str = "stdio"  # stdio, websocket, http
    url: Optional[str] = None  # for websocket/http transports


@dataclass
class MCPTool:
    """Represents a tool exposed by an MCP server"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    server: str


@dataclass
class MCPResource:
    """Represents a resource exposed by an MCP server"""
    uri: str
    name: str
    mimeType: Optional[str] = None
    server: Optional[str] = None


class MCPTransport(ABC):
    """Abstract base class for MCP transports"""
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        pass
    
    @abstractmethod
    async def close(self):
        pass


class StdioTransport(MCPTransport):
    """stdio-based MCP transport"""
    
    def __init__(self, command: List[str], args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args or []
        self.env = env
        self.process = None
        self.request_id = 0
        self.pending_responses = {}
        
    async def connect(self):
        """Start the MCP server process"""
        cmd = self.command + self.args
        
        # Ensure node is in PATH
        env = self.env.copy() if self.env else {}
        if 'PATH' not in env:
            env['PATH'] = os.environ.get('PATH', '')
        # Add node path if not already there
        node_path = '/usr/local/share/nvm/versions/node/v22.16.0/bin'
        if node_path not in env['PATH']:
            env['PATH'] = f"{node_path}:{env['PATH']}"
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Start reading responses and errors
        asyncio.create_task(self._read_responses())
        asyncio.create_task(self._read_errors())
        
        # Initialize connection
        response = await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "cwmai-mcp-client",
                "version": "1.0.0"
            }
        })
        
        return response
    
    async def _read_responses(self):
        """Read responses from the server"""
        while self.process and self.process.stdout:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                    
                response = json.loads(line.decode())
                request_id = response.get("id")
                
                if request_id in self.pending_responses:
                    self.pending_responses[request_id].set_result(response)
                    
            except Exception as e:
                logger.error(f"Error reading response: {e}")
    
    async def _read_errors(self):
        """Read error output from the server"""
        while self.process and self.process.stderr:
            try:
                line = await self.process.stderr.readline()
                if not line:
                    break
                error_msg = line.decode().strip()
                if error_msg:
                    logger.debug(f"MCP server stderr: {error_msg}")
            except Exception as e:
                logger.error(f"Error reading stderr: {e}")
                break
    
    async def send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Send a request to the server"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        # Create future for response
        future = asyncio.Future()
        self.pending_responses[self.request_id] = future
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            del self.pending_responses[self.request_id]
            raise Exception("MCP request timed out")
        del self.pending_responses[self.request_id]
        
        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")
            
        return response.get("result", {})
    
    async def close(self):
        """Close the connection"""
        if self.process:
            self.process.terminate()
            await self.process.wait()


class WebSocketTransport(MCPTransport):
    """WebSocket-based MCP transport"""
    
    def __init__(self, url: str):
        self.url = url
        self.websocket = None
        self.request_id = 0
        self.pending_responses = {}
        
    async def connect(self):
        """Connect to the WebSocket server"""
        self.websocket = await websockets.connect(self.url)
        
        # Start reading responses
        asyncio.create_task(self._read_responses())
        
        # Initialize connection
        response = await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "cwmai-mcp-client",
                "version": "1.0.0"
            }
        })
        
        return response
    
    async def _read_responses(self):
        """Read responses from the server"""
        async for message in self.websocket:
            try:
                response = json.loads(message)
                request_id = response.get("id")
                
                if request_id in self.pending_responses:
                    self.pending_responses[request_id].set_result(response)
                    
            except Exception as e:
                logger.error(f"Error reading response: {e}")
    
    async def send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Send a request to the server"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        # Create future for response
        future = asyncio.Future()
        self.pending_responses[self.request_id] = future
        
        # Send request
        await self.websocket.send(json.dumps(request))
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            del self.pending_responses[self.request_id]
            raise Exception("MCP request timed out")
        del self.pending_responses[self.request_id]
        
        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")
            
        return response.get("result", {})
    
    async def close(self):
        """Close the connection"""
        if self.websocket:
            await self.websocket.close()


class MCPClient:
    """Main MCP client for managing multiple servers"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.transports: Dict[str, MCPTransport] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        
    def add_server(self, server: MCPServer):
        """Add an MCP server configuration"""
        self.servers[server.name] = server
        
    async def connect(self, server_name: str) -> Dict:
        """Connect to a specific MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"Unknown server: {server_name}")
            
        server = self.servers[server_name]
        
        # Create transport
        if server.transport == "stdio":
            transport = StdioTransport(server.command, server.args, server.env)
        elif server.transport == "websocket":
            transport = WebSocketTransport(server.url)
        else:
            raise ValueError(f"Unsupported transport: {server.transport}")
            
        # Connect
        init_response = await transport.connect()
        self.transports[server_name] = transport
        
        # Discover tools
        tools_response = await transport.send_request("tools/list")
        for tool in tools_response.get("tools", []):
            tool_obj = MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                inputSchema=tool.get("inputSchema", {}),
                server=server_name
            )
            self.tools[f"{server_name}:{tool['name']}"] = tool_obj
            
        # Discover resources (if supported)
        try:
            resources_response = await transport.send_request("resources/list")
            for resource in resources_response.get("resources", []):
                resource_obj = MCPResource(
                    uri=resource["uri"],
                    name=resource.get("name", resource["uri"]),
                    mimeType=resource.get("mimeType"),
                    server=server_name
                )
                self.resources[resource["uri"]] = resource_obj
        except Exception as e:
            # Some servers don't support resources
            logger.debug(f"Server {server_name} doesn't support resources: {e}")
            
        return init_response
        
    async def connect_all(self):
        """Connect to all configured servers"""
        tasks = [self.connect(name) for name in self.servers]
        await asyncio.gather(*tasks)
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on an MCP server"""
        if ":" in tool_name:
            # Full tool name with server prefix
            full_name = tool_name
        else:
            # Find tool by name only
            matching_tools = [k for k in self.tools if k.endswith(f":{tool_name}")]
            if not matching_tools:
                raise ValueError(f"Tool not found: {tool_name}")
            if len(matching_tools) > 1:
                raise ValueError(f"Ambiguous tool name: {tool_name}. Matches: {matching_tools}")
            full_name = matching_tools[0]
            
        tool = self.tools.get(full_name)
        if not tool:
            raise ValueError(f"Tool not found: {full_name}")
            
        transport = self.transports.get(tool.server)
        if not transport:
            raise ValueError(f"Server not connected: {tool.server}")
            
        response = await transport.send_request("tools/call", {
            "name": tool.name,
            "arguments": arguments
        })
        
        return response.get("content", [])
        
    async def read_resource(self, uri: str) -> Any:
        """Read a resource from an MCP server"""
        resource = self.resources.get(uri)
        if not resource:
            raise ValueError(f"Resource not found: {uri}")
            
        transport = self.transports.get(resource.server)
        if not transport:
            raise ValueError(f"Server not connected: {resource.server}")
            
        response = await transport.send_request("resources/read", {
            "uri": uri
        })
        
        return response.get("contents", [])
        
    def list_tools(self) -> List[MCPTool]:
        """List all available tools"""
        return list(self.tools.values())
        
    def list_resources(self) -> List[MCPResource]:
        """List all available resources"""
        return list(self.resources.values())
        
    async def close(self):
        """Close all connections"""
        for transport in self.transports.values():
            await transport.close()
        self.transports.clear()
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()