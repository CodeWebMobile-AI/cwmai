"""
MCP Monitoring and Health Check System

Provides real-time monitoring of MCP server health, performance metrics,
and automatic recovery mechanisms.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from scripts.mcp_integration import MCPIntegrationHub


class ServerStatus(Enum):
    """MCP server health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServerHealth:
    """Health information for an MCP server."""
    name: str
    status: ServerStatus
    last_check: datetime
    response_time: Optional[float] = None
    error_count: int = 0
    consecutive_failures: int = 0
    available_tools: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    

@dataclass
class MCPMetrics:
    """Performance metrics for MCP operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: Optional[float] = None
    max_response_time: Optional[float] = None
    requests_by_server: Dict[str, int] = field(default_factory=dict)
    errors_by_server: Dict[str, int] = field(default_factory=dict)
    

class MCPMonitor:
    """Monitors MCP server health and performance."""
    
    def __init__(self, mcp_hub: Optional[MCPIntegrationHub] = None,
                 check_interval: int = 60,
                 unhealthy_threshold: int = 3):
        """Initialize MCP monitor.
        
        Args:
            mcp_hub: MCP integration hub to monitor
            check_interval: Seconds between health checks
            unhealthy_threshold: Consecutive failures before marking unhealthy
        """
        self.mcp_hub = mcp_hub
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.logger = logging.getLogger(__name__)
        
        self.server_health: Dict[str, ServerHealth] = {}
        self.metrics = MCPMetrics()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start the monitoring loop."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("🚀 MCP monitoring started")
        
    async def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("🛑 MCP monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self.check_all_servers()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
                
    async def check_all_servers(self) -> Dict[str, ServerHealth]:
        """Check health of all MCP servers.
        
        Returns:
            Current health status of all servers
        """
        if not self.mcp_hub:
            self.logger.warning("No MCP hub configured for monitoring")
            return {}
            
        # Check each configured server
        for server_name in self.mcp_hub.client.servers:
            await self.check_server_health(server_name)
            
        return dict(self.server_health)
        
    async def check_server_health(self, server_name: str) -> ServerHealth:
        """Check health of a specific MCP server.
        
        Args:
            server_name: Name of the server to check
            
        Returns:
            Server health information
        """
        start_time = time.time()
        
        # Get or create health record
        if server_name not in self.server_health:
            self.server_health[server_name] = ServerHealth(
                name=server_name,
                status=ServerStatus.UNKNOWN,
                last_check=datetime.now(timezone.utc)
            )
            
        health = self.server_health[server_name]
        
        try:
            # Get transport for the server
            transport = self.mcp_hub.client.transports.get(server_name)
            if not transport:
                raise Exception(f"No transport for server {server_name}")
                
            # Perform health check by listing tools
            tools_response = await transport.send_request("tools/list")
            tools = tools_response.get("tools", [])
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update health record
            health.status = ServerStatus.HEALTHY
            health.response_time = response_time
            health.consecutive_failures = 0
            health.available_tools = [tool["name"] for tool in tools]
            health.last_check = datetime.now(timezone.utc)
            health.last_error = None
            
            # Update metrics
            self._update_metrics(server_name, True, response_time)
            
            self.logger.debug(f"✅ {server_name}: Healthy ({len(tools)} tools, {response_time:.2f}s)")
            
        except Exception as e:
            # Handle health check failure
            health.consecutive_failures += 1
            health.error_count += 1
            health.last_error = str(e)
            health.last_check = datetime.now(timezone.utc)
            
            # Determine status based on failure count
            if health.consecutive_failures >= self.unhealthy_threshold:
                health.status = ServerStatus.UNHEALTHY
            else:
                health.status = ServerStatus.DEGRADED
                
            # Update metrics
            self._update_metrics(server_name, False, time.time() - start_time)
            
            self.logger.warning(f"❌ {server_name}: {health.status.value} - {e}")
            
            # Attempt recovery if unhealthy
            if health.status == ServerStatus.UNHEALTHY:
                await self._attempt_recovery(server_name)
                
        return health
        
    def _update_metrics(self, server_name: str, success: bool, response_time: float):
        """Update performance metrics.
        
        Args:
            server_name: Server that was checked
            success: Whether the check succeeded
            response_time: Time taken for the check
        """
        self.metrics.total_requests += 1
        self.metrics.total_response_time += response_time
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            self.metrics.errors_by_server[server_name] = \
                self.metrics.errors_by_server.get(server_name, 0) + 1
                
        self.metrics.requests_by_server[server_name] = \
            self.metrics.requests_by_server.get(server_name, 0) + 1
            
        # Update min/max response times
        if self.metrics.min_response_time is None or response_time < self.metrics.min_response_time:
            self.metrics.min_response_time = response_time
        if self.metrics.max_response_time is None or response_time > self.metrics.max_response_time:
            self.metrics.max_response_time = response_time
            
    async def _attempt_recovery(self, server_name: str):
        """Attempt to recover an unhealthy server.
        
        Args:
            server_name: Server to recover
        """
        self.logger.info(f"🔧 Attempting recovery for {server_name}...")
        
        try:
            # Close existing connection
            transport = self.mcp_hub.client.transports.get(server_name)
            if transport:
                await transport.close()
                del self.mcp_hub.client.transports[server_name]
                
            # Reconnect
            await self.mcp_hub.client.connect(server_name)
            
            # Verify recovery
            await self.check_server_health(server_name)
            
            if self.server_health[server_name].status == ServerStatus.HEALTHY:
                self.logger.info(f"✅ Successfully recovered {server_name}")
            else:
                self.logger.warning(f"⚠️  Recovery attempted but {server_name} still unhealthy")
                
        except Exception as e:
            self.logger.error(f"Recovery failed for {server_name}: {e}")
            
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all server statuses.
        
        Returns:
            Status summary
        """
        healthy_count = sum(1 for h in self.server_health.values() 
                          if h.status == ServerStatus.HEALTHY)
        degraded_count = sum(1 for h in self.server_health.values() 
                           if h.status == ServerStatus.DEGRADED)
        unhealthy_count = sum(1 for h in self.server_health.values() 
                            if h.status == ServerStatus.UNHEALTHY)
        
        return {
            "total_servers": len(self.server_health),
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count,
            "servers": {
                name: {
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "response_time": health.response_time,
                    "consecutive_failures": health.consecutive_failures,
                    "available_tools": len(health.available_tools),
                    "last_error": health.last_error
                }
                for name, health in self.server_health.items()
            }
        }
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.
        
        Returns:
            Metrics summary
        """
        success_rate = (self.metrics.successful_requests / self.metrics.total_requests * 100) \
                      if self.metrics.total_requests > 0 else 0
        avg_response_time = (self.metrics.total_response_time / self.metrics.total_requests) \
                           if self.metrics.total_requests > 0 else 0
                           
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": f"{success_rate:.1f}%",
            "avg_response_time": f"{avg_response_time:.3f}s",
            "min_response_time": f"{self.metrics.min_response_time:.3f}s" if self.metrics.min_response_time else "N/A",
            "max_response_time": f"{self.metrics.max_response_time:.3f}s" if self.metrics.max_response_time else "N/A",
            "requests_by_server": dict(self.metrics.requests_by_server),
            "errors_by_server": dict(self.metrics.errors_by_server)
        }
        
    async def wait_for_healthy(self, timeout: int = 30) -> bool:
        """Wait for all servers to become healthy.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if all servers are healthy, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            await self.check_all_servers()
            
            all_healthy = all(h.status == ServerStatus.HEALTHY 
                            for h in self.server_health.values())
            if all_healthy:
                return True
                
            await asyncio.sleep(2)
            
        return False
        
    def export_health_report(self) -> str:
        """Export a detailed health report.
        
        Returns:
            JSON health report
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_summary": self.get_status_summary(),
            "metrics_summary": self.get_metrics_summary(),
            "server_details": {
                name: {
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "response_time": health.response_time,
                    "error_count": health.error_count,
                    "consecutive_failures": health.consecutive_failures,
                    "available_tools": health.available_tools,
                    "last_error": health.last_error
                }
                for name, health in self.server_health.items()
            }
        }
        
        return json.dumps(report, indent=2)


# Convenience function for quick health checks
async def check_mcp_health(mcp_hub: MCPIntegrationHub) -> Dict[str, Any]:
    """Perform a quick health check of all MCP servers.
    
    Args:
        mcp_hub: MCP integration hub to check
        
    Returns:
        Health status summary
    """
    monitor = MCPMonitor(mcp_hub)
    await monitor.check_all_servers()
    return monitor.get_status_summary()