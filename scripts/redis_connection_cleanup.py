#!/usr/bin/env python3
"""
Redis Connection Cleanup

Clean up orphaned Redis connections and reset connection state.
"""

import asyncio
import logging
import redis.asyncio as redis
from typing import List, Dict, Any

from scripts.redis_integration import (
    get_redis_client,
    close_redis_client,
    get_connection_pool,
    close_connection_pool,
    RedisConfig,
    get_redis_config
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisConnectionCleanup:
    """Clean up Redis connections and reset state."""
    
    def __init__(self):
        """Initialize cleanup utility."""
        self.config = get_redis_config()
        self.cleaned_connections = 0
        self.errors: List[str] = []
    
    async def cleanup_all_connections(self):
        """Clean up all Redis connections."""
        logger.info("Starting Redis connection cleanup...")
        
        try:
            # Step 1: Close global client
            logger.info("Closing global Redis client...")
            await close_redis_client()
            
            # Step 2: Close connection pool
            logger.info("Closing connection pool...")
            await close_connection_pool()
            
            # Step 3: Get server-side connection info
            await self._check_server_connections()
            
            # Step 4: Kill idle connections on server
            await self._kill_idle_connections()
            
            # Report results
            self._report_results()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.errors.append(str(e))
    
    async def _check_server_connections(self):
        """Check server-side connection info."""
        try:
            # Create temporary connection
            conn = redis.Redis(
                host=self.config.connection_config.host,
                port=self.config.connection_config.port,
                password=self.config.connection_config.password,
                decode_responses=True
            )
            
            # Get client list
            client_list = await conn.execute_command('CLIENT', 'LIST')
            clients = self._parse_client_list(client_list)
            
            logger.info(f"Found {len(clients)} active Redis connections")
            
            # Analyze connections
            by_name = {}
            idle_clients = []
            
            for client in clients:
                name = client.get('name', 'unnamed')
                by_name[name] = by_name.get(name, 0) + 1
                
                # Check for idle connections
                idle_time = int(client.get('idle', 0))
                if idle_time > 300:  # 5 minutes
                    idle_clients.append(client)
            
            logger.info("Connection breakdown by name:")
            for name, count in by_name.items():
                logger.info(f"  {name}: {count}")
            
            if idle_clients:
                logger.warning(f"Found {len(idle_clients)} idle connections (>5 minutes)")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Error checking server connections: {e}")
            self.errors.append(f"Server check: {e}")
    
    async def _kill_idle_connections(self, idle_threshold: int = 300):
        """Kill idle connections on the server.
        
        Args:
            idle_threshold: Idle time in seconds before killing connection
        """
        try:
            # Create temporary connection
            conn = redis.Redis(
                host=self.config.connection_config.host,
                port=self.config.connection_config.port,
                password=self.config.connection_config.password,
                decode_responses=True
            )
            
            # Get client list
            client_list = await conn.execute_command('CLIENT', 'LIST')
            clients = self._parse_client_list(client_list)
            
            killed_count = 0
            
            for client in clients:
                idle_time = int(client.get('idle', 0))
                client_id = client.get('id')
                
                # Skip our own connection
                if client_id and idle_time > idle_threshold:
                    try:
                        # Kill idle client
                        result = await conn.execute_command('CLIENT', 'KILL', 'ID', client_id)
                        if result:
                            killed_count += 1
                            logger.info(f"Killed idle connection {client_id} (idle: {idle_time}s)")
                    except Exception as e:
                        logger.warning(f"Failed to kill client {client_id}: {e}")
            
            if killed_count > 0:
                logger.info(f"Killed {killed_count} idle connections")
                self.cleaned_connections += killed_count
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Error killing idle connections: {e}")
            self.errors.append(f"Kill connections: {e}")
    
    def _parse_client_list(self, client_list: str) -> List[Dict[str, str]]:
        """Parse CLIENT LIST output."""
        clients = []
        
        for line in client_list.strip().split('\n'):
            if not line:
                continue
                
            client = {}
            for pair in line.split():
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    client[key] = value
            
            if client:
                clients.append(client)
        
        return clients
    
    def _report_results(self):
        """Report cleanup results."""
        logger.info("\n" + "="*60)
        logger.info("Redis Connection Cleanup Report")
        logger.info("="*60)
        
        logger.info(f"\nCleaned connections: {self.cleaned_connections}")
        
        if self.errors:
            logger.error(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  - {error}")
        else:
            logger.info("\nNo errors encountered")
        
        logger.info("\n" + "="*60)


async def reset_connection_limits():
    """Reset Redis connection limits on the server."""
    logger.info("Resetting Redis connection limits...")
    
    try:
        config = get_redis_config()
        conn = redis.Redis(
            host=config.connection_config.host,
            port=config.connection_config.port,
            password=config.connection_config.password,
            decode_responses=True
        )
        
        # Get current config
        max_clients = await conn.config_get('maxclients')
        timeout = await conn.config_get('timeout')
        
        logger.info(f"Current settings:")
        logger.info(f"  maxclients: {max_clients.get('maxclients', 'default')}")
        logger.info(f"  timeout: {timeout.get('timeout', 'default')}")
        
        # Set recommended values
        await conn.config_set('timeout', '300')  # 5 minutes
        logger.info("Set client timeout to 300 seconds")
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"Error resetting limits: {e}")


async def main():
    """Main cleanup function."""
    import sys
    
    cleanup = RedisConnectionCleanup()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'reset':
            await reset_connection_limits()
        elif sys.argv[1] == 'force':
            # Force kill all non-system connections
            logger.warning("Force killing all connections...")
            await cleanup._kill_idle_connections(idle_threshold=0)
    else:
        await cleanup.cleanup_all_connections()


if __name__ == "__main__":
    asyncio.run(main())