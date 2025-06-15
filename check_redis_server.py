#!/usr/bin/env python3
"""
Check Redis Server Status

Simple script to check Redis server connection limits and current connections.
"""

import asyncio
import redis.asyncio as redis


async def check_redis_server():
    """Check Redis server status directly."""
    try:
        # Connect directly to Redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Ping server
        await client.ping()
        print("✅ Redis server is responding")
        
        # Get client info
        info = await client.info('clients')
        print("\n=== Client Information ===")
        print(f"Connected clients: {info.get('connected_clients', 'unknown')}")
        print(f"Client longest output list: {info.get('client_longest_output_list', 'unknown')}")
        print(f"Client biggest input buf: {info.get('client_biggest_input_buf', 'unknown')}")
        print(f"Blocked clients: {info.get('blocked_clients', 'unknown')}")
        
        # Get memory info
        mem_info = await client.info('memory')
        print("\n=== Memory Information ===")
        print(f"Used memory: {mem_info.get('used_memory_human', 'unknown')}")
        print(f"Used memory peak: {mem_info.get('used_memory_peak_human', 'unknown')}")
        print(f"Used memory RSS: {mem_info.get('used_memory_rss_human', 'unknown')}")
        
        # Get stats
        stats = await client.info('stats')
        print("\n=== Connection Statistics ===")
        print(f"Total connections received: {stats.get('total_connections_received', 'unknown')}")
        print(f"Total commands processed: {stats.get('total_commands_processed', 'unknown')}")
        print(f"Rejected connections: {stats.get('rejected_connections', 'unknown')}")
        
        # Check maxclients config
        config = await client.config_get('maxclients')
        max_clients = config.get('maxclients', 'unknown')
        print(f"\n=== Configuration ===")
        print(f"Max clients: {max_clients}")
        
        # Get current client list count
        client_list = await client.client_list()
        print(f"Current client connections: {len(client_list)}")
        
        # Close connection
        await client.close()
        
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")


if __name__ == "__main__":
    asyncio.run(check_redis_server())