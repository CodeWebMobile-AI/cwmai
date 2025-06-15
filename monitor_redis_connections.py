#!/usr/bin/env python3
"""
Monitor Redis Connections in Real-Time

This script monitors Redis connections to help identify connection leaks.
"""

import asyncio
import logging
import redis.asyncio as redis
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def monitor_connections():
    """Monitor Redis connections in real-time."""
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    try:
        print("Starting Redis connection monitoring...")
        print("Press Ctrl+C to stop\n")
        
        previous_count = 0
        previous_clients = set()
        
        while True:
            # Get current client list
            client_list = await client.client_list()
            current_count = len(client_list)
            
            # Extract client identifiers
            current_clients = set()
            for c in client_list:
                client_id = f"{c.get('addr', 'unknown')}:{c.get('fd', 'unknown')}"
                current_clients.add(client_id)
            
            # Detect changes
            new_clients = current_clients - previous_clients
            closed_clients = previous_clients - current_clients
            
            # Get server stats
            info = await client.info('clients')
            stats = await client.info('stats')
            
            # Display status
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if current_count != previous_count or new_clients or closed_clients:
                print(f"\n[{timestamp}] Connection change detected:")
                print(f"  Total connections: {previous_count} → {current_count}")
                
                if new_clients:
                    print(f"  New connections ({len(new_clients)}):")
                    for client_id in list(new_clients)[:5]:  # Show first 5
                        print(f"    + {client_id}")
                    if len(new_clients) > 5:
                        print(f"    ... and {len(new_clients) - 5} more")
                
                if closed_clients:
                    print(f"  Closed connections ({len(closed_clients)}):")
                    for client_id in list(closed_clients)[:5]:  # Show first 5
                        print(f"    - {client_id}")
                    if len(closed_clients) > 5:
                        print(f"    ... and {len(closed_clients) - 5} more")
            else:
                # Just show a dot to indicate we're still monitoring
                print(".", end="", flush=True)
            
            # Show warnings
            if current_count > 50:
                print(f"\n⚠️  WARNING: High connection count: {current_count}")
            
            rejected = stats.get('rejected_connections', 0)
            if rejected > 0:
                print(f"\n❌ ALERT: {rejected} connections have been rejected!")
            
            # Update for next iteration
            previous_count = current_count
            previous_clients = current_clients
            
            await asyncio.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(monitor_connections())