#!/usr/bin/env python3
"""
Simple Worker Monitor - Lightweight monitoring without complex initialization
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
import sys
from pathlib import Path

# Configure logging
log_file_path = Path('/workspaces/cwmai/worker_monitor.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SimpleWorkerMonitor')


async def monitor_workers():
    """Simple worker monitoring that reads from Redis directly."""
    logger.info("Starting simple worker monitor")
    
    try:
        # Import Redis client
        sys.path.append('/workspaces/cwmai')
        from scripts.redis_integration.redis_client import get_redis_client
        
        # Get Redis client
        redis_client = await get_redis_client()
        logger.info("Connected to Redis")
        
        while True:
            try:
                # Get worker information directly from Redis
                worker_keys = await redis_client.smembers("cwmai:active_workers")
                logger.info(f"Found {len(worker_keys)} active workers")
                
                worker_status = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'active_workers': len(worker_keys),
                    'workers': {}
                }
                
                # Get each worker's status
                for worker_key in worker_keys:
                    worker_id = worker_key.decode() if isinstance(worker_key, bytes) else worker_key
                    worker_data = await redis_client.hgetall(f"cwmai:workers:{worker_id}")
                    
                    if worker_data:
                        # Decode bytes to strings
                        decoded_data = {}
                        for k, v in worker_data.items():
                            key = k.decode() if isinstance(k, bytes) else k
                            value = v.decode() if isinstance(v, bytes) else v
                            decoded_data[key] = value
                        
                        worker_status['workers'][worker_id] = decoded_data
                
                # Get queue stats
                queue_stats = {}
                for priority in ['critical', 'high', 'medium', 'low', 'background']:
                    queue_key = f"cwmai:work_queue:{priority}"
                    length = await redis_client.xlen(queue_key)
                    queue_stats[priority] = length
                
                worker_status['queue_stats'] = queue_stats
                worker_status['total_queued'] = sum(queue_stats.values())
                
                # Log the status
                logger.info("=" * 80)
                logger.info(f"WORKER STATUS UPDATE - {worker_status['timestamp']}")
                logger.info("=" * 80)
                logger.info(f"Active Workers: {worker_status['active_workers']}")
                logger.info(f"Total Queued Tasks: {worker_status['total_queued']}")
                
                logger.info("\nQueue Breakdown:")
                for priority, count in queue_stats.items():
                    if count > 0:
                        logger.info(f"  {priority}: {count} tasks")
                
                logger.info("\nWorker Details:")
                for worker_id, data in worker_status['workers'].items():
                    status = data.get('status', 'unknown')
                    current_task = data.get('current_task', 'None')
                    if current_task != 'None':
                        try:
                            task_data = json.loads(current_task)
                            task_title = task_data.get('title', 'Unknown')[:50]
                        except:
                            task_title = str(current_task)[:50]
                    else:
                        task_title = 'Idle'
                    
                    logger.info(f"  {worker_id}: {status} - {task_title}")
                
                logger.info("=" * 80 + "\n")
                
            except Exception as e:
                logger.error(f"Error monitoring workers: {e}")
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
            
    except Exception as e:
        logger.error(f"Fatal error in worker monitor: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(monitor_workers())