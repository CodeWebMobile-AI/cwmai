#!/usr/bin/env python3
"""
Integrate Redis Resilience Solution into the System

This script updates the system to use the new resilient Redis components.
"""

import os
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_file(file_path: str, replacements: list):
    """Update a file with the given replacements."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        for old, new in replacements:
            content = re.sub(old, new, content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Updated: {file_path}")
            return True
        else:
            logger.debug(f"No changes needed: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")
        return False


def main():
    """Main integration function."""
    logger.info("Starting Redis Resilience Integration...")
    
    # Define replacements
    replacements = [
        # Import replacements
        (
            r'from scripts\.redis_task_persistence import RedisTaskPersistence',
            'from scripts.redis_integration.redis_resilient_task_persistence import ResilientRedisTaskPersistence'
        ),
        (
            r'from scripts\.redis_integration\.redis_client import (?:RedisClient, )?get_redis_client',
            'from scripts.redis_integration.redis_connection_manager import get_connection_manager'
        ),
        # Class replacements
        (
            r'RedisTaskPersistence\(',
            'ResilientRedisTaskPersistence('
        ),
        (
            r'self\.redis = await get_redis_client\(\)',
            'self.redis = await get_connection_manager()'
        ),
        (
            r'redis_client = await get_redis_client\(\)',
            'redis_client = await get_connection_manager()'
        ),
    ]
    
    # Files to update
    target_files = [
        'scripts/continuous_orchestrator.py',
        'scripts/redis_work_queue.py',
        'scripts/worker_status_monitor.py',
        'scripts/github_issue_queue.py',
        'scripts/redis_worker_coordinator.py',
        'scripts/redis_async_state_manager.py',
        'scripts/redis_distributed_workflows.py',
        'scripts/redis_event_stream_processor.py'
    ]
    
    updated_count = 0
    for file_path in target_files:
        full_path = Path(file_path)
        if full_path.exists():
            if update_file(str(full_path), replacements):
                updated_count += 1
        else:
            logger.warning(f"File not found: {file_path}")
    
    # Create a wrapper for backward compatibility
    wrapper_content = '''"""
Backward compatibility wrapper for Redis task persistence.

Maps old imports to new resilient implementation.
"""

# Re-export the resilient version with the old name
from scripts.redis_integration.redis_resilient_task_persistence import (
    ResilientRedisTaskPersistence as RedisTaskPersistence
)

# Also provide the new name
ResilientRedisTaskPersistence = RedisTaskPersistence

__all__ = ['RedisTaskPersistence', 'ResilientRedisTaskPersistence']
'''
    
    wrapper_path = 'scripts/redis_task_persistence_wrapper.py'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    logger.info(f"Created compatibility wrapper: {wrapper_path}")
    
    logger.info(f"\nIntegration complete! Updated {updated_count} files.")
    logger.info("\nTo fully activate the resilient solution:")
    logger.info("1. Review the changes made to the files")
    logger.info("2. Test the system with Redis running")
    logger.info("3. Test the system with Redis stopped to verify fallback")
    logger.info("4. Monitor using: python monitor_redis_resilience.py")


if __name__ == "__main__":
    main()