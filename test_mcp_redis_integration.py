"""
Test MCP-Redis Integration
Tests the new MCP-Redis features in CWMAI
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.mcp_redis_integration import MCPRedisIntegration
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.redis_work_queue import RedisWorkQueue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_mcp_redis():
    """Test basic MCP-Redis functionality"""
    logger.info("Testing basic MCP-Redis functionality...")
    
    try:
        async with MCPRedisIntegration() as redis:
            # Test health check
            logger.info("Testing health check...")
            health = await redis.health_check()
            logger.info(f"Redis health: {health}")
            
            # Test state management
            logger.info("Testing state management...")
            test_state = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "status": "testing_mcp"
            }
            
            # Save state
            saved = await redis.save_state("test:mcp:state", test_state, ttl=300)
            logger.info(f"State saved: {saved}")
            
            # Retrieve state
            retrieved = await redis.get_state("test:mcp:state")
            logger.info(f"State retrieved: {retrieved}")
            
            # Test natural language query
            logger.info("Testing natural language query...")
            result = await redis.execute("Count all keys starting with 'cwmai:'")
            logger.info(f"Natural language query result: {result}")
            
            return True
            
    except Exception as e:
        logger.error(f"Basic MCP-Redis test failed: {e}")
        return False


async def test_work_queue_with_mcp():
    """Test RedisWorkQueue with MCP-Redis integration"""
    logger.info("Testing RedisWorkQueue with MCP-Redis...")
    
    # Enable MCP-Redis
    os.environ["USE_MCP_REDIS"] = "true"
    
    try:
        # Initialize work queue
        queue = RedisWorkQueue()
        await queue.initialize()
        
        # Create test work items
        test_items = [
            WorkItem(
                id=f"mcp-test-{i}",
                task_type="test_mcp",
                title=f"Test MCP Task {i}",
                description=f"Testing MCP-Redis integration with task {i}",
                priority=TaskPriority.MEDIUM if i % 2 == 0 else TaskPriority.HIGH,
                metadata={"test": True, "mcp": True}
            )
            for i in range(5)
        ]
        
        # Add work items
        logger.info("Adding test work items...")
        for item in test_items:
            await queue.add_work(item)
        
        # Flush buffer to ensure items are in Redis
        await queue._flush_buffer()
        
        # Test MCP-enhanced features
        if queue.mcp_redis:
            logger.info("Testing MCP-enhanced features...")
            
            # Find similar tasks
            similar = await queue.find_similar_tasks("Testing MCP-Redis integration", limit=3)
            logger.info(f"Similar tasks found: {len(similar)}")
            
            # Get intelligent insights
            insights = await queue.get_intelligent_queue_insights()
            logger.info(f"Queue insights: {insights}")
            
            # Test optimized task assignment
            best_task = await queue.optimize_task_assignment("test-worker-1", "test_mcp")
            logger.info(f"Best task for worker: {best_task.title if best_task else 'None'}")
        else:
            logger.warning("MCP-Redis not initialized in work queue")
        
        # Get regular stats
        stats = await queue.get_queue_stats()
        logger.info(f"Queue stats: {stats}")
        
        # Cleanup
        await queue.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Work queue MCP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_advanced_mcp_features():
    """Test advanced MCP-Redis features"""
    logger.info("Testing advanced MCP-Redis features...")
    
    try:
        async with MCPRedisIntegration() as redis:
            # Test event streaming
            logger.info("Testing event streaming...")
            test_event = {
                "type": "test_event",
                "source": "mcp_test",
                "data": {"message": "Testing MCP event streaming"}
            }
            
            event_id = await redis.create_event_stream(test_event)
            logger.info(f"Event created with ID: {event_id}")
            
            # Get recent events
            recent_events = await redis.get_recent_events(count=10)
            logger.info(f"Recent events count: {len(recent_events)}")
            
            # Test system metrics
            logger.info("Testing system metrics...")
            metrics = await redis.get_system_metrics()
            logger.info(f"System metrics: {metrics}")
            
            # Test worker management
            logger.info("Testing worker management...")
            updated = await redis.update_worker_status(
                "test-mcp-worker",
                "active",
                current_task="mcp-test-1"
            )
            logger.info(f"Worker status updated: {updated}")
            
            # Get all workers
            workers = await redis.get_all_workers()
            logger.info(f"Active workers: {len(workers)}")
            
            # Test natural language operations
            logger.info("Testing complex natural language query...")
            complex_result = await redis.execute("""
                Find all work items that:
                - Have high or critical priority
                - Were created in the last hour
                - Are not yet assigned to any worker
                Return count and list the first 5
            """)
            logger.info(f"Complex query result: {complex_result}")
            
            return True
            
    except Exception as e:
        logger.error(f"Advanced MCP-Redis test failed: {e}")
        return False


async def main():
    """Run all MCP-Redis tests"""
    logger.info("Starting MCP-Redis integration tests...")
    
    results = {
        "basic": await test_basic_mcp_redis(),
        "work_queue": await test_work_queue_with_mcp(),
        "advanced": await test_advanced_mcp_features()
    }
    
    # Summary
    logger.info("\n=== Test Results ===")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)