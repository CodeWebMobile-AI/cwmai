"""
Simple test for MCP-Redis without requiring Redis server
"""

import asyncio
import logging
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.mcp_redis_integration import MCPRedisIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_mcp_initialization():
    """Test that MCP-Redis can initialize even without Redis server"""
    logger.info("Testing MCP-Redis initialization...")
    
    try:
        # Use a fake Redis URL to test initialization
        redis = MCPRedisIntegration(redis_url="redis://fake-server:6379")
        
        # Try to initialize (this will start the MCP server process)
        await redis.initialize()
        
        logger.info("MCP-Redis initialized successfully!")
        
        # List available tools
        if redis.hub and redis.hub.client:
            tools = redis.hub.client.list_tools()
            logger.info(f"Available MCP-Redis tools: {[tool.name for tool in tools]}")
        
        # Close connection
        await redis.close()
        
        return True
        
    except Exception as e:
        logger.error(f"MCP-Redis initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_command_structure():
    """Test MCP-Redis command structure without executing"""
    logger.info("Testing MCP-Redis command structure...")
    
    # Test creating commands without execution
    from scripts.work_item_types import WorkItem, TaskPriority
    
    test_item = WorkItem(
        id="test-123",
        task_type="test",
        title="Test Task",
        description="Test Description",
        priority=TaskPriority.HIGH
    )
    
    # Create command strings
    add_command = f"""
    Add this work item to Redis:
    - Store in stream 'cwmai:work_queue:{test_item.priority.name.lower()}'
    - Fields:
      - id: {test_item.id}
      - task_type: {test_item.task_type}
      - title: {test_item.title}
      - description: {test_item.description}
      - priority: {test_item.priority.name}
    """
    
    logger.info(f"Sample add command:\n{add_command}")
    
    query_command = """
    Get statistics for all work queues:
    - For each priority stream (critical, high, medium, low, background):
      - Count total items
      - Count pending items
    - Return as structured data
    """
    
    logger.info(f"Sample query command:\n{query_command}")
    
    return True


async def main():
    """Run simple MCP-Redis tests"""
    logger.info("Starting simple MCP-Redis tests...")
    
    # Test initialization
    init_success = await test_mcp_initialization()
    
    # Test command structure
    command_success = await test_mcp_command_structure()
    
    # Summary
    logger.info("\n=== Test Results ===")
    logger.info(f"Initialization: {'PASSED' if init_success else 'FAILED'}")
    logger.info(f"Command Structure: {'PASSED' if command_success else 'FAILED'}")
    
    return init_success and command_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)