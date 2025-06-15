"""
Demo: How to use MCP-Redis in CWMAI
This demonstrates the usage patterns without requiring a running Redis server
"""

import asyncio
import logging
from datetime import datetime
from scripts.work_item_types import WorkItem, TaskPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_mcp_redis_usage():
    """Demonstrate MCP-Redis usage patterns"""
    
    logger.info("=== MCP-Redis Usage Demo for CWMAI ===\n")
    
    # 1. Basic Setup
    logger.info("1. BASIC SETUP")
    logger.info("To enable MCP-Redis in your system:")
    logger.info("   - Set environment variable: USE_MCP_REDIS=true")
    logger.info("   - Ensure Redis server is running")
    logger.info("   - The system will automatically use MCP-Redis for enhanced features\n")
    
    # 2. Work Queue Operations
    logger.info("2. WORK QUEUE OPERATIONS")
    logger.info("When MCP-Redis is enabled, RedisWorkQueue gains new capabilities:")
    
    # Example work item
    example_item = WorkItem(
        id="demo-123",
        task_type="enhancement",
        title="Add new feature X",
        description="Implement feature X with proper testing and documentation",
        priority=TaskPriority.HIGH,
        repository="cwmai/example-repo"
    )
    
    logger.info(f"\nExample: Adding work item '{example_item.title}'")
    logger.info("Traditional method: Complex serialization and Redis commands")
    logger.info("MCP-Redis method: Natural language command sent to MCP server")
    
    # Show the natural language command
    nl_command = f"""
    Add this work item to Redis:
    - Store in stream 'cwmai:work_queue:{example_item.priority.name.lower()}'
    - Fields:
      - id: {example_item.id}
      - task_type: {example_item.task_type}
      - title: {example_item.title}
      - description: {example_item.description}
      - priority: {example_item.priority.name}
      - repository: {example_item.repository}
    - Return the stream entry ID
    """
    logger.info(f"Natural language command:{nl_command}")
    
    # 3. Intelligent Task Assignment
    logger.info("\n3. INTELLIGENT TASK ASSIGNMENT")
    logger.info("MCP-Redis enables smarter task assignment:")
    
    assignment_command = """
    Find the best task for worker 'worker-123' with specialization 'python-backend':
    - Consider task priority and age
    - Match worker specialization to task type or repository
    - Avoid tasks that have failed multiple times
    - Prefer tasks similar to what this worker has successfully completed before
    - Return the single best matching task
    """
    logger.info(f"Natural language assignment:{assignment_command}")
    
    # 4. Similarity Search
    logger.info("\n4. SIMILARITY SEARCH")
    logger.info("Find similar tasks using vector search:")
    
    search_command = """
    Search for tasks similar to 'Fix Redis connection pooling issues':
    - Search in all work queue streams
    - Use text similarity on title and description fields
    - Limit to 10 results
    - Return tasks ordered by similarity score
    """
    logger.info(f"Similarity search command:{search_command}")
    
    # 5. Queue Analytics
    logger.info("\n5. INTELLIGENT QUEUE ANALYTICS")
    logger.info("Get deep insights about your work queue:")
    
    analytics_command = """
    Analyze the work queue and provide:
    - Task distribution by priority and type
    - Average wait time by priority
    - Worker utilization rates
    - Bottlenecks or stuck tasks
    - Recommendations for optimization
    """
    logger.info(f"Analytics command:{analytics_command}")
    
    # 6. System Monitoring
    logger.info("\n6. SYSTEM MONITORING")
    logger.info("Monitor system health with natural language:")
    
    monitor_command = """
    Perform health check:
    - Test Redis connectivity
    - Check memory usage
    - Count total keys
    - Check for any blocked clients
    - Return health status and metrics
    """
    logger.info(f"Health check command:{monitor_command}")
    
    # 7. Advanced Features
    logger.info("\n7. ADVANCED FEATURES")
    logger.info("MCP-Redis enables features not possible with traditional Redis clients:")
    logger.info("   - Natural language queries for complex operations")
    logger.info("   - Vector similarity search for tasks")
    logger.info("   - Intelligent task routing based on worker history")
    logger.info("   - Automatic optimization recommendations")
    logger.info("   - Cross-stream analytics and insights")
    
    # 8. Integration Example
    logger.info("\n8. INTEGRATION IN YOUR CODE")
    logger.info("Example usage in your continuous_orchestrator.py:")
    
    code_example = '''
# In your existing code
async def process_work(self):
    # Traditional approach
    work_items = await self.work_queue.get_work_for_worker(worker_id)
    
    # With MCP-Redis enabled
    if self.work_queue.mcp_redis:
        # Get intelligently assigned task
        best_task = await self.work_queue.optimize_task_assignment(
            worker_id, 
            specialization="python-ai"
        )
        
        # Find similar completed tasks for reference
        similar = await self.work_queue.find_similar_tasks(
            best_task.description
        )
        
        # Get insights for better processing
        insights = await self.work_queue.get_intelligent_queue_insights()
    '''
    logger.info(code_example)
    
    # 9. Benefits Summary
    logger.info("\n9. BENEFITS OF MCP-REDIS IN CWMAI")
    logger.info("   ✓ Simplifies complex Redis operations")
    logger.info("   ✓ Adds AI-native features (similarity search, intelligent routing)")
    logger.info("   ✓ Reduces code complexity and maintenance")
    logger.info("   ✓ Provides natural language interface for debugging")
    logger.info("   ✓ Enables advanced analytics without custom code")
    logger.info("   ✓ Maintains compatibility with existing Redis infrastructure")
    
    logger.info("\n=== End of Demo ===")


async def show_configuration():
    """Show how to configure MCP-Redis"""
    logger.info("\n=== MCP-Redis Configuration ===")
    
    config_steps = """
1. Install MCP-Redis server (already done):
   npm install -g @modelcontextprotocol/server-redis

2. Set environment variables in .env.local:
   USE_MCP_REDIS=true
   REDIS_URL=redis://your-redis-server:6379

3. Ensure Redis server is running:
   docker run -d -p 6379:6379 redis:latest
   # OR
   redis-server

4. The system will automatically use MCP-Redis when available

5. To test if MCP-Redis is active:
   - Check logs for "MCP-Redis integration enabled"
   - Try using enhanced features like find_similar_tasks()
   - Monitor for natural language commands in debug logs
"""
    logger.info(config_steps)


async def main():
    """Run the demo"""
    await demo_mcp_redis_usage()
    await show_configuration()


if __name__ == "__main__":
    asyncio.run(main())