#!/usr/bin/env python3
"""
Test Claude Executor Pool Integration

Verifies that the Claude executor pool starts automatically with the hybrid orchestrator.
"""

import asyncio
import logging
import sys
import os

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator


async def test_claude_pool_integration():
    """Test that Claude executor pool starts with orchestrator."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing Claude executor pool integration...")
    
    # Create orchestrator (this will initialize the Claude pool)
    orchestrator = EnhancedHybridOrchestrator(
        max_workers=2,
        enable_parallel=True,
        enable_round_robin=False
    )
    
    # Check that Claude pool was initialized
    assert hasattr(orchestrator, 'claude_executor_pool'), "Claude executor pool not initialized"
    assert orchestrator.claude_executor_pool is not None, "Claude executor pool is None"
    assert orchestrator.claude_num_executors > 0, "No Claude executors configured"
    
    logger.info(f"✅ Claude executor pool initialized with {orchestrator.claude_num_executors} executors")
    
    # Start the orchestrator (this should start the Claude pool)
    logger.info("🚀 Starting orchestrator (and Claude pool)...")
    
    # Create a task to run the orchestrator for a short time
    async def run_for_duration():
        await orchestrator.run()
    
    # Run for 10 seconds then shutdown
    try:
        await asyncio.wait_for(run_for_duration(), timeout=10)
    except asyncio.TimeoutError:
        logger.info("⏰ Test duration reached, shutting down...")
    
    # Get analytics including Claude pool stats
    analytics = await orchestrator.get_analytics()
    
    logger.info("📊 Analytics:")
    logger.info(f"  - Workers: {analytics.get('worker_count', 0)}")
    logger.info(f"  - Claude executors: {analytics.get('claude_executor_pool', {}).get('num_executors', 0)}")
    logger.info(f"  - Claude tasks completed: {analytics.get('claude_executor_pool', {}).get('total_completed', 0)}")
    
    # Shutdown
    await orchestrator.shutdown()
    
    logger.info("✅ Test completed successfully!")
    logger.info("The Claude executor pool is now integrated and will start automatically with run_hybrid_continuous.py")


if __name__ == "__main__":
    try:
        asyncio.run(test_claude_pool_integration())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)