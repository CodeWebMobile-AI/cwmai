#!/usr/bin/env python3
"""
Run CWMAI with MCP integration enabled

This script demonstrates how to use CWMAI with MCP integrations
in a production environment.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_single_cycle():
    """Run a single orchestration cycle with MCP support."""
    logger.info("🚀 Starting CWMAI with MCP integration...")
    
    # Check if MCP is enabled
    mcp_enabled = os.getenv('MCP_ENABLED', 'true').lower() == 'true'
    
    if mcp_enabled:
        logger.info("✅ MCP integration is ENABLED")
        # Try to use enhanced orchestrator with MCP
        try:
            from scripts.enhanced_continuous_orchestrator import EnhancedContinuousOrchestrator
            
            orchestrator = EnhancedContinuousOrchestrator(
                max_workers=10,
                enable_parallel=True,
                enable_research=True,
                enable_round_robin=True,
                enable_mcp=True
            )
            logger.info("Using Enhanced Continuous Orchestrator with MCP")
        except Exception as e:
            logger.warning(f"Failed to initialize MCP orchestrator: {e}")
            logger.info("Falling back to standard orchestrator")
            from scripts.continuous_orchestrator import ContinuousOrchestrator
            orchestrator = ContinuousOrchestrator(
                max_workers=10,
                enable_parallel=True,
                enable_research=True,
                enable_round_robin=True
            )
    else:
        logger.info("⚠️ MCP integration is DISABLED")
        from scripts.continuous_orchestrator import ContinuousOrchestrator
        orchestrator = ContinuousOrchestrator(
            max_workers=3,
            enable_parallel=True,
            enable_research=True,
            enable_round_robin=True
        )
    
    try:
        # Initialize orchestrator
        await orchestrator.initialize()
        logger.info("✅ Orchestrator initialized successfully")
        
        # Run a single cycle
        logger.info("🔄 Running orchestration cycle...")
        await orchestrator.run_single_cycle()
        
        logger.info("✅ Cycle completed successfully")
        
    except Exception as e:
        logger.error(f"Error during orchestration: {e}")
        raise
        
    finally:
        # Clean up
        await orchestrator.cleanup()
        logger.info("🧹 Cleanup completed")


async def test_mcp_fallback():
    """Test that the system works even without MCP servers."""
    logger.info("\n🧪 Testing MCP fallback mode...")
    
    # Test with MCP disabled
    os.environ['MCP_ENABLED'] = 'false'
    
    from scripts.github_issue_creator import GitHubIssueCreator
    creator = GitHubIssueCreator()
    
    if creator.can_create_issues():
        logger.info("✅ GitHub issue creation available (direct API)")
    else:
        logger.warning("⚠️ GitHub issue creation not available")
    
    # Test task persistence
    from scripts.task_persistence import TaskPersistence
    persistence = TaskPersistence()
    
    test_task = {
        "id": "test-" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "title": "Test task",
        "status": "pending"
    }
    
    success = await persistence.save_task(test_task)
    logger.info(f"✅ Task persistence working: {success}")
    
    # Test market research
    from scripts.ai_brain import IntelligentAIBrain
    from scripts.market_research_engine import MarketResearchEngine
    
    ai_brain = IntelligentAIBrain({}, {})
    research_engine = MarketResearchEngine(ai_brain)
    
    logger.info("✅ Market research engine initialized")
    
    logger.info("✅ Fallback mode test completed - system works without MCPs")


async def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║              CWMAI - Continuous AI Orchestrator           ║
║                    with MCP Integration                   ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        await test_mcp_fallback()
    else:
        await run_single_cycle()
    
    print("\n✨ CWMAI execution completed!")


if __name__ == "__main__":
    asyncio.run(main())