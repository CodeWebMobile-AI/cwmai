#!/usr/bin/env python3
"""
Start the Self-Improving CWMAI System

This launches the complete self-improving orchestrator that includes:
- Continuous task orchestration
- Autonomous improvement system
- Code evolution engine
- Intelligent monitoring
- Research integration
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/self_improving_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce noise from some loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('scripts.repository_exclusion_manager').setLevel(logging.WARNING)


async def main():
    """Main entry point for self-improving system."""
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Self-Improving CWMAI System")
    logger.info("=" * 60)
    
    # Import and start the orchestrator
    from scripts.self_improving_orchestrator import get_self_improving_orchestrator
    
    orchestrator = get_self_improving_orchestrator()
    
    try:
        # Start the system
        await orchestrator.start()
        
        logger.info("✅ Self-improving system is now running!")
        logger.info("📊 Monitor progress at: http://localhost:8080/dashboard")
        logger.info("📝 Logs are being written to: self_improving_system.log")
        
        # Print initial status
        await asyncio.sleep(5)  # Let systems initialize
        
        # Get initial intelligence report
        report = await orchestrator.get_intelligence_report()
        
        logger.info("\n📈 Initial Intelligence Report:")
        logger.info(f"   Overall Intelligence Score: {report['overall_intelligence_score']:.1f}/100")
        logger.info(f"   Status: {report['status']}")
        logger.info(f"   Health: {report['system_health']['status']}")
        
        logger.info("\n🔄 Active Self-Improvement Systems:")
        logger.info("   ✓ Autonomous Improvement System - Learning from patterns")
        logger.info("   ✓ Code Evolution Engine - Optimizing code quality")
        logger.info("   ✓ Intelligent Monitoring - Tracking all metrics")
        logger.info("   ✓ Research Evolution Engine - Expanding knowledge")
        
        logger.info("\n💡 The system will now:")
        logger.info("   • Monitor performance and detect patterns")
        logger.info("   • Automatically fix errors and optimize code")
        logger.info("   • Research new techniques and integrate them")
        logger.info("   • Generate insights and apply improvements")
        logger.info("   • Evolve to become more intelligent over time")
        
        # Run until interrupted
        logger.info("\n⏳ System is running. Press Ctrl+C to stop.")
        
        # Periodic status updates
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Get updated report
            report = await orchestrator.get_intelligence_report()
            
            logger.info("\n📊 Status Update:")
            logger.info(f"   Intelligence Score: {report['overall_intelligence_score']:.1f}/100 ({report['status']})")
            logger.info(f"   Improvements Applied: {report['coordination_metrics']['improvements_applied']}")
            logger.info(f"   Code Evolutions: {report['evolution_engine']['evolutions_applied']}")
            logger.info(f"   Patterns Discovered: {report['evolution_engine']['patterns_discovered']}")
            logger.info(f"   Active Insights: {report['system_health']['insights_available']}")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down gracefully...")
        await orchestrator.stop()
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        await orchestrator.stop()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)