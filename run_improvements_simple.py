#!/usr/bin/env python3
"""
Simple improvement runner that bypasses problematic components
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix SSL
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.environ['NLTK_DATA'] = '/tmp/nltk_data'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for noisy in ['scripts', 'urllib3']:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from scripts.state_manager import StateManager
from scripts.real_improvement_engine import RealImprovementEngine


async def run_improvements_simple(duration_seconds: int = 60):
    """Run improvements for specified duration."""
    logger.info(f"🚀 Running improvements for {duration_seconds} seconds")
    
    state_manager = StateManager()
    engine = RealImprovementEngine(state_manager)
    
    # Bypass the learning system by setting empty results
    engine.feedback_system = None  # Disable feedback system
    
    start_time = asyncio.get_event_loop().time()
    improvements_created = []
    errors_encountered = []
    
    while asyncio.get_event_loop().time() - start_time < duration_seconds:
        try:
            logger.info(f"\n--- Attempt #{len(improvements_created) + len(errors_encountered) + 1} ---")
            
            # Call the improvement creation directly
            improvement = await engine.create_real_improvement()
            
            if improvement.get('success'):
                improvements_created.append(improvement)
                logger.info(f"✅ Created: {improvement['type']} - {improvement.get('name', 'N/A')}")
                
                # Log details
                if improvement['type'] == 'tool_created':
                    logger.info(f"   Tool: {improvement.get('description', 'No description')}")
                elif improvement['type'] == 'workflow_created':
                    logger.info(f"   Workflow: {improvement.get('description', 'No description')}")
                elif improvement['type'] == 'error_fix':
                    logger.info(f"   Fixed: {improvement.get('error_type', 'Unknown error')}")
                    
            else:
                logger.warning(f"⚠️ Failed: {improvement.get('reason', 'Unknown reason')}")
                
        except Exception as e:
            errors_encountered.append(str(e))
            logger.error(f"❌ Error: {e}")
            
        # Small delay to prevent CPU spinning
        await asyncio.sleep(0.5)
        
    # Summary
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"🏁 Completed after {elapsed:.1f} seconds")
    logger.info(f"✅ Improvements created: {len(improvements_created)}")
    logger.info(f"❌ Errors encountered: {len(errors_encountered)}")
    
    if improvements_created:
        logger.info("\n📦 Created improvements:")
        for imp in improvements_created:
            logger.info(f"  - {imp['type']}: {imp.get('name', 'N/A')}")
            
    # Get final stats
    stats = engine.get_improvement_stats()
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"  Total improvements: {stats['total_improvements']}")
    logger.info(f"  Tools created: {stats['tools_created']}")
    logger.info(f"  Workflows created: {stats['workflows_created']}")
    logger.info(f"  Error fixes: {stats['error_fixes']}")


async def main():
    """Main entry point."""
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    await run_improvements_simple(60)


if __name__ == '__main__':
    asyncio.run(main())