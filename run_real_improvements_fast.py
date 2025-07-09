#!/usr/bin/env python3
"""
Fast Real Improvement Engine Runner

This script runs the real improvement engine directly with minimal initialization
for quick improvement creation.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix SSL certificate issues for NLTK
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Disable NLTK download attempts
os.environ['NLTK_DATA'] = '/tmp/nltk_data'

# Configure logging - less verbose
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for noisy_logger in ['scripts.http_ai_client', 'scripts.tool_calling_system', 
                     'scripts.auto_wrap_creator', 'scripts.tool_wrapper',
                     'scripts.method_self_healer']:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from scripts.state_manager import StateManager
from scripts.real_improvement_engine import RealImprovementEngine


async def run_fast_improvement():
    """Run a single improvement quickly."""
    logger.info("🚀 Fast Real Improvement Engine")
    logger.info("=" * 50)
    
    # Initialize only essential components
    state_manager = StateManager()
    improvement_engine = RealImprovementEngine(state_manager)
    
    # Get current state
    state = state_manager.get_state()
    
    # Analyze improvement areas
    logger.info("📊 Analyzing system for improvements...")
    areas = await improvement_engine._identify_improvement_areas(state)
    
    if areas:
        logger.info(f"Found {len(areas)} improvement opportunities:")
        for i, area in enumerate(areas[:3], 1):
            logger.info(f"  {i}. {area['type']} (priority: {area['priority']:.2f})")
    
    # Create improvement
    logger.info("\n🔨 Creating real improvement...")
    improvement = await improvement_engine.create_real_improvement()
    
    if improvement.get('success'):
        logger.info(f"✅ SUCCESS: Created {improvement['type']}")
        if 'name' in improvement:
            logger.info(f"   Name: {improvement['name']}")
        if 'description' in improvement:
            logger.info(f"   Description: {improvement['description']}")
    else:
        logger.error(f"❌ FAILED: {improvement.get('reason', 'Unknown error')}")
        
    # Show stats
    stats = improvement_engine.get_improvement_stats()
    logger.info(f"\n📈 Total Real Improvements: {stats['total_improvements']}")
    
    return improvement


async def run_continuous_fast(interval: int = 0):
    """Run continuous improvements with minimal overhead."""
    if interval > 0:
        logger.info(f"🔄 Starting continuous improvements (every {interval}s)")
    else:
        logger.info("🔄 Starting continuous improvements (no delay between cycles)")
    
    # Initialize once
    state_manager = StateManager()
    improvement_engine = RealImprovementEngine(state_manager)
    
    cycle = 0
    created = []
    
    while True:
        cycle += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"Cycle #{cycle} - {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Create improvement
            improvement = await improvement_engine.create_real_improvement()
            
            if improvement.get('success'):
                created.append(improvement)
                logger.info(f"✅ Created: {improvement.get('type', 'unknown')} - {improvement.get('name', 'N/A')}")
                
                # Show what was actually created
                if improvement.get('type') == 'tool_created':
                    logger.info(f"   📦 New tool: {improvement.get('description', 'No description')}")
                elif improvement.get('type') == 'workflow_created':
                    logger.info(f"   🔄 New workflow: {improvement.get('description', 'No description')}")
                elif improvement.get('type') == 'error_fix':
                    logger.info(f"   🔧 Fixed error: {improvement.get('error_type', 'Unknown')}")
            else:
                logger.warning("⚠️ No improvement created")
                
            # Quick stats
            stats = improvement_engine.get_improvement_stats()
            logger.info(f"Total: {stats['total_improvements']} | "
                       f"Tools: {stats['tools_created']} | "
                       f"Workflows: {stats['workflows_created']}")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            
        if interval > 0:
            logger.info(f"Next cycle in {interval}s...")
            await asyncio.sleep(interval)
        else:
            # No delay - immediately start next improvement
            logger.info("Starting next improvement immediately...")
            await asyncio.sleep(0.1)  # Tiny delay to prevent CPU spinning


async def main():
    """Main entry point."""
    import argparse
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv('.env.local')
    
    parser = argparse.ArgumentParser(description='Fast real improvement runner')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuously')
    parser.add_argument('--interval', type=int, default=0,
                       help='Interval between improvements in seconds (0 = no delay)')
    
    args = parser.parse_args()
    
    try:
        if args.continuous:
            await run_continuous_fast(args.interval)
        else:
            await run_fast_improvement()
    except KeyboardInterrupt:
        logger.info("\n👋 Stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())