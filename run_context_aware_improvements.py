#!/usr/bin/env python3
"""
Run Context-Aware Improvements Continuously

This script runs the improvement system continuously but with
context-aware decision making based on actual system needs.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("Received shutdown signal. Stopping...")
    running = False


async def run_improvement_cycle():
    """Run a single improvement cycle."""
    from scripts.state_manager import StateManager
    from scripts.real_improvement_engine import RealImprovementEngine
    from scripts.improvement_needs_analyzer import ImprovementNeedsAnalyzer
    
    # Initialize components
    state_manager = StateManager()
    engine = RealImprovementEngine(state_manager)
    analyzer = ImprovementNeedsAnalyzer(state_manager)
    
    # Analyze needs first
    logger.info("Analyzing system needs...")
    needs = await analyzer.analyze_improvement_needs()
    
    if needs:
        logger.info(f"Found {len(needs)} improvement needs:")
        for i, need in enumerate(needs[:3], 1):  # Show top 3
            logger.info(f"  {i}. {need['type']} - {need['description']} (priority: {need.get('priority', 0):.2f})")
    else:
        logger.info("No specific improvement needs identified")
    
    # Create improvement
    logger.info("Creating context-aware improvement...")
    try:
        improvement = await engine.create_real_improvement()
        
        if improvement.get('success'):
            logger.info(f"✓ Created {improvement.get('type')}: {improvement.get('name')}")
            logger.info(f"  Description: {improvement.get('description')}")
            if improvement.get('from_need'):
                logger.info("  ↳ Based on identified system need")
        else:
            logger.warning(f"✗ Failed to create improvement: {improvement.get('reason', 'Unknown error')}")
            
        return improvement
        
    except Exception as e:
        logger.error(f"Error creating improvement: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


async def main():
    """Main improvement loop."""
    logger.info("=== Context-Aware Continuous Improvement System ===")
    logger.info("Starting improvement engine with intelligent need analysis...")
    logger.info("Press Ctrl+C to stop\n")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    improvement_count = 0
    improvement_types = {}
    start_time = datetime.now(timezone.utc)
    
    while running:
        cycle_start = datetime.now(timezone.utc)
        logger.info(f"\n--- Improvement Cycle {improvement_count + 1} ---")
        
        # Run improvement cycle
        improvement = await run_improvement_cycle()
        
        if improvement.get('success'):
            improvement_count += 1
            imp_type = improvement.get('type', 'unknown')
            improvement_types[imp_type] = improvement_types.get(imp_type, 0) + 1
            
            # Log statistics
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            rate = improvement_count / (elapsed / 60) if elapsed > 0 else 0
            
            logger.info(f"\nStatistics:")
            logger.info(f"  Total improvements: {improvement_count}")
            logger.info(f"  Rate: {rate:.1f} improvements/minute")
            logger.info(f"  Types created:")
            for typ, count in improvement_types.items():
                logger.info(f"    - {typ}: {count}")
        
        # Wait a moment before next cycle
        if running:
            logger.info("\nWaiting 5 seconds before next improvement...")
            await asyncio.sleep(5)
    
    # Final summary
    logger.info("\n=== Final Summary ===")
    logger.info(f"Created {improvement_count} improvements")
    logger.info(f"Improvement breakdown:")
    for typ, count in improvement_types.items():
        percentage = (count / improvement_count * 100) if improvement_count > 0 else 0
        logger.info(f"  - {typ}: {count} ({percentage:.1f}%)")
    
    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds() / 60
    logger.info(f"Total runtime: {elapsed_total:.1f} minutes")
    logger.info(f"Average rate: {improvement_count / elapsed_total:.1f} improvements/minute")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete.")