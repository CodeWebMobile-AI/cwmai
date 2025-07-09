#!/usr/bin/env python3
"""
Test the integrated coordinated discovery system.

This test verifies that the hybrid orchestrator now uses
the coordinated discovery system instead of the old polling approach.
"""

import asyncio
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Filter out some noise
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


async def test_integrated_system():
    """Test the integrated coordinated discovery."""
    from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
    
    print("=" * 70)
    print("TESTING INTEGRATED COORDINATED DISCOVERY")
    print("=" * 70)
    print("\nThis test verifies the new coordinated discovery is working")
    print("instead of the old polling approach.\n")
    
    # Create orchestrator
    orchestrator = EnhancedHybridOrchestrator(
        max_workers=3,  # Use fewer workers for testing
        enable_parallel=True,
        enable_round_robin=False
    )
    
    # Track discovery attempts
    original_trigger = orchestrator.coordinated_discovery.coordinator.trigger_discovery
    discovery_count = 0
    discovery_reasons = []
    
    async def tracking_trigger(reason: str):
        nonlocal discovery_count
        discovery_count += 1
        discovery_reasons.append(reason)
        print(f"\n📍 Discovery #{discovery_count} triggered: {reason}")
        return await original_trigger(reason)
    
    orchestrator.coordinated_discovery.coordinator.trigger_discovery = tracking_trigger
    
    print("\n🚀 Starting orchestrator with coordinated discovery...")
    
    # Start the orchestrator
    await orchestrator.start()
    
    # Run for 30 seconds
    print("\n⏱️  Running for 30 seconds to observe behavior...")
    print("   (In old system, you'd see ~20 discovery attempts)")
    print("   (In new system, you should see 1-2 attempts)\n")
    
    await asyncio.sleep(30)
    
    # Show results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Discovery Statistics:")
    print(f"   Total discovery attempts: {discovery_count}")
    print(f"   Rate: {discovery_count * 2} attempts/minute")
    
    if discovery_reasons:
        print(f"\n📋 Discovery Reasons:")
        for i, reason in enumerate(discovery_reasons, 1):
            print(f"   {i}. {reason}")
    
    # Get coordinator stats
    stats = orchestrator.coordinated_discovery.get_stats()
    coord_stats = stats['coordinator']
    
    print(f"\n🎯 Coordinator Details:")
    print(f"   Success rate: {coord_stats['success_rate']}")
    print(f"   Work items found: {coord_stats['work_items_found']}")
    print(f"   Current backoff: {coord_stats['next_check_interval']}")
    print(f"   Research aware: {stats['research_scheduler']['research_active']}")
    
    # Comparison
    old_rate = 20  # ~20 attempts in 30 seconds with old system
    new_rate = discovery_count
    reduction = (1 - new_rate / old_rate) * 100 if old_rate > 0 else 0
    
    print(f"\n✨ Efficiency Improvement:")
    print(f"   Old system (expected): ~{old_rate} attempts")
    print(f"   New system (actual): {new_rate} attempts")
    print(f"   Reduction: {reduction:.0f}%")
    
    # Shutdown
    print("\n🛑 Shutting down...")
    await orchestrator.shutdown()
    
    print("\n✅ Test complete!")
    
    # Return success if we reduced attempts significantly
    return reduction > 50  # At least 50% reduction


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_integrated_system())
    sys.exit(0 if success else 1)