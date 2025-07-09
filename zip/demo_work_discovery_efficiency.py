#!/usr/bin/env python3
"""
Demonstrate the efficiency improvements of the coordinated work discovery system.
"""

import asyncio
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


class WorkDiscoveryDemo:
    """Demonstrate work discovery patterns."""
    
    def __init__(self):
        self.old_system_attempts = 0
        self.new_system_attempts = 0
        
    async def simulate_old_system(self, duration_seconds: int = 60):
        """Simulate the old system behavior."""
        logger.info("🔴 OLD SYSTEM SIMULATION - 10 workers polling every 30 seconds")
        logger.info("=" * 60)
        
        start_time = asyncio.get_event_loop().time()
        worker_count = 10
        check_interval = 30
        
        # Simulate for the duration
        while (asyncio.get_event_loop().time() - start_time) < duration_seconds:
            # All 10 workers check
            for worker_id in range(worker_count):
                self.old_system_attempts += 1
                logger.info(f"Worker {worker_id}: Checking for work... (attempt #{self.old_system_attempts})")
                
            # Wait for next interval
            await asyncio.sleep(check_interval)
            
        attempts_per_hour = (self.old_system_attempts / duration_seconds) * 3600
        logger.info(f"\n📊 Old System Results:")
        logger.info(f"   Total attempts in {duration_seconds}s: {self.old_system_attempts}")
        logger.info(f"   Rate: {attempts_per_hour:.0f} attempts/hour")
        
    async def simulate_new_system(self, duration_seconds: int = 60):
        """Simulate the new coordinated system."""
        logger.info("\n🟢 NEW SYSTEM SIMULATION - Coordinated discovery")
        logger.info("=" * 60)
        
        start_time = asyncio.get_event_loop().time()
        consecutive_empty = 0
        base_interval = 60  # 1 minute base
        
        # Simulate research phases
        research_phases = [
            (15, "market_discovery", 240),  # Research at 15s, takes 4 minutes
            (300, "technology_trends", 180)  # Research at 5 min, takes 3 minutes
        ]
        
        next_check_time = start_time
        
        while (asyncio.get_event_loop().time() - start_time) < duration_seconds:
            current_time = asyncio.get_event_loop().time() - start_time
            
            # Check for research phases
            for research_time, research_type, research_duration in research_phases:
                if research_time <= current_time < research_time + research_duration:
                    # During research - don't check
                    logger.info(f"📚 Research '{research_type}' in progress - skipping discovery")
                    await asyncio.sleep(30)
                    continue
                    
            # Time for a check?
            if (asyncio.get_event_loop().time() - start_time) >= (next_check_time - start_time):
                self.new_system_attempts += 1
                
                # Simulate finding work (30% chance)
                import random
                found_work = random.random() < 0.3
                
                if found_work:
                    logger.info(f"✅ Coordinator: Found work! (attempt #{self.new_system_attempts})")
                    consecutive_empty = 0
                    interval = base_interval
                else:
                    logger.info(f"❌ Coordinator: No work found (attempt #{self.new_system_attempts})")
                    consecutive_empty += 1
                    # Exponential backoff
                    interval = min(base_interval * (2 ** consecutive_empty), 600)
                    
                logger.info(f"   Next check in {interval}s (backoff level: {consecutive_empty})")
                next_check_time = asyncio.get_event_loop().time() + interval
                
            await asyncio.sleep(1)
            
        attempts_per_hour = (self.new_system_attempts / duration_seconds) * 3600
        logger.info(f"\n📊 New System Results:")
        logger.info(f"   Total attempts in {duration_seconds}s: {self.new_system_attempts}")
        logger.info(f"   Rate: {attempts_per_hour:.0f} attempts/hour")
        
        return attempts_per_hour


async def main():
    """Run the demonstration."""
    demo = WorkDiscoveryDemo()
    
    print("\n" + "=" * 70)
    print("WORK DISCOVERY EFFICIENCY DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how the new coordinated system reduces API calls")
    print("from 1,200/hour to ~20-30/hour through intelligent scheduling.\n")
    
    # Run old system for 60 seconds (simulated)
    await demo.simulate_old_system(60)
    
    # Run new system for 60 seconds (simulated)
    new_rate = await demo.simulate_new_system(60)
    
    # Show comparison
    print("\n" + "=" * 70)
    print("EFFICIENCY COMPARISON")
    print("=" * 70)
    
    old_rate = 1200  # 10 workers * 120 checks/hour
    reduction = (1 - new_rate / old_rate) * 100
    
    print(f"\n📈 Results:")
    print(f"   Old System: {old_rate} attempts/hour")
    print(f"   New System: {new_rate:.0f} attempts/hour")
    print(f"   Reduction:  {reduction:.0f}%")
    
    print("\n✨ Key Improvements:")
    print("   1. ONE coordinator instead of 10 workers")
    print("   2. Exponential backoff when no work found")
    print("   3. Pauses during 3-4 minute research phases")
    print("   4. Event-driven triggers for important events")
    print("   5. Smart queue management")
    
    # Save results
    results = {
        'demo_time': datetime.now().isoformat(),
        'old_system_rate': old_rate,
        'new_system_rate': new_rate,
        'reduction_percent': reduction,
        'key_features': [
            'Centralized coordination',
            'Research-aware scheduling',
            'Exponential backoff',
            'Event-driven discovery',
            'Queue intelligence'
        ]
    }
    
    with open('work_discovery_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Results saved to work_discovery_demo_results.json")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())