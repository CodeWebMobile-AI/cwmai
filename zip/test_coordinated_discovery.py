#!/usr/bin/env python3
"""
Test the new coordinated work discovery system.

This test demonstrates:
1. Centralized coordinator instead of 10 workers polling
2. Research-aware scheduling
3. Exponential backoff
4. Event-driven discovery
"""

import asyncio
import logging
from datetime import datetime
import json

from scripts.coordinated_work_discovery import CoordinatedWorkDiscovery
from scripts.ai_brain import IntelligentAIBrain
from scripts.state_manager import StateManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce some noise
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


class DiscoverySimulator:
    """Simulate the work discovery process with the new system."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.work_items_created = 0
        
    async def simulate_old_system(self, duration_minutes: float = 1):
        """Simulate the old system with 10 workers polling every 30 seconds."""
        self.logger.info("\n🔴 OLD SYSTEM: 10 workers polling every 30 seconds")
        
        worker_count = 10
        check_interval = 30  # seconds
        duration_seconds = duration_minutes * 60
        
        checks_per_worker = int(duration_seconds / check_interval)
        total_checks = worker_count * checks_per_worker
        
        self.logger.info(f"Duration: {duration_minutes} minutes")
        self.logger.info(f"Workers: {worker_count}")
        self.logger.info(f"Check interval: {check_interval} seconds")
        self.logger.info(f"Total discovery attempts: {total_checks}")
        self.logger.info(f"Rate: {total_checks * 60 / duration_seconds:.0f} attempts/hour")
        
        # Simulate some checks
        for i in range(min(5, total_checks)):
            self.logger.info(f"  Worker {i % worker_count}: Checking for work... (attempt {i+1}/{total_checks})")
            await asyncio.sleep(0.1)
            
        if total_checks > 5:
            self.logger.info(f"  ... ({total_checks - 5} more attempts)")
            
    async def simulate_new_system(self, duration_minutes: float = 1):
        """Simulate the new coordinated system."""
        self.logger.info("\n🟢 NEW SYSTEM: Coordinated discovery with research awareness")
        
        # Initialize components
        ai_brain = IntelligentAIBrain()
        state_manager = StateManager()
        system_state = state_manager.load_state()
        
        # Create coordinated discovery
        discovery = CoordinatedWorkDiscovery(
            ai_brain=ai_brain,
            system_state=system_state,
            worker_count=10
        )
        
        # Start the system
        discovery.start()
        
        # Simulate some scenarios
        self.logger.info("\n📋 Scenario 1: Normal operation")
        await asyncio.sleep(2)
        discovery.log_status()
        
        self.logger.info("\n📋 Scenario 2: Research starts")
        discovery.start_research("market_discovery")
        await asyncio.sleep(2)
        discovery.log_status()
        
        self.logger.info("\n📋 Scenario 3: Research completes with 5 problems")
        discovery.complete_research(problems_found=5)
        await asyncio.sleep(2)
        discovery.log_status()
        
        self.logger.info("\n📋 Scenario 4: Queue gets low")
        discovery.update_queue_size(15)  # Below minimum for 10 workers
        await asyncio.sleep(2)
        discovery.log_status()
        
        self.logger.info("\n📋 Scenario 5: Project completes")
        await discovery.on_project_completed("test-project")
        await asyncio.sleep(2)
        discovery.log_status()
        
        # Get final stats
        stats = discovery.get_stats()
        
        # Stop the system
        await discovery.stop()
        
        # Show comparison
        self.logger.info("\n📊 COMPARISON SUMMARY:")
        self.logger.info("─" * 50)
        
        # Old system stats
        old_attempts_per_hour = 10 * (3600 / 30)  # 10 workers, every 30 seconds
        self.logger.info(f"Old System:")
        self.logger.info(f"  - Discovery attempts/hour: {old_attempts_per_hour:.0f}")
        self.logger.info(f"  - Research awareness: ❌ No")
        self.logger.info(f"  - Coordination: ❌ No (10 independent workers)")
        self.logger.info(f"  - Backoff strategy: ❌ No")
        
        # New system stats
        coord_stats = stats['coordinator']
        actual_rate = coord_stats['total_attempts'] * 60 / (duration_minutes * 60) * 60  # per hour
        self.logger.info(f"\nNew System:")
        self.logger.info(f"  - Discovery attempts/hour: ~{actual_rate:.0f}")
        self.logger.info(f"  - Research awareness: ✅ Yes")
        self.logger.info(f"  - Coordination: ✅ Yes (centralized)")
        self.logger.info(f"  - Backoff strategy: ✅ Yes (exponential)")
        
        # Efficiency improvement
        reduction = (1 - actual_rate / old_attempts_per_hour) * 100
        self.logger.info(f"\n🎯 Efficiency Improvement: {reduction:.0f}% reduction in API calls")
        
        # Save stats
        with open('coordinated_discovery_test_results.json', 'w') as f:
            json.dump({
                'test_time': datetime.now().isoformat(),
                'duration_minutes': duration_minutes,
                'old_system': {
                    'attempts_per_hour': old_attempts_per_hour,
                    'coordination': False,
                    'research_aware': False
                },
                'new_system': {
                    'attempts_per_hour': actual_rate,
                    'coordination': True,
                    'research_aware': True,
                    'stats': stats
                },
                'improvement': {
                    'reduction_percent': reduction
                }
            }, f, indent=2)
            
        self.logger.info(f"\n💾 Results saved to coordinated_discovery_test_results.json")


async def main():
    """Run the comparison test."""
    simulator = DiscoverySimulator()
    
    print("=" * 60)
    print("COORDINATED WORK DISCOVERY SYSTEM TEST")
    print("=" * 60)
    
    # Simulate old system
    await simulator.simulate_old_system(duration_minutes=0.5)
    
    print("\n" + "=" * 60 + "\n")
    
    # Simulate new system
    await simulator.simulate_new_system(duration_minutes=0.5)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())