# Work Discovery Migration Guide

## Overview
This guide explains how to migrate from the current inefficient work discovery (1,200 attempts/hour) to the new coordinated system (~20-30 attempts/hour).

## Current Problem
- 10 workers independently check for work every 30 seconds
- Results in ~1,200 discovery attempts per hour
- Most attempts find 0 work items
- No coordination with research phases
- Wastes API calls and processing time

## New Solution Components

### 1. WorkDiscoveryCoordinator (`scripts/work_discovery_coordinator.py`)
- Centralized coordinator - ONE component checks for work, not 10
- Exponential backoff when no work found
- Event-driven discovery triggers
- Comprehensive statistics tracking

### 2. ResearchAwareScheduler (`scripts/research_aware_scheduler.py`)
- Tracks active research phases
- Prevents work discovery during research (3-4 minutes)
- Learns typical research durations from history
- Triggers immediate discovery after research completes

### 3. CoordinatedWorkDiscovery (`scripts/coordinated_work_discovery.py`)
- Integrates all components
- Manages work queue intelligently
- Provides unified interface

## Migration Steps

### Step 1: Update ContinuousOrchestrator

Replace the current worker polling logic in `continuous_orchestrator.py`:

```python
# OLD CODE - Remove this pattern:
async def worker_loop(worker_id):
    while True:
        # Check for work every 30 seconds
        work = await work_finder.find_work()
        await asyncio.sleep(30)

# NEW CODE - Use coordinated discovery:
from scripts.coordinated_work_discovery import CoordinatedWorkDiscovery

class ContinuousOrchestrator:
    def __init__(self):
        # ... existing init ...
        
        # Initialize coordinated discovery
        self.discovery = CoordinatedWorkDiscovery(
            ai_brain=self.ai_brain,
            system_state=self.state_manager.state,
            worker_count=self.worker_count
        )
        
    async def start(self):
        # Start coordinated discovery
        self.discovery.start()
        
        # Workers now just consume from queue, don't discover
        for i in range(self.worker_count):
            asyncio.create_task(self.worker_consume_only(i))
```

### Step 2: Update Worker Logic

Workers should only consume work, not discover it:

```python
async def worker_consume_only(self, worker_id: int):
    """Worker that only consumes from queue."""
    while self.running:
        # Get work from queue
        work_item = await self.get_next_work_item()
        
        if work_item:
            await self.process_work_item(worker_id, work_item)
        else:
            # No work - wait a bit
            await asyncio.sleep(5)
            
        # Notify if idle
        if self.is_worker_idle(worker_id):
            await self.discovery.on_workers_idle(self.get_idle_count())
```

### Step 3: Integrate Research Notifications

Update research components to notify the coordinator:

```python
# In gemini_deep_research.py or research orchestrator:
async def start_research(self):
    # Notify discovery system
    self.discovery.start_research("market_discovery")
    
    # Run research
    problems = await self._run_research()
    
    # Notify completion
    self.discovery.complete_research(len(problems))
```

### Step 4: Add Event Triggers

Add event-based discovery triggers:

```python
# When a project completes
async def on_project_complete(self, project_name: str):
    await self.discovery.on_project_completed(project_name)
    
# When queue gets low
async def monitor_queue(self):
    queue_size = len(self.work_queue)
    self.discovery.update_queue_size(queue_size)
```

### Step 5: Update Configuration

Add new environment variables:

```bash
# Base interval between discovery attempts (seconds)
DISCOVERY_BASE_INTERVAL=60

# Maximum interval between attempts (seconds)
DISCOVERY_MAX_INTERVAL=600

# Expected research duration (seconds)
RESEARCH_EXPECTED_DURATION=240

# Minimum queue size (multiplier of worker count)
QUEUE_MIN_MULTIPLIER=2

# Optimal queue size (multiplier of worker count)
QUEUE_OPTIMAL_MULTIPLIER=5
```

## Testing the Migration

1. **Run the test script:**
   ```bash
   export $(cat .env.local | grep -v '^#' | xargs) && python3 test_coordinated_discovery.py
   ```

2. **Monitor the improvement:**
   - Check logs for discovery frequency
   - Verify research phases block discovery
   - Confirm event triggers work

3. **Check statistics:**
   ```python
   # Get discovery stats
   stats = discovery.get_stats()
   print(f"Discovery attempts: {stats['coordinator']['total_attempts']}")
   print(f"Success rate: {stats['coordinator']['success_rate']}")
   ```

## Benefits After Migration

1. **Reduced API Calls**: From ~1,200/hour to ~20-30/hour (95%+ reduction)
2. **Smarter Scheduling**: No wasted checks during research
3. **Better Worker Utilization**: Workers work instead of searching
4. **Predictive Queue Management**: Maintains optimal work levels
5. **Event-Driven**: Responds immediately to important events

## Rollback Plan

If issues arise, you can temporarily disable coordinated discovery:

```python
# Set environment variable
USE_COORDINATED_DISCOVERY=false

# Or in code
if os.getenv('USE_COORDINATED_DISCOVERY', 'true').lower() == 'false':
    # Use old polling method
    pass
else:
    # Use new coordinated discovery
    pass
```

## Monitoring

After migration, monitor these metrics:
- Discovery attempts per hour
- Worker idle time
- Queue depth over time
- Research phase durations
- API call reduction

## Next Steps

1. Implement the migration in `continuous_orchestrator.py`
2. Update worker loops to remove discovery
3. Add research phase notifications
4. Deploy and monitor
5. Fine-tune intervals based on actual usage