# Smarter Work Discovery Strategy

## Current Issues
- Workers check for work every 30 seconds
- 10 workers × 120 checks/hour = 1,200 discovery attempts per hour
- Most attempts find 0 work items
- Research takes 3-4 minutes but checking happens constantly

## Proposed Solution

### 1. Centralized Discovery Coordinator
```python
class WorkDiscoveryCoordinator:
    def __init__(self):
        self.last_discovery = None
        self.consecutive_empty = 0
        self.research_in_progress = False
        self.base_interval = 60  # 1 minute
        
    def get_next_check_interval(self):
        """Exponential backoff when no work found."""
        if self.research_in_progress:
            return 300  # 5 minutes - wait for research
        
        if self.consecutive_empty == 0:
            return self.base_interval
        elif self.consecutive_empty < 5:
            return min(self.base_interval * (2 ** self.consecutive_empty), 600)
        else:
            return 600  # Max 10 minutes
```

### 2. Event-Driven Discovery
```python
# Trigger discovery on events, not timers
async def on_project_completed():
    await discover_follow_up_work()
    
async def on_research_completed(problems):
    await generate_work_batch(problems)
    
async def on_worker_idle_threshold():
    if idle_workers > total_workers * 0.5:
        await discover_work()
```

### 3. Intelligent Queue Management
```python
class SmartWorkQueue:
    def __init__(self):
        self.min_queue_size = workers_count * 2
        self.optimal_queue_size = workers_count * 5
        
    async def maintain_queue(self):
        if self.size < self.min_queue_size:
            # Proactively generate work
            needed = self.optimal_queue_size - self.size
            await self.generate_work_batch(needed)
```

### 4. Research-Aware Scheduling
```python
class ResearchAwareScheduler:
    def __init__(self):
        self.research_duration = 240  # 4 minutes typical
        self.last_research_start = None
        
    def should_check_for_work(self):
        if self.research_in_progress():
            # Don't check while research is running
            elapsed = time.time() - self.last_research_start
            return elapsed > self.research_duration
        return True
```

### 5. Work Discovery Analytics
```python
# Track discovery efficiency
discovery_stats = {
    'attempts': 0,
    'successful': 0,
    'work_items_found': 0,
    'time_spent': 0
}

# Adjust strategy based on success rate
if success_rate < 0.1:  # Less than 10% successful
    increase_backoff_multiplier()
```

## Benefits
- Reduce discovery attempts from 1,200/hour to ~20/hour
- Save API calls and processing time
- Workers spend more time working, less time searching
- Predictable work generation aligned with research completion
- Self-adjusting based on workload patterns

## Implementation Priority
1. Add exponential backoff (immediate fix)
2. Implement centralized coordinator
3. Add event-driven triggers
4. Implement queue prediction
5. Add analytics and self-tuning