# Task Monitoring System Guide

## Overview

The Task Monitoring System provides comprehensive tracking and analytics for all task execution in the CWMAI system. It tracks execution metrics, dependencies, quality, learning patterns, detailed logs, lifecycle events, worker performance, and predictive analytics.

## Key Features

### 1. Task Execution Metrics
- **Timing**: Estimated vs actual duration, queue wait time
- **Resources**: Peak CPU/memory usage, API calls per provider
- **Execution Details**: Retry counts, error types, Claude execution stats

### 2. Task Dependencies
- Track parent/child relationships
- Block/require/related dependency types
- Automatic resolution tracking

### 3. Quality Metrics
- Linting scores
- Code complexity analysis
- Test coverage before/after
- Performance impact assessment
- Automated test results

### 4. Learning & Pattern Detection
- Common failure patterns by task type
- Success patterns and best practices
- Prompt patterns that work well
- Time-of-day performance patterns

### 5. Detailed Execution Logging
- Step-by-step progress tracking
- Intermediate outputs and checkpoints
- Decision points and choices
- Error details and recovery actions

### 6. Task Lifecycle Events
- Created, started, paused, resumed
- Blocked, unblocked, reviewed
- Completed, failed, rolled back
- Custom events with metadata

### 7. Worker Performance
- Tasks per worker tracking
- Worker specialization analysis
- Efficiency scores over time
- Best worker selection for task types

### 8. Predictive Analytics
- Task duration predictions
- Failure risk assessment
- Optimal scheduling times
- Resource requirement predictions

## Integration Guide

### Basic Integration

```python
from scripts.monitoring_integration import MonitoringIntegration

# In your worker class
class MyWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.monitoring = MonitoringIntegration()
    
    async def execute_task(self, work_item):
        # Monitoring happens automatically with decorator
        return await self._execute_with_monitoring(work_item)
    
    @MonitoringIntegration().monitor_task_execution
    async def _execute_with_monitoring(self, work_item):
        # Your task execution code here
        pass
```

### Manual Step Tracking

```python
# Track execution steps
with self.monitoring.track_execution_step(task_id, "validation"):
    # Validation code
    pass

with self.monitoring.track_execution_step(task_id, "processing"):
    # Processing code
    pass
```

### API Call Tracking

```python
# Track API calls
self.monitoring.track_api_call(task_id, "openai")
self.monitoring.track_api_call(task_id, "gemini")
```

### Claude Execution Tracking

```python
# Track Claude Code CLI execution
start_time = time.time()
claude_result = await execute_claude(...)
duration = time.time() - start_time

self.monitoring.track_claude_execution(
    task_id, 
    success=claude_result['success'],
    duration=duration
)
```

### Dependency Management

```python
# Add dependencies between tasks
self.monitoring.add_task_dependency(
    parent_task_id, 
    child_task_id, 
    "blocks"
)

# Check blocking tasks
blocking = self.monitoring.get_blocking_tasks(task_id)
if blocking:
    raise Exception(f"Task blocked by: {blocking}")
```

### Lifecycle Events

```python
# Track custom events
self.monitoring.track_lifecycle_event(task_id, "waiting_for_review", {
    "reviewer": "user123",
    "reason": "major changes"
})

self.monitoring.track_lifecycle_event(task_id, "approved", {
    "reviewer": "user123",
    "comments": "LGTM"
})
```

### Quality Measurement

```python
# Measure code quality after task
quality = await self.monitoring.measure_quality(
    task_id, 
    repository_path
)
```

### Predictive Features

```python
# Get predictions before executing
predicted_duration = self.monitoring.predict_task_duration(
    task_type, 
    repository
)

failure_risk = self.monitoring.get_failure_risk(task_type)

best_worker = self.monitoring.get_best_worker(task_type)

optimal_time = self.monitoring.get_optimal_schedule_time(task_type)
```

## Data Storage

All monitoring data is stored in the `task_monitoring/` directory:

- `task_{id}_metrics.json` - Execution metrics
- `steps_{id}.json` - Execution steps
- `events_{id}.json` - Lifecycle events
- `quality_{id}.json` - Quality metrics
- `dependencies_{id}.json` - Task dependencies
- `learned_patterns.json` - Learning data
- `worker_performance.json` - Worker metrics
- `task_history.json` - Historical data

## Reporting

### Task Report
```python
report = self.monitoring.get_task_report(task_id)
# Contains: metrics, quality, steps, events, dependencies
```

### System Insights
```python
insights = self.monitoring.get_system_insights()
# Contains: task type stats, worker performance, 
# common failures, predictions
```

## Best Practices

1. **Always track execution steps** - Use context managers for automatic completion tracking
2. **Track all API calls** - Helps identify expensive operations
3. **Record meaningful lifecycle events** - Include metadata for context
4. **Check dependencies before execution** - Prevents wasted work
5. **Use predictions for scheduling** - Optimize worker assignment and timing
6. **Review failure patterns regularly** - Identify systemic issues
7. **Monitor worker specialization** - Assign tasks to best performers

## Example Full Integration

```python
class FullyMonitoredWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.monitoring = MonitoringIntegration()
    
    @MonitoringIntegration().monitor_task_execution
    async def execute_work_item(self, work_item):
        task_id = work_item.id
        
        # Check dependencies
        blocking = self.monitoring.get_blocking_tasks(task_id)
        if blocking:
            self.monitoring.track_lifecycle_event(task_id, "blocked", {
                "blocking_tasks": blocking
            })
            raise Exception(f"Blocked by: {blocking}")
        
        # Validation step
        with self.monitoring.track_execution_step(task_id, "validation"):
            self._validate_work_item(work_item)
        
        # AI Processing
        with self.monitoring.track_execution_step(task_id, "ai_processing"):
            self.monitoring.track_api_call(task_id, "openai")
            result = await self._process_with_ai(work_item)
        
        # Claude execution
        if self.should_use_claude(work_item):
            with self.monitoring.track_execution_step(task_id, "claude_execution"):
                start = time.time()
                claude_result = await self._execute_with_claude(work_item)
                self.monitoring.track_claude_execution(
                    task_id,
                    success=claude_result['success'],
                    duration=time.time() - start
                )
        
        # Quality check
        with self.monitoring.track_execution_step(task_id, "quality_check"):
            quality = await self.monitoring.measure_quality(
                task_id,
                work_item.repository
            )
        
        return {"success": True, "quality": quality}
```

## Monitoring Dashboard (Future)

The monitoring data can be visualized through:
- Task execution timelines
- Worker performance charts
- Failure pattern analysis
- Resource usage graphs
- Predictive accuracy tracking

For now, use the JSON reports and system insights for analysis.