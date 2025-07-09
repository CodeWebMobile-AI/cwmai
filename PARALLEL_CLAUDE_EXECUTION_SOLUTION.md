# Parallel Claude Execution Solution

## Problem Identified

The Claude executions were being **serialized by our code**, not by Claude itself. The issue was in the worker implementation where each worker would:

1. Pick up a task from the queue
2. Create an issue
3. Execute Claude (blocking for 48+ minutes)
4. Only then move to the next task

This meant Worker 03 had to wait for Worker 02 to complete its entire Claude execution before starting its own.

## Solution Implemented

We've decoupled issue creation from Claude execution by introducing:

### 1. Claude Execution Queue (`claude_execution_queue.py`)
- Manages a queue of Claude tasks to be executed
- Tracks task status (pending, running, completed, failed)
- Persists queue state to disk
- No automatic timeouts (Claude can run as long as needed)

### 2. Claude Executor Pool (`claude_executor.py`)
- Multiple Claude executors that process tasks from the queue
- Each executor runs independently and in parallel
- Provides progress logging every 30 seconds
- Handles errors and retries gracefully

### 3. Modified Local Issue Creator
- Now queues Claude execution instead of executing directly
- Returns immediately after creating issue and queuing Claude task
- Workers can move on to create more issues while Claude executes

## How It Works

### Before (Serial Execution):
```
Worker 02: Create Issue → Execute Claude (48 min) → Complete
Worker 03: Waiting...                                → Create Issue → Execute Claude
```

### After (Parallel Execution):
```
Worker 02: Create Issue → Queue Claude → Next Task
Worker 03: Create Issue → Queue Claude → Next Task

Claude Executor 01: Execute Claude Task 1 (48 min)
Claude Executor 02: Execute Claude Task 2 (48 min) [PARALLEL]
Claude Executor 03: Execute Claude Task 3 (48 min) [PARALLEL]
```

## Benefits

1. **True Parallel Execution**: Multiple Claude processes run simultaneously
2. **Better Resource Utilization**: Workers don't sit idle waiting for Claude
3. **Scalability**: Can easily add more Claude executors for more parallelism
4. **No Timeouts**: Claude tasks can run as long as needed without interruption
5. **Progress Visibility**: Real-time monitoring of Claude execution progress

## Configuration

### Environment Variables
- `CLAUDE_EXECUTABLE`: Path to Claude executable (default: 'claude')
- `ENABLE_CLAUDE_EXECUTION`: Enable/disable Claude execution (default: 'true')

### Orchestrator Settings
- `num_claude_executors`: Number of parallel Claude executors (default: 3)
- Can be adjusted based on system resources and needs

## Usage

### 1. Using Parallel Claude Orchestrator
```python
from scripts.parallel_claude_orchestrator import ParallelClaudeOrchestrator

orchestrator = ParallelClaudeOrchestrator(
    max_workers=10,           # Workers for creating issues
    num_claude_executors=3    # Parallel Claude executors
)
await orchestrator.start()
```

### 2. Direct Claude Executor Pool Usage
```python
from scripts.claude_execution_queue import ClaudeExecutionQueue
from scripts.claude_executor import ClaudeExecutorPool

queue = ClaudeExecutionQueue()
pool = ClaudeExecutorPool(num_executors=3, queue=queue)
await pool.start()
```

### 3. Testing
```bash
# Test parallel execution
python test_parallel_claude_execution.py

# Monitor Claude processes
ps aux | grep claude
```

## Performance Impact

With 3 Claude executors running in parallel:
- **Before**: 3 tasks × 48 minutes = 144 minutes (serial)
- **After**: 48 minutes (parallel) = **3x speedup**

The speedup scales with the number of executors, limited by system resources.

## Important Notes

1. **No Timeouts**: Claude executions are NOT timed out by default. They can run as long as needed.
2. **Progress Logging**: Executors log progress every 30 seconds
3. **Error Handling**: Failed tasks are automatically retried (configurable)
4. **Queue Persistence**: Queue state is saved to disk and survives restarts

## Future Enhancements

1. **Dynamic Executor Scaling**: Add/remove executors based on queue size
2. **Priority Queue**: Process high-priority tasks first
3. **Resource Monitoring**: Track CPU/memory usage per executor
4. **Distributed Execution**: Run executors on multiple machines