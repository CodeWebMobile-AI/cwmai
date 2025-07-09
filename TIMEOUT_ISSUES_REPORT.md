# Timeout Issues in Hybrid Continuous System

## 1. Claude Execution Timeouts

### Local Issue Creator (`scripts/local_issue_creator.py`)
- **Line 706**: Comment says "NO TIMEOUT - let Claude run as long as it needs"
- **Line 725**: Uses `process.communicate(input=prompt)` without explicit timeout
- **Issue**: No explicit `timeout=None` parameter, relying on default behavior
- **Fix**: Add explicit `timeout=None` to ensure no timeout

### Error Logging Issues
- **Lines 779-784**: When Claude fails, only logs `result.stderr` 
- **Issue**: Doesn't capture full error context, stack traces, or subprocess details
- **Fix**: Enhanced error logging needed

## 2. HTTP AI Client Timeouts (`scripts/http_ai_client.py`)

### Request Timeouts
- **Line 164**: Default timeout of 120 seconds (2 minutes) for AI requests
- **Line 171**: `requests.post(..., timeout=timeout)`
- **Issue**: 2-minute timeout is too short for complex AI tasks
- **Fix**: Increase timeout or make configurable

### Retry Logic
- **Lines 166-180**: Has retry logic with exponential backoff
- **Max retries**: 3 attempts
- **Issue**: Still limited by the 120-second timeout per attempt

## 3. Process Monitoring Scripts

### run_hybrid_5min.py
- **Line 76**: `process.terminate()` after timeout
- **Line 79**: `process.kill()` if terminate fails
- **Issue**: Forcefully kills processes after timeout
- **Purpose**: This is a test script with intentional 5-minute limit

### monitor_hybrid_run.py  
- **Line 119**: `process.terminate()`
- **Issue**: Can kill long-running Claude processes
- **Fix**: Should allow Claude processes to complete

### improvement_validator.py
- **Line 370**: `process.terminate()`
- **Context**: Terminates validation processes that exceed limits

## 4. Missing Timeout Configurations

### No Environment Variable Controls
- No `CLAUDE_EXECUTION_TIMEOUT` environment variable
- No way to configure timeouts without code changes
- No differentiation between task types (simple vs complex)

### No Task-Based Timeout Logic
- All tasks treated the same regardless of complexity
- "Initial Project Setup" tasks need 10+ minutes
- Simple tasks complete in seconds

## 5. Workflow Task Timeouts (`scripts/continuous_orchestrator.py`)

### Hardcoded Timeouts
- **Lines 1859-1889**: Fixed timeouts for workflow tasks:
  - Code Review: 7200 seconds (2 hours)
  - Unit Testing: 3600 seconds (1 hour)  
  - Integration Testing: 5400 seconds (1.5 hours)
- **Issue**: Not applicable to Claude execution but shows timeout patterns

## 6. Error Capture Deficiencies

### Incomplete Error Information
1. **local_issue_creator.py** (lines 786-791):
   ```python
   except Exception as e:
       self.logger.error(f"Error executing Claude: {e}")
       return {'success': False, 'error': str(e)}
   ```
   - Missing: Stack trace, exception type, subprocess details

2. **No stdout capture on failure**:
   - When Claude returns non-zero exit code, stdout might contain useful error info
   - Currently only logs stderr

3. **No intermediate output capture**:
   - Claude might output progress before failing
   - This information is lost

## 7. System-Level Issues

### Python Default Behaviors
- `subprocess.communicate()` without timeout parameter uses system default
- Different OS/Python versions may have different defaults
- No guarantee of truly unlimited execution time

### Background Process Management
- Multiple Claude processes accumulating (seen in ps output)
- No cleanup of orphaned processes
- No process tracking/management system

## Recommendations

### 1. Immediate Fixes
```python
# Explicit no timeout
stdout, stderr = process.communicate(input=prompt, timeout=None)

# Enhanced error logging
import traceback
except Exception as e:
    self.logger.error(f"Error executing Claude: {e}")
    self.logger.error(f"Exception type: {type(e).__name__}")
    self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
    if hasattr(e, 'stdout'):
        self.logger.error(f"Process stdout: {e.stdout}")
    if hasattr(e, 'stderr'):
        self.logger.error(f"Process stderr: {e.stderr}")
```

### 2. Configuration Improvements
```python
# Add environment variable support
import os
claude_timeout = os.getenv('CLAUDE_EXECUTION_TIMEOUT')
if claude_timeout == 'none':
    timeout_value = None
elif claude_timeout:
    timeout_value = int(claude_timeout)
else:
    timeout_value = None  # Default to no timeout
```

### 3. Task-Based Timeouts
```python
# Different timeouts based on task complexity
def get_timeout_for_task(task_type, task_description):
    if 'initial project setup' in task_description.lower():
        return None  # No timeout for complex setup
    elif task_type == 'FEATURE':
        return 600  # 10 minutes for features
    elif task_type == 'BUG_FIX':
        return 300  # 5 minutes for bug fixes
    else:
        return None  # Default to no timeout
```

### 4. Process Management
```python
# Track Claude processes
active_claude_processes = {}

# Before starting
process_id = str(uuid.uuid4())
active_claude_processes[process_id] = {
    'process': process,
    'start_time': time.time(),
    'task': work_item.title
}

# After completion
if process_id in active_claude_processes:
    del active_claude_processes[process_id]
```

### 5. Progress Monitoring
```python
# Enhanced progress monitoring with output capture
def monitor_claude_output(process, logger):
    """Monitor Claude output in real-time."""
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        output_lines.append(line)
        logger.debug(f"Claude output: {line.strip()}")
        if len(output_lines) % 100 == 0:
            logger.info(f"Claude has produced {len(output_lines)} lines of output")
    return ''.join(output_lines)
```

### 6. HTTP Client Timeout Increase
```python
# In http_ai_client.py
DEFAULT_TIMEOUT = int(os.getenv('AI_REQUEST_TIMEOUT', '600'))  # 10 minutes default

def _make_request_with_retry(self, ..., timeout: int = DEFAULT_TIMEOUT):
    # Use configurable timeout
```

## Summary

The main issues causing Claude execution failures are:

1. **No explicit timeout=None** in subprocess.communicate()
2. **HTTP AI client 120-second timeout** too short for complex tasks
3. **Insufficient error logging** - missing stdout, stack traces, and context
4. **No timeout configuration** - can't adjust without code changes
5. **Process termination** in monitoring scripts
6. **No differentiation** between simple and complex tasks

All of these can be fixed to allow Claude to run as long as needed with proper error reporting.