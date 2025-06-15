# Smart Reset Capabilities

The conversational AI assistant now includes intelligent system reset functionality that can analyze, recommend, and execute various types of system resets through natural conversation.

## Overview

The assistant can:
- Analyze system health and determine if a reset is needed
- Recommend the appropriate type of reset based on the situation
- Execute resets safely with confirmation and dry-run options
- Preserve important data when requested
- Handle emergency situations

## Reset Types

### 1. **Full Reset** (`FULL`)
- Clears all data, logs, cache, and state files
- Stops running processes
- Complete fresh start
- Use when: Major issues, testing from scratch

### 2. **Selective Reset** (`SELECTIVE`)
- Preserves AI response cache and/or knowledge base
- Clears execution state and logs
- Keeps learned data
- Use when: Want to fix issues but keep valuable data

### 3. **Emergency Reset** (`EMERGENCY`)
- Force stops all processes
- Clears all data immediately
- No graceful shutdown
- Use when: System is completely stuck

### 4. **Logs Only** (`LOGS_ONLY`)
- Clears only log files (.log, .txt)
- Preserves all other data
- Quick cleanup
- Use when: Too many log files accumulating

### 5. **State Only** (`STATE_ONLY`)
- Clears state files (.json)
- Preserves logs and cache
- Fixes state corruption
- Use when: State files are corrupted

### 6. **Cache Only** (`CACHE_ONLY`)
- Clears cache files (.cache, .pkl)
- Preserves logs and state
- Frees memory
- Use when: Cache is too large or corrupted

## Natural Language Examples

### Basic Reset Requests
```
You > Reset the system
Assistant > I can help you reset the system. Based on my analysis, a full reset would clear all data 
and give you a fresh start. This will delete all logs, state files, and cache data. 
Would you like me to proceed?

You > Yes, go ahead
Assistant > Starting full system reset...
✓ Reset completed successfully! Deleted 45 files and freed 125.3 MB.
```

### Selective Reset
```
You > Clean everything but keep the knowledge base
Assistant > I'll do a selective reset that preserves your knowledge base and AI cache while 
clearing execution state and logs. This will fix most issues while keeping your learned data.
Would you like me to proceed?
```

### Emergency Situation
```
You > Everything is stuck! The system won't respond!
Assistant > I understand this is an emergency. I'll perform an emergency reset which will:
⚠️ Force stop all processes immediately
⚠️ Clear all data without graceful shutdown
This is a last resort option. Are you sure you want to proceed?
```

### Smart Analysis
```
You > The system seems really slow
Assistant > Let me analyze the system... I found:
- High memory usage (2.5 GB)
- Large log files (500+ MB)
- State file hasn't updated in 15 minutes

I recommend a selective reset to clear logs and state while preserving your AI cache.
This should resolve the performance issues. Would you like me to proceed?
```

## How It Works

### 1. System Analysis
The assistant checks:
- Continuous AI health score
- Log file sizes
- State file corruption
- Error frequency
- Available disk space
- Process resource usage

### 2. Smart Recommendations
Based on the analysis, it recommends:
- **No reset needed**: System is healthy
- **Logs only**: Just log accumulation
- **Selective**: Preserve valuable data
- **Full**: Major issues detected
- **Emergency**: System unresponsive

### 3. Safe Execution
- Always shows what will be deleted first
- Dry-run option available
- Graceful process shutdown (except emergency)
- Creates fresh state files after reset
- Detailed success/failure reporting

## Integration with Conversation

The reset functionality is seamlessly integrated into the conversational flow:

```python
# The assistant recognizes reset-related inputs
"reset", "clean", "clear", "broken", "stuck", "fix", "emergency"

# Maps to appropriate handlers
- Questions about problems → Analyze and recommend
- Direct reset requests → Confirm and execute
- Emergency situations → Quick action with warnings
```

## Benefits

1. **No Manual Commands**: No need to remember `python reset_system.py` flags
2. **Intelligent Analysis**: Assistant determines what type of reset is best
3. **Safe Operation**: Always confirms before destructive operations
4. **Preserves Value**: Can keep expensive AI cache and learned knowledge
5. **Natural Language**: Just describe the problem in your own words

## Example Scenarios

### Scenario 1: Routine Maintenance
```
You > Can you clean up the old logs?
Assistant > I'll clear the log files for you. This will remove 23 log files 
and free up 156 MB of space. Your system state and cache will be preserved.
Shall I proceed?
```

### Scenario 2: System Issues
```
You > Nothing is working properly anymore
Assistant > Let me check what's wrong... I see multiple issues:
- System health is at 45%
- Several corrupted state files
- High error rate in logs

I recommend a full reset to resolve these issues. This will give you a 
completely fresh start. Would you like me to do that?
```

### Scenario 3: Cost-Conscious Reset
```
You > Reset but don't delete the AI cache, I don't want to pay for regenerating it
Assistant > Smart choice! I'll do a selective reset that preserves your AI response 
cache while clearing everything else. This will fix most issues while saving you 
API costs. Ready to proceed?
```

## Technical Details

The reset system uses:
- `reset_system.py` for actual reset operations
- Process management via psutil
- File pattern matching for selective deletion
- State analysis from multiple sources
- Graceful degradation when components unavailable

The assistant makes reset operations feel natural and safe, guiding users to the right solution for their specific situation.