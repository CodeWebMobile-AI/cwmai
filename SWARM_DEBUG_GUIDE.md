# Dynamic Swarm Debug Guide

## Overview

The `DynamicSwarmIntelligence` class has been enhanced with comprehensive logging capabilities to help debug the "list index out of range" error and other swarm-related issues.

## Key Debug Features Added

### 1. Enhanced Logging Throughout the Swarm Pipeline

- **Agent Analysis Logging**: Tracks each agent's analysis process
- **AI Response Logging**: Logs raw AI responses and parsing results
- **Cross-Pollination Logging**: Monitors data flow between agents
- **Performance Tracking**: Records metrics and identifies problem agents

### 2. Safety Checks in Critical Methods

#### Fixed `_format_other_insights()` Method
The main source of the "list index out of range" error was in this method at line 194:

```python
# OLD CODE (DANGEROUS):
main_challenge = insight.get('challenges', ['None'])[0]

# NEW CODE (SAFE):
challenges = insight.get('challenges', [])
if challenges:
    main_challenge = challenges[0]
else:
    main_challenge = "No challenges identified"
    logging.warning(f"[SWARM_DEBUG] Agent has EMPTY challenges list!")
```

### 3. Debug Logging Categories

All debug logs use the `[SWARM_DEBUG]` prefix for easy filtering:

- `INFO`: High-level process flow
- `DEBUG`: Detailed data structures and responses
- `WARNING`: Empty lists or missing data
- `ERROR`: Exceptions and failures

## How to Enable Debug Logging

### Method 1: Using the Built-in Debug Method

```python
from scripts.dynamic_swarm import DynamicSwarmIntelligence

# Create swarm
swarm = DynamicSwarmIntelligence(ai_brain)

# Enable debug logging
swarm.enable_debug_logging("DEBUG")  # or "INFO", "WARNING", "ERROR"

# Run analysis with logging
result = await swarm.process_task_swarm(task, context)
```

### Method 2: Manual Logging Configuration

```python
import logging

# Configure logging manually
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Filter for swarm debug messages only
class SwarmDebugFilter(logging.Filter):
    def filter(self, record):
        return '[SWARM_DEBUG]' in record.getMessage()

handler = logging.StreamHandler()
handler.addFilter(SwarmDebugFilter())
logging.getLogger().addHandler(handler)
```

## Debugging the "List Index Out of Range" Error

### What to Look For

1. **Empty Challenge Lists**:
   ```
   WARNING - [SWARM_DEBUG] Agent agent_123 parsed EMPTY challenges list
   WARNING - [SWARM_DEBUG] Insight 2 from ARCHITECT has EMPTY challenges list!
   ```

2. **AI Response Parsing Errors**:
   ```
   ERROR - [SWARM_DEBUG] Agent agent_456 parse error: JSON decode error
   ERROR - [SWARM_DEBUG] Agent agent_456 raw response that failed to parse: ...
   ```

3. **Agent Performance Issues**:
   ```
   INFO - [SWARM_DEBUG] Agent agent_789 individual analysis: 0 challenges, 0 key_points, confidence: 0, error: True
   ```

### Common Root Causes

1. **AI Model Response Quality**: Some models may not generate proper JSON responses
2. **Prompt Engineering Issues**: Agents may not receive clear enough instructions
3. **Model Availability**: AI models may be unavailable or rate-limited
4. **Data Structure Mismatches**: Response format doesn't match expected structure

## Using the Test Script

Run the comprehensive debug test:

```bash
cd /workspaces/cwmai
python test_swarm_debug.py
```

This will:
- Enable debug logging
- Run a test analysis
- Show detailed logging output
- Identify any issues with empty lists
- Provide performance metrics

## Debug Summary API

Get a comprehensive overview of swarm state:

```python
debug_summary = swarm.get_debug_summary()
print(json.dumps(debug_summary, indent=2))
```

Returns:
- Agent configuration and performance
- Recent analysis summaries
- Performance metrics
- Error tracking data

## Performance Monitoring

Track agent performance over time:

```python
analytics = swarm.get_swarm_analytics()
agent_performance = analytics['agent_performance']

for agent_id, metrics in agent_performance.items():
    print(f"Agent {agent_id}:")
    print(f"  - Average Confidence: {metrics['average_confidence']}")
    print(f"  - Average Alignment: {metrics['average_alignment']}")
    print(f"  - Total Analyses: {metrics['total_analyses']}")
```

## Troubleshooting Tips

### If You See "List Index Out of Range" Errors

1. **Enable debug logging** and look for empty list warnings
2. **Check AI response quality** - look for parse errors
3. **Identify problematic agents** - some may consistently fail
4. **Verify model availability** - ensure all AI models are accessible

### If Swarm Analysis is Slow

1. **Check duration metrics** in debug summary
2. **Monitor AI response times** in detailed logs
3. **Look for retry attempts** or timeout errors
4. **Consider using faster AI models** for development

### If Results are Poor Quality

1. **Review agent confidence scores** in performance tracking
2. **Check alignment scores** with system charter
3. **Monitor consensus building** in cross-pollination phase
4. **Adjust agent prompts** based on logged responses

## Log File Analysis

To analyze logs after running:

```bash
# Filter for swarm debug messages only
grep '\[SWARM_DEBUG\]' logfile.log

# Find empty list warnings
grep 'EMPTY.*list' logfile.log

# Find parse errors
grep 'parse error' logfile.log

# Track specific agent performance
grep 'Agent agent_123' logfile.log
```

## Next Steps

After identifying issues:

1. **Fix AI Response Quality**: Improve prompts or switch models
2. **Add More Safety Checks**: Prevent crashes from unexpected data
3. **Optimize Performance**: Cache responses or use faster models
4. **Monitor in Production**: Set up proper logging infrastructure