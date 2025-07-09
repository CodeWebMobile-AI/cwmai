# Adaptive Parameter Evolution System

## Overview

The Adaptive Parameter Evolution System allows tools to automatically evolve when the AI attempts to use parameters that don't exist but make logical sense. Instead of failing with an error, the system analyzes the intent, updates the tool code if appropriate, and retries the operation.

## Problem Solved

Previously, when the AI tried to use a parameter that didn't exist (like `filter="analyze_repository_health"` on `list_tools`), the system would:
- ❌ Fail with an error
- ❌ Require manual code updates
- ❌ Not learn from the attempt

Now, the system:
- ✅ Detects useful parameter additions
- ✅ Automatically updates tool code
- ✅ Learns from usage patterns
- ✅ Maintains backward compatibility

## How It Works

### 1. Parameter Mismatch Detection
When a tool is called with unexpected parameters, the `ParameterEvolutionHandler` intercepts the call before it fails.

```python
# AI tries this:
list_tools(filter="repository", sort="name")

# System detects: filter and sort don't exist
```

### 2. Intent Analysis
The `ParameterIntentAnalyzer` determines if the parameters make sense:
- Recognizes common patterns (filter, limit, sort, etc.)
- Uses AI to analyze domain-specific parameters
- Tracks usage patterns over time

### 3. Tool Enhancement
If parameters are deemed useful, the `EnhancedToolEnhancer`:
- Parses the tool's current code using AST
- Generates updated code with new parameters
- Validates the changes
- Creates a backup of the original

### 4. Code Update Example

**Before:**
```python
def list_tools(limit: int = 10):
    """List available tools."""
    tools = get_all_tools()
    return dict(list(tools.items())[:limit])
```

**After Automatic Evolution:**
```python
def list_tools(limit: int = 10, filter: str = ""):
    """List available tools.
    
    Args:
        limit: Maximum number of tools to return
        filter: Search string to filter tools by name
    """
    if not isinstance(filter, str):
        raise TypeError("filter must be a string")
        
    tools = get_all_tools()
    
    if filter:
        filtered_tools = {
            name: desc for name, desc in tools.items() 
            if filter.lower() in name.lower()
        }
    else:
        filtered_tools = tools
        
    return dict(list(filtered_tools.items())[:limit])
```

### 5. Tool Reload & Retry
After enhancement:
1. Tool is reloaded with new capabilities
2. Original call is retried automatically
3. AI gets the result it expected

## Components

### ParameterEvolutionHandler
- Detects parameter mismatches
- Coordinates the evolution process
- Maintains evolution history
- Handles retry logic

### ParameterIntentAnalyzer
- Analyzes parameter names for common patterns
- Determines if parameters make logical sense
- Tracks pattern frequency

### EnhancedToolEnhancer
- Performs actual code modifications
- Uses AST for safe code updates
- Maintains code style and conventions
- Creates backups before changes

### ToolCodeUpdater
- Specialized for parameter additions
- Ensures backward compatibility
- Adds proper validation
- Updates documentation

## Usage Examples

### Example 1: Adding Filter to list_tools
```python
# AI attempts (filter doesn't exist)
result = await tool_system.call_tool('list_tools', filter='repository')

# System automatically:
# 1. Detects 'filter' is missing but useful
# 2. Updates list_tools to support filtering
# 3. Retries the call
# 4. Returns filtered results
```

### Example 2: Multiple New Parameters
```python
# AI attempts (neither exists)
result = await tool_system.call_tool('get_repositories', 
                                   sort='stars', 
                                   include_archived=False)

# System enhances tool with both parameters if they make sense
```

### Example 3: Invalid Parameter Rejection
```python
# AI attempts (nonsensical parameter)
result = await tool_system.call_tool('get_system_status', 
                                   pizza_topping='pepperoni')

# System rejects this and provides guidance on valid parameters
```

## Configuration

### Evolution History
Parameter evolution patterns are tracked in `parameter_evolution_history.json`:
```json
{
  "evolutions": [
    {
      "timestamp": "2024-01-20T10:30:00Z",
      "tool": "list_tools",
      "added_params": ["filter"],
      "reason": "Filter tools by name pattern",
      "confidence": 0.95
    }
  ],
  "patterns": {
    "list_tools:filter": {
      "occurrences": 5,
      "first_seen": "2024-01-20T10:30:00Z",
      "reason": "Common filtering pattern"
    }
  }
}
```

### Enabling/Disabling
The system is automatically enabled when the enhanced tools are available. To disable:
```python
tool_system.parameter_evolution = None
```

## Best Practices

1. **Parameter Naming**: Use clear, descriptive parameter names that follow common patterns
2. **Default Values**: New parameters always get sensible defaults for backward compatibility
3. **Validation**: Enhanced tools include parameter validation
4. **Documentation**: Docstrings are automatically updated

## Limitations

1. **Complex Logic**: Very complex parameter interactions may require manual implementation
2. **Breaking Changes**: System won't make changes that break existing functionality
3. **Type Safety**: Parameter types are inferred and may need manual refinement

## Future Enhancements

1. **Learning from Usage**: Track which evolved parameters are actually used
2. **Cross-Tool Patterns**: Apply successful patterns to similar tools
3. **Rollback Capability**: Automatic rollback if evolved tool fails
4. **Parameter Deprecation**: Remove unused parameters over time

## Testing

Run the test suite:
```bash
python test_parameter_evolution.py
```

Run the demo:
```bash
python demo_parameter_evolution.py
```

## Integration

The system integrates seamlessly with:
- Tool calling system
- Tool evolution system
- Semantic tool matcher
- Multi-tool orchestrator

No additional setup required - it works automatically when tools are called with unexpected parameters.