# Tool Generation Solution Documentation

## Problem Statement
The IntelligentToolGenerator had three critical issues:
1. Code extraction needs improvement
2. The assistant calls tools directly instead of using tool_calling_system
3. Missing proper imports for RepositoryAnalyzer and StateManager

The key requirement was to fix these "permanently without hard coding to specific scenarios."

## Root Cause
The AI was making assumptions about method names (analyze, update_repository, etc.) instead of discovering them. The conversational assistant wasn't actually executing the discovery commands in the prompts.

## Solution Overview
We implemented a pre-discovery phase that runs grep commands directly using subprocess to find actual method signatures, then provides these discovered APIs to the AI in the prompt.

## Implementation Details

### 1. Pre-Discovery Phase (`_run_discovery_phase`)
```python
async def _run_discovery_phase(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Run discovery phase to find actual APIs."""
    import subprocess
    
    discovered = {
        'StateManager': {},
        'RepositoryAnalyzer': {},
        'tools': []
    }
    
    # Discover StateManager methods
    state_result = subprocess.run(
        ["grep", "def ", "scripts/state_manager.py"],
        capture_output=True, text=True
    )
    
    # Parse and extract method signatures
    for line in state_result.stdout.split('\n'):
        if 'def ' in line and '__' not in line and '(' in line:
            method_match = re.search(r'def\s+(\w+)\s*\((.*?)\)', line)
            if method_match:
                method_name = method_match.group(1)
                params = method_match.group(2)
                discovered['StateManager'][method_name] = params
```

### 2. Enhanced Prompt Structure
The prompt now includes:
- Clear distinction between MODULES (import directly) and TOOLS (call via tool_system)
- Discovered APIs provided upfront
- Mandatory discovery output format
- Method inventory requirement
- Validation rules

### 3. Validation Enhancements
- `_validate_discovery_happened()`: Checks if AI ran discovery commands
- Enhanced `_validate_tool()`: Catches common mistakes like direct tool calls
- Retry logic if discovery was skipped

## Key Learnings

### What Didn't Work
1. **Relying on AI to execute commands**: The conversational assistant didn't actually run the grep commands
2. **Generic prompts**: Even detailed instructions weren't enough without actual data
3. **Post-generation fixes**: Trying to fix issues after generation was error-prone

### What Worked
1. **Pre-discovery with subprocess**: Running grep commands ourselves to get real method signatures
2. **Providing discovered APIs upfront**: Giving the AI actual method names to use
3. **Structured validation**: Catching common mistakes before runtime
4. **Clear module vs tool distinction**: Explicit categorization in the prompt

## Result
The final generated tool correctly:
- Used only discovered methods: `analyze_repository()`, `get_state()`, `update_state()`
- Imported modules properly
- Called tools via tool_system
- Had no placeholder code or assumptions

## Future Improvements
1. Cache discovery results for performance
2. Extend discovery to more module types
3. Add discovery for method parameters and return types
4. Create a discovery service that other components can use