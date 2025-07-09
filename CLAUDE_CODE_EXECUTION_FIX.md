# Claude Code Execution Fix Documentation

## Issue Discovered (2025-06-22)

Claude Code CLI was not making actual file changes when executed by the continuous AI system. It was running successfully but only asking for permission to edit files instead of making changes.

## Root Cause

1. **Incorrect invocation**: Claude Code was being called with the prompt as a command-line argument instead of via stdin
2. **Permission mode**: Claude Code was running in interactive mode, waiting for permission to edit files

## Solution Implemented

### 1. Fixed Claude Invocation Method
Changed from:
```python
result = subprocess.run(
    [self.claude_executable, prompt],
    cwd=str(repo_path),
    capture_output=True,
    text=True,
    timeout=300
)
```

To:
```python
process = subprocess.Popen(
    [self.claude_executable, '--dangerously-skip-permissions'],
    cwd=str(repo_path),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = process.communicate(input=prompt, timeout=300)
```

### 2. Added Permission Flag
- Used `--dangerously-skip-permissions` flag to allow automatic file editing
- This bypasses the interactive permission prompts

### 3. Enhanced Logging
Added detailed logging to track:
- Claude executable location
- Working directory
- Prompt content and length
- Execution time and return code
- Git status after execution
- Files changed count
- Claude's output

## Performance Metrics

- **Before fix**: 67-194 seconds (waiting for permissions)
- **After fix**: 24-34 seconds (direct execution)
- **Success rate**: 100% after fix

## Key Learnings

1. Claude Code expects input via stdin, not command-line arguments
2. The `--dangerously-skip-permissions` flag is required for autonomous operation
3. Always check git status after execution to verify actual changes
4. Claude's output message indicates whether it's waiting for permissions

## Testing Command

```bash
export $(cat .env.local | grep -v '^#' | xargs) && \
PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai \
arch -arm64 /usr/bin/python3 test_claude_execution.py
```

## Verification Steps

1. Check git status shows modified files (not just untracked)
2. Claude output should say "Task completed successfully" not "I need permission"
3. Execution time should be under 60 seconds for simple tasks

## Files Modified

- `/scripts/local_issue_creator.py` - Updated `_execute_with_claude()` method
- Added enhanced logging throughout execution flow

## Important Notes

- The `--dangerously-skip-permissions` flag should only be used in controlled environments
- Always run in repositories where file changes are safe and can be reverted
- Monitor Claude's actions via git status and logs

---
Generated: 2025-06-22
Context: Debugging why Claude Code wasn't making actual file changes in brand-guardian-ai repository