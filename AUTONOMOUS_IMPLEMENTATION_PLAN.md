# Autonomous Implementation Plan

## Current vs Desired Behavior

### Current (Partial Automation):
```python
# In LocalIssueCreator.execute_work_item():
1. Check if issue exists
2. If not, create issue JSON
3. Return "success" ← But no code was written!
```

### Desired (Full Automation):
```python
# Enhanced execute_work_item():
1. Check if issue exists
2. If not, create issue JSON
3. Parse issue requirements
4. Generate implementation plan
5. Write actual code using AI
6. Run tests
7. Commit changes
8. Update issue status
9. Return "success" with actual work done
```

## Implementation Plan for Full Autonomy

### Step 1: Create AICodeExecutor Class

```python
# scripts/ai_code_executor.py
import asyncio
from pathlib import Path
from typing import Dict, Any
import git

class AICodeExecutor:
    """Executes actual code implementation for issues."""
    
    def __init__(self, ai_client):
        self.ai_client = ai_client
        
    async def implement_issue(self, issue_data: Dict[str, Any], repo_path: Path) -> Dict[str, Any]:
        """
        Autonomously implement the requirements described in an issue.
        
        Flow:
        1. Analyze issue requirements
        2. Examine codebase context
        3. Generate implementation
        4. Write files
        5. Run tests
        6. Commit changes
        """
        # 1. Parse issue requirements
        task_type = issue_data.get('labels', [])
        title = issue_data['title']
        description = issue_data['body']
        
        # 2. Analyze codebase
        context = await self.analyze_codebase(repo_path, task_type)
        
        # 3. Generate implementation plan
        plan = await self.generate_implementation_plan(
            title, description, context, task_type
        )
        
        # 4. Write code
        files_changed = await self.write_code_files(plan, repo_path)
        
        # 5. Validate changes
        validation_result = await self.validate_changes(repo_path, files_changed)
        
        # 6. Commit if valid
        if validation_result['valid']:
            commit_result = await self.commit_changes(
                repo_path, files_changed, issue_data
            )
            return {
                'success': True,
                'files_changed': files_changed,
                'commit': commit_result['sha'],
                'message': f"Implemented: {title}"
            }
        else:
            return {
                'success': False,
                'error': validation_result['errors'],
                'files_attempted': files_changed
            }
```

### Step 2: Enhance LocalIssueCreator

```python
# Modify scripts/local_issue_creator.py
class LocalIssueCreator:
    def __init__(self, autonomous_mode=True):
        self.autonomous_mode = autonomous_mode
        if autonomous_mode:
            self.code_executor = AICodeExecutor(self.ai_client)
    
    async def execute_work_item(self, work_item: WorkItem) -> Dict[str, Any]:
        # ... existing issue creation code ...
        
        if self.autonomous_mode and issue_created:
            # NEW: Actually implement the issue!
            print(f"🤖 Autonomously implementing: {work_item.title}")
            
            implementation_result = await self.code_executor.implement_issue(
                issue_data=issue_data,
                repo_path=repo_path
            )
            
            if implementation_result['success']:
                print(f"✅ Successfully implemented and committed: {implementation_result['commit']}")
                return {
                    'success': True,
                    'task_id': work_item.id,
                    'issue_number': issue.number,
                    'implementation': implementation_result,
                    'value_created': 2.0  # Higher value for actual implementation
                }
```

### Step 3: Task Type Implementations

For each task type, the system would:

#### FEATURE Tasks:
```python
async def implement_feature(self, description, repo_path):
    # 1. Identify where to add feature
    # 2. Generate new code
    # 3. Update imports/exports
    # 4. Add tests
    # 5. Update documentation
```

#### BUG_FIX Tasks:
```python
async def implement_bug_fix(self, description, repo_path):
    # 1. Locate buggy code
    # 2. Analyze root cause
    # 3. Generate fix
    # 4. Add regression test
    # 5. Verify fix doesn't break other tests
```

#### TESTING Tasks:
```python
async def implement_tests(self, description, repo_path):
    # 1. Find code to test
    # 2. Analyze test patterns in repo
    # 3. Generate comprehensive tests
    # 4. Run tests to ensure they pass
```

## Complete Autonomous Flow

### 1. System Startup
```bash
./run_hybrid_continuous.py --workers 20 --autonomous
```

### 2. Worker Picks Up Task
```
Worker-1: Found issue "Add authentication to API"
Worker-1: Creating issue JSON...
Worker-1: Analyzing repository structure...
Worker-1: Generating implementation plan...
```

### 3. AI Implementation
```
Worker-1: Creating auth middleware...
Worker-1: Writing file: src/middleware/auth.js
Worker-1: Creating auth tests...
Worker-1: Writing file: tests/auth.test.js
Worker-1: Running tests... ✓ All tests pass
```

### 4. Git Operations
```
Worker-1: Committing changes...
Worker-1: Created commit: a1b2c3d "Add authentication to API endpoints"
Worker-1: Updating issue status to 'implemented'
```

### 5. Continuous Loop
```
Worker-1: Task completed! Looking for next task...
Worker-2: Implementing "Fix login validation bug"...
Worker-3: Adding "User profile component"...
```

## Configuration for Autonomous Mode

```yaml
# config/autonomous_settings.yaml
autonomous_mode:
  enabled: true
  
  code_generation:
    max_file_size: 10000  # lines
    languages_supported:
      - python
      - javascript
      - typescript
      - react
    
  validation:
    run_tests: true
    run_linter: true
    check_types: true
    
  git_operations:
    auto_commit: true
    branch_per_task: false
    push_to_remote: false  # Safety: don't auto-push
    
  safety_limits:
    max_files_per_task: 20
    max_lines_changed: 1000
    require_test_pass: true
```

## Current Blockers to Full Autonomy

1. **LocalIssueCreator only creates issues** - needs to be extended
2. **No code generation component** - need AICodeExecutor
3. **No validation pipeline** - need test runners
4. **No git integration** - need commit automation

## Quick Path to Autonomy

The fastest way to make it autonomous:

1. **Use Claude Code CLI** as the executor:
```python
# In LocalIssueCreator
import subprocess

def execute_with_claude(self, issue_description, repo_path):
    cmd = ["claude", issue_description]
    result = subprocess.run(cmd, cwd=repo_path, capture_output=True)
    return result.returncode == 0
```

2. **Or use the AI client directly**:
```python
# Generate code using existing AI client
code = await self.ai_client.generate(
    prompt=f"Implement this feature: {issue_description}",
    context=codebase_context
)
await self.write_files(code)
```

## Summary

**Current State**: Semi-autonomous (creates issues only)
**Desired State**: Fully autonomous (implements solutions)
**Missing Piece**: Code execution component that takes issues and writes actual code

The system has all the intelligence to identify what needs to be done, but lacks the execution layer to actually do it!