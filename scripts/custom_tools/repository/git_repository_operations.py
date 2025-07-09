"""
AI-Generated Tool: git_repository_operations
Description: Perform Git operations on local repositories
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

__description__ = "Execute Git operations on repositories - status, commit, branch, push, pull, etc."
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["status", "add", "commit", "push", "pull", "branch", "checkout", "merge", "log", "diff", "stash", "remote"],
        "description": "Git operation to perform"
    },
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "files": {
        "type": "array",
        "required": False,
        "description": "Files to add (for add operation)"
    },
    "message": {
        "type": "string",
        "required": False,
        "description": "Commit message"
    },
    "branch_name": {
        "type": "string",
        "required": False,
        "description": "Branch name for branch operations"
    },
    "remote": {
        "type": "string",
        "required": False,
        "default": "origin",
        "description": "Remote name"
    },
    "limit": {
        "type": "integer",
        "required": False,
        "default": 10,
        "description": "Number of commits to show in log"
    }
}
__examples__ = [
    "git_repository_operations(operation='status', repo_name='my-app')",
    "git_repository_operations(operation='add', repo_name='my-app', files=['.''])",
    "git_repository_operations(operation='commit', repo_name='my-app', message='Add new feature')",
    "git_repository_operations(operation='branch', repo_name='my-app', branch_name='feature/new-ui')",
    "git_repository_operations(operation='log', repo_name='my-app', limit=5)"
]

# Base path
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"

async def git_repository_operations(**kwargs) -> Dict[str, Any]:
    """
    Perform Git operations on local repositories.
    """
    operation = kwargs.get('operation')
    repo_name = kwargs.get('repo_name')
    
    if not operation:
        return {
            "status": "error",
            "message": "operation parameter is required"
        }
    
    if not repo_name:
        return {
            "status": "error",
            "message": "repo_name parameter is required"
        }
    
    # Check if repository exists
    repo_path = REPOSITORIES_PATH / repo_name
    if not repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    # Check if it's a git repository
    if not (repo_path / '.git').exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' is not a Git repository",
            "suggestion": "Initialize with: git init"
        }
    
    # Route to appropriate operation
    operations = {
        'status': git_status,
        'add': git_add,
        'commit': git_commit,
        'push': git_push,
        'pull': git_pull,
        'branch': git_branch,
        'checkout': git_checkout,
        'merge': git_merge,
        'log': git_log,
        'diff': git_diff,
        'stash': git_stash,
        'remote': git_remote
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        kwargs['repo_path'] = repo_path
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing git {operation}: {str(e)}"
        }

async def git_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get repository status."""
    repo_path = params['repo_path']
    
    try:
        # Get current branch
        branch_result = subprocess.run(
            ['git', 'branch', '--show-current'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        current_branch = branch_result.stdout.strip()
        
        # Get status
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        # Parse status
        changes = {
            'staged': [],
            'modified': [],
            'untracked': []
        }
        
        for line in status_result.stdout.splitlines():
            if line:
                status_code = line[:2]
                file_path = line[3:]
                
                if status_code[0] in ['A', 'M', 'D', 'R']:
                    changes['staged'].append({
                        'status': status_code[0],
                        'file': file_path
                    })
                elif status_code[1] == 'M':
                    changes['modified'].append(file_path)
                elif status_code == '??':
                    changes['untracked'].append(file_path)
        
        # Check if ahead/behind
        upstream_result = subprocess.run(
            ['git', 'rev-list', '--left-right', '--count', 'HEAD...@{u}'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        ahead = 0
        behind = 0
        if upstream_result.returncode == 0 and upstream_result.stdout:
            parts = upstream_result.stdout.strip().split()
            if len(parts) == 2:
                ahead = int(parts[0])
                behind = int(parts[1])
        
        return {
            "status": "success",
            "repository": params.get('repo_name'),
            "current_branch": current_branch,
            "changes": changes,
            "clean": not any(changes.values()),
            "ahead": ahead,
            "behind": behind,
            "summary": _get_status_summary(changes, ahead, behind)
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git command failed: {e.stderr}"
        }

async def git_add(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add files to staging area."""
    repo_path = params['repo_path']
    files = params.get('files', ['.'])
    
    if not files:
        files = ['.']
    
    try:
        result = subprocess.run(
            ['git', 'add'] + files,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed to add files: {result.stderr}"
            }
        
        # Get updated status
        status = await git_status(params)
        
        return {
            "status": "success",
            "message": "Files added to staging area",
            "files_added": files,
            "staged_files": len(status.get('changes', {}).get('staged', []))
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git add failed: {e.stderr}"
        }

async def git_commit(params: Dict[str, Any]) -> Dict[str, Any]:
    """Commit changes."""
    repo_path = params['repo_path']
    message = params.get('message')
    
    if not message:
        return {
            "status": "error",
            "message": "Commit message is required"
        }
    
    try:
        # Check if there are changes to commit
        status = await git_status(params)
        if not status.get('changes', {}).get('staged'):
            return {
                "status": "error",
                "message": "No changes staged for commit",
                "suggestion": "Use git add to stage changes first"
            }
        
        # Commit
        result = subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Commit failed: {result.stderr}"
            }
        
        # Extract commit hash
        commit_hash = None
        for line in result.stdout.splitlines():
            if line.strip().startswith('['):
                parts = line.split()
                if len(parts) > 1:
                    commit_hash = parts[1].rstrip(']')
                    break
        
        return {
            "status": "success",
            "message": "Changes committed successfully",
            "commit_message": message,
            "commit_hash": commit_hash,
            "output": result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git commit failed: {e.stderr}"
        }

async def git_push(params: Dict[str, Any]) -> Dict[str, Any]:
    """Push commits to remote."""
    repo_path = params['repo_path']
    remote = params.get('remote', 'origin')
    branch = params.get('branch_name')
    
    try:
        # Get current branch if not specified
        if not branch:
            branch_result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            branch = branch_result.stdout.strip()
        
        # Push
        result = subprocess.run(
            ['git', 'push', remote, branch],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Check if remote exists
            if 'fatal: \'origin\' does not appear to be a git repository' in result.stderr:
                return {
                    "status": "error",
                    "message": "No remote repository configured",
                    "suggestion": "Add remote with: git remote add origin <url>"
                }
            return {
                "status": "error",
                "message": f"Push failed: {result.stderr}"
            }
        
        return {
            "status": "success",
            "message": f"Pushed to {remote}/{branch}",
            "remote": remote,
            "branch": branch,
            "output": result.stderr  # Git push outputs to stderr
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git push failed: {e.stderr}"
        }

async def git_pull(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pull changes from remote."""
    repo_path = params['repo_path']
    remote = params.get('remote', 'origin')
    
    try:
        result = subprocess.run(
            ['git', 'pull', remote],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Pull failed: {result.stderr}"
            }
        
        # Parse output for summary
        updated_files = 0
        insertions = 0
        deletions = 0
        
        for line in result.stdout.splitlines():
            if 'files changed' in line:
                parts = line.split(',')
                for part in parts:
                    if 'files changed' in part:
                        updated_files = int(part.split()[0])
                    elif 'insertions' in part:
                        insertions = int(part.split()[0])
                    elif 'deletions' in part:
                        deletions = int(part.split()[0])
        
        return {
            "status": "success",
            "message": "Pull completed",
            "remote": remote,
            "files_updated": updated_files,
            "insertions": insertions,
            "deletions": deletions,
            "output": result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git pull failed: {e.stderr}"
        }

async def git_branch(params: Dict[str, Any]) -> Dict[str, Any]:
    """Manage branches."""
    repo_path = params['repo_path']
    branch_name = params.get('branch_name')
    
    try:
        if branch_name:
            # Create new branch
            result = subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if 'already exists' in result.stderr:
                    return {
                        "status": "error",
                        "message": f"Branch '{branch_name}' already exists"
                    }
                return {
                    "status": "error",
                    "message": f"Failed to create branch: {result.stderr}"
                }
            
            return {
                "status": "success",
                "message": f"Created and switched to branch '{branch_name}'",
                "branch": branch_name
            }
        else:
            # List branches
            result = subprocess.run(
                ['git', 'branch', '-a'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            branches = []
            current_branch = None
            
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith('*'):
                    current_branch = line[2:]
                    branches.append({
                        "name": current_branch,
                        "current": True,
                        "type": "local"
                    })
                elif line.startswith('remotes/'):
                    branches.append({
                        "name": line.replace('remotes/', ''),
                        "current": False,
                        "type": "remote"
                    })
                elif line:
                    branches.append({
                        "name": line,
                        "current": False,
                        "type": "local"
                    })
            
            return {
                "status": "success",
                "current_branch": current_branch,
                "branches": branches,
                "total_branches": len(branches)
            }
            
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git branch operation failed: {e.stderr}"
        }

async def git_checkout(params: Dict[str, Any]) -> Dict[str, Any]:
    """Switch branches."""
    repo_path = params['repo_path']
    branch_name = params.get('branch_name')
    
    if not branch_name:
        return {
            "status": "error",
            "message": "branch_name is required for checkout"
        }
    
    try:
        result = subprocess.run(
            ['git', 'checkout', branch_name],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Checkout failed: {result.stderr}"
            }
        
        return {
            "status": "success",
            "message": f"Switched to branch '{branch_name}'",
            "branch": branch_name
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git checkout failed: {e.stderr}"
        }

async def git_merge(params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge branches."""
    repo_path = params['repo_path']
    branch_name = params.get('branch_name')
    
    if not branch_name:
        return {
            "status": "error",
            "message": "branch_name is required for merge"
        }
    
    try:
        # Get current branch
        current_result = subprocess.run(
            ['git', 'branch', '--show-current'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        current_branch = current_result.stdout.strip()
        
        # Merge
        result = subprocess.run(
            ['git', 'merge', branch_name],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            if 'CONFLICT' in result.stdout:
                return {
                    "status": "error",
                    "message": "Merge conflicts detected",
                    "conflicts": True,
                    "output": result.stdout
                }
            return {
                "status": "error",
                "message": f"Merge failed: {result.stderr}"
            }
        
        return {
            "status": "success",
            "message": f"Merged '{branch_name}' into '{current_branch}'",
            "from_branch": branch_name,
            "to_branch": current_branch,
            "output": result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git merge failed: {e.stderr}"
        }

async def git_log(params: Dict[str, Any]) -> Dict[str, Any]:
    """Show commit history."""
    repo_path = params['repo_path']
    limit = params.get('limit', 10)
    
    try:
        result = subprocess.run(
            ['git', 'log', f'-{limit}', '--oneline', '--decorate', '--graph'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed to get log: {result.stderr}"
            }
        
        # Parse commits
        commits = []
        for line in result.stdout.splitlines():
            if line:
                # Remove graph characters
                clean_line = line.lstrip('*| \\/')
                parts = clean_line.strip().split(' ', 1)
                if len(parts) == 2:
                    commits.append({
                        "hash": parts[0],
                        "message": parts[1]
                    })
        
        # Get detailed info for latest commit
        if commits:
            detail_result = subprocess.run(
                ['git', 'show', '--stat', commits[0]['hash']],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            latest_details = detail_result.stdout if detail_result.returncode == 0 else None
        else:
            latest_details = None
        
        return {
            "status": "success",
            "commits": commits,
            "total_shown": len(commits),
            "latest_commit_details": latest_details
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git log failed: {e.stderr}"
        }

async def git_diff(params: Dict[str, Any]) -> Dict[str, Any]:
    """Show differences."""
    repo_path = params['repo_path']
    
    try:
        # Get unstaged changes
        unstaged_result = subprocess.run(
            ['git', 'diff'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        # Get staged changes
        staged_result = subprocess.run(
            ['git', 'diff', '--cached'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        # Parse diff stats
        stats_result = subprocess.run(
            ['git', 'diff', '--stat'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "status": "success",
            "unstaged_changes": unstaged_result.stdout if unstaged_result.stdout else "No unstaged changes",
            "staged_changes": staged_result.stdout if staged_result.stdout else "No staged changes",
            "summary": stats_result.stdout if stats_result.stdout else "No changes"
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git diff failed: {e.stderr}"
        }

async def git_stash(params: Dict[str, Any]) -> Dict[str, Any]:
    """Manage stash."""
    repo_path = params['repo_path']
    message = params.get('message', 'WIP')
    
    try:
        # Stash changes
        result = subprocess.run(
            ['git', 'stash', 'push', '-m', message],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if 'No local changes to save' in result.stdout:
            return {
                "status": "info",
                "message": "No changes to stash"
            }
        
        # List stashes
        list_result = subprocess.run(
            ['git', 'stash', 'list'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        stashes = []
        for line in list_result.stdout.splitlines():
            if line:
                stashes.append(line)
        
        return {
            "status": "success",
            "message": "Changes stashed",
            "stash_message": message,
            "total_stashes": len(stashes),
            "stashes": stashes[:5]  # Show latest 5
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git stash failed: {e.stderr}"
        }

async def git_remote(params: Dict[str, Any]) -> Dict[str, Any]:
    """Manage remotes."""
    repo_path = params['repo_path']
    
    try:
        result = subprocess.run(
            ['git', 'remote', '-v'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        remotes = {}
        for line in result.stdout.splitlines():
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    url = parts[1]
                    type_ = parts[2].strip('()') if len(parts) > 2 else 'unknown'
                    
                    if name not in remotes:
                        remotes[name] = {}
                    remotes[name][type_] = url
        
        return {
            "status": "success",
            "remotes": remotes,
            "total_remotes": len(remotes)
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Git remote failed: {e.stderr}"
        }

def _get_status_summary(changes: Dict[str, List], ahead: int, behind: int) -> str:
    """Generate a human-readable status summary."""
    parts = []
    
    if changes['staged']:
        parts.append(f"{len(changes['staged'])} staged")
    if changes['modified']:
        parts.append(f"{len(changes['modified'])} modified")
    if changes['untracked']:
        parts.append(f"{len(changes['untracked'])} untracked")
    
    if ahead > 0:
        parts.append(f"{ahead} ahead")
    if behind > 0:
        parts.append(f"{behind} behind")
    
    return ", ".join(parts) if parts else "Working tree clean"

# Test function
async def main():
    # Test git status
    result = await git_repository_operations(
        operation='status',
        repo_name='brand-guardian-ai'
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())