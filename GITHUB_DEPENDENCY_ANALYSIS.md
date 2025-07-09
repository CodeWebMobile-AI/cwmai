# GitHub API Dependency Analysis for CWMAI

## Overview
This analysis identifies all remaining GitHub API calls, authentication usage, and dependencies in the CWMAI project.

## Key Findings

### 1. Environment Variables Still Referenced
- `GITHUB_TOKEN` - Used in 89+ files
- `CLAUDE_PAT` - Used as fallback GitHub token
- `GITHUB_REPOSITORY` - Set to 'CodeWebMobile-AI/cwmai'

### 2. PyGithub Dependency
- **Still in requirements.txt**: `PyGithub==1.59.0`
- **Imported in 21+ files** including core components

### 3. Storage Mode Configuration
While `storage_factory.py` is hardcoded to use `LOCAL` mode:
```python
mode = StorageMode.LOCAL  # ALWAYS use local mode
```

The codebase still contains extensive GitHub integration code that could be activated if this hardcoding is removed.

### 4. Major Components Using GitHub API

#### State Manager (`scripts/state_manager.py`)
- Lines 15, 60-62, 81-100: GitHub authentication and repository discovery
- `discover_organization_repositories()` - Discovers GitHub repos
- Uses `Github` client to interact with GitHub API

#### Task Manager (`scripts/task_manager.py`)
- Line 17: `from github import Github`
- Lines 86-100: GitHub token initialization
- Contains methods for creating GitHub issues

#### Project Creator (`scripts/project_creator.py`)
- Line 13: `from github import Github, GithubException`
- Lines 36-37: GitHub client initialization
- Contains methods for forking repositories and creating GitHub projects

#### Repository Analyzer (`scripts/repository_analyzer.py`)
- Analyzes GitHub repositories
- Uses GitHub API for repository metadata

### 5. GitHub API Calls Found
Common GitHub API methods still in use:
- `.get_organization()` - Get GitHub organization
- `.get_repo()` - Get repository details
- `.create_issue()` - Create GitHub issues
- `.get_issues()` - Retrieve issues
- `.create_fork()` - Fork repositories

### 6. Local Mode Implementation
Despite having local mode infrastructure:
- `local_issue_creator.py` - Local issue storage
- `local_github_simulator.py` - Simulates GitHub API locally
- `hybrid_state_manager.py` - SQLite-based state management

The system still maintains GitHub dependencies that could be activated.

## Recommendations

### To Fully Remove GitHub Dependencies:

1. **Remove PyGithub from requirements.txt**
   ```bash
   # Remove line: PyGithub==1.59.0
   ```

2. **Remove GitHub imports from core files**:
   - `scripts/state_manager.py`
   - `scripts/task_manager.py`
   - `scripts/project_creator.py`
   - `scripts/repository_analyzer.py`
   - And 17+ other files

3. **Remove GitHub environment variables from .env files**:
   - `GITHUB_TOKEN`
   - `CLAUDE_PAT`
   - `GITHUB_REPOSITORY`

4. **Update storage factory default**:
   Currently line 127 in `storage_factory.py`:
   ```python
   self.storage_mode = storage_mode or os.getenv('STORAGE_MODE', StorageMode.GITHUB)
   ```
   Should be:
   ```python
   self.storage_mode = storage_mode or os.getenv('STORAGE_MODE', StorageMode.LOCAL)
   ```

5. **Remove GitHub-specific methods** from:
   - StateManager: `discover_organization_repositories()`
   - TaskManager: GitHub issue creation methods
   - ProjectCreator: Fork-based project creation

6. **Update continuous orchestrator**:
   Lines 88-93 in `continuous_orchestrator_no_redis.py` still check for GitHub mode

## Current State
The system is in a **hybrid state** where:
- Storage factories are hardcoded to LOCAL mode
- GitHub code is still present but bypassed
- Environment still expects GitHub tokens
- Dependencies still include PyGithub

This creates potential for:
- Confusion about which mode is active
- Accidental GitHub API calls if hardcoding is changed
- Unnecessary dependencies and complexity