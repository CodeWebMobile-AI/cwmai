# Local Repository Structure

## Overview
The LocalGitHubSimulator creates a file-based structure that mimics GitHub repositories. Here's how it's organized:

## Directory Structure

```
local_repos/
├── repositories/           # All local repositories
│   └── CodeWebMobile-AI/  # Organization directory
│       └── repo-name/     # Individual repository
│           ├── .git/      # Git repository (initialized)
│           ├── README.md  # Initial readme file
│           ├── metadata.json  # Repository metadata
│           └── issues/    # Issues directory
│               ├── issue-1.json
│               ├── issue-2.json
│               └── ...
├── state/                 # System state files
│   └── repositories.json  # Registry of all repositories
├── archives/              # Deleted repositories (by date)
│   └── 20250622/
│       └── archived-repo/
└── templates/             # Repository templates

```

## Key Files

### 1. Repository Metadata (`metadata.json`)
Located in each repository root, contains:
```json
{
  "name": "test-local-repo",
  "full_name": "CodeWebMobile-AI/test-local-repo",
  "description": "Test repository for local mode",
  "created_at": "2025-06-22T14:51:50.168395+00:00",
  "updated_at": "2025-06-22T14:51:50.168417+00:00",
  "next_issue_number": 1,
  "organization": "CodeWebMobile-AI"
}
```

### 2. Issue Files (`issues/issue-N.json`)
Each issue is stored as a separate JSON file:
```json
{
  "id": "uuid-here",
  "number": 1,
  "title": "Issue Title",
  "body": "Issue description",
  "labels": ["bug", "enhancement"],
  "state": "open",
  "created_at": "2025-06-22T14:51:50.168395+00:00",
  "updated_at": "2025-06-22T14:51:50.168417+00:00",
  "closed_at": null,
  "assignee": null,
  "html_url": "local://issue/1"
}
```

### 3. State Registry (`state/repositories.json`)
Central registry of all repositories:
```json
{
  "repositories": [
    {
      "name": "test-local-repo",
      "full_name": "CodeWebMobile-AI/test-local-repo",
      "description": "Test repository description",
      "path": "local_repos/repositories/CodeWebMobile-AI/test-local-repo",
      "created_at": "2025-06-22T14:51:50.168395+00:00",
      "updated_at": "2025-06-22T14:51:50.168417+00:00",
      "default_branch": "main",
      "private": false,
      "html_url": "file://local_repos/repositories/CodeWebMobile-AI/test-local-repo"
    }
  ],
  "total_count": 29,
  "last_updated": "2025-06-22T14:51:50.315448+00:00",
  "metadata": {
    "version": "1.0",
    "storage_mode": "local"
  }
}
```

## How It Works

1. **Repository Creation**:
   - Creates directory structure
   - Initializes git repository
   - Creates README.md
   - Creates metadata.json
   - Creates issues directory
   - Updates state registry

2. **Issue Creation**:
   - Reads metadata for next issue number
   - Creates issue JSON file
   - Updates metadata with incremented issue number
   - Issue stored in `issues/issue-N.json`

3. **Repository Operations**:
   - List: Reads from state registry
   - Get: Looks up in state registry and returns LocalRepository object
   - Delete: Moves to archives directory with timestamp

## Benefits

1. **No API Rate Limits**: All operations are local file system operations
2. **Full Git Repositories**: Each repo is a real git repository
3. **JSON-Based Storage**: Easy to inspect and modify
4. **Archival System**: Deleted repos are archived, not permanently deleted
5. **State Persistence**: All state is preserved between sessions

## Testing

The state file shows that the system is already tracking 29 repositories, but the physical directories don't exist. This suggests the repositories were created in a previous session and the directories were cleaned up. The LocalGitHubSimulator expects to find these directories at the paths specified in the state file.

To test with a fresh repository:
1. Create using `local_simulator.create_repository(name, description)`
2. Create issues using `local_simulator.create_issue(repo_name, title, body)`
3. List repos using `local_simulator.list_repositories()`
4. Get issues using `local_simulator.get_issues(repo_name)`