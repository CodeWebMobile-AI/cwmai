# Repository Discovery Prevention Guide

## Problem Summary
Repositories can exist in the filesystem but not be discovered by the system if they're not registered in `repositories.json`. This causes the AI system to miss repositories and not generate work for them.

## Root Cause
The system relies on `/local_repos/state/repositories.json` as the single source of truth for repository discovery. When repositories are created outside the normal workflow or when the registration fails, they become "invisible" to the system.

## Prevention Strategies

### 1. Automatic Registration on Creation
**Implemented in:** `project_creator.py`
```python
# After creating a repository, automatically register it
from scripts.auto_register_repositories import RepositoryAutoRegistrar
registrar = RepositoryAutoRegistrar()
registrar.register_new_repositories()
```

### 2. Periodic Repository Reconciliation
**Script:** `scripts/auto_register_repositories.py`

Run periodically to catch any unregistered repositories:
```bash
# One-time check
python3 scripts/auto_register_repositories.py

# Continuous monitoring (checks every 5 minutes)
python3 scripts/auto_register_repositories.py --watch

# Dry run to see what would be registered
python3 scripts/auto_register_repositories.py --dry-run
```

### 3. System Startup Verification
Add to system startup scripts:
```bash
# In startup scripts or cwmai wrapper
python3 scripts/auto_register_repositories.py
```

### 4. Repository Creation Hook
The `LocalGitHubSimulator.create_repository()` method should automatically update `repositories.json`. This is the primary registration mechanism.

### 5. Monitoring and Alerts
Monitor for discrepancies between filesystem and registered repositories:
```python
# Example monitoring code
registrar = RepositoryAutoRegistrar()
found = registrar.scan_for_repositories()
registered = registrar.load_registered_repositories()

# Alert if mismatch
filesystem_count = sum(len(repos) for repos in found.values())
registered_count = sum(len(repos) for repos in registered.values())

if filesystem_count != registered_count:
    print(f"WARNING: Repository count mismatch! Filesystem: {filesystem_count}, Registered: {registered_count}")
```

## Best Practices

### 1. Always Use Official APIs
- Use `ProjectCreator` for creating new projects
- Use `LocalGitHubSimulator` for repository operations
- Avoid manual directory creation without registration

### 2. Regular Maintenance
- Run reconciliation scripts daily or on system startup
- Monitor logs for repository creation failures
- Check for orphaned directories periodically

### 3. Error Handling
- If repository creation fails, ensure cleanup removes partial directories
- Log all repository operations for audit trail
- Implement retry logic for registration failures

### 4. Integration Points
Ensure registration happens at these critical points:
- After successful repository creation
- After repository restoration from backup
- After bulk imports or migrations
- During system recovery procedures

## Manual Recovery

If repositories are missing from discovery:
```bash
# Quick fix - register all unregistered repositories
python3 reconcile_repository_discovery.py

# Or use the auto-registrar
python3 scripts/auto_register_repositories.py
```

## Testing

To test the prevention mechanisms:
```bash
# Create a test repository manually
mkdir -p local_repos/repositories/CodeWebMobile-AI/test-repo
cd local_repos/repositories/CodeWebMobile-AI/test-repo
git init
echo "# Test" > README.md

# Verify it's not discovered
python3 -c "from scripts.local_github_simulator import LocalGitHubSimulator; sim = LocalGitHubSimulator(); print(len(sim.list_repositories()))"

# Run auto-registration
python3 scripts/auto_register_repositories.py

# Verify it's now discovered
python3 -c "from scripts.local_github_simulator import LocalGitHubSimulator; sim = LocalGitHubSimulator(); print(len(sim.list_repositories()))"
```

## Configuration

Environment variables for auto-registration:
```bash
# How often to check for new repositories (in minutes)
export REPO_CHECK_INTERVAL=5

# Enable automatic registration on system startup
export AUTO_REGISTER_REPOS=true

# Log level for registration operations
export REGISTRATION_LOG_LEVEL=INFO
```

## Future Improvements

1. **File System Watcher**: Implement real-time detection of new repository directories
2. **Health Check Endpoint**: Add API endpoint to verify repository registration status
3. **Metrics Collection**: Track registration failures and delays
4. **Automatic Cleanup**: Remove registered entries for deleted repositories
5. **Distributed Locking**: Prevent concurrent registration conflicts in multi-worker setups