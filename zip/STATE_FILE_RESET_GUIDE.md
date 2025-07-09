# State File Reset Guide for Claude

## Important: Correct State File Structures

When resetting state files in the zip/local_repos/state directory, use these exact structures:

### repositories.json
```json
{
  "repositories": [],
  "total_count": 0,
  "last_updated": null,
  "metadata": {
    "version": "1.0",
    "storage_mode": "local"
  }
}
```
**NOT** just `[]`

### system_state.json
```json
{
  "charter": {
    "primary_goal": "innovation",
    "secondary_goal": "community_engagement", 
    "constraints": ["maintain_quality", "ensure_security"]
  },
  "projects": {},
  "repositories": {},  // Must be object {}, NOT array []
  "system_performance": {
    "total_cycles": 0,
    "successful_actions": 0,
    "failed_actions": 0,
    "learning_metrics": {
      "decision_accuracy": 0.0,
      "resource_efficiency": 0.0,
      "goal_achievement": 0.0
    }
  },
  "task_queue": [],
  "external_context": {
    "last_updated": null,
    "market_trends": [],
    "security_alerts": [],
    "technology_updates": []
  },
  "completed_tasks": [],
  "active_tasks": [],
  "metrics": {
    "total_repositories": 0,
    "total_tasks_completed": 0,
    "total_tasks_active": 0
  },
  "version": "1.0.0",
  "last_updated": null
}
```

### continuous_orchestrator_state.json
```json
{
  "metrics": {
    "total_work_completed": 0,
    "total_errors": 0,
    "total_work_created": 0,
    "work_per_hour": 0,
    "average_completion_time": 0.0,
    "uptime_hours": 0
  },
  "enhanced_metrics": {
    "smart_tasks_selected": 0,
    "dependency_chains_executed": 0,
    "critical_paths_analyzed": 0,
    "context_updates": 0,
    "strategy_switches": 0,
    "task_assignments": 0
  },
  "work_queue": [],
  "completed_work": [],
  "workers": {},
  "last_updated": null
}
```

### active_tasks.json
```json
[]
```

### completed_tasks.json
```json
[]
```

### task_hashes.json
```json
{}
```

### task_history.json
```json
[]
```

## Common Mistakes to Avoid

1. **repositories field in system_state.json**: Must be `{}` (object), not `[]` (array)
2. **repositories.json**: Must be a full object structure with metadata, not just `[]`
3. **Empty objects vs arrays**: Be careful about which fields should be `{}` vs `[]`

## Reset Command

To properly reset all state files:

```bash
cd /Users/fortressarigo/PhpstormProjects/cwmai/zip/local_repos/state

# Reset repositories.json with proper structure
echo '{
  "repositories": [],
  "total_count": 0,
  "last_updated": null,
  "metadata": {
    "version": "1.0",
    "storage_mode": "local"
  }
}' > repositories.json

# Reset simple array files
echo '[]' > active_tasks.json
echo '[]' > completed_tasks.json
echo '[]' > task_history.json
echo '{}' > task_hashes.json

# For system_state.json and continuous_orchestrator_state.json, 
# use the full structures shown above
```

## Why This Matters

The code expects specific data structures. When you reset to just `[]` or `{}`, it causes TypeErrors like:
- `'list' object is not a mapping`
- `list indices must be integers or slices, not str`

Always check what structure the code expects before resetting!