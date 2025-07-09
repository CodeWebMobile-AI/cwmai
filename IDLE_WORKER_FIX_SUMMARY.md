# Idle Worker Fix Summary

## Problem Identified
The CWMAI system had workers remaining 81% idle despite having:
- Market research capabilities
- Portfolio analysis features  
- Project creation abilities
- Non-repository task support

## Root Causes
1. Research tasks were not being created for idle workers
2. The continuous orchestrator's `_handle_research_task` just redirected to knowledge-driven evolution
3. Discovery interval was hardcoded instead of using environment variable
4. Regular workers couldn't execute non-repository tasks

## Fixes Applied

### 1. Updated IntelligentWorkFinder (scripts/intelligent_work_finder.py)
- Modified `_discover_research_opportunities` to call `_create_research_work_items`
- Ensured research tasks are always added when there's capacity
- Research tasks include:
  - Market Research (MARKET_RESEARCH)
  - Portfolio Analysis (PORTFOLIO_ANALYSIS)  
  - Project Creation (PROJECT_CREATION)

### 2. Enhanced Continuous Orchestrator (scripts/continuous_orchestrator_no_redis.py)
- Rewrote `_handle_research_task` to actually perform research:
  - Market research discovers opportunities and queues project creation
  - Portfolio analysis identifies gaps and creates projects
  - Direct project creation when requested
- Updated `_execute_work_item` to properly route research tasks
- Added support for non-repository work items

### 3. Fixed Discovery Interval
- Updated `scripts/hybrid_orchestrator_enhanced.py` to use `DISCOVERY_INTERVAL_SECONDS` env var
- Allows faster discovery during testing

### 4. Work Item Routing
Research tasks are identified by:
- Task type in ['MARKET_RESEARCH', 'PORTFOLIO_ANALYSIS', 'PROJECT_CREATION']
- Metadata containing `no_repository_needed: true`
- Metadata containing `task_type: 'research'`

## Expected Behavior
With these fixes:
1. When workers are idle, research tasks are automatically created
2. Market research generates project opportunities
3. These opportunities are converted to NEW_PROJECT work items
4. Regular workers can execute research tasks
5. The system maintains high utilization

## Testing
Run the 6-minute test with:
```bash
export $(cat .env.local | grep -v '^#' | xargs) && \
PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai \
arch -arm64 /usr/bin/python3 test_6min_idle_fix.py
```

Success criteria:
- Average idle < 50%
- At least 10 tasks completed
- Research tasks created and queued

## Environment Variables
For optimal performance:
- `DISCOVERY_INTERVAL_SECONDS=60` (faster discovery)
- `LIFECYCLE_CACHE_MINUTES=60` (reduce API calls)
- `PORTFOLIO_CACHE_MINUTES=60` (reduce API calls)
- `WORK_GEN_CACHE_HOURS=1` (reduce API calls)