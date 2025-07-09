# AI API Call Optimization Guide

## Current Unnecessary AI Calls

### 1. **Work Discovery Every 30 Seconds** ❌
- **Problem**: System discovers work every 30 seconds regardless of changes
- **Impact**: 120 discovery cycles per hour × AI calls per cycle
- **Fix**: Implement adaptive discovery frequency
```bash
# Add to .env.local
DISCOVERY_INTERVAL_ACTIVE=300    # 5 minutes for active repos
DISCOVERY_INTERVAL_INACTIVE=3600 # 1 hour for inactive repos
```

### 2. **DynamicWorkGenerator - No Caching** ❌ (FIXED)
- **Problem**: Made AI calls for every repository every discovery cycle
- **Impact**: 4 repos × 120 cycles/hour = 480 AI calls/hour just for work generation
- **Fix**: Added 6-hour cache (configurable via `WORK_GEN_CACHE_HOURS`)

### 3. **Short Cache Durations** ⚠️
Current defaults are too short:
- ProjectLifecycleAnalyzer: 15 minutes (now 24 hours with our fix)
- PortfolioIntelligence: 5 minutes
- MarketResearchEngine: 24 hours (good)

**Recommended Settings**:
```bash
# Add to .env.local
LIFECYCLE_CACHE_MINUTES=1440    # 24 hours (1440 minutes)
PORTFOLIO_CACHE_MINUTES=360     # 6 hours
WORK_GEN_CACHE_HOURS=12        # 12 hours
```

### 4. **Redundant Repository Analysis** ❌
- **Problem**: Analyzes all repositories even if unchanged
- **Fix**: Track last commit/issue update and skip if unchanged
```python
# Pseudo-code for optimization
if repo.last_commit_date < last_analysis_date:
    return cached_analysis  # Skip AI call
```

### 5. **Portfolio Analysis on Every Discovery** ⚠️
- **Problem**: Deep portfolio analysis runs too frequently
- **Fix**: Only run when:
  - New repository added
  - Repository archived/deleted
  - Once per day maximum

## Estimated API Call Reduction

### Before Optimizations (4 repositories):
- Per discovery cycle: ~20 AI calls
- Per hour (120 cycles): ~2,400 AI calls
- Per day: ~57,600 AI calls 😱

### After Optimizations:
- Per discovery cycle: 2-5 AI calls (with caching)
- Per hour (12 cycles): ~60 AI calls
- Per day: ~1,440 AI calls
- **Reduction: 97.5%** ✅

## Priority Fixes

1. **Increase Discovery Interval** (Highest Impact)
   ```python
   # In intelligent_work_finder.py
   self.discovery_interval = 300  # Change from 30 to 300 seconds
   ```

2. **Enable All Caches**
   ```bash
   # Add to .env.local
   export LIFECYCLE_CACHE_MINUTES=1440
   export PORTFOLIO_CACHE_MINUTES=360
   export WORK_GEN_CACHE_HOURS=12
   ```

3. **Skip Unchanged Repositories**
   - Track repository state hash
   - Only analyze if state changed

4. **Batch Similar Operations**
   - Group similar repositories
   - Use one AI call for multiple similar repos

5. **Implement Smart Triggers**
   - Only analyze on webhook events
   - Use file watchers for local repos
   - Analyze on-demand instead of polling

## Quick Win Script

Create `optimize_ai_calls.sh`:
```bash
#!/bin/bash
# Optimize AI API calls

# Set optimal cache durations
export LIFECYCLE_CACHE_MINUTES=1440    # 24 hours
export PORTFOLIO_CACHE_MINUTES=360     # 6 hours  
export WORK_GEN_CACHE_HOURS=12        # 12 hours

# Set discovery intervals
export DISCOVERY_INTERVAL_SECONDS=300  # 5 minutes

# Run with optimizations
export $(cat .env.local | grep -v '^#' | xargs)
python3 run_hybrid_continuous.py --workers 10
```

## Monitoring API Usage

Add logging to track API calls:
```python
# In http_ai_client.py
self.api_call_count = 0
self.api_call_log = []

async def generate_response(...):
    self.api_call_count += 1
    self.api_call_log.append({
        'timestamp': datetime.now(),
        'provider': provider,
        'purpose': 'work_discovery',
        'cached': False
    })
```

## Long-term Solutions

1. **Event-Driven Architecture**
   - Replace polling with webhooks
   - React to changes instead of checking constantly

2. **Local Intelligence**
   - Build local models from AI responses
   - Reduce dependency on external AI

3. **Batch Processing**
   - Queue non-urgent analyses
   - Process in batches during off-peak

4. **Tiered Analysis**
   - Quick local analysis first
   - AI analysis only when needed

By implementing these optimizations, the system will be much more efficient and cost-effective while maintaining its intelligence.