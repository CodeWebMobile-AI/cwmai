#!/bin/bash
# Optimize AI API calls and system performance

echo "🚀 Optimizing CWMAI for reduced API calls and better performance..."

# Export optimized cache durations
export LIFECYCLE_CACHE_MINUTES=1440    # 24 hours (was 15 minutes)
export PORTFOLIO_CACHE_MINUTES=360     # 6 hours (was 5 minutes)
export WORK_GEN_CACHE_HOURS=12        # 12 hours (new cache)

# Set discovery intervals (requires code change)
export DISCOVERY_INTERVAL_SECONDS=300  # 5 minutes (was 30 seconds)

echo "✅ Environment optimizations set:"
echo "  - Lifecycle cache: 24 hours"
echo "  - Portfolio cache: 6 hours"
echo "  - Work generation cache: 12 hours"
echo "  - Discovery interval: 5 minutes (pending code change)"

# Load environment and run
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai
arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 10
