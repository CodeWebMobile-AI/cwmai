#!/bin/bash
# Temporarily disable Redis circuit breaker for debugging

echo "🔧 Disabling Redis circuit breaker..."
export REDIS_CIRCUIT_BREAKER_ENABLED=false
export REDIS_MAX_CONNECTIONS=500
export REDIS_CIRCUIT_BREAKER_THRESHOLD=100

echo "✅ Circuit breaker disabled with settings:"
echo "   REDIS_CIRCUIT_BREAKER_ENABLED=$REDIS_CIRCUIT_BREAKER_ENABLED"
echo "   REDIS_MAX_CONNECTIONS=$REDIS_MAX_CONNECTIONS"
echo "   REDIS_CIRCUIT_BREAKER_THRESHOLD=$REDIS_CIRCUIT_BREAKER_THRESHOLD"
echo ""
echo "To re-enable, unset these variables or set REDIS_CIRCUIT_BREAKER_ENABLED=true"