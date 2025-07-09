#!/bin/bash
# Start the CWMAI Continuous System with proper configuration

echo "🚀 Starting CWMAI Continuous System..."

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables
export $(cat "$DIR/.env.local" | grep -v '^#' | xargs)
export NLTK_DATA="$HOME/nltk_data"

# Check if Redis is running (required for MCP)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis is not running. Starting Redis..."
        redis-server --daemonize yes
        sleep 2
    fi
else
    echo "⚠️  Redis not installed. Some features may be limited."
fi

# Parse command line arguments
MODE="enhanced"
WORKERS=5
PARALLEL="true"
RESEARCH="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        --basic)
            MODE="basic"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --no-parallel)
            PARALLEL="false"
            shift
            ;;
        --no-research)
            RESEARCH="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --basic          Use basic orchestrator (no MCP)"
            echo "  --workers N      Number of parallel workers (default: 5)"
            echo "  --no-parallel    Disable parallel processing"
            echo "  --no-research    Disable research evolution engine"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Workers: $WORKERS"
echo "  Parallel: $PARALLEL"
echo "  Research: $RESEARCH"
echo ""

# Run with ARM64 architecture
exec arch -arm64 python "$DIR/run_continuous_ai.py" \
    --mode "$MODE" \
    --workers "$WORKERS" \
    $([ "$PARALLEL" = "false" ] && echo "--no-parallel") \
    $([ "$RESEARCH" = "false" ] && echo "--no-research")