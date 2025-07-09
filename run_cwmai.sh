#!/bin/bash
# CWMAI launcher with proper environment

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project directory
cd "$DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Load environment
if [ -f ".env.local" ]; then
    export $(cat .env.local | grep -v '^#' | xargs)
elif [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi
export NLTK_DATA="$HOME/nltk_data"

# Clean up idle Redis connections on startup
echo "🧹 Cleaning up idle Redis connections..."
python cleanup_redis_now.py >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Redis connections cleaned"
else
    echo "⚠️  Redis cleanup failed (Redis might not be running)"
fi

# Run cwmai with the venv Python
exec python cwmai "$@"
