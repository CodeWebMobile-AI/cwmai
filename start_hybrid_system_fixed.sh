#!/bin/bash
# Start the Hybrid Continuous AI System (Fixed Version)

echo "🚀 Starting Hybrid Continuous AI System (Fixed)..."
echo "This combines:"
echo "  - 24/7 Continuous Operation"
echo "  - Dynamic God Mode Intelligence"
echo "  - Learning & Evolution"
echo "  - SQLite/In-Memory State Management (Redis-free)"
echo ""

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check SQLite database
if [ -f "$DIR/cwmai_state.db" ]; then
    echo "✅ SQLite state database found"
else
    echo "📝 Creating new SQLite state database..."
fi

# Default settings
WORKERS=5
LOG_LEVEL="INFO"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --workers N      Number of parallel workers (default: 5)"
            echo "  --log-level L    Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
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
echo "  Workers: $WORKERS"
echo "  Log level: $LOG_LEVEL"
echo "  State DB: cwmai_state.db"
echo "  Work Queue: In-memory with persistence"
echo ""

# Create a Python wrapper that properly loads environment
cat > "$DIR/run_with_env.py" << 'EOF'
#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main script
import run_hybrid_continuous
run_hybrid_continuous.main()
EOF

# Set PYTHONPATH and run with environment loaded
export PYTHONPATH="$DIR"
export NLTK_DATA="$HOME/nltk_data"

# Run with ARM64 architecture
exec arch -arm64 /usr/bin/python3 "$DIR/run_with_env.py" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL"