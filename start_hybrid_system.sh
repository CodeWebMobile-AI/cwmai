#!/bin/bash
# Start the Hybrid Continuous AI System

echo "🚀 Starting Enhanced Hybrid Continuous AI System..."
echo "This combines:"
echo "  - 24/7 Continuous Operation"
echo "  - Dynamic God Mode Intelligence"
echo "  - Learning & Evolution"
echo "  - SQLite/In-Memory State Management (Redis-free)"
echo ""
echo "✨ NEW ENHANCEMENTS:"
echo "  - Research Evolution Engine with Persistent Knowledge"
echo "  - Pattern Learning from System Behavior"
echo "  - Feedback Learning and Performance Tracking"
echo "  - Swarm Intelligence with 5 Specialized AI Agents"
echo "  - Multi-Objective Task Optimization"
echo ""

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables (filter out comments and empty lines)
set -a
source <(grep -v '^#' "$DIR/.env.local" | grep -v '^$')
set +a
export PYTHONPATH="$DIR"
export NLTK_DATA="$HOME/nltk_data"

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

# Create a Python wrapper that loads environment properly
cat > "$DIR/.run_with_env_temp.py" << EOF
#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('$DIR/.env.local')

# Verify at least one API key is loaded
api_keys_loaded = any([
    os.getenv('ANTHROPIC_API_KEY'),
    os.getenv('OPENAI_API_KEY'), 
    os.getenv('GOOGLE_API_KEY'),
    os.getenv('GEMINI_API_KEY'),
    os.getenv('DEEPSEEK_API_KEY')
])

if not api_keys_loaded:
    print("❌ ERROR: No API keys found in .env.local")
    print("Please check your .env.local file contains valid API keys")
    sys.exit(1)

# Run the main script
sys.argv = ['run_hybrid_continuous.py', '--workers', '$WORKERS', '--log-level', '$LOG_LEVEL']
exec(open('$DIR/run_hybrid_continuous.py').read())
EOF

# Run with ARM64 architecture
exec arch -arm64 /usr/bin/python3 "$DIR/.run_with_env_temp.py"