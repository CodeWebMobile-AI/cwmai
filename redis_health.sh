#!/bin/bash
# Redis health monitoring wrapper

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

case "$1" in
    "check"|"quick"|"")
        echo "🔍 Running quick Redis health check..."
        python3 debug_redis_connections.py
        ;;
    
    "monitor"|"detailed"|"full")
        echo "📊 Starting detailed Redis monitoring (5 minutes)..."
        echo "Press Ctrl+C to stop early"
        python3 monitor_redis_health.py
        ;;
    
    "cleanup"|"clean")
        echo "🧹 Cleaning up idle Redis connections..."
        python3 cleanup_redis_now.py
        ;;
    
    "watch"|"live")
        echo "👁️  Live Redis monitoring (press Ctrl+C to stop)"
        while true; do
            clear
            echo "=== Redis Live Monitor - $(date) ==="
            echo ""
            # Get connection count
            CONNECTIONS=$(redis-cli client list | wc -l)
            echo "Total connections: $CONNECTIONS"
            
            # Get idle connections
            IDLE=$(redis-cli client list | awk -F'idle=' '{print $2}' | awk '{print $1}' | awk '$1 > 60' | wc -l)
            echo "Idle connections (>60s): $IDLE"
            
            # Memory usage
            MEMORY=$(redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
            echo "Memory usage: $MEMORY"
            
            # Quick ping test
            echo -n "Ping test: "
            if redis-cli ping >/dev/null 2>&1; then
                echo "✅ OK"
            else
                echo "❌ Failed"
            fi
            
            echo ""
            echo "Recent connections:"
            redis-cli client list | tail -5 | awk -F' ' '{print $1" "$3" "$5}'
            
            sleep 5
        done
        ;;
    
    "schedule"|"scheduler")
        echo "⏰ Setting up automatic Redis cleanup..."
        ./setup_redis_scheduler.sh
        ;;
    
    "logs")
        echo "📜 Showing Redis cleanup logs..."
        if [ -f "/tmp/cwmai-redis-cleanup.log" ]; then
            tail -n 50 /tmp/cwmai-redis-cleanup.log
        else
            echo "No cleanup logs found yet"
        fi
        ;;
    
    "help"|"-h"|"--help")
        echo "Redis Health Monitor - Usage:"
        echo ""
        echo "  ./redis_health.sh [command]"
        echo ""
        echo "Commands:"
        echo "  check    - Quick health check (default)"
        echo "  monitor  - Detailed 5-minute monitoring"
        echo "  cleanup  - Clean up idle connections now"
        echo "  watch    - Live monitoring (updates every 5s)"
        echo "  schedule - Set up automatic cleanup"
        echo "  logs     - View cleanup logs"
        echo "  help     - Show this help"
        echo ""
        echo "Examples:"
        echo "  ./redis_health.sh         # Quick check"
        echo "  ./redis_health.sh monitor # Full monitoring"
        echo "  ./redis_health.sh watch   # Live view"
        ;;
    
    *)
        echo "Unknown command: $1"
        echo "Run './redis_health.sh help' for usage"
        exit 1
        ;;
esac