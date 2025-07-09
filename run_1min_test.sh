#!/bin/bash
echo "Starting real improvements for 60 seconds..."
export $(cat .env.local | grep -v '^#' | xargs)

# Start the improvement process
python3 run_real_improvements.py --interval 10 &
PID=$!

echo "Started with PID: $PID"

# Let it run for 60 seconds
sleep 60

# Kill the process
kill $PID 2>/dev/null

echo "Stopped after 60 seconds"

# Check what was created
echo ""
echo "=== IMPROVEMENTS CREATED ==="
echo "Checking for new tools..."
ls -la scripts/custom_tools/*.py | tail -5

echo ""
echo "Checking system state..."
python3 -c "
import json
with open('system_state.json', 'r') as f:
    state = json.load(f)
print(f\"Total cycles: {state.get('stats', {}).get('total_cycles', 0)}\")
print(f\"Successful actions: {state.get('stats', {}).get('successful_actions', 0)}\")
"

echo ""
echo "Checking for errors in log..."
tail -20 improvement_output.log | grep -E "(ERROR|Error|❌)" || echo "No errors found in last 20 lines"