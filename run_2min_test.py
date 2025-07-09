#!/usr/bin/env python3
import subprocess
import time
import signal
import sys
import os

# Use bash to properly export environment variables
cmd = [
    'bash', '-c',
    'export $(cat .env.local | grep -v "^#" | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 10'
]

print("Starting hybrid continuous system for 2 minutes...")
print("=" * 80)

start_time = time.time()
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

output_lines = []

try:
    while time.time() - start_time < 120:  # Run for 2 minutes
        line = process.stdout.readline()
        if line:
            print(line.rstrip())
            output_lines.append(line.rstrip())
        elif process.poll() is not None:
            # Process ended
            break
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

# Terminate the process
print("\n" + "=" * 80)
print("Stopping hybrid continuous system...")
process.terminate()
try:
    process.wait(timeout=5)
except subprocess.TimeoutExpired:
    process.kill()
    process.wait()

# Save output to file
with open('hybrid_run_output.log', 'w') as f:
    f.write('\n'.join(output_lines))

print(f"Total lines captured: {len(output_lines)}")
print("Output saved to hybrid_run_output.log")