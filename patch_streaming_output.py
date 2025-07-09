#!/usr/bin/env python3
"""
Patch the local_issue_creator.py to show Claude's real-time output.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Read the current file
with open('scripts/local_issue_creator.py', 'r') as f:
    content = f.read()

# Find the log_progress function and replace it
old_code = """            def log_progress():
                elapsed = 0
                while not stop_progress.is_set():
                    elapsed += 30
                    if not stop_progress.is_set():
                        self.logger.info(f"Claude still executing... ({elapsed}s elapsed)")
                    stop_progress.wait(30)"""

new_code = """            def log_progress():
                elapsed = 0
                last_output_time = time.time()
                while not stop_progress.is_set():
                    # Check if Claude is producing output
                    if process.poll() is None:  # Still running
                        current_time = time.time()
                        if current_time - last_output_time > 30:
                            elapsed = int(current_time - start_time)
                            self.logger.info(f"[CLAUDE STATUS] Still working... ({elapsed}s elapsed)")
                            last_output_time = current_time
                    stop_progress.wait(5)  # Check more frequently"""

# Also need to modify the Popen call to enable real-time output
old_popen = """            process = subprocess.Popen(
                [self.claude_executable, '--dangerously-skip-permissions'],
                cwd=str(repo_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )"""

new_popen = """            process = subprocess.Popen(
                [self.claude_executable, '--dangerously-skip-permissions'],
                cwd=str(repo_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered for real-time output
            )
            
            # Stream output in real-time
            import select
            import fcntl
            
            # Make stdout non-blocking
            fl = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
            fcntl.fcntl(process.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            
            # Collect output while streaming
            stdout_lines = []
            stderr_lines = []
            
            def read_output():
                try:
                    while True:
                        reads = [process.stdout, process.stderr]
                        ret = select.select(reads, [], [], 0.1)
                        
                        for fd in ret[0]:
                            if fd == process.stdout:
                                line = fd.readline()
                                if line:
                                    stdout_lines.append(line)
                                    self.logger.info(f"[CLAUDE] {line.rstrip()}")
                            elif fd == process.stderr:
                                line = fd.readline()
                                if line:
                                    stderr_lines.append(line)
                                    self.logger.warning(f"[CLAUDE ERR] {line.rstrip()}")
                        
                        if process.poll() is not None:
                            break
                except Exception as e:
                    self.logger.debug(f"Error reading output: {e}")"""

# Replace the code
if old_code in content:
    content = content.replace(old_code, new_code)
    print("✓ Replaced log_progress function")
else:
    print("✗ Could not find log_progress function to replace")

if old_popen in content:
    content = content.replace(old_popen, new_popen)
    print("✓ Replaced Popen call")
else:
    print("✗ Could not find Popen call to replace")

# Also need to update the communicate call
old_comm = """            # Explicitly set no timeout - let Claude run as long as needed
            stdout, stderr = process.communicate(input=prompt, timeout=None)
            stop_progress.set()"""

new_comm = """            # Send prompt and start reading output
            process.stdin.write(prompt)
            process.stdin.close()
            
            # Read output in real-time
            output_thread = threading.Thread(target=read_output)
            output_thread.daemon = True
            output_thread.start()
            
            # Wait for process to complete
            process.wait()
            stop_progress.set()
            output_thread.join(timeout=1)
            
            # Collect final output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)"""

if old_comm in content:
    content = content.replace(old_comm, new_comm)
    print("✓ Replaced communicate call")
else:
    print("✗ Could not find communicate call to replace")

# Write the patched file
with open('scripts/local_issue_creator_patched.py', 'w') as f:
    f.write(content)

print("\n✅ Created patched file: scripts/local_issue_creator_patched.py")
print("\nTo apply the patch:")
print("  cp scripts/local_issue_creator_patched.py scripts/local_issue_creator.py")
print("\nThen restart the system to see Claude's real-time output!")