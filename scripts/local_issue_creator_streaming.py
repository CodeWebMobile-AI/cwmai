"""
Enhanced Local Issue Creator with Real-Time Claude Output Streaming

This version shows Claude's actual output as it works, not just "still executing" messages.
"""

import subprocess
import threading
import queue
import sys
import time
from typing import Optional
import logging


class StreamingClaudeExecutor:
    """Execute Claude and stream its output in real-time."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def execute_with_streaming(self, claude_path: str, prompt: str, cwd: str) -> subprocess.CompletedProcess:
        """Execute Claude and stream output in real-time.
        
        Args:
            claude_path: Path to Claude executable
            prompt: The prompt to send to Claude
            cwd: Working directory
            
        Returns:
            CompletedProcess with results
        """
        self.logger.info("Starting Claude with real-time output streaming...")
        
        # Start Claude process
        process = subprocess.Popen(
            [claude_path, '--dangerously-skip-permissions'],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Send prompt and close stdin
        process.stdin.write(prompt)
        process.stdin.close()
        
        # Create queues for output
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        stdout_lines = []
        stderr_lines = []
        
        def enqueue_output(pipe, queue, lines_list, pipe_name):
            """Read from pipe and put lines in queue."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        queue.put(line)
                        lines_list.append(line)
                        # Log Claude's actual output
                        self.logger.info(f"[CLAUDE {pipe_name}] {line.rstrip()}")
            except Exception as e:
                self.logger.error(f"Error reading from {pipe_name}: {e}")
            finally:
                pipe.close()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=enqueue_output, 
            args=(process.stdout, stdout_queue, stdout_lines, "OUT")
        )
        stderr_thread = threading.Thread(
            target=enqueue_output, 
            args=(process.stderr, stderr_queue, stderr_lines, "ERR")
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Monitor progress
        start_time = time.time()
        last_activity = time.time()
        
        while process.poll() is None:
            # Check for new output
            current_time = time.time()
            had_activity = False
            
            # Process stdout queue
            while not stdout_queue.empty():
                try:
                    stdout_queue.get_nowait()
                    had_activity = True
                    last_activity = current_time
                except queue.Empty:
                    break
            
            # Process stderr queue
            while not stderr_queue.empty():
                try:
                    stderr_queue.get_nowait()
                    had_activity = True
                    last_activity = current_time
                except queue.Empty:
                    break
            
            # Log status if no activity for 30 seconds
            if current_time - last_activity > 30:
                elapsed = int(current_time - start_time)
                self.logger.info(f"[CLAUDE STATUS] Still processing... ({elapsed}s elapsed, no output for {int(current_time - last_activity)}s)")
                last_activity = current_time
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.1)
        
        # Wait for threads to finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        # Get return code
        return_code = process.wait()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"[CLAUDE COMPLETE] Finished in {elapsed_time:.1f}s with return code: {return_code}")
        
        # Return completed process
        return subprocess.CompletedProcess(
            args=[claude_path],
            returncode=return_code,
            stdout=''.join(stdout_lines),
            stderr=''.join(stderr_lines)
        )


# Example usage modification for local_issue_creator.py:
def execute_with_claude_streaming(self, work_item, repo_full_name, issue):
    """Enhanced version with real-time streaming."""
    try:
        # ... (setup code remains the same) ...
        
        # Use streaming executor
        executor = StreamingClaudeExecutor(self.logger)
        result = executor.execute_with_streaming(
            claude_path=self.claude_executable,
            prompt=self._prepare_claude_prompt(work_item, issue),
            cwd=str(repo_path)
        )
        
        # ... (result handling remains the same) ...
        
    except Exception as e:
        self.logger.error(f"Error in streaming execution: {e}")
        # ... (error handling) ...