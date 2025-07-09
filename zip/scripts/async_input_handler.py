"""
Async Input Handler for Concurrent Input

Allows users to type new commands while the system is processing.
"""

import asyncio
import sys
import threading
import queue
from typing import Optional


class AsyncInputHandler:
    """Handle input asynchronously to allow concurrent typing."""
    
    def __init__(self):
        self.input_queue = queue.Queue()
        self.running = False
        self.input_thread = None
        
    def start(self):
        """Start the input handler thread."""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
    def stop(self):
        """Stop the input handler."""
        self.running = False
        
    def _input_loop(self):
        """Background thread that reads input."""
        while self.running:
            try:
                # Show prompt
                sys.stdout.write("\n> ")
                sys.stdout.flush()
                
                # Read input
                line = input()
                if line:
                    self.input_queue.put(line)
            except EOFError:
                break
            except Exception:
                pass
                
    async def get_input(self, timeout: float = 0.1) -> Optional[str]:
        """Get input if available, non-blocking."""
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def has_input(self) -> bool:
        """Check if there's input waiting."""
        return not self.input_queue.empty()