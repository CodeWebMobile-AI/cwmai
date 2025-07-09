#!/usr/bin/env python3
"""
Simple API call limiter to prevent excessive AI API calls during work discovery.
"""

import time
from typing import Dict, Any
import asyncio

class APICallLimiter:
    """Limit API calls to prevent excessive usage."""
    
    def __init__(self, max_calls_per_minute: int = 10):
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = []
        self._lock = asyncio.Lock()
    
    async def check_and_wait(self):
        """Check if we can make an API call, wait if necessary."""
        async with self._lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) >= self.max_calls_per_minute:
                # Need to wait
                oldest_call = self.call_times[0]
                wait_time = 60 - (now - oldest_call) + 1
                print(f"⏳ Rate limit reached, waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                # Recurse to recheck
                return await self.check_and_wait()
            
            # Record this call
            self.call_times.append(now)
            return True

# Global limiter instance
api_limiter = APICallLimiter(max_calls_per_minute=10)
