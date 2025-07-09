#!/usr/bin/env python3
"""
Streaming Response Generator for Natural Language Todo System

Provides character-by-character streaming output with interruption support.
"""

import asyncio
import sys
import time
from typing import AsyncGenerator, Optional, Callable
from enum import Enum


class StreamState(Enum):
    """States for the streaming system."""
    IDLE = "idle"
    STREAMING = "streaming"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


class StreamingResponseGenerator:
    """Handles streaming responses with interruption support."""
    
    def __init__(self, typing_speed: float = 0.03):
        """
        Initialize the streaming generator.
        
        Args:
            typing_speed: Delay between characters (seconds)
        """
        self.typing_speed = typing_speed
        self.state = StreamState.IDLE
        self._interrupt_event = asyncio.Event()
        self._current_task: Optional[asyncio.Task] = None
        
    async def stream_response(
        self, 
        text: str, 
        indicator: str = "🤖",
        show_cursor: bool = True
    ) -> bool:
        """
        Stream text character by character with interruption support.
        
        Args:
            text: The text to stream
            indicator: Prefix indicator (e.g., "🤖")
            show_cursor: Whether to show typing cursor
            
        Returns:
            True if completed, False if interrupted
        """
        self.state = StreamState.STREAMING
        self._interrupt_event.clear()
        
        # Show indicator
        sys.stdout.write(f"{indicator} ")
        sys.stdout.flush()
        
        try:
            for i, char in enumerate(text):
                # Check for interruption
                if self._interrupt_event.is_set():
                    self.state = StreamState.INTERRUPTED
                    sys.stdout.write("\n[Interrupted]")
                    sys.stdout.flush()
                    return False
                
                # Write character
                sys.stdout.write(char)
                
                # Show cursor if enabled
                if show_cursor and i < len(text) - 1:
                    sys.stdout.write("▌")
                    sys.stdout.flush()
                    await asyncio.sleep(self.typing_speed)
                    sys.stdout.write("\b \b")  # Erase cursor
                
                sys.stdout.flush()
                
                # Small delay for natural typing effect
                if char in '.!?':
                    await asyncio.sleep(self.typing_speed * 3)
                elif char in ',;:':
                    await asyncio.sleep(self.typing_speed * 2)
                else:
                    await asyncio.sleep(self.typing_speed)
            
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.state = StreamState.COMPLETED
            return True
            
        except asyncio.CancelledError:
            self.state = StreamState.INTERRUPTED
            return False
    
    def interrupt(self):
        """Interrupt the current streaming."""
        if self.state == StreamState.STREAMING:
            self._interrupt_event.set()
    
    async def stream_with_state(
        self, 
        text: str,
        state_indicator: Optional[Callable] = None
    ) -> bool:
        """
        Stream with visual state indicators.
        
        Args:
            text: Text to stream
            state_indicator: Optional callback for state updates
            
        Returns:
            True if completed, False if interrupted
        """
        if state_indicator:
            await state_indicator("processing")
        
        result = await self.stream_response(text)
        
        if state_indicator:
            await state_indicator("completed" if result else "interrupted")
        
        return result


class VisualStateIndicator:
    """Provides visual indicators for system state."""
    
    def __init__(self):
        self.indicators = {
            "idle": "⚪",
            "processing": "🔄",
            "queued": "📋",
            "completed": "✅",
            "interrupted": "⚠️",
            "error": "❌"
        }
        self._spinner_task: Optional[asyncio.Task] = None
        self._spinning = False
    
    async def show_state(self, state: str, message: str = ""):
        """Show a state indicator with optional message."""
        indicator = self.indicators.get(state, "❓")
        output = f"\r{indicator} {message}"
        sys.stdout.write(output)
        sys.stdout.flush()
    
    async def show_spinner(self, message: str = "Processing"):
        """Show an animated spinner."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinning = True
        i = 0
        
        while self._spinning:
            sys.stdout.write(f"\r{spinner_chars[i]} {message}")
            sys.stdout.flush()
            i = (i + 1) % len(spinner_chars)
            await asyncio.sleep(0.1)
    
    def stop_spinner(self):
        """Stop the spinner animation."""
        self._spinning = False
        sys.stdout.write("\r" + " " * 50 + "\r")  # Clear line
        sys.stdout.flush()
    
    async def show_queue_status(self, pending: int, processing: int):
        """Display queue status."""
        if pending > 0:
            status = f"📋 Queue: {pending} pending"
            if processing > 0:
                status += f", {processing} processing"
            sys.stdout.write(f"\r{status}")
            sys.stdout.flush()


class InterruptHandler:
    """Handles input interruption during streaming."""
    
    def __init__(self):
        self.pending_input = ""
        self.has_pending = False
        self._input_lock = asyncio.Lock()
    
    async def capture_input(self) -> Optional[str]:
        """
        Capture input without blocking streaming.
        
        Returns:
            Captured input or None
        """
        # This would integrate with the terminal's raw mode
        # For now, return None (would need platform-specific implementation)
        return None
    
    async def check_for_interrupt(self) -> bool:
        """Check if user wants to interrupt."""
        async with self._input_lock:
            return self.has_pending
    
    def queue_input(self, text: str):
        """Queue input for later processing."""
        self.pending_input = text
        self.has_pending = True
    
    def get_pending_input(self) -> Optional[str]:
        """Get and clear pending input."""
        if self.has_pending:
            self.has_pending = False
            return self.pending_input
        return None


async def demo_streaming():
    """Demonstrate streaming capabilities."""
    print("=== Streaming Response Demo ===\n")
    
    generator = StreamingResponseGenerator(typing_speed=0.03)
    indicator = VisualStateIndicator()
    
    # Demo 1: Basic streaming
    print("1. Basic streaming:")
    await generator.stream_response(
        "I'm processing your todo list request. Let me add that task for you..."
    )
    
    # Demo 2: With state indicators
    print("\n2. With state indicators:")
    await indicator.show_state("processing", "Adding task")
    await asyncio.sleep(1)
    await indicator.show_state("completed", "Task added successfully")
    print()
    
    # Demo 3: Queue status
    print("\n3. Queue status display:")
    await indicator.show_queue_status(pending=3, processing=1)
    await asyncio.sleep(2)
    print()
    
    # Demo 4: Spinner animation
    print("\n4. Processing with spinner:")
    spinner_task = asyncio.create_task(indicator.show_spinner("Analyzing your request"))
    await asyncio.sleep(3)
    indicator.stop_spinner()
    print("✅ Analysis complete!")
    
    # Demo 5: Different speeds
    print("\n5. Variable typing speeds:")
    fast_gen = StreamingResponseGenerator(typing_speed=0.01)
    await fast_gen.stream_response("This is fast typing! ", indicator="⚡")
    
    slow_gen = StreamingResponseGenerator(typing_speed=0.05)
    await slow_gen.stream_response("This... is... slow... typing...", indicator="🐌")


if __name__ == "__main__":
    asyncio.run(demo_streaming())