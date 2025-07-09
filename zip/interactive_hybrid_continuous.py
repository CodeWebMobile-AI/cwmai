#!/usr/bin/env python3
"""
Interactive Hybrid Continuous System with Claude Code-like Terminal Interface

This provides a beautiful terminal interface where you can:
- Send messages while the system is processing
- See live worker status
- Watch streaming responses
- Queue multiple commands
"""

import asyncio
import sys
import os
import threading
import queue
from pathlib import Path
from datetime import datetime
import argparse

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.enhanced_conversational_assistant import EnhancedConversationalAssistant
from scripts.streaming_response_generator import StreamingResponseGenerator, VisualStateIndicator
from scripts.hybrid_continuous_orchestrator import HybridContinuousOrchestrator


class InteractiveTerminal:
    """Handles non-blocking terminal input."""
    
    def __init__(self):
        self.input_queue = queue.Queue()
        self.input_thread = None
        self.running = False
        
    def start(self):
        """Start the input handler."""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
    def stop(self):
        """Stop the input handler."""
        self.running = False
        
    def _input_loop(self):
        """Background thread for input."""
        while self.running:
            try:
                line = input()
                if line:
                    self.input_queue.put(line)
            except EOFError:
                break
            except:
                pass
                
    def get_input(self, timeout=0.1):
        """Get input if available."""
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class InteractiveHybridSystem:
    """Interactive hybrid continuous system with Claude Code interface."""
    
    def __init__(self, num_workers=10):
        self.num_workers = num_workers
        self.orchestrator = None
        self.assistant = None
        self.terminal = InteractiveTerminal()
        self.streamer = StreamingResponseGenerator(typing_speed=0.02)
        self.indicator = VisualStateIndicator()
        self.running = False
        self.worker_status = {}
        self.pending_messages = []
        
    async def initialize(self):
        """Initialize all components."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🚀 Interactive Hybrid Continuous System")
        print("=" * 80)
        print(f"\n⚙️  Initializing with {self.num_workers} workers...")
        
        # Initialize orchestrator
        self.orchestrator = HybridContinuousOrchestrator(
            max_workers=self.num_workers,
            enable_streaming=True
        )
        
        # Initialize assistant
        self.assistant = EnhancedConversationalAssistant()
        await self.assistant.initialize()
        
        # Start orchestrator
        await self.orchestrator.start()
        
        print("✅ System initialized successfully!")
        print("\n" + "=" * 80)
        
    def display_interface(self):
        """Display the main interface."""
        print("\n🤖 Claude Code-Style Terminal Interface")
        print("=" * 80)
        
        print("\n📊 System Status:")
        print(f"  • Workers: {self.num_workers} active")
        print(f"  • Tools: {len(self.assistant.tool_system.tools)} available")
        print(f"  • Streaming: ✅ Enabled")
        print(f"  • Background Processing: ✅ Active")
        
        print("\n✨ Features:")
        print("  • Type while the system is processing")
        print("  • Multiple messages are queued automatically")
        print("  • See live worker status and progress")
        print("  • Character-by-character streaming responses")
        
        print("\n📝 Commands:")
        print("  • Natural language for tasks and questions")
        print("  • 'status' - Show system status")
        print("  • 'workers' - Show worker activity")
        print("  • 'quit' or 'exit' - Stop the system")
        
        print("\n" + "-" * 80)
        print("Ready! Start typing your messages...\n")
        
    async def run(self):
        """Run the interactive system."""
        await self.initialize()
        self.display_interface()
        
        # Start terminal input handler
        self.terminal.start()
        self.running = True
        
        # Create status update task
        status_task = asyncio.create_task(self.update_status_periodically())
        
        try:
            while self.running:
                # Check for input
                user_input = self.terminal.get_input(timeout=0.05)
                
                if user_input:
                    await self.handle_input(user_input)
                
                # Process any pending work
                await self.process_background_work()
                
                # Small delay for responsiveness
                await asyncio.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Shutting down gracefully...")
        finally:
            self.running = False
            status_task.cancel()
            self.terminal.stop()
            
            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.stop()
                
            print("\n👋 System stopped. Goodbye!")
    
    async def handle_input(self, user_input: str):
        """Handle user input."""
        # Show input with prompt
        print(f"\n> {user_input}")
        
        # Check for special commands
        if user_input.lower() in ['quit', 'exit']:
            self.running = False
            return
            
        elif user_input.lower() == 'status':
            await self.show_system_status()
            return
            
        elif user_input.lower() == 'workers':
            await self.show_worker_status()
            return
        
        # Queue the message if system is busy
        if self.is_processing():
            self.pending_messages.append(user_input)
            print(f"📋 Message queued ({len(self.pending_messages)} pending)")
            await self.indicator.show_queue_status(len(self.pending_messages), self.active_workers())
        else:
            # Process immediately
            await self.process_message(user_input)
    
    async def process_message(self, message: str):
        """Process a message with streaming output."""
        # Show processing indicator
        spinner_task = asyncio.create_task(
            self.indicator.show_spinner("Processing your request")
        )
        
        try:
            # Get response from assistant
            response = await self.assistant.handle_conversation(message, stream_output=False)
            
            # Stop spinner
            self.indicator.stop_spinner()
            await asyncio.sleep(0.1)
            
            # Stream the response
            if response:
                await self.streamer.stream_response(response, indicator="🤖")
                
        except Exception as e:
            self.indicator.stop_spinner()
            await self.streamer.stream_response(f"Error: {str(e)}", indicator="❌")
        
        # Process next pending message if any
        if self.pending_messages:
            next_message = self.pending_messages.pop(0)
            print(f"\n📋 Processing queued message: {next_message[:50]}...")
            await self.process_message(next_message)
    
    async def process_background_work(self):
        """Process background work from orchestrator."""
        # This would integrate with your orchestrator's work queue
        # For now, we'll simulate background processing
        pass
    
    async def update_status_periodically(self):
        """Update status line periodically."""
        while self.running:
            try:
                # Update worker status
                if hasattr(self.orchestrator, 'get_worker_status'):
                    self.worker_status = await self.orchestrator.get_worker_status()
                
                # Display status line
                active = self.active_workers()
                pending = len(self.pending_messages)
                
                if active > 0 or pending > 0:
                    status = f"\r💼 Workers: {active} active"
                    if pending > 0:
                        status += f" | 📋 Queue: {pending} pending"
                    status += " " * 20  # Clear line
                    
                    sys.stdout.write(status)
                    sys.stdout.flush()
                
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                break
            except:
                pass
    
    def is_processing(self):
        """Check if system is currently processing."""
        # Check if any workers are active or messages are pending
        return self.active_workers() > 0 or len(self.pending_messages) > 0
    
    def active_workers(self):
        """Get number of active workers."""
        # This would check actual worker status
        # For now, return a simulated value
        return sum(1 for status in self.worker_status.values() if status.get('active', False))
    
    async def show_system_status(self):
        """Show detailed system status."""
        print("\n📊 System Status")
        print("-" * 40)
        
        # Get assistant status
        status = await self.assistant.get_status()
        
        print(f"Performance Metrics:")
        print(f"  • Total interactions: {status['performance']['total_interactions']}")
        print(f"  • Successful: {status['performance']['successful_interactions']}")
        print(f"  • Avg response time: {status['performance']['average_response_time']:.2f}s")
        
        print(f"\nLearning Status:")
        print(f"  • Insights gathered: {status['learning']['total_insights']}")
        print(f"  • Improvements applied: {status['learning']['applied_improvements']}")
        
        print(f"\nSystem Resources:")
        print(f"  • Available tools: {status['available_tools']}")
        print(f"  • Conversation history: {status['conversation_history_size']} entries")
        print(f"  • Background tasks: {status['background_tasks']}")
    
    async def show_worker_status(self):
        """Show worker activity."""
        print("\n👷 Worker Status")
        print("-" * 40)
        
        if not self.worker_status:
            print("No worker activity data available.")
            return
        
        for worker_id, status in self.worker_status.items():
            state = "🟢 Active" if status.get('active') else "⚪ Idle"
            task = status.get('current_task', 'None')
            print(f"Worker {worker_id}: {state} - {task}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Interactive Hybrid Continuous System')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    args = parser.parse_args()
    
    system = InteractiveHybridSystem(num_workers=args.workers)
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())