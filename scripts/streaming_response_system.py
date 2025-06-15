"""
Streaming Response System for Conversational AI

Provides real-time streaming responses with parallel execution
for better user experience.
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import logging


class StreamEventType(Enum):
    """Types of events in the stream."""
    THINKING = "thinking"
    CONTEXT_GATHERING = "context_gathering"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE_CHUNK = "response_chunk"
    ERROR = "error"
    COMPLETE = "complete"
    SUGGESTION = "suggestion"


@dataclass
class StreamEvent:
    """Represents an event in the response stream."""
    type: StreamEventType
    content: str
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata or {},
            "timestamp": self.timestamp.isoformat()
        }


class StreamingResponseSystem:
    """Handles streaming responses with parallel execution."""
    
    def __init__(self, tool_system=None, context_collector=None):
        """Initialize the streaming system.
        
        Args:
            tool_system: ToolCallingSystem instance
            context_collector: DynamicContextCollector instance
        """
        self.tool_system = tool_system
        self.context_collector = context_collector
        self.logger = logging.getLogger(__name__)
        self.active_streams: Dict[str, asyncio.Queue] = {}
        
    async def stream_response(self, user_input: str, session_id: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response for user input with parallel processing.
        
        Args:
            user_input: The user's input
            session_id: Unique session identifier
            
        Yields:
            StreamEvent objects as they occur
        """
        # Create a queue for this stream
        stream_queue = asyncio.Queue()
        self.active_streams[session_id] = stream_queue
        
        try:
            # Start parallel tasks
            tasks = [
                asyncio.create_task(self._think_about_response(user_input, stream_queue)),
                asyncio.create_task(self._gather_context(user_input, stream_queue)),
                asyncio.create_task(self._process_response(user_input, stream_queue))
            ]
            
            # Stream events as they come
            while True:
                try:
                    event = await asyncio.wait_for(stream_queue.get(), timeout=0.1)
                    
                    if event.type == StreamEventType.COMPLETE:
                        break
                        
                    yield event
                    
                except asyncio.TimeoutError:
                    # Check if all tasks are done
                    if all(task.done() for task in tasks):
                        yield StreamEvent(StreamEventType.COMPLETE, "Response complete")
                        break
                        
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            yield StreamEvent(StreamEventType.ERROR, str(e))
            
        finally:
            # Cleanup
            del self.active_streams[session_id]
            
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    async def _think_about_response(self, user_input: str, queue: asyncio.Queue):
        """Think about the response in parallel."""
        await queue.put(StreamEvent(
            StreamEventType.THINKING,
            "Processing your request...",
            {"stage": "initial_analysis"}
        ))
        
        # Simulate thinking time
        await asyncio.sleep(0.5)
        
        # Analyze intent
        intents = await self._analyze_intent(user_input)
        if intents:
            await queue.put(StreamEvent(
                StreamEventType.THINKING,
                f"I understand you want to {intents[0]}",
                {"intents": intents}
            ))
    
    async def _gather_context(self, user_input: str, queue: asyncio.Queue):
        """Gather context in parallel."""
        if not self.context_collector:
            return
            
        await queue.put(StreamEvent(
            StreamEventType.CONTEXT_GATHERING,
            "Gathering relevant information...",
            {"stage": "context_collection"}
        ))
        
        # Start gathering context
        context = await self.context_collector.gather_context_for_query(user_input)
        
        # Report what we found
        summary = context.get('context_summary', {})
        updates = []
        
        if summary.get('repository_count', 0) > 0:
            updates.append(f"Found {summary['repository_count']} repositories")
        if summary.get('task_count', 0) > 0:
            updates.append(f"Found {summary['task_count']} active tasks")
        if summary.get('continuous_ai_running'):
            updates.append("Continuous AI is running")
            
        if updates:
            await queue.put(StreamEvent(
                StreamEventType.CONTEXT_GATHERING,
                " • ".join(updates),
                {"context_summary": summary}
            ))
    
    async def _process_response(self, user_input: str, queue: asyncio.Queue):
        """Process the main response with tool execution."""
        await asyncio.sleep(1)  # Wait for context to start gathering
        
        # Determine if tools need to be called
        tool_calls = await self._determine_tool_calls(user_input)
        
        if tool_calls:
            for tool_call in tool_calls:
                await queue.put(StreamEvent(
                    StreamEventType.TOOL_EXECUTION,
                    f"Executing: {tool_call['name']}",
                    {"tool": tool_call['name'], "params": tool_call.get('params', {})}
                ))
                
                # Execute tool
                if self.tool_system:
                    result = await self.tool_system.call_tool(
                        tool_call['name'],
                        **tool_call.get('params', {})
                    )
                    
                    # Stream the result
                    await self._stream_tool_result(result, tool_call['name'], queue)
        
        # Generate and stream the main response
        await self._generate_streaming_response(user_input, queue)
        
        # Add suggestions
        suggestions = await self._generate_suggestions(user_input)
        if suggestions:
            await queue.put(StreamEvent(
                StreamEventType.SUGGESTION,
                "You might also want to:",
                {"suggestions": suggestions}
            ))
    
    async def _stream_tool_result(self, result: Dict[str, Any], tool_name: str, queue: asyncio.Queue):
        """Stream tool execution results."""
        if result.get('success'):
            content = self._format_tool_result(result.get('result'), tool_name)
            
            # Stream in chunks if large
            if len(content) > 200:
                chunks = self._chunk_text(content, 200)
                for chunk in chunks:
                    await queue.put(StreamEvent(
                        StreamEventType.RESPONSE_CHUNK,
                        chunk,
                        {"source": "tool_result", "tool": tool_name}
                    ))
                    await asyncio.sleep(0.05)  # Small delay for readability
            else:
                await queue.put(StreamEvent(
                    StreamEventType.RESPONSE_CHUNK,
                    content,
                    {"source": "tool_result", "tool": tool_name}
                ))
        else:
            await queue.put(StreamEvent(
                StreamEventType.ERROR,
                f"Error executing {tool_name}: {result.get('error')}",
                {"tool": tool_name}
            ))
    
    def _format_tool_result(self, result: Any, tool_name: str) -> str:
        """Format tool results for display."""
        if isinstance(result, list):
            if len(result) == 0:
                return f"No results found from {tool_name}"
            else:
                formatted = f"Results from {tool_name}:\n"
                for item in result[:5]:
                    if isinstance(item, dict):
                        formatted += f"• {item.get('name', item.get('title', str(item)))}\n"
                    else:
                        formatted += f"• {item}\n"
                if len(result) > 5:
                    formatted += f"... and {len(result) - 5} more"
                return formatted
        elif isinstance(result, dict):
            # Format key details
            formatted = f"{tool_name} result:\n"
            for key, value in result.items():
                if key not in ['raw_data', 'metadata']:  # Skip verbose keys
                    formatted += f"• {key}: {value}\n"
            return formatted
        else:
            return f"{tool_name}: {result}"
    
    async def _generate_streaming_response(self, user_input: str, queue: asyncio.Queue):
        """Generate and stream the main AI response."""
        # This would integrate with your AI system
        response_parts = [
            "Based on the information I've gathered,",
            "here's what I found:",
            "The system is currently operational",
            "with all services running normally."
        ]
        
        for part in response_parts:
            await queue.put(StreamEvent(
                StreamEventType.RESPONSE_CHUNK,
                part + " ",
                {"source": "ai_response"}
            ))
            await asyncio.sleep(0.1)  # Simulate typing
    
    async def _analyze_intent(self, user_input: str) -> List[str]:
        """Analyze user intent quickly."""
        lower_input = user_input.lower()
        intents = []
        
        intent_keywords = {
            "check system status": ["status", "running", "health"],
            "view repositories": ["repositories", "repos", "projects"],
            "manage tasks": ["tasks", "todos", "work"],
            "get help": ["help", "what can", "how to"],
            "create something": ["create", "make", "build"],
            "analyze data": ["analyze", "review", "check"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in lower_input for keyword in keywords):
                intents.append(intent)
                
        return intents
    
    async def _determine_tool_calls(self, user_input: str) -> List[Dict[str, Any]]:
        """Determine which tools need to be called."""
        tool_calls = []
        lower_input = user_input.lower()
        
        # Map patterns to tool calls
        if "status" in lower_input or "running" in lower_input:
            tool_calls.append({"name": "get_system_status", "params": {}})
            
        if "repositories" in lower_input or "repos" in lower_input:
            tool_calls.append({"name": "get_repositories", "params": {"limit": 10}})
            
        if "tasks" in lower_input:
            tool_calls.append({"name": "get_tasks", "params": {"status": "active"}})
            
        if "continuous ai" in lower_input:
            tool_calls.append({"name": "get_continuous_ai_status", "params": {}})
            
        return tool_calls
    
    async def _generate_suggestions(self, user_input: str) -> List[str]:
        """Generate follow-up suggestions."""
        suggestions = []
        lower_input = user_input.lower()
        
        if "status" in lower_input:
            suggestions.extend([
                "View detailed performance metrics",
                "Check continuous AI health",
                "See recent system activity"
            ])
        elif "repositories" in lower_input:
            suggestions.extend([
                "Analyze a specific repository",
                "Search for code patterns",
                "Create a new issue"
            ])
        elif "tasks" in lower_input:
            suggestions.extend([
                "Create a new task",
                "View completed tasks",
                "Check task performance metrics"
            ])
            
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into readable chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    async def cancel_stream(self, session_id: str):
        """Cancel an active stream."""
        if session_id in self.active_streams:
            queue = self.active_streams[session_id]
            await queue.put(StreamEvent(
                StreamEventType.COMPLETE,
                "Stream cancelled by user"
            ))


class StreamingFormatter:
    """Formats stream events for different output formats."""
    
    @staticmethod
    def format_for_terminal(event: StreamEvent) -> str:
        """Format event for terminal output."""
        type_symbols = {
            StreamEventType.THINKING: "🤔",
            StreamEventType.CONTEXT_GATHERING: "📊",
            StreamEventType.TOOL_EXECUTION: "🔧",
            StreamEventType.RESPONSE_CHUNK: "💬",
            StreamEventType.ERROR: "❌",
            StreamEventType.COMPLETE: "✅",
            StreamEventType.SUGGESTION: "💡"
        }
        
        symbol = type_symbols.get(event.type, "•")
        
        if event.type == StreamEventType.RESPONSE_CHUNK:
            return event.content  # No prefix for response chunks
        else:
            return f"{symbol} {event.content}"
    
    @staticmethod
    def format_for_web(event: StreamEvent) -> Dict[str, Any]:
        """Format event for web/API output."""
        return {
            "type": event.type.value,
            "content": event.content,
            "metadata": event.metadata,
            "timestamp": event.timestamp.isoformat(),
            "formatted_time": event.timestamp.strftime("%H:%M:%S")
        }