"""
Streaming Task Processor - Real-time Task Processing with Streaming

This module provides streaming capabilities for task generation to reduce
perceived latency and enable real-time processing of large AI responses.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, AsyncIterator, Callable
from dataclasses import dataclass, field
from collections import deque
import re


@dataclass
class StreamChunk:
    """Represents a chunk of streamed data."""
    content: str
    chunk_type: str  # 'partial', 'complete', 'error'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StreamingTask:
    """Task being processed via streaming."""
    id: str
    partial_content: str = ""
    completed_sections: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "streaming"  # 'streaming', 'complete', 'error'
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    chunks_received: int = 0


class StreamingTaskProcessor:
    """Processes task generation with streaming for improved responsiveness."""
    
    def __init__(self, ai_client, callback_handler: Optional[Callable] = None):
        """Initialize streaming processor.
        
        Args:
            ai_client: AI client with streaming support
            callback_handler: Optional callback for processing chunks
        """
        self.ai_client = ai_client
        self.callback_handler = callback_handler
        self.logger = logging.getLogger(__name__)
        
        # Streaming configuration
        self.chunk_size = 100  # Characters per chunk
        self.buffer_size = 10  # Chunks to buffer
        self.parse_incrementally = True
        
        # Active streams
        self.active_streams: Dict[str, StreamingTask] = {}
        
        # Pattern matchers for incremental parsing
        self.task_patterns = {
            'title': re.compile(r'"title"\s*:\s*"([^"]+)"'),
            'type': re.compile(r'"type"\s*:\s*"([^"]+)"'),
            'priority': re.compile(r'"priority"\s*:\s*"([^"]+)"'),
            'repository': re.compile(r'"repository"\s*:\s*"([^"]+)"'),
            'description': re.compile(r'"description"\s*:\s*"([^"]+(?:\\.[^"]+)*)"')
        }
    
    async def generate_tasks_streaming(self, prompt: str, 
                                     context: Dict[str, Any],
                                     on_task_ready: Optional[Callable] = None) -> AsyncIterator[Dict[str, Any]]:
        """Generate tasks with streaming for real-time processing.
        
        Args:
            prompt: Task generation prompt
            context: Generation context
            on_task_ready: Callback when a task is ready
            
        Yields:
            Completed tasks as they become available
        """
        stream_id = f"stream_{datetime.now(timezone.utc).timestamp()}"
        streaming_task = StreamingTask(id=stream_id)
        self.active_streams[stream_id] = streaming_task
        
        try:
            # Start streaming request
            async for chunk in self._stream_ai_response(prompt, context):
                # Process chunk
                processed = await self._process_chunk(streaming_task, chunk)
                
                # Check for completed tasks
                if processed and 'completed_tasks' in processed:
                    for task in processed['completed_tasks']:
                        if on_task_ready:
                            await on_task_ready(task)
                        yield task
                
                # Update progress if callback provided
                if self.callback_handler:
                    await self.callback_handler({
                        'type': 'progress',
                        'stream_id': stream_id,
                        'chunks': streaming_task.chunks_received,
                        'partial_tasks': len(streaming_task.completed_sections)
                    })
            
            # Process any remaining content
            final_tasks = self._finalize_stream(streaming_task)
            for task in final_tasks:
                if on_task_ready:
                    await on_task_ready(task)
                yield task
                
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            streaming_task.status = "error"
            if self.callback_handler:
                await self.callback_handler({
                    'type': 'error',
                    'stream_id': stream_id,
                    'error': str(e)
                })
        finally:
            # Cleanup
            del self.active_streams[stream_id]
    
    async def _stream_ai_response(self, prompt: str, context: Dict[str, Any]) -> AsyncIterator[str]:
        """Stream AI response chunks.
        
        Args:
            prompt: AI prompt
            context: Context data
            
        Yields:
            Response chunks
        """
        # This is a simulation - real implementation would use actual streaming API
        # For now, we'll simulate streaming by breaking up a response
        
        full_response = await self.ai_client.generate_enhanced_response(prompt)
        content = full_response.get('content', '')
        
        # Simulate streaming by yielding chunks
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            yield chunk
            # Simulate network delay
            await asyncio.sleep(0.05)
    
    async def _process_chunk(self, streaming_task: StreamingTask, 
                           chunk: str) -> Optional[Dict[str, Any]]:
        """Process a streaming chunk.
        
        Args:
            streaming_task: Current streaming task
            chunk: New chunk of data
            
        Returns:
            Processed result if tasks are ready
        """
        streaming_task.chunks_received += 1
        streaming_task.partial_content += chunk
        
        if not self.parse_incrementally:
            return None
        
        # Try to extract completed tasks incrementally
        completed_tasks = []
        
        # Look for complete JSON objects
        content = streaming_task.partial_content
        
        # Simple approach: look for complete task objects
        # In production, use a proper streaming JSON parser
        task_matches = re.finditer(r'\{[^{}]*"title"[^{}]*\}', content, re.DOTALL)
        
        for match in task_matches:
            task_json = match.group()
            try:
                # Try to parse as complete task
                task_data = json.loads(task_json)
                
                # Extract task fields progressively
                task = self._extract_task_fields(task_json)
                if task and self._is_valid_task(task):
                    completed_tasks.append(task)
                    
                    # Remove processed content
                    streaming_task.partial_content = streaming_task.partial_content.replace(
                        task_json, '', 1
                    )
            except json.JSONDecodeError:
                # Not a complete JSON object yet
                continue
        
        if completed_tasks:
            streaming_task.completed_sections.extend(completed_tasks)
            return {'completed_tasks': completed_tasks}
        
        return None
    
    def _extract_task_fields(self, task_json: str) -> Dict[str, Any]:
        """Extract task fields from partial JSON.
        
        Args:
            task_json: Partial or complete task JSON
            
        Returns:
            Extracted task fields
        """
        task = {}
        
        # Extract fields using patterns
        for field, pattern in self.task_patterns.items():
            match = pattern.search(task_json)
            if match:
                value = match.group(1)
                # Unescape JSON strings
                value = value.replace('\\"', '"').replace('\\n', '\n')
                task[field] = value
        
        # Set defaults for missing fields
        task.setdefault('type', 'TASK')
        task.setdefault('priority', 'medium')
        task.setdefault('status', 'pending')
        
        return task
    
    def _is_valid_task(self, task: Dict[str, Any]) -> bool:
        """Check if extracted task is valid.
        
        Args:
            task: Task to validate
            
        Returns:
            True if valid
        """
        required_fields = ['title']
        return all(field in task and task[field] for field in required_fields)
    
    def _finalize_stream(self, streaming_task: StreamingTask) -> List[Dict[str, Any]]:
        """Finalize streaming and extract any remaining tasks.
        
        Args:
            streaming_task: Streaming task to finalize
            
        Returns:
            Any remaining tasks
        """
        remaining_tasks = []
        
        # Try to parse any remaining content
        if streaming_task.partial_content.strip():
            try:
                # Attempt to parse as JSON array or object
                remaining_content = streaming_task.partial_content.strip()
                
                # Try array first
                if remaining_content.startswith('['):
                    tasks = json.loads(remaining_content)
                    if isinstance(tasks, list):
                        remaining_tasks.extend(tasks)
                # Try single object
                elif remaining_content.startswith('{'):
                    task = json.loads(remaining_content)
                    if isinstance(task, dict):
                        remaining_tasks.append(task)
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse remaining content: {len(remaining_content)} chars")
        
        streaming_task.status = "complete"
        elapsed = (datetime.now(timezone.utc) - streaming_task.start_time).total_seconds()
        self.logger.info(f"Stream {streaming_task.id} complete: {elapsed:.2f}s, "
                        f"{streaming_task.chunks_received} chunks, "
                        f"{len(streaming_task.completed_sections) + len(remaining_tasks)} tasks")
        
        return remaining_tasks
    
    async def process_large_analysis_streaming(self, repository_data: Dict[str, Any],
                                             chunk_processor: Optional[Callable] = None) -> Dict[str, Any]:
        """Process large repository analysis with streaming.
        
        Args:
            repository_data: Large repository data to analyze
            chunk_processor: Optional processor for intermediate results
            
        Returns:
            Complete analysis
        """
        # Break large data into chunks for streaming analysis
        data_chunks = self._chunk_repository_data(repository_data)
        
        analysis_results = {
            'health_metrics': {},
            'technical_insights': [],
            'recommendations': [],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        for i, chunk in enumerate(data_chunks):
            # Process chunk
            chunk_prompt = f"""
            Analyze this portion ({i+1}/{len(data_chunks)}) of repository data:
            
            {json.dumps(chunk, indent=2)}
            
            Provide incremental analysis focusing on:
            1. Health indicators
            2. Technical patterns
            3. Improvement opportunities
            
            Format as JSON with clear sections.
            """
            
            # Stream process the chunk
            async for result in self.generate_tasks_streaming(chunk_prompt, chunk):
                # Merge results incrementally
                if 'health_metrics' in result:
                    analysis_results['health_metrics'].update(result['health_metrics'])
                if 'insights' in result:
                    analysis_results['technical_insights'].extend(result['insights'])
                if 'recommendations' in result:
                    analysis_results['recommendations'].extend(result['recommendations'])
                
                # Call chunk processor if provided
                if chunk_processor:
                    await chunk_processor(result)
        
        return analysis_results
    
    def _chunk_repository_data(self, data: Dict[str, Any], 
                             max_size: int = 10000) -> List[Dict[str, Any]]:
        """Break repository data into processable chunks.
        
        Args:
            data: Repository data
            max_size: Maximum chunk size in characters
            
        Returns:
            List of data chunks
        """
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for key, value in data.items():
            value_str = json.dumps(value)
            value_size = len(value_str)
            
            if current_size + value_size > max_size and current_chunk:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = {}
                current_size = 0
            
            current_chunk[key] = value
            current_size += value_size
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def get_active_streams(self) -> List[Dict[str, Any]]:
        """Get information about active streams.
        
        Returns:
            Active stream information
        """
        return [
            {
                'id': stream_id,
                'status': stream.status,
                'chunks_received': stream.chunks_received,
                'tasks_completed': len(stream.completed_sections),
                'duration': (datetime.now(timezone.utc) - stream.start_time).total_seconds()
            }
            for stream_id, stream in self.active_streams.items()
        ]
    
    async def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream.
        
        Args:
            stream_id: Stream to cancel
            
        Returns:
            True if cancelled
        """
        if stream_id in self.active_streams:
            self.active_streams[stream_id].status = "cancelled"
            del self.active_streams[stream_id]
            self.logger.info(f"Cancelled stream: {stream_id}")
            return True
        return False


class StreamingProgressTracker:
    """Tracks progress of streaming operations."""
    
    def __init__(self):
        self.progress_data = {}
        self.callbacks = []
    
    def register_callback(self, callback: Callable) -> None:
        """Register a progress callback."""
        self.callbacks.append(callback)
    
    async def update_progress(self, stream_id: str, progress: Dict[str, Any]) -> None:
        """Update progress for a stream."""
        self.progress_data[stream_id] = progress
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                await callback(stream_id, progress)
            except Exception as e:
                logging.error(f"Progress callback error: {e}")
    
    def get_progress(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a stream."""
        return self.progress_data.get(stream_id)
    
    def clear_completed(self) -> None:
        """Clear completed stream progress."""
        completed = [sid for sid, data in self.progress_data.items() 
                    if data.get('status') in ['complete', 'error', 'cancelled']]
        for sid in completed:
            del self.progress_data[sid]