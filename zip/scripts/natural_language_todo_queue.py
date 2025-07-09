"""
Natural Language Todo Queue System

Implements concurrent message processing similar to Claude Code's approach,
allowing users to send new todo commands while previous ones are still processing.
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from queue import PriorityQueue, Empty
import logging


class MessagePriority(Enum):
    """Priority levels for todo messages."""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class MessageStatus(Enum):
    """Status of messages in the queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TodoMessage:
    """Represents a todo-related message from the user."""
    id: str
    content: str
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    
    def __lt__(self, other):
        """Enable priority queue sorting."""
        return self.priority.value < other.priority.value


class ConcurrentTodoQueue:
    """
    Thread-safe message queue for natural language todo processing.
    Mimics Claude Code's concurrent message handling.
    """
    
    def __init__(self, persist_path: Optional[Path] = None):
        """Initialize the concurrent todo queue."""
        self.logger = logging.getLogger(__name__)
        self.persist_path = persist_path or Path("todo_queue_state.json")
        
        # Thread-safe priority queue
        self._queue = PriorityQueue()
        self._processing = {}  # Messages currently being processed
        self._completed = {}   # Completed messages
        self._lock = threading.Lock()
        
        # Message deduplication
        self._message_hashes = set()
        
        # Load persisted state
        self._load_state()
        
    def _generate_message_id(self, content: str) -> str:
        """Generate unique message ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(f"{content}_{timestamp}".encode()).hexdigest()[:8]
        
    def _generate_message_hash(self, content: str) -> str:
        """Generate hash for deduplication."""
        # Normalize content for deduplication
        normalized = content.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
        
    def enqueue(self, content: str, priority: MessagePriority = MessagePriority.NORMAL) -> TodoMessage:
        """
        Add a new message to the queue.
        Returns the created message.
        """
        # Check for duplicates
        msg_hash = self._generate_message_hash(content)
        
        with self._lock:
            if msg_hash in self._message_hashes:
                self.logger.debug(f"Duplicate message detected: {content[:50]}...")
                # Find and return the existing message
                for msg in list(self._queue.queue) + list(self._processing.values()):
                    if self._generate_message_hash(msg.content) == msg_hash:
                        return msg
                        
        # Create new message
        message = TodoMessage(
            id=self._generate_message_id(content),
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=priority
        )
        
        # Add to queue
        with self._lock:
            self._queue.put((priority.value, time.time(), message))
            self._message_hashes.add(msg_hash)
            
        self.logger.info(f"Enqueued message: {message.id} - {content[:50]}...")
        self._persist_state()
        
        return message
        
    def dequeue(self, timeout: float = 1.0) -> Optional[TodoMessage]:
        """
        Get the next message to process.
        Returns None if no messages available within timeout.
        """
        try:
            _, _, message = self._queue.get(timeout=timeout)
            
            with self._lock:
                message.status = MessageStatus.PROCESSING
                message.processing_started = datetime.now(timezone.utc)
                self._processing[message.id] = message
                
            self.logger.debug(f"Dequeued message: {message.id}")
            self._persist_state()
            
            return message
            
        except Empty:
            return None
            
    def complete(self, message_id: str, result: Any = None, error: Optional[str] = None):
        """Mark a message as completed."""
        with self._lock:
            if message_id not in self._processing:
                self.logger.warning(f"Message {message_id} not found in processing")
                return
                
            message = self._processing.pop(message_id)
            message.processing_completed = datetime.now(timezone.utc)
            
            if error:
                message.status = MessageStatus.FAILED
                message.error = error
                self.logger.error(f"Message {message_id} failed: {error}")
            else:
                message.status = MessageStatus.COMPLETED
                message.result = result
                self.logger.info(f"Message {message_id} completed successfully")
                
            self._completed[message_id] = message
            
        self._persist_state()
        
    def get_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            return {
                "pending": self._queue.qsize(),
                "processing": len(self._processing),
                "completed": len(self._completed),
                "oldest_pending": self._get_oldest_pending(),
                "active_processors": list(self._processing.keys())
            }
            
    def _get_oldest_pending(self) -> Optional[str]:
        """Get timestamp of oldest pending message."""
        if self._queue.empty():
            return None
            
        try:
            # Peek at queue without removing
            items = list(self._queue.queue)
            if items:
                oldest = min(items, key=lambda x: x[2].timestamp)
                return oldest[2].timestamp.isoformat()
        except:
            pass
            
        return None
        
    def _persist_state(self):
        """Save queue state to disk."""
        try:
            state = {
                "pending": [],
                "processing": [],
                "completed": []
            }
            
            # Convert queue items
            with self._lock:
                for _, _, msg in list(self._queue.queue):
                    state["pending"].append(self._message_to_dict(msg))
                    
                for msg in self._processing.values():
                    state["processing"].append(self._message_to_dict(msg))
                    
                for msg in list(self._completed.values())[-100:]:  # Keep last 100
                    state["completed"].append(self._message_to_dict(msg))
                    
            with open(self.persist_path, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to persist state: {e}")
            
    def _load_state(self):
        """Load queue state from disk."""
        if not self.persist_path.exists():
            return
            
        try:
            with open(self.persist_path, 'r') as f:
                state = json.load(f)
                
            # Restore pending messages
            for msg_dict in state.get("pending", []):
                msg = self._dict_to_message(msg_dict)
                if msg:
                    self._queue.put((msg.priority.value, time.time(), msg))
                    self._message_hashes.add(self._generate_message_hash(msg.content))
                    
            # Restore processing messages as pending (they were interrupted)
            for msg_dict in state.get("processing", []):
                msg = self._dict_to_message(msg_dict)
                if msg:
                    msg.status = MessageStatus.PENDING
                    msg.processing_started = None
                    self._queue.put((msg.priority.value, time.time(), msg))
                    self._message_hashes.add(self._generate_message_hash(msg.content))
                    
            # Restore completed messages
            for msg_dict in state.get("completed", []):
                msg = self._dict_to_message(msg_dict)
                if msg:
                    self._completed[msg.id] = msg
                    
            self.logger.info(f"Loaded {self._queue.qsize()} pending messages from state")
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            
    def _message_to_dict(self, msg: TodoMessage) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": msg.id,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "priority": msg.priority.name,
            "status": msg.status.value,
            "result": msg.result,
            "error": msg.error,
            "processing_started": msg.processing_started.isoformat() if msg.processing_started else None,
            "processing_completed": msg.processing_completed.isoformat() if msg.processing_completed else None
        }
        
    def _dict_to_message(self, data: Dict[str, Any]) -> Optional[TodoMessage]:
        """Convert dictionary to message object."""
        try:
            return TodoMessage(
                id=data["id"],
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                priority=MessagePriority[data["priority"]],
                status=MessageStatus(data["status"]),
                result=data.get("result"),
                error=data.get("error"),
                processing_started=datetime.fromisoformat(data["processing_started"]) if data.get("processing_started") else None,
                processing_completed=datetime.fromisoformat(data["processing_completed"]) if data.get("processing_completed") else None
            )
        except Exception as e:
            self.logger.error(f"Failed to deserialize message: {e}")
            return None
            
    def clear_completed(self, keep_last: int = 100):
        """Clear old completed messages, keeping the most recent ones."""
        with self._lock:
            if len(self._completed) > keep_last:
                # Sort by completion time and keep most recent
                sorted_msgs = sorted(
                    self._completed.values(),
                    key=lambda m: m.processing_completed or m.timestamp,
                    reverse=True
                )
                self._completed = {m.id: m for m in sorted_msgs[:keep_last]}
                
        self._persist_state()


class TodoQueueMonitor:
    """Monitors and reports on queue health and performance."""
    
    def __init__(self, queue: ConcurrentTodoQueue):
        self.queue = queue
        self.logger = logging.getLogger(__name__)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue performance metrics."""
        status = self.queue.get_status()
        
        # Calculate average processing time
        completed = list(self.queue._completed.values())
        if completed:
            processing_times = []
            for msg in completed[-20:]:  # Last 20 messages
                if msg.processing_started and msg.processing_completed:
                    duration = (msg.processing_completed - msg.processing_started).total_seconds()
                    processing_times.append(duration)
                    
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        else:
            avg_processing_time = 0
            
        return {
            **status,
            "avg_processing_time": avg_processing_time,
            "queue_depth": status["pending"],
            "throughput": len(completed) / 3600 if completed else 0  # Messages per hour
        }
        
    async def monitor_loop(self, interval: float = 30):
        """Run monitoring loop."""
        while True:
            try:
                metrics = self.get_metrics()
                self.logger.info(f"Queue metrics: {metrics}")
                
                # Alert on high queue depth
                if metrics["queue_depth"] > 50:
                    self.logger.warning(f"High queue depth: {metrics['queue_depth']} messages pending")
                    
                # Alert on slow processing
                if metrics["avg_processing_time"] > 10:
                    self.logger.warning(f"Slow processing: {metrics['avg_processing_time']:.1f}s average")
                    
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                
            await asyncio.sleep(interval)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create queue
    queue = ConcurrentTodoQueue()
    
    # Add some test messages
    queue.enqueue("Add a task to review the documentation")
    queue.enqueue("Mark the API task as complete", MessagePriority.HIGH)
    queue.enqueue("Show me my tasks for today")
    
    # Simulate processing
    msg = queue.dequeue()
    if msg:
        print(f"Processing: {msg.content}")
        queue.complete(msg.id, result={"action": "added", "task": "Review documentation"})
        
    # Check status
    print(f"Queue status: {queue.get_status()}")