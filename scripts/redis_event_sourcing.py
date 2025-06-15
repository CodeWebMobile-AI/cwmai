"""
Redis Event Sourcing Implementation

Event sourcing patterns for distributed worker intelligence with Redis Streams,
providing event replay, state reconstruction, and temporal analytics capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from scripts.redis_integration.redis_client import get_redis_client
from scripts.redis_intelligence_hub import IntelligenceEvent, EventType, EventPriority
from scripts.redis_lockfree_adapter import create_lockfree_state_manager


class EventStoreError(Exception):
    """Event store specific errors."""
    pass


class SnapshotStrategy(Enum):
    """Snapshot creation strategies."""
    TIME_BASED = "time_based"          # Create snapshots every N hours
    EVENT_COUNT = "event_count"        # Create snapshots every N events
    SIZE_BASED = "size_based"          # Create snapshots when stream size exceeds threshold
    MANUAL = "manual"                  # Manual snapshot creation only


@dataclass
class EventSnapshot:
    """Event store snapshot."""
    snapshot_id: str
    entity_id: str
    snapshot_time: datetime
    event_sequence: int
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'entity_id': self.entity_id,
            'snapshot_time': self.snapshot_time.isoformat(),
            'event_sequence': self.event_sequence,
            'state': json.dumps(self.state),
            'metadata': json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventSnapshot':
        """Create snapshot from dictionary."""
        return cls(
            snapshot_id=data['snapshot_id'],
            entity_id=data['entity_id'],
            snapshot_time=datetime.fromisoformat(data['snapshot_time']),
            event_sequence=data['event_sequence'],
            state=json.loads(data['state']),
            metadata=json.loads(data['metadata'])
        )


@dataclass
class EventProjection:
    """Event projection configuration."""
    projection_id: str
    name: str
    event_types: List[EventType]
    projection_function: Callable
    state_key: str
    enabled: bool = True
    last_processed_sequence: int = 0


class RedisEventStore:
    """Redis-based event store with event sourcing capabilities."""
    
    def __init__(self,
                 store_id: str = None,
                 snapshot_strategy: SnapshotStrategy = SnapshotStrategy.EVENT_COUNT,
                 snapshot_frequency: int = 100,
                 max_events_in_memory: int = 1000,
                 enable_projections: bool = True):
        """Initialize Redis Event Store.
        
        Args:
            store_id: Unique store identifier
            snapshot_strategy: Strategy for creating snapshots
            snapshot_frequency: Frequency for snapshot creation
            max_events_in_memory: Maximum events to keep in memory
            enable_projections: Enable real-time projections
        """
        self.store_id = store_id or f"event_store_{uuid.uuid4().hex[:8]}"
        self.snapshot_strategy = snapshot_strategy
        self.snapshot_frequency = snapshot_frequency
        self.max_events_in_memory = max_events_in_memory
        self.enable_projections = enable_projections
        
        self.logger = logging.getLogger(f"{__name__}.RedisEventStore")
        
        # Redis components
        self.redis_client = None
        self.state_manager = None
        
        # Stream keys
        self.event_stream_key = f"events:{self.store_id}"
        self.snapshot_key = f"snapshots:{self.store_id}"
        self.projection_key = f"projections:{self.store_id}"
        
        # Event store state
        self._event_sequence = 0
        self._projections: Dict[str, EventProjection] = {}
        self._projection_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # In-memory caching
        self._recent_events: List[IntelligenceEvent] = []
        self._entity_states: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self._metrics = {
            'events_stored': 0,
            'events_replayed': 0,
            'snapshots_created': 0,
            'projections_updated': 0,
            'average_write_time_ms': 0.0,
            'average_read_time_ms': 0.0
        }
    
    async def initialize(self):
        """Initialize event store components."""
        try:
            self.logger.info(f"Initializing Redis Event Store: {self.store_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            self.state_manager = create_lockfree_state_manager(f"event_store_{self.store_id}")
            await self.state_manager.initialize()
            
            # Initialize event stream
            await self._initialize_event_stream()
            
            # Load existing event sequence
            await self._load_event_sequence()
            
            # Initialize projections
            if self.enable_projections:
                await self._initialize_projections()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info(f"Event Store {self.store_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Event Store: {e}")
            raise
    
    async def _initialize_event_stream(self):
        """Initialize Redis stream for events."""
        try:
            # Create stream with initial entry if it doesn't exist
            try:
                await self.redis_client.xadd(
                    self.event_stream_key,
                    {'init': 'true'},
                    maxlen=self.max_events_in_memory * 10  # Keep more events for replay
                )
            except Exception:
                pass  # Stream might already exist
            
        except Exception as e:
            self.logger.error(f"Error initializing event stream: {e}")
            raise
    
    async def _load_event_sequence(self):
        """Load current event sequence number."""
        try:
            # Get latest event sequence from Redis
            sequence = await self.state_manager.get(
                f"event_store.{self.store_id}.sequence", 
                0
            )
            self._event_sequence = sequence
            
            self.logger.debug(f"Loaded event sequence: {self._event_sequence}")
            
        except Exception as e:
            self.logger.error(f"Error loading event sequence: {e}")
            self._event_sequence = 0
    
    async def append_event(self, event: IntelligenceEvent) -> int:
        """Append event to event store.
        
        Args:
            event: Intelligence event to store
            
        Returns:
            Event sequence number
        """
        start_time = time.time()
        
        try:
            # Assign sequence number
            self._event_sequence += 1
            event_sequence = self._event_sequence
            
            # Prepare event data for storage
            event_data = event.to_dict()
            event_data['sequence'] = event_sequence
            event_data['store_id'] = self.store_id
            
            # Store event in Redis stream
            stream_id = await self.redis_client.xadd(
                self.event_stream_key,
                event_data
            )
            
            # Update sequence in state manager
            await self.state_manager.update(
                f"event_store.{self.store_id}.sequence",
                event_sequence,
                distributed=True
            )
            
            # Add to in-memory cache
            self._recent_events.append(event)
            if len(self._recent_events) > self.max_events_in_memory:
                self._recent_events.pop(0)
            
            # Update metrics
            self._metrics['events_stored'] += 1
            write_time = (time.time() - start_time) * 1000
            self._metrics['average_write_time_ms'] = (
                (self._metrics['average_write_time_ms'] * (self._metrics['events_stored'] - 1) + write_time) 
                / self._metrics['events_stored']
            )
            
            # Check if snapshot should be created
            await self._check_snapshot_creation(event, event_sequence)
            
            # Update projections
            if self.enable_projections:
                await self._update_projections(event, event_sequence)
            
            self.logger.debug(f"Stored event {event.event_id} with sequence {event_sequence}")
            return event_sequence
            
        except Exception as e:
            self.logger.error(f"Error appending event: {e}")
            raise EventStoreError(f"Failed to append event: {e}")
    
    async def get_events(self, 
                        entity_id: str = None,
                        from_sequence: int = 0,
                        to_sequence: int = None,
                        event_types: List[EventType] = None,
                        limit: int = None) -> AsyncGenerator[IntelligenceEvent, None]:
        """Get events from event store with filtering.
        
        Args:
            entity_id: Filter by entity (worker) ID
            from_sequence: Start sequence number
            to_sequence: End sequence number
            event_types: Filter by event types
            limit: Maximum number of events to return
            
        Yields:
            IntelligenceEvent objects
        """
        start_time = time.time()
        events_read = 0
        
        try:
            # Determine read strategy
            if from_sequence == 0 and not to_sequence and not entity_id and not event_types:
                # Read from in-memory cache if available
                for event in self._recent_events:
                    if limit and events_read >= limit:
                        break
                    yield event
                    events_read += 1
            else:
                # Read from Redis stream
                async for event in self._read_events_from_stream(
                    from_sequence, to_sequence, entity_id, event_types, limit
                ):
                    yield event
                    events_read += 1
            
            # Update metrics
            if events_read > 0:
                read_time = (time.time() - start_time) * 1000
                self._metrics['events_replayed'] += events_read
                self._metrics['average_read_time_ms'] = (
                    (self._metrics['average_read_time_ms'] * (self._metrics['events_replayed'] - events_read) + read_time)
                    / self._metrics['events_replayed']
                )
            
        except Exception as e:
            self.logger.error(f"Error getting events: {e}")
            raise EventStoreError(f"Failed to get events: {e}")
    
    async def _read_events_from_stream(self,
                                     from_sequence: int = 0,
                                     to_sequence: int = None,
                                     entity_id: str = None,
                                     event_types: List[EventType] = None,
                                     limit: int = None) -> AsyncGenerator[IntelligenceEvent, None]:
        """Read events from Redis stream with filtering."""
        try:
            events_yielded = 0
            
            # Read from Redis stream
            # Start from beginning or specific sequence
            start_id = str(from_sequence) if from_sequence > 0 else '0'
            
            # Read stream in chunks
            chunk_size = 100
            current_id = start_id
            
            while True:
                # Read chunk from stream
                messages = await self.redis_client.xread(
                    {self.event_stream_key: current_id},
                    count=chunk_size,
                    block=None
                )
                
                if not messages:
                    break
                
                stream_name, msgs = messages[0]
                
                if not msgs:
                    break
                
                for msg_id, fields in msgs:
                    try:
                        # Skip init message
                        if fields.get('init') == 'true':
                            current_id = msg_id
                            continue
                        
                        # Convert to event
                        event = IntelligenceEvent.from_dict(fields)
                        event_sequence = int(fields.get('sequence', 0))
                        
                        # Apply filters
                        if to_sequence and event_sequence > to_sequence:
                            return
                        
                        if entity_id and event.worker_id != entity_id:
                            current_id = msg_id
                            continue
                        
                        if event_types and event.event_type not in event_types:
                            current_id = msg_id
                            continue
                        
                        yield event
                        events_yielded += 1
                        
                        if limit and events_yielded >= limit:
                            return
                        
                        current_id = msg_id
                        
                    except Exception as e:
                        self.logger.error(f"Error processing event from stream: {e}")
                        current_id = msg_id
                        continue
                
                # If we got less than chunk_size, we've reached the end
                if len(msgs) < chunk_size:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error reading events from stream: {e}")
            raise
    
    async def replay_events(self, entity_id: str, from_sequence: int = 0) -> Dict[str, Any]:
        """Replay events to reconstruct entity state.
        
        Args:
            entity_id: Entity (worker) ID to replay
            from_sequence: Start sequence for replay
            
        Returns:
            Reconstructed entity state
        """
        try:
            self.logger.info(f"Replaying events for entity {entity_id} from sequence {from_sequence}")
            
            # Check for existing snapshot
            snapshot = await self._get_latest_snapshot(entity_id)
            
            if snapshot and snapshot.event_sequence >= from_sequence:
                # Start from snapshot
                entity_state = snapshot.state.copy()
                replay_from = snapshot.event_sequence + 1
                self.logger.debug(f"Starting replay from snapshot at sequence {snapshot.event_sequence}")
            else:
                # Start from beginning
                entity_state = {}
                replay_from = from_sequence
            
            # Replay events
            events_replayed = 0
            async for event in self.get_events(
                entity_id=entity_id,
                from_sequence=replay_from
            ):
                # Apply event to state
                entity_state = await self._apply_event_to_state(entity_state, event)
                events_replayed += 1
            
            self.logger.info(f"Replayed {events_replayed} events for entity {entity_id}")
            
            # Cache state
            self._entity_states[entity_id] = entity_state
            
            return entity_state
            
        except Exception as e:
            self.logger.error(f"Error replaying events for entity {entity_id}: {e}")
            raise EventStoreError(f"Failed to replay events: {e}")
    
    async def create_snapshot(self, entity_id: str, force: bool = False) -> EventSnapshot:
        """Create snapshot of entity state.
        
        Args:
            entity_id: Entity ID to snapshot
            force: Force snapshot creation regardless of strategy
            
        Returns:
            Created snapshot
        """
        try:
            # Get current state
            if entity_id not in self._entity_states:
                entity_state = await self.replay_events(entity_id)
            else:
                entity_state = self._entity_states[entity_id]
            
            # Create snapshot
            snapshot = EventSnapshot(
                snapshot_id=str(uuid.uuid4()),
                entity_id=entity_id,
                snapshot_time=datetime.now(timezone.utc),
                event_sequence=self._event_sequence,
                state=entity_state,
                metadata={
                    'store_id': self.store_id,
                    'snapshot_strategy': self.snapshot_strategy.value,
                    'created_by': 'event_store'
                }
            )
            
            # Store snapshot in Redis
            await self.redis_client.hset(
                f"{self.snapshot_key}:{entity_id}",
                snapshot.snapshot_id,
                json.dumps(snapshot.to_dict())
            )
            
            # Update latest snapshot reference
            await self.redis_client.set(
                f"{self.snapshot_key}:{entity_id}:latest",
                snapshot.snapshot_id
            )
            
            self._metrics['snapshots_created'] += 1
            
            self.logger.info(f"Created snapshot {snapshot.snapshot_id} for entity {entity_id}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error creating snapshot for entity {entity_id}: {e}")
            raise EventStoreError(f"Failed to create snapshot: {e}")
    
    async def _get_latest_snapshot(self, entity_id: str) -> Optional[EventSnapshot]:
        """Get latest snapshot for entity."""
        try:
            # Get latest snapshot ID
            snapshot_id = await self.redis_client.get(f"{self.snapshot_key}:{entity_id}:latest")
            
            if not snapshot_id:
                return None
            
            # Get snapshot data
            snapshot_data = await self.redis_client.hget(
                f"{self.snapshot_key}:{entity_id}",
                snapshot_id
            )
            
            if not snapshot_data:
                return None
            
            return EventSnapshot.from_dict(json.loads(snapshot_data))
            
        except Exception as e:
            self.logger.error(f"Error getting latest snapshot for entity {entity_id}: {e}")
            return None
    
    async def _apply_event_to_state(self, state: Dict[str, Any], event: IntelligenceEvent) -> Dict[str, Any]:
        """Apply event to entity state."""
        try:
            # Create copy of state
            new_state = state.copy()
            
            # Apply event based on type
            if event.event_type == EventType.WORKER_REGISTRATION:
                new_state.update({
                    'worker_id': event.worker_id,
                    'registration_time': event.timestamp.isoformat(),
                    'capabilities': event.data.get('capabilities', []),
                    'status': 'registered'
                })
            
            elif event.event_type == EventType.WORKER_HEARTBEAT:
                new_state.update({
                    'last_heartbeat': event.timestamp.isoformat(),
                    'status': 'active'
                })
                if 'performance' in event.data:
                    new_state['performance'] = event.data['performance']
            
            elif event.event_type == EventType.TASK_ASSIGNMENT:
                if 'assigned_tasks' not in new_state:
                    new_state['assigned_tasks'] = []
                new_state['assigned_tasks'].append({
                    'task_id': event.data.get('task_id'),
                    'assigned_time': event.timestamp.isoformat(),
                    'task_type': event.data.get('task_type'),
                    'status': 'assigned'
                })
            
            elif event.event_type == EventType.TASK_COMPLETION:
                task_id = event.data.get('task_id')
                if 'assigned_tasks' in new_state:
                    for task in new_state['assigned_tasks']:
                        if task.get('task_id') == task_id:
                            task.update({
                                'status': 'completed',
                                'completion_time': event.timestamp.isoformat(),
                                'result': event.data.get('result')
                            })
                            break
                
                # Update completion stats
                if 'task_stats' not in new_state:
                    new_state['task_stats'] = {'completed': 0, 'failed': 0}
                new_state['task_stats']['completed'] += 1
            
            elif event.event_type == EventType.PERFORMANCE_METRIC:
                if 'performance_history' not in new_state:
                    new_state['performance_history'] = []
                
                new_state['performance_history'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'metric_type': event.data.get('metric_type'),
                    'value': event.data.get('value'),
                    'metadata': event.data.get('metadata', {})
                })
                
                # Keep only recent metrics
                if len(new_state['performance_history']) > 100:
                    new_state['performance_history'] = new_state['performance_history'][-100:]
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Error applying event to state: {e}")
            return state
    
    async def _check_snapshot_creation(self, event: IntelligenceEvent, event_sequence: int):
        """Check if snapshot should be created based on strategy."""
        try:
            should_create = False
            
            if self.snapshot_strategy == SnapshotStrategy.EVENT_COUNT:
                should_create = (event_sequence % self.snapshot_frequency) == 0
            
            elif self.snapshot_strategy == SnapshotStrategy.TIME_BASED:
                # Check if enough time has passed since last snapshot
                last_snapshot = await self._get_latest_snapshot(event.worker_id)
                if not last_snapshot:
                    should_create = True
                else:
                    time_diff = datetime.now(timezone.utc) - last_snapshot.snapshot_time
                    should_create = time_diff.total_seconds() >= (self.snapshot_frequency * 3600)
            
            if should_create:
                await self.create_snapshot(event.worker_id)
                
        except Exception as e:
            self.logger.error(f"Error checking snapshot creation: {e}")
    
    def register_projection(self, projection: EventProjection):
        """Register event projection.
        
        Args:
            projection: Event projection configuration
        """
        self._projections[projection.projection_id] = projection
        self.logger.info(f"Registered projection: {projection.name}")
    
    async def _initialize_projections(self):
        """Initialize projection processing."""
        try:
            # Load projection states
            for projection_id, projection in self._projections.items():
                last_sequence = await self.state_manager.get(
                    f"projections.{self.store_id}.{projection_id}.last_sequence",
                    0
                )
                projection.last_processed_sequence = last_sequence
            
            # Start projection processors
            for projection in self._projections.values():
                if projection.enabled:
                    task = asyncio.create_task(self._projection_processor(projection))
                    self._projection_tasks.append(task)
            
        except Exception as e:
            self.logger.error(f"Error initializing projections: {e}")
    
    async def _projection_processor(self, projection: EventProjection):
        """Process events for projection."""
        while not self._shutdown:
            try:
                # Process new events for this projection
                async for event in self.get_events(
                    from_sequence=projection.last_processed_sequence + 1,
                    event_types=projection.event_types,
                    limit=100
                ):
                    try:
                        # Apply projection function
                        if asyncio.iscoroutinefunction(projection.projection_function):
                            await projection.projection_function(event, projection.state_key)
                        else:
                            projection.projection_function(event, projection.state_key)
                        
                        # Update last processed sequence
                        projection.last_processed_sequence = self._event_sequence
                        await self.state_manager.update(
                            f"projections.{self.store_id}.{projection.projection_id}.last_sequence",
                            projection.last_processed_sequence
                        )
                        
                        self._metrics['projections_updated'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error in projection {projection.name}: {e}")
                
                await asyncio.sleep(1)  # Check for new events every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in projection processor {projection.name}: {e}")
                await asyncio.sleep(5)
    
    async def _update_projections(self, event: IntelligenceEvent, event_sequence: int):
        """Update projections with new event."""
        try:
            for projection in self._projections.values():
                if not projection.enabled:
                    continue
                
                if event.event_type in projection.event_types:
                    try:
                        if asyncio.iscoroutinefunction(projection.projection_function):
                            await projection.projection_function(event, projection.state_key)
                        else:
                            projection.projection_function(event, projection.state_key)
                        
                        self._metrics['projections_updated'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error updating projection {projection.name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error updating projections: {e}")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        pass  # Placeholder for background tasks like cleanup, compaction, etc.
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event store metrics."""
        return {
            'store_id': self.store_id,
            'event_sequence': self._event_sequence,
            'recent_events_cached': len(self._recent_events),
            'entity_states_cached': len(self._entity_states),
            'projections_registered': len(self._projections),
            'performance': self._metrics.copy()
        }
    
    async def shutdown(self):
        """Shutdown event store."""
        self.logger.info(f"Shutting down Event Store: {self.store_id}")
        self._shutdown = True
        
        # Stop projection tasks
        for task in self._projection_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info(f"Event Store {self.store_id} shutdown complete")


# Global event store instance
_global_event_store: Optional[RedisEventStore] = None


async def get_event_store(**kwargs) -> RedisEventStore:
    """Get global event store instance."""
    global _global_event_store
    
    if _global_event_store is None:
        _global_event_store = RedisEventStore(**kwargs)
        await _global_event_store.initialize()
    
    return _global_event_store


async def create_event_store(**kwargs) -> RedisEventStore:
    """Create new event store instance."""
    store = RedisEventStore(**kwargs)
    await store.initialize()
    return store