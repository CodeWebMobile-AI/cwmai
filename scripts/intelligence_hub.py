"""
Intelligence Hub Module

Central intelligence hub with event-driven architecture for cross-component learning.
Provides pattern recognition, causal relationship tracking, and real-time knowledge updates.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import weakref


class EventType(Enum):
    """Types of intelligence hub events."""
    DECISION_MADE = "decision_made"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    PATTERN_DETECTED = "pattern_detected"
    INSIGHT_GENERATED = "insight_generated"
    LEARNING_UPDATE = "learning_update"
    PERFORMANCE_METRIC = "performance_metric"
    SWARM_ANALYSIS = "swarm_analysis"
    AI_RESPONSE = "ai_response"
    STATE_CHANGE = "state_change"
    ERROR_OCCURRED = "error_occurred"


class InsightType(Enum):
    """Types of insights that can be generated."""
    CAUSAL_RELATIONSHIP = "causal_relationship"
    PATTERN_CORRELATION = "pattern_correlation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    FAILURE_PREDICTION = "failure_prediction"
    SUCCESS_PATTERN = "success_pattern"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    USER_BEHAVIOR = "user_behavior"


@dataclass
class IntelligenceEvent:
    """Represents an event in the intelligence hub."""
    id: str
    event_type: EventType
    source_component: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class Insight:
    """Represents a generated insight."""
    id: str
    insight_type: InsightType
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    supporting_events: List[str]  # Event IDs
    actionable_recommendations: List[str]
    timestamp: datetime
    source_analysis: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Pattern:
    """Represents a detected pattern."""
    id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    first_seen: datetime
    last_seen: datetime
    related_events: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelationship:
    """Represents a causal relationship between events."""
    cause_pattern: str
    effect_pattern: str
    strength: float  # 0.0 to 1.0
    confidence: float
    sample_size: int
    time_delay: float  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligenceHub:
    """Central intelligence hub for cross-component learning and knowledge sharing."""
    
    def __init__(self, 
                 max_events: int = 10000,
                 max_insights: int = 1000,
                 pattern_detection_window: int = 3600):
        """Initialize intelligence hub.
        
        Args:
            max_events: Maximum number of events to keep in memory
            max_insights: Maximum number of insights to keep
            pattern_detection_window: Time window for pattern detection (seconds)
        """
        self.max_events = max_events
        self.max_insights = max_insights
        self.pattern_detection_window = pattern_detection_window
        
        # Event storage and management
        self.events: deque = deque(maxlen=max_events)
        self.event_index: Dict[str, IntelligenceEvent] = {}
        self.events_by_type: Dict[EventType, List[str]] = defaultdict(list)
        self.events_by_component: Dict[str, List[str]] = defaultdict(list)
        
        # Insights and patterns
        self.insights: deque = deque(maxlen=max_insights)
        self.insight_index: Dict[str, Insight] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.causal_relationships: List[CausalRelationship] = []
        
        # Event subscribers
        self.event_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.component_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.pattern_subscribers: List[Callable] = []
        self.insight_subscribers: List[Callable] = []
        
        # Async coordination
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processor_task: Optional[asyncio.Task] = None
        self.pattern_analyzer_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            'total_events': 0,
            'events_processed': 0,
            'insights_generated': 0,
            'patterns_detected': 0,
            'relationships_discovered': 0,
            'avg_processing_time': 0.0
        }
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.IntelligenceHub")
        
        # State
        self.running = False
    
    async def start(self):
        """Start the intelligence hub."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting Intelligence Hub")
        
        # Start background processors
        self.processor_task = asyncio.create_task(self._event_processor())
        self.pattern_analyzer_task = asyncio.create_task(self._pattern_analyzer())
        
        self.logger.info("Intelligence Hub started successfully")
    
    async def stop(self):
        """Stop the intelligence hub."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping Intelligence Hub")
        
        # Cancel background tasks
        if self.processor_task and not self.processor_task.done():
            self.processor_task.cancel()
        if self.pattern_analyzer_task and not self.pattern_analyzer_task.done():
            self.pattern_analyzer_task.cancel()
        
        # Process remaining events
        await self._process_remaining_events()
        
        self.logger.info("Intelligence Hub stopped")
    
    async def emit_event(self, 
                        event_type: EventType, 
                        source_component: str, 
                        data: Dict[str, Any],
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Emit an event to the intelligence hub.
        
        Args:
            event_type: Type of event
            source_component: Component that generated the event
            data: Event data
            metadata: Optional metadata
            
        Returns:
            Event ID
        """
        event = IntelligenceEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            source_component=source_component,
            timestamp=datetime.now(timezone.utc),
            data=data,
            metadata=metadata or {}
        )
        
        # Add to queue for processing
        await self.processing_queue.put(event)
        self.metrics['total_events'] += 1
        
        self.logger.debug(f"Event emitted: {event_type.value} from {source_component}")
        return event.id
    
    async def _event_processor(self):
        """Background event processor."""
        while self.running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                start_time = time.time()
                await self._process_event(event)
                
                # Update metrics
                processing_time = time.time() - start_time
                self._update_processing_metrics(processing_time)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    async def _process_event(self, event: IntelligenceEvent):
        """Process a single event."""
        # Store event
        self.events.append(event)
        self.event_index[event.id] = event
        self.events_by_type[event.event_type].append(event.id)
        self.events_by_component[event.source_component].append(event.id)
        
        # Notify subscribers
        await self._notify_event_subscribers(event)
        
        # Trigger pattern analysis for relevant events
        if event.event_type in [EventType.DECISION_MADE, EventType.TASK_COMPLETED, 
                               EventType.TASK_FAILED, EventType.PERFORMANCE_METRIC]:
            await self._analyze_event_for_patterns(event)
        
        event.processed = True
        self.metrics['events_processed'] += 1
    
    async def _notify_event_subscribers(self, event: IntelligenceEvent):
        """Notify all relevant subscribers of an event."""
        try:
            # Notify event type subscribers
            for callback in self.event_subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event subscriber: {e}")
            
            # Notify component subscribers
            for callback in self.component_subscribers[event.source_component]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    self.logger.error(f"Error in component subscriber: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error notifying subscribers: {e}")
    
    async def _pattern_analyzer(self):
        """Background pattern analyzer."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._detect_patterns()
                await self._analyze_causal_relationships()
                await self._generate_insights()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in pattern analyzer: {e}")
    
    async def _analyze_event_for_patterns(self, event: IntelligenceEvent):
        """Analyze a new event for immediate pattern detection."""
        # Look for similar events in recent window
        window_start = event.timestamp - timedelta(seconds=self.pattern_detection_window)
        
        similar_events = []
        for stored_event in reversed(self.events):
            if stored_event.timestamp < window_start:
                break
            
            if (stored_event.event_type == event.event_type and 
                stored_event.source_component == event.source_component):
                similar_events.append(stored_event)
        
        # If we have enough similar events, look for patterns
        if len(similar_events) >= 3:
            await self._detect_sequence_pattern(similar_events)
    
    async def _detect_patterns(self):
        """Detect patterns in recent events."""
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(seconds=self.pattern_detection_window)
        
        # Get recent events
        recent_events = [
            event for event in self.events 
            if event.timestamp >= window_start
        ]
        
        if len(recent_events) < 5:
            return
        
        # Detect frequency patterns
        await self._detect_frequency_patterns(recent_events)
        
        # Detect sequence patterns
        await self._detect_sequence_patterns(recent_events)
        
        # Detect correlation patterns
        await self._detect_correlation_patterns(recent_events)
    
    async def _detect_frequency_patterns(self, events: List[IntelligenceEvent]):
        """Detect frequency-based patterns."""
        # Count event types per component
        component_event_counts = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            component_event_counts[event.source_component][event.event_type] += 1
        
        # Find high-frequency patterns
        for component, event_counts in component_event_counts.items():
            for event_type, count in event_counts.items():
                if count >= 5:  # Threshold for pattern detection
                    pattern_id = f"freq_{component}_{event_type.value}"
                    
                    if pattern_id not in self.patterns:
                        pattern = Pattern(
                            id=pattern_id,
                            pattern_type="frequency",
                            description=f"High frequency of {event_type.value} events in {component}",
                            frequency=count,
                            confidence=min(count / 10, 1.0),
                            first_seen=events[0].timestamp,
                            last_seen=events[-1].timestamp,
                            related_events=[e.id for e in events if e.source_component == component and e.event_type == event_type]
                        )
                        
                        self.patterns[pattern_id] = pattern
                        self.metrics['patterns_detected'] += 1
                        await self._notify_pattern_subscribers(pattern)
    
    async def _detect_sequence_pattern(self, events: List[IntelligenceEvent]):
        """Detect sequence patterns in events."""
        if len(events) < 3:
            return
        
        # Look for temporal sequences
        events_sorted = sorted(events, key=lambda e: e.timestamp)
        
        # Simple sequence detection: A -> B -> C pattern
        for i in range(len(events_sorted) - 2):
            e1, e2, e3 = events_sorted[i], events_sorted[i+1], events_sorted[i+2]
            
            # Check if events happen within reasonable time windows
            time_diff1 = (e2.timestamp - e1.timestamp).total_seconds()
            time_diff2 = (e3.timestamp - e2.timestamp).total_seconds()
            
            if 1 <= time_diff1 <= 300 and 1 <= time_diff2 <= 300:  # 1 second to 5 minutes
                pattern_id = f"seq_{e1.event_type.value}_{e2.event_type.value}_{e3.event_type.value}"
                
                if pattern_id not in self.patterns:
                    pattern = Pattern(
                        id=pattern_id,
                        pattern_type="sequence",
                        description=f"Sequence: {e1.event_type.value} -> {e2.event_type.value} -> {e3.event_type.value}",
                        frequency=1,
                        confidence=0.7,
                        first_seen=e1.timestamp,
                        last_seen=e3.timestamp,
                        related_events=[e1.id, e2.id, e3.id]
                    )
                    
                    self.patterns[pattern_id] = pattern
                    self.metrics['patterns_detected'] += 1
                    await self._notify_pattern_subscribers(pattern)
                else:
                    # Update existing pattern
                    self.patterns[pattern_id].frequency += 1
                    self.patterns[pattern_id].last_seen = e3.timestamp
                    self.patterns[pattern_id].confidence = min(self.patterns[pattern_id].frequency / 10, 1.0)
    
    async def _detect_sequence_patterns(self, events: List[IntelligenceEvent]):
        """Detect sequence patterns in a list of events."""
        # Group events by component
        component_events = defaultdict(list)
        for event in events:
            component_events[event.source_component].append(event)
        
        # Analyze sequences within each component
        for component, comp_events in component_events.items():
            comp_events.sort(key=lambda e: e.timestamp)
            await self._detect_sequence_pattern(comp_events)
    
    async def _detect_correlation_patterns(self, events: List[IntelligenceEvent]):
        """Detect correlation patterns between different event types."""
        # Group events by type
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event.event_type].append(event)
        
        # Look for correlations between different event types
        event_types = list(events_by_type.keys())
        for i in range(len(event_types)):
            for j in range(i + 1, len(event_types)):
                type1, type2 = event_types[i], event_types[j]
                events1, events2 = events_by_type[type1], events_by_type[type2]
                
                # Calculate temporal correlation
                correlation = await self._calculate_temporal_correlation(events1, events2)
                if correlation > 0.7:
                    pattern_id = f"corr_{type1.value}_{type2.value}"
                    
                    pattern = Pattern(
                        id=pattern_id,
                        pattern_type="correlation",
                        description=f"Correlation between {type1.value} and {type2.value}",
                        frequency=len(events1) + len(events2),
                        confidence=correlation,
                        first_seen=min(events1[0].timestamp, events2[0].timestamp),
                        last_seen=max(events1[-1].timestamp, events2[-1].timestamp),
                        related_events=[e.id for e in events1 + events2],
                        metadata={'correlation_score': correlation}
                    )
                    
                    self.patterns[pattern_id] = pattern
                    self.metrics['patterns_detected'] += 1
                    await self._notify_pattern_subscribers(pattern)
    
    async def _calculate_temporal_correlation(self, events1: List[IntelligenceEvent], 
                                           events2: List[IntelligenceEvent]) -> float:
        """Calculate temporal correlation between two event lists."""
        if not events1 or not events2:
            return 0.0
        
        # Simple correlation: count events that happen within time windows
        correlation_count = 0
        total_pairs = 0
        
        for e1 in events1:
            for e2 in events2:
                total_pairs += 1
                time_diff = abs((e1.timestamp - e2.timestamp).total_seconds())
                if time_diff <= 300:  # Within 5 minutes
                    correlation_count += 1
        
        return correlation_count / total_pairs if total_pairs > 0 else 0.0
    
    async def _analyze_causal_relationships(self):
        """Analyze causal relationships between patterns."""
        patterns = list(self.patterns.values())
        
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j:
                    cause_pattern = patterns[i]
                    effect_pattern = patterns[j]
                    
                    # Analyze if pattern i might cause pattern j
                    relationship = await self._analyze_causality(cause_pattern, effect_pattern)
                    if relationship and relationship.strength > 0.6:
                        self.causal_relationships.append(relationship)
                        self.metrics['relationships_discovered'] += 1
    
    async def _analyze_causality(self, cause_pattern: Pattern, effect_pattern: Pattern) -> Optional[CausalRelationship]:
        """Analyze potential causality between two patterns."""
        # Get events for both patterns
        cause_events = [self.event_index[eid] for eid in cause_pattern.related_events if eid in self.event_index]
        effect_events = [self.event_index[eid] for eid in effect_pattern.related_events if eid in self.event_index]
        
        if not cause_events or not effect_events:
            return None
        
        # Analyze temporal precedence and correlation
        causal_pairs = 0
        total_comparisons = 0
        time_delays = []
        
        for cause_event in cause_events:
            for effect_event in effect_events:
                total_comparisons += 1
                time_diff = (effect_event.timestamp - cause_event.timestamp).total_seconds()
                
                # Effect should come after cause within reasonable timeframe
                if 0 < time_diff <= 3600:  # 1 hour window
                    causal_pairs += 1
                    time_delays.append(time_diff)
        
        if total_comparisons == 0:
            return None
        
        strength = causal_pairs / total_comparisons
        avg_delay = sum(time_delays) / len(time_delays) if time_delays else 0
        
        if strength > 0.3:  # Minimum threshold for potential causality
            return CausalRelationship(
                cause_pattern=cause_pattern.id,
                effect_pattern=effect_pattern.id,
                strength=strength,
                confidence=min(strength * 1.5, 1.0),
                sample_size=total_comparisons,
                time_delay=avg_delay
            )
        
        return None
    
    async def _generate_insights(self):
        """Generate insights from patterns and relationships."""
        # Generate insights from patterns
        for pattern in self.patterns.values():
            if pattern.confidence > 0.8 and pattern.frequency > 10:
                insight = await self._generate_pattern_insight(pattern)
                if insight:
                    await self._add_insight(insight)
        
        # Generate insights from causal relationships
        for relationship in self.causal_relationships:
            if relationship.confidence > 0.7:
                insight = await self._generate_causal_insight(relationship)
                if insight:
                    await self._add_insight(insight)
    
    async def _generate_pattern_insight(self, pattern: Pattern) -> Optional[Insight]:
        """Generate insight from a pattern."""
        if pattern.pattern_type == "frequency":
            return Insight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.PERFORMANCE_OPTIMIZATION,
                title=f"High Activity Pattern Detected",
                description=f"Component {pattern.id.split('_')[1]} shows high frequency of {pattern.id.split('_')[2]} events. This could indicate high workload or potential inefficiency.",
                confidence=pattern.confidence,
                supporting_events=pattern.related_events,
                actionable_recommendations=[
                    "Monitor resource usage in this component",
                    "Consider optimizing the frequency of these operations",
                    "Investigate if this pattern indicates normal behavior or issues"
                ],
                timestamp=datetime.now(timezone.utc),
                source_analysis="pattern_analyzer"
            )
        elif pattern.pattern_type == "sequence":
            return Insight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.SUCCESS_PATTERN,
                title=f"Sequence Pattern Identified",
                description=f"Consistent sequence pattern detected: {pattern.description}. This could be optimized or automated.",
                confidence=pattern.confidence,
                supporting_events=pattern.related_events,
                actionable_recommendations=[
                    "Consider automating this sequence",
                    "Optimize the timing between steps",
                    "Monitor for deviations from this pattern"
                ],
                timestamp=datetime.now(timezone.utc),
                source_analysis="pattern_analyzer"
            )
        
        return None
    
    async def _generate_causal_insight(self, relationship: CausalRelationship) -> Optional[Insight]:
        """Generate insight from a causal relationship."""
        cause_pattern = self.patterns.get(relationship.cause_pattern)
        effect_pattern = self.patterns.get(relationship.effect_pattern)
        
        if not cause_pattern or not effect_pattern:
            return None
        
        return Insight(
            id=str(uuid.uuid4()),
            insight_type=InsightType.CAUSAL_RELATIONSHIP,
            title=f"Causal Relationship Discovered",
            description=f"Pattern '{cause_pattern.description}' appears to cause '{effect_pattern.description}' with {relationship.strength:.2f} strength and average delay of {relationship.time_delay:.1f} seconds.",
            confidence=relationship.confidence,
            supporting_events=cause_pattern.related_events + effect_pattern.related_events,
            actionable_recommendations=[
                "Use this relationship for predictive optimization",
                "Monitor cause patterns to anticipate effects",
                "Consider if this causality can be optimized or controlled"
            ],
            timestamp=datetime.now(timezone.utc),
            source_analysis="causality_analyzer",
            metadata={
                'cause_pattern': relationship.cause_pattern,
                'effect_pattern': relationship.effect_pattern,
                'strength': relationship.strength,
                'time_delay': relationship.time_delay
            }
        )
    
    async def _add_insight(self, insight: Insight):
        """Add insight to the hub."""
        self.insights.append(insight)
        self.insight_index[insight.id] = insight
        self.metrics['insights_generated'] += 1
        
        await self._notify_insight_subscribers(insight)
        
        self.logger.info(f"Generated insight: {insight.title}")
    
    async def _notify_pattern_subscribers(self, pattern: Pattern):
        """Notify pattern subscribers."""
        for callback in self.pattern_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(pattern)
                else:
                    callback(pattern)
            except Exception as e:
                self.logger.error(f"Error in pattern subscriber: {e}")
    
    async def _notify_insight_subscribers(self, insight: Insight):
        """Notify insight subscribers."""
        for callback in self.insight_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(insight)
                else:
                    callback(insight)
            except Exception as e:
                self.logger.error(f"Error in insight subscriber: {e}")
    
    def _update_processing_metrics(self, processing_time: float):
        """Update processing metrics."""
        total_time = self.metrics['avg_processing_time'] * (self.metrics['events_processed'] - 1)
        self.metrics['avg_processing_time'] = (total_time + processing_time) / self.metrics['events_processed']
    
    async def _process_remaining_events(self):
        """Process remaining events in queue."""
        while not self.processing_queue.empty():
            try:
                event = self.processing_queue.get_nowait()
                await self._process_event(event)
            except asyncio.QueueEmpty:
                break
    
    # Public API methods
    
    def subscribe_to_events(self, event_type: EventType, callback: Callable):
        """Subscribe to specific event types."""
        self.event_subscribers[event_type].append(callback)
    
    def subscribe_to_component(self, component: str, callback: Callable):
        """Subscribe to events from specific components."""
        self.component_subscribers[component].append(callback)
    
    def subscribe_to_patterns(self, callback: Callable):
        """Subscribe to pattern detection."""
        self.pattern_subscribers.append(callback)
    
    def subscribe_to_insights(self, callback: Callable):
        """Subscribe to insight generation."""
        self.insight_subscribers.append(callback)
    
    def get_recent_events(self, hours: int = 1) -> List[IntelligenceEvent]:
        """Get recent events within specified hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [event for event in self.events if event.timestamp >= cutoff]
    
    def get_insights(self, insight_type: Optional[InsightType] = None) -> List[Insight]:
        """Get insights, optionally filtered by type."""
        if insight_type:
            return [insight for insight in self.insights if insight.insight_type == insight_type]
        return list(self.insights)
    
    def get_patterns(self, pattern_type: Optional[str] = None) -> List[Pattern]:
        """Get patterns, optionally filtered by type."""
        if pattern_type:
            return [pattern for pattern in self.patterns.values() if pattern.pattern_type == pattern_type]
        return list(self.patterns.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get hub metrics."""
        return {
            **self.metrics,
            'queue_size': self.processing_queue.qsize(),
            'total_patterns': len(self.patterns),
            'total_insights': len(self.insights),
            'total_relationships': len(self.causal_relationships)
        }


# Global instance
_global_intelligence_hub: Optional[IntelligenceHub] = None


async def get_intelligence_hub() -> IntelligenceHub:
    """Get or create global intelligence hub."""
    global _global_intelligence_hub
    if _global_intelligence_hub is None:
        _global_intelligence_hub = IntelligenceHub()
        await _global_intelligence_hub.start()
    return _global_intelligence_hub


# Convenience functions
async def emit_intelligence_event(event_type: EventType, 
                                source_component: str, 
                                data: Dict[str, Any],
                                metadata: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to emit intelligence events."""
    hub = await get_intelligence_hub()
    return await hub.emit_event(event_type, source_component, data, metadata)