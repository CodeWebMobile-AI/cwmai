"""
Research Scheduler - Adaptive scheduling system for intelligent research timing.

This module manages when research should be conducted based on system needs,
performance indicators, and resource availability. It implements both scheduled
and event-driven research triggers.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading
import queue
from enum import Enum


class ResearchJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles ResearchPriority enums."""
    
    def default(self, obj):
        if isinstance(obj, ResearchPriority):
            return obj.value
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ResearchPriority(Enum):
    """Research priority levels."""
    IMMEDIATE = "immediate"
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    def __str__(self):
        """Return the enum value as string for JSON serialization."""
        return self.value
    
    def __json__(self):
        """Custom JSON serialization method."""
        return self.value


class ResearchScheduler:
    """Adaptive scheduling system for research execution."""
    
    def __init__(self, state_manager=None, knowledge_store=None):
        self.state_manager = state_manager
        self.knowledge_store = knowledge_store
        
        # Research intervals in seconds
        self.base_intervals = {
            'critical_performance': 2 * 3600,      # Every 2 hours
            'task_analysis': 4 * 3600,             # Every 4 hours
            'security_updates': 6 * 3600,          # Every 6 hours
            'technology_trends': 8 * 3600,         # Every 8 hours
            'market_opportunities': 24 * 3600,     # Daily
            'portfolio_review': 48 * 3600          # Every 2 days
        }
        
        # Current intervals (may be adapted)
        self.current_intervals = self.base_intervals.copy()
        
        # Schedule tracking
        self.last_research_times = {}
        self.research_queue = queue.PriorityQueue()
        self.active_research = {}
        
        # Event triggers
        self.event_triggers = {
            'performance_drop': self._on_performance_drop,
            'task_failure_spike': self._on_task_failure_spike,
            'claude_failure': self._on_claude_failure,
            'security_alert': self._on_security_alert,
            'system_error': self._on_system_error
        }
        
        # Performance tracking for adaptive scheduling
        self.performance_history = []
        self.research_effectiveness = {}
        
        # Resource limits
        self.max_concurrent_research = 3
        self.research_budget = {
            'api_calls_per_hour': 100,
            'max_research_duration': 1800  # 30 minutes
        }
        
        # Current usage tracking
        self.current_usage = {
            'api_calls_this_hour': 0,
            'hour_reset_time': datetime.now().replace(minute=0, second=0)
        }
        
    def should_research(self, research_type: str, context: Dict = None) -> bool:
        """
        Determine if research should be performed.
        
        Args:
            research_type: Type of research to check
            context: Current system context
            
        Returns:
            Boolean indicating if research should proceed
        """
        # Check resource limits
        if not self._has_resources_available():
            return False
        
        # Check if already researching this topic
        if research_type in self.active_research:
            return False
        
        # Check scheduled interval
        if self._is_scheduled_time(research_type):
            return True
        
        # Check for event triggers
        if context and self._check_event_triggers(research_type, context):
            return True
        
        # Check for urgent needs
        if self._has_urgent_research_need(research_type, context):
            return True
        
        return False
    
    def schedule_research(self, research_type: str, priority: ResearchPriority, 
                         context: Dict = None, callback: Callable = None) -> str:
        """
        Schedule research for execution.
        
        Args:
            research_type: Type of research
            priority: Priority level
            context: Additional context
            callback: Function to call when research completes
            
        Returns:
            Research ID for tracking
        """
        research_id = f"{research_type}_{int(time.time() * 1000)}"
        
        research_item = {
            'id': research_id,
            'type': research_type,
            'priority': priority,
            'context': context or {},
            'callback': callback,
            'scheduled_at': datetime.now().isoformat(),
            'estimated_duration': self._estimate_research_duration(research_type)
        }
        
        # Add to priority queue (lower priority value = higher priority)
        priority_value = self._get_priority_value(priority)
        self.research_queue.put((priority_value, time.time(), research_item))
        
        return research_id
    
    def get_next_research(self) -> Optional[Dict]:
        """Get the next research item to execute."""
        if self.research_queue.empty():
            return None
        
        # Check resource availability
        if not self._has_resources_available():
            return None
        
        # Check concurrent research limit
        if len(self.active_research) >= self.max_concurrent_research:
            return None
        
        try:
            _, _, research_item = self.research_queue.get_nowait()
            
            # Mark as active
            self.active_research[research_item['id']] = {
                'item': research_item,
                'started_at': datetime.now(),
                'status': 'running'
            }
            
            return research_item
            
        except queue.Empty:
            return None
    
    def complete_research(self, research_id: str, success: bool, 
                         results: Dict = None, effectiveness: float = None):
        """
        Mark research as completed and update metrics.
        
        Args:
            research_id: ID of completed research
            success: Whether research was successful
            results: Research results
            effectiveness: Effectiveness score (0-1)
        """
        if research_id not in self.active_research:
            return
        
        research_info = self.active_research[research_id]
        research_item = research_info['item']
        
        # Calculate duration
        start_time = research_info['started_at']
        duration = (datetime.now() - start_time).total_seconds()
        
        # Update completion info
        completion_info = {
            'completed_at': datetime.now().isoformat(),
            'duration_seconds': duration,
            'success': success,
            'effectiveness': effectiveness or (0.8 if success else 0.2),
            'results': results or {}
        }
        
        # Update research effectiveness tracking
        research_type = research_item['type']
        if research_type not in self.research_effectiveness:
            self.research_effectiveness[research_type] = []
        
        self.research_effectiveness[research_type].append({
            'timestamp': completion_info['completed_at'],
            'effectiveness': completion_info['effectiveness'],
            'duration': duration
        })
        
        # Keep only recent effectiveness data
        if len(self.research_effectiveness[research_type]) > 20:
            self.research_effectiveness[research_type] = \
                self.research_effectiveness[research_type][-20:]
        
        # Update last research time
        self.last_research_times[research_type] = datetime.now()
        
        # Adapt scheduling based on effectiveness
        self._adapt_schedule(research_type, completion_info)
        
        # Execute callback if provided
        if research_item.get('callback'):
            try:
                research_item['callback'](research_id, completion_info)
            except Exception as e:
                print(f"Error executing research callback: {e}")
        
        # Remove from active research
        del self.active_research[research_id]
    
    def trigger_immediate_research(self, research_type: str, reason: str, 
                                  context: Dict = None) -> str:
        """
        Trigger immediate research due to an event.
        
        Args:
            research_type: Type of research needed
            reason: Reason for immediate research
            context: Event context
            
        Returns:
            Research ID
        """
        context = context or {}
        context['trigger_reason'] = reason
        context['triggered_at'] = datetime.now().isoformat()
        
        return self.schedule_research(
            research_type, 
            ResearchPriority.IMMEDIATE, 
            context
        )
    
    def _is_scheduled_time(self, research_type: str) -> bool:
        """Check if it's time for scheduled research."""
        if research_type not in self.current_intervals:
            return False
        
        last_time = self.last_research_times.get(research_type)
        if not last_time:
            return True  # Never done before
        
        interval = self.current_intervals[research_type]
        next_time = last_time + timedelta(seconds=interval)
        
        return datetime.now() >= next_time
    
    def _check_event_triggers(self, research_type: str, context: Dict) -> bool:
        """Check if any event triggers apply."""
        system_health = context.get('system_health', 'normal')
        
        # Performance-based triggers
        if research_type == 'critical_performance':
            claude_success = context.get('metrics', {}).get('claude_success_rate', 1)
            task_completion = context.get('metrics', {}).get('task_completion_rate', 1)
            
            if claude_success == 0 or task_completion < 0.1:
                return True
        
        # Health-based triggers
        if system_health in ['critical', 'degraded'] and research_type in [
            'critical_performance', 'task_analysis'
        ]:
            return True
        
        # Failure spike triggers
        recent_failures = context.get('recent_failures', [])
        if len(recent_failures) > 5 and research_type == 'task_analysis':
            return True
        
        return False
    
    def _has_urgent_research_need(self, research_type: str, context: Dict) -> bool:
        """Check for urgent research needs."""
        if not context:
            return False
        
        # Check if knowledge store lacks critical information
        if self.knowledge_store:
            relevant_research = self.knowledge_store.retrieve_research(
                research_type=research_type,
                min_quality=0.7,
                limit=1
            )
            
            if not relevant_research:
                # No good research available for critical areas
                if research_type in ['critical_performance', 'task_analysis']:
                    return True
        
        return False
    
    def _has_resources_available(self) -> bool:
        """Check if resources are available for research."""
        # Reset hourly counters
        current_hour = datetime.now().replace(minute=0, second=0)
        if current_hour > self.current_usage['hour_reset_time']:
            self.current_usage['api_calls_this_hour'] = 0
            self.current_usage['hour_reset_time'] = current_hour
        
        # Check API call limits
        if (self.current_usage['api_calls_this_hour'] >= 
            self.research_budget['api_calls_per_hour']):
            return False
        
        # Check concurrent research limit
        if len(self.active_research) >= self.max_concurrent_research:
            return False
        
        return True
    
    def _estimate_research_duration(self, research_type: str) -> int:
        """Estimate research duration in seconds."""
        duration_estimates = {
            'critical_performance': 600,    # 10 minutes
            'task_analysis': 900,           # 15 minutes
            'security_updates': 300,        # 5 minutes
            'technology_trends': 1200,      # 20 minutes
            'market_opportunities': 1800,   # 30 minutes
            'portfolio_review': 2400        # 40 minutes
        }
        
        return duration_estimates.get(research_type, 900)
    
    def _get_priority_value(self, priority: ResearchPriority) -> int:
        """Convert priority enum to numeric value for queue ordering."""
        priority_values = {
            ResearchPriority.IMMEDIATE: 1,
            ResearchPriority.CRITICAL: 2,
            ResearchPriority.HIGH: 3,
            ResearchPriority.MEDIUM: 4,
            ResearchPriority.LOW: 5
        }
        return priority_values.get(priority, 4)
    
    def _adapt_schedule(self, research_type: str, completion_info: Dict):
        """Adapt scheduling based on research effectiveness."""
        if research_type not in self.research_effectiveness:
            return
        
        # Calculate average effectiveness for this research type
        effectiveness_data = self.research_effectiveness[research_type]
        recent_effectiveness = [item['effectiveness'] for item in effectiveness_data[-5:]]
        avg_effectiveness = sum(recent_effectiveness) / len(recent_effectiveness)
        
        # Adapt interval based on effectiveness
        base_interval = self.base_intervals.get(research_type, 3600)
        
        if avg_effectiveness > 0.8:
            # High effectiveness - can research less frequently
            new_interval = min(base_interval * 1.5, base_interval * 2)
        elif avg_effectiveness < 0.4:
            # Low effectiveness - research more frequently to improve
            new_interval = max(base_interval * 0.5, base_interval * 0.3)
        else:
            # Normal effectiveness - keep base interval
            new_interval = base_interval
        
        self.current_intervals[research_type] = int(new_interval)
    
    def _on_performance_drop(self, context: Dict):
        """Handle performance drop events."""
        self.trigger_immediate_research(
            'critical_performance',
            'Performance degradation detected',
            context
        )
    
    def _on_task_failure_spike(self, context: Dict):
        """Handle task failure spike events."""
        self.trigger_immediate_research(
            'task_analysis',
            'Task failure rate spike detected',
            context
        )
    
    def _on_claude_failure(self, context: Dict):
        """Handle Claude interaction failures."""
        self.trigger_immediate_research(
            'critical_performance',
            'Claude interaction failure',
            context
        )
    
    def _on_security_alert(self, context: Dict):
        """Handle security alerts."""
        self.trigger_immediate_research(
            'security_updates',
            'Security alert triggered',
            context
        )
    
    def _on_system_error(self, context: Dict):
        """Handle system errors."""
        self.trigger_immediate_research(
            'critical_performance',
            'System error occurred',
            context
        )
    
    def trigger_event(self, event_type: str, context: Dict):
        """Trigger an event that may cause immediate research."""
        if event_type in self.event_triggers:
            self.event_triggers[event_type](context)
    
    def get_schedule_status(self) -> Dict:
        """Get current scheduling status."""
        return {
            'active_research': {
                research_id: {
                    'type': info['item']['type'],
                    'started_at': info['started_at'].isoformat(),
                    'status': info['status']
                }
                for research_id, info in self.active_research.items()
            },
            'queue_size': self.research_queue.qsize(),
            'current_intervals': self.current_intervals,
            'last_research_times': {
                research_type: time.isoformat() if time else None
                for research_type, time in self.last_research_times.items()
            },
            'resource_usage': self.current_usage,
            'research_effectiveness': {
                research_type: sum(item['effectiveness'] for item in data[-5:]) / len(data[-5:])
                for research_type, data in self.research_effectiveness.items()
                if data
            }
        }
    
    def update_resource_usage(self, api_calls: int = 0):
        """Update resource usage tracking."""
        self.current_usage['api_calls_this_hour'] += api_calls
    
    def get_next_scheduled_times(self) -> Dict[str, datetime]:
        """Get next scheduled research times for each type."""
        next_times = {}
        
        for research_type, interval in self.current_intervals.items():
            last_time = self.last_research_times.get(research_type)
            if last_time:
                next_time = last_time + timedelta(seconds=interval)
            else:
                next_time = datetime.now()  # Can run immediately
            
            next_times[research_type] = next_time
        
        return next_times
    
    def force_research_schedule_reset(self):
        """Reset all research schedules (for testing/emergency)."""
        self.last_research_times.clear()
        self.current_intervals = self.base_intervals.copy()
        
        # Clear queue
        while not self.research_queue.empty():
            try:
                self.research_queue.get_nowait()
            except queue.Empty:
                break