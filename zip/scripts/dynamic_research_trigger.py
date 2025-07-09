"""
Dynamic Research Trigger - Intelligently triggers research based on system performance and needs.

This module monitors system metrics and events to dynamically trigger research
when specific conditions are met, replacing fixed-interval research with need-based research.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque, defaultdict
import logging


class DynamicResearchTrigger:
    """Monitors system state and triggers research based on dynamic conditions."""
    
    def __init__(self, state_manager=None, research_engine=None):
        self.state_manager = state_manager
        self.research_engine = research_engine
        
        # Trigger conditions with thresholds
        self.trigger_conditions = {
            "performance_drop": {
                "enabled": True,
                "check_interval": 60,  # Check every minute
                "conditions": {
                    "claude_success_rate_drop": {"threshold": 20, "window": 300},  # 20% drop in 5 minutes
                    "task_failure_spike": {"threshold": 30, "window": 600},  # 30% failure in 10 minutes
                    "error_rate_increase": {"threshold": 50, "window": 300},  # 50% increase in errors
                }
            },
            "anomaly_detection": {
                "enabled": True,
                "check_interval": 120,  # Check every 2 minutes
                "conditions": {
                    "unusual_pattern": {"sensitivity": 0.8},
                    "new_error_types": {"threshold": 3},  # 3 new error types
                    "performance_outlier": {"std_deviations": 2}
                }
            },
            "opportunity_based": {
                "enabled": True,
                "check_interval": 300,  # Check every 5 minutes
                "conditions": {
                    "idle_resources": {"threshold": 0.7},  # 70% idle
                    "learning_plateau": {"duration": 3600},  # No improvement for 1 hour
                    "new_capability_needed": {"confidence": 0.8}
                }
            },
            "event_based": {
                "enabled": True,
                "events": {
                    "critical_error": {"immediate": True},
                    "new_project_added": {"delay": 60},
                    "major_task_failure": {"immediate": True},
                    "system_restart": {"delay": 300}
                }
            }
        }
        
        # Cooldown management
        self.cooldowns = {
            "global": {"duration": 300, "last_triggered": None},  # 5 min global cooldown
            "per_type": defaultdict(lambda: {"duration": 600, "last_triggered": None})  # 10 min per type
        }
        
        # Metrics tracking
        self.metrics_history = {
            "claude_success_rate": deque(maxlen=100),
            "task_completion_rate": deque(maxlen=100),
            "error_count": deque(maxlen=100),
            "performance_score": deque(maxlen=100)
        }
        
        # Event queue
        self.event_queue = deque(maxlen=1000)
        
        # Trigger history
        self.trigger_history = []
        
        # Statistics
        self.stats = {
            "total_triggers": 0,
            "triggers_by_type": defaultdict(int),
            "false_positives": 0,
            "successful_triggers": 0,
            "cooldown_blocks": 0
        }
        
        # Running state
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self):
        """Start monitoring for research triggers."""
        self.is_monitoring = True
        self.logger.info("Starting dynamic research trigger monitoring")
        
        # Start monitoring tasks
        tasks = [
            self._monitor_performance_drops(),
            self._monitor_anomalies(),
            self._monitor_opportunities(),
            self._monitor_events()
        ]
        
        self.monitoring_task = asyncio.gather(*tasks)
        
        try:
            await self.monitoring_task
        except asyncio.CancelledError:
            self.logger.info("Dynamic research trigger monitoring stopped")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
    
    async def _monitor_performance_drops(self):
        """Monitor for performance drops."""
        config = self.trigger_conditions["performance_drop"]
        
        while self.is_monitoring and config["enabled"]:
            try:
                # Collect current metrics
                metrics = self._collect_current_metrics()
                
                # Check each condition
                for condition_name, params in config["conditions"].items():
                    if await self._check_performance_condition(condition_name, params, metrics):
                        await self._trigger_research("performance_drop", {
                            "condition": condition_name,
                            "metrics": metrics,
                            "params": params
                        })
                
                await asyncio.sleep(config["check_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(config["check_interval"])
    
    async def _monitor_anomalies(self):
        """Monitor for anomalies in system behavior."""
        config = self.trigger_conditions["anomaly_detection"]
        
        while self.is_monitoring and config["enabled"]:
            try:
                # Detect anomalies
                anomalies = self._detect_anomalies()
                
                if anomalies:
                    await self._trigger_research("anomaly", {
                        "anomalies": anomalies,
                        "timestamp": datetime.now().isoformat()
                    })
                
                await asyncio.sleep(config["check_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in anomaly monitoring: {e}")
                await asyncio.sleep(config["check_interval"])
    
    async def _monitor_opportunities(self):
        """Monitor for research opportunities."""
        config = self.trigger_conditions["opportunity_based"]
        
        while self.is_monitoring and config["enabled"]:
            try:
                # Check for opportunities
                opportunities = self._identify_opportunities()
                
                for opportunity in opportunities:
                    await self._trigger_research("opportunity", opportunity)
                
                await asyncio.sleep(config["check_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in opportunity monitoring: {e}")
                await asyncio.sleep(config["check_interval"])
    
    async def _monitor_events(self):
        """Monitor system events."""
        config = self.trigger_conditions["event_based"]
        
        while self.is_monitoring and config["enabled"]:
            try:
                # Process event queue
                while self.event_queue:
                    event = self.event_queue.popleft()
                    
                    if event["type"] in config["events"]:
                        event_config = config["events"][event["type"]]
                        
                        if event_config.get("immediate"):
                            await self._trigger_research("event", event)
                        else:
                            delay = event_config.get("delay", 0)
                            await asyncio.sleep(delay)
                            await self._trigger_research("event", event)
                
                await asyncio.sleep(10)  # Check events every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in event monitoring: {e}")
                await asyncio.sleep(10)
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            "timestamp": datetime.now(),
            "claude_success_rate": 0,
            "task_completion_rate": 0,
            "error_count": 0,
            "active_tasks": 0,
            "system_health": "unknown"
        }
        
        if self.state_manager:
            try:
                state = self.state_manager.load_state()
                
                # Claude success rate
                claude_data = state.get("performance", {}).get("claude_interactions", {})
                if claude_data.get("total_attempts", 0) > 0:
                    metrics["claude_success_rate"] = (
                        claude_data.get("successful", 0) / claude_data["total_attempts"] * 100
                    )
                
                # Task completion rate
                task_data = state.get("performance", {}).get("task_completion", {})
                if task_data.get("total_tasks", 0) > 0:
                    metrics["task_completion_rate"] = (
                        task_data.get("completed_tasks", 0) / task_data["total_tasks"] * 100
                    )
                
                # Error count
                metrics["error_count"] = len(state.get("recent_errors", []))
                
                # Active tasks
                tasks = state.get("task_state", {}).get("tasks", [])
                metrics["active_tasks"] = len([t for t in tasks if t.get("status") == "in_progress"])
                
                # System health
                if metrics["claude_success_rate"] < 30:
                    metrics["system_health"] = "critical"
                elif metrics["claude_success_rate"] < 70:
                    metrics["system_health"] = "degraded"
                else:
                    metrics["system_health"] = "healthy"
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
        
        # Store in history
        self.metrics_history["claude_success_rate"].append(
            (metrics["timestamp"], metrics["claude_success_rate"])
        )
        self.metrics_history["task_completion_rate"].append(
            (metrics["timestamp"], metrics["task_completion_rate"])
        )
        self.metrics_history["error_count"].append(
            (metrics["timestamp"], metrics["error_count"])
        )
        
        return metrics
    
    async def _check_performance_condition(self, condition_name: str, 
                                         params: Dict, metrics: Dict) -> bool:
        """Check if a performance condition is met."""
        if condition_name == "claude_success_rate_drop":
            return self._check_metric_drop(
                "claude_success_rate", 
                params["threshold"], 
                params["window"]
            )
        
        elif condition_name == "task_failure_spike":
            current_failure = 100 - metrics.get("task_completion_rate", 100)
            history = self.metrics_history["task_completion_rate"]
            
            if len(history) > 5:
                window_start = datetime.now() - timedelta(seconds=params["window"])
                window_data = [100 - rate for ts, rate in history if ts > window_start]
                
                if window_data:
                    avg_failure = sum(window_data) / len(window_data)
                    return current_failure > avg_failure + params["threshold"]
        
        elif condition_name == "error_rate_increase":
            return self._check_metric_increase(
                "error_count",
                params["threshold"],
                params["window"]
            )
        
        return False
    
    def _check_metric_drop(self, metric: str, threshold: float, window: int) -> bool:
        """Check if a metric has dropped by threshold within window."""
        history = self.metrics_history.get(metric, [])
        
        if len(history) < 2:
            return False
        
        current_value = history[-1][1]
        window_start = datetime.now() - timedelta(seconds=window)
        
        # Get values from window
        window_values = [val for ts, val in history if ts > window_start]
        
        if window_values:
            max_value = max(window_values)
            if max_value > 0:
                drop_percentage = ((max_value - current_value) / max_value) * 100
                return drop_percentage >= threshold
        
        return False
    
    def _check_metric_increase(self, metric: str, threshold: float, window: int) -> bool:
        """Check if a metric has increased by threshold within window."""
        history = self.metrics_history.get(metric, [])
        
        if len(history) < 2:
            return False
        
        current_value = history[-1][1]
        window_start = datetime.now() - timedelta(seconds=window)
        
        # Get values from window
        window_values = [val for ts, val in history if ts > window_start]
        
        if window_values:
            min_value = min(window_values)
            if min_value > 0:
                increase_percentage = ((current_value - min_value) / min_value) * 100
                return increase_percentage >= threshold
            elif min_value == 0 and current_value > 0:
                return True
        
        return False
    
    def _detect_anomalies(self) -> List[Dict]:
        """Detect anomalies in system behavior."""
        anomalies = []
        
        # Check for unusual patterns in metrics
        for metric_name, history in self.metrics_history.items():
            if len(history) > 10:
                # Convert deque to list for slicing
                history_list = list(history)
                values = [val for _, val in history_list[-20:]]
                
                # Simple statistical anomaly detection
                if values:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    
                    current_value = values[-1]
                    
                    # Check if current value is an outlier
                    if std_dev > 0:
                        z_score = abs((current_value - mean) / std_dev)
                        if z_score > 2:  # 2 standard deviations
                            anomalies.append({
                                "type": "statistical_outlier",
                                "metric": metric_name,
                                "value": current_value,
                                "z_score": z_score,
                                "mean": mean,
                                "std_dev": std_dev
                            })
        
        # Check for new error patterns
        if self.state_manager:
            state = self.state_manager.load_state()
            recent_errors = state.get("recent_errors", [])
            
            if recent_errors:
                # Group errors by type
                error_types = defaultdict(int)
                for error in recent_errors[-50:]:
                    error_type = error.get("type", "unknown")
                    error_types[error_type] += 1
                
                # Check for new error types
                historical_error_types = getattr(self, "_historical_error_types", set())
                new_error_types = set(error_types.keys()) - historical_error_types
                
                if len(new_error_types) >= 3:
                    anomalies.append({
                        "type": "new_error_types",
                        "new_types": list(new_error_types),
                        "count": len(new_error_types)
                    })
                
                self._historical_error_types = set(error_types.keys())
        
        return anomalies
    
    def _identify_opportunities(self) -> List[Dict]:
        """Identify research opportunities."""
        opportunities = []
        
        if self.state_manager:
            state = self.state_manager.load_state()
            metrics = self._collect_current_metrics()
            
            # Check for idle resources
            if metrics["active_tasks"] == 0 and metrics["system_health"] == "healthy":
                opportunities.append({
                    "type": "idle_resources",
                    "description": "System is idle and healthy, good time for proactive research",
                    "priority": "low"
                })
            
            # Check for learning plateau
            history = self.metrics_history["claude_success_rate"]
            if len(history) > 20:
                # Convert deque to list for slicing
                history_list = list(history)
                recent_values = [val for _, val in history_list[-20:]]
                if recent_values:
                    # Check if values are stable (low variance)
                    mean = sum(recent_values) / len(recent_values)
                    variance = sum((x - mean) ** 2 for x in recent_values) / len(recent_values)
                    
                    if variance < 5 and mean < 80:  # Stable but not optimal
                        opportunities.append({
                            "type": "learning_plateau",
                            "description": "Performance has plateaued, research needed for breakthrough",
                            "current_performance": mean,
                            "priority": "medium"
                        })
            
            # Check for capability gaps
            task_failures = []
            tasks = state.get("task_state", {}).get("tasks", [])
            for task in tasks[-50:]:  # Last 50 tasks
                if task.get("status") == "failed":
                    task_failures.append(task)
            
            if len(task_failures) > 5:
                # Analyze failure patterns
                failure_reasons = defaultdict(int)
                for task in task_failures:
                    reason = task.get("failure_reason", "unknown")
                    failure_reasons[reason] += 1
                
                # Most common failure reason might indicate capability gap
                if failure_reasons:
                    most_common = max(failure_reasons.items(), key=lambda x: x[1])
                    if most_common[1] >= 3:
                        opportunities.append({
                            "type": "capability_gap",
                            "description": f"Repeated failures due to: {most_common[0]}",
                            "failure_count": most_common[1],
                            "priority": "high"
                        })
        
        return opportunities
    
    async def _trigger_research(self, trigger_type: str, context: Dict):
        """Trigger research if not in cooldown."""
        # Check cooldowns
        if not self._check_cooldown(trigger_type):
            self.stats["cooldown_blocks"] += 1
            return
        
        # Prepare research trigger
        trigger_info = {
            "trigger_type": trigger_type,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "metrics_snapshot": self._get_metrics_snapshot()
        }
        
        # Log trigger
        self.logger.info(f"Triggering research: {trigger_type} - {context}")
        
        # Execute research if engine is available
        if self.research_engine:
            try:
                # Determine research priority
                priority = self._determine_priority(trigger_type, context)
                
                # Create research request
                research_request = self._create_research_request(trigger_type, context, priority)
                
                # Trigger emergency research for critical issues
                if priority == "critical":
                    await self.research_engine.execute_emergency_research(research_request)
                else:
                    # Add to research queue
                    await self._queue_research(research_request)
                
                # Update statistics
                self.stats["total_triggers"] += 1
                self.stats["triggers_by_type"][trigger_type] += 1
                self.stats["successful_triggers"] += 1
                
                # Record trigger
                self.trigger_history.append(trigger_info)
                
                # Update cooldowns
                self._update_cooldowns(trigger_type)
                
            except Exception as e:
                self.logger.error(f"Error triggering research: {e}")
        else:
            self.logger.warning("Research engine not available")
    
    def _check_cooldown(self, trigger_type: str) -> bool:
        """Check if trigger is allowed based on cooldowns."""
        now = datetime.now()
        
        # Check global cooldown
        global_cooldown = self.cooldowns["global"]
        if global_cooldown["last_triggered"]:
            time_since = (now - global_cooldown["last_triggered"]).total_seconds()
            if time_since < global_cooldown["duration"]:
                return False
        
        # Check per-type cooldown
        type_cooldown = self.cooldowns["per_type"][trigger_type]
        if type_cooldown["last_triggered"]:
            time_since = (now - type_cooldown["last_triggered"]).total_seconds()
            if time_since < type_cooldown["duration"]:
                return False
        
        return True
    
    def _update_cooldowns(self, trigger_type: str):
        """Update cooldown timestamps."""
        now = datetime.now()
        self.cooldowns["global"]["last_triggered"] = now
        self.cooldowns["per_type"][trigger_type]["last_triggered"] = now
    
    def _determine_priority(self, trigger_type: str, context: Dict) -> str:
        """Determine research priority based on trigger type and context."""
        if trigger_type == "event" and context.get("type") == "critical_error":
            return "critical"
        
        if trigger_type == "performance_drop":
            metrics = context.get("metrics", {})
            if metrics.get("system_health") == "critical":
                return "critical"
            elif metrics.get("claude_success_rate", 100) < 30:
                return "high"
            else:
                return "medium"
        
        if trigger_type == "anomaly":
            anomalies = context.get("anomalies", [])
            if any(a["type"] == "new_error_types" for a in anomalies):
                return "high"
            else:
                return "medium"
        
        if trigger_type == "opportunity":
            return context.get("priority", "low")
        
        return "medium"
    
    def _create_research_request(self, trigger_type: str, context: Dict, priority: str) -> Dict:
        """Create a research request based on trigger."""
        request = {
            "trigger_type": trigger_type,
            "priority": priority,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "research_areas": []
        }
        
        # Determine research areas based on trigger
        if trigger_type == "performance_drop":
            condition = context.get("condition", "")
            if "claude" in condition:
                request["research_areas"] = ["claude_interactions", "efficiency"]
            elif "task" in condition:
                request["research_areas"] = ["task_performance", "efficiency"]
            elif "error" in condition:
                request["research_areas"] = ["error_handling", "reliability"]
        
        elif trigger_type == "anomaly":
            request["research_areas"] = ["anomaly_analysis", "pattern_detection"]
        
        elif trigger_type == "opportunity":
            opp_type = context.get("type", "")
            if opp_type == "learning_plateau":
                request["research_areas"] = ["breakthrough_strategies", "innovation"]
            elif opp_type == "capability_gap":
                request["research_areas"] = ["capability_development", "skill_acquisition"]
        
        elif trigger_type == "event":
            event_type = context.get("type", "")
            if event_type == "new_project_added":
                request["research_areas"] = ["project_optimization", "growth"]
            elif event_type == "critical_error":
                request["research_areas"] = ["error_recovery", "resilience"]
        
        return request
    
    async def _queue_research(self, research_request: Dict):
        """Queue research request for execution."""
        # For now, just log it
        self.logger.info(f"Queuing research request: {research_request}")
        
        # In a full implementation, this would add to a priority queue
        # that the research engine processes
    
    def _get_metrics_snapshot(self) -> Dict:
        """Get current metrics snapshot."""
        snapshot = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                current = history[-1][1]
                values = [val for _, val in history]
                
                snapshot[metric_name] = {
                    "current": current,
                    "average": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
        
        return snapshot
    
    def add_event(self, event_type: str, event_data: Dict):
        """Add an event to the monitoring queue."""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        self.event_queue.append(event)
        self.logger.info(f"Event added: {event_type}")
    
    def get_statistics(self) -> Dict:
        """Get trigger statistics."""
        return {
            **self.stats,
            "recent_triggers": self.trigger_history[-10:],
            "cooldown_status": {
                "global": self._get_cooldown_status(self.cooldowns["global"]),
                "per_type": {
                    trigger_type: self._get_cooldown_status(cooldown)
                    for trigger_type, cooldown in self.cooldowns["per_type"].items()
                }
            },
            "monitoring_status": "active" if self.is_monitoring else "inactive"
        }
    
    def _get_cooldown_status(self, cooldown: Dict) -> Dict:
        """Get cooldown status."""
        if not cooldown["last_triggered"]:
            return {"active": False, "remaining": 0}
        
        elapsed = (datetime.now() - cooldown["last_triggered"]).total_seconds()
        remaining = max(0, cooldown["duration"] - elapsed)
        
        return {
            "active": remaining > 0,
            "remaining": remaining
        }
    
    def adjust_sensitivity(self, trigger_type: str, adjustment: float):
        """Adjust trigger sensitivity."""
        if trigger_type in self.trigger_conditions:
            # Adjust thresholds based on adjustment factor
            conditions = self.trigger_conditions[trigger_type].get("conditions", {})
            for condition, params in conditions.items():
                if "threshold" in params:
                    params["threshold"] *= adjustment
                elif "sensitivity" in params:
                    params["sensitivity"] *= adjustment
            
            self.logger.info(f"Adjusted {trigger_type} sensitivity by {adjustment}")
    
    def enable_trigger(self, trigger_type: str, enabled: bool = True):
        """Enable or disable a trigger type."""
        if trigger_type in self.trigger_conditions:
            self.trigger_conditions[trigger_type]["enabled"] = enabled
            self.logger.info(f"Trigger {trigger_type} {'enabled' if enabled else 'disabled'}")