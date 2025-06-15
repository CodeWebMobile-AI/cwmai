"""
Intelligence Integration Module

Connects the intelligence hub to all system components for comprehensive learning.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

# Import components
try:
    from intelligence_hub import IntelligenceHub, EventType, get_intelligence_hub
    from async_state_manager import AsyncStateManager, get_async_state_manager
    from ai_brain import IntelligentAIBrain
    INTEGRATION_AVAILABLE = True
    
    # Import swarm intelligence separately to handle potential issues
    try:
        from swarm_intelligence import RealSwarmIntelligence
        SWARM_AVAILABLE = True
    except ImportError:
        SWARM_AVAILABLE = False
        RealSwarmIntelligence = None
        
except ImportError:
    INTEGRATION_AVAILABLE = False
    SWARM_AVAILABLE = False


class IntelligenceIntegrator:
    """Manages integration between intelligence hub and system components."""
    
    def __init__(self):
        """Initialize intelligence integrator."""
        self.logger = logging.getLogger(f"{__name__}.IntelligenceIntegrator")
        self.intelligence_hub: Optional[IntelligenceHub] = None
        self.async_state_manager: Optional[AsyncStateManager] = None
        self.connected_components = set()
        
    async def initialize(self):
        """Initialize intelligence integration."""
        if not INTEGRATION_AVAILABLE:
            self.logger.warning("Intelligence integration not available - missing dependencies")
            return False
        
        try:
            self.logger.info("Initializing intelligence integration...")
            
            # Get hub and state manager
            self.intelligence_hub = await get_intelligence_hub()
            self.async_state_manager = await get_async_state_manager()
            
            # Set up event subscriptions
            await self._setup_event_subscriptions()
            
            self.logger.info("Intelligence integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize intelligence integration: {e}")
            return False
    
    async def connect_swarm_intelligence(self, swarm):
        """Connect swarm intelligence to the hub."""
        if not self.intelligence_hub:
            return False
        
        try:
            # Set intelligence hub reference in swarm
            await swarm.set_intelligence_hub(self.intelligence_hub)
            
            # Subscribe to swarm events
            self.intelligence_hub.subscribe_to_component(
                "swarm_intelligence",
                self._handle_swarm_event
            )
            
            self.connected_components.add("swarm_intelligence")
            self.logger.info("Swarm intelligence connected to intelligence hub")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect swarm intelligence: {e}")
            return False
    
    async def connect_ai_brain(self, ai_brain):
        """Connect AI brain to the hub."""
        if not self.intelligence_hub:
            return False
        
        try:
            # Store reference for event emission
            ai_brain.intelligence_hub = self.intelligence_hub
            
            # Subscribe to AI brain events
            self.intelligence_hub.subscribe_to_component(
                "ai_brain",
                self._handle_ai_brain_event
            )
            
            self.connected_components.add("ai_brain")
            self.logger.info("AI brain connected to intelligence hub")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect AI brain: {e}")
            return False
    
    async def connect_production_orchestrator(self, orchestrator):
        """Connect production orchestrator to the hub."""
        if not self.intelligence_hub:
            return False
        
        try:
            # Store references
            orchestrator.intelligence_hub = self.intelligence_hub
            orchestrator.async_state_manager = self.async_state_manager
            
            # Subscribe to orchestrator events
            self.intelligence_hub.subscribe_to_component(
                "production_orchestrator",
                self._handle_orchestrator_event
            )
            
            self.connected_components.add("production_orchestrator")
            self.logger.info("Production orchestrator connected to intelligence hub")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect production orchestrator: {e}")
            return False
    
    async def _setup_event_subscriptions(self):
        """Set up global event subscriptions."""
        if not self.intelligence_hub:
            return
        
        # Subscribe to all decision events
        self.intelligence_hub.subscribe_to_events(
            EventType.DECISION_MADE,
            self._handle_decision_event
        )
        
        # Subscribe to task completion events
        self.intelligence_hub.subscribe_to_events(
            EventType.TASK_COMPLETED,
            self._handle_task_completion_event
        )
        
        # Subscribe to failure events
        self.intelligence_hub.subscribe_to_events(
            EventType.TASK_FAILED,
            self._handle_failure_event
        )
        
        # Subscribe to pattern detection
        self.intelligence_hub.subscribe_to_patterns(
            self._handle_pattern_detection
        )
        
        # Subscribe to insight generation
        self.intelligence_hub.subscribe_to_insights(
            self._handle_insight_generation
        )
    
    async def _handle_swarm_event(self, event):
        """Handle swarm intelligence events."""
        try:
            # Extract insights from swarm analysis
            if event.data.get("phase") == "complete":
                # Store swarm performance data
                if self.async_state_manager:
                    await self.async_state_manager.update(
                        "intelligence_hub.swarm_performance",
                        {
                            "last_analysis": event.timestamp.isoformat(),
                            "duration": event.data.get("duration", 0),
                            "consensus_priority": event.data.get("consensus_priority", 5)
                        }
                    )
                
                self.logger.debug(f"Processed swarm analysis event: {event.id}")
                
        except Exception as e:
            self.logger.error(f"Error handling swarm event: {e}")
    
    async def _handle_ai_brain_event(self, event):
        """Handle AI brain events."""
        try:
            # Track AI brain decision patterns
            if self.async_state_manager:
                await self.async_state_manager.update(
                    "intelligence_hub.ai_brain_activity",
                    {
                        "last_activity": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "data_summary": str(event.data)[:100]
                    }
                )
            
            self.logger.debug(f"Processed AI brain event: {event.id}")
            
        except Exception as e:
            self.logger.error(f"Error handling AI brain event: {e}")
    
    async def _handle_orchestrator_event(self, event):
        """Handle production orchestrator events."""
        try:
            # Track orchestrator state changes
            if self.async_state_manager:
                await self.async_state_manager.update(
                    "intelligence_hub.orchestrator_state",
                    {
                        "last_change": event.timestamp.isoformat(),
                        "action": event.data.get("action", "unknown"),
                        "mode": event.data.get("mode", "unknown")
                    }
                )
            
            self.logger.debug(f"Processed orchestrator event: {event.id}")
            
        except Exception as e:
            self.logger.error(f"Error handling orchestrator event: {e}")
    
    async def _handle_decision_event(self, event):
        """Handle decision-making events."""
        try:
            # Store decision patterns for learning
            decision_data = {
                "timestamp": event.timestamp.isoformat(),
                "source": event.source_component,
                "decision_data": event.data
            }
            
            if self.async_state_manager:
                # Get existing decisions
                existing_decisions = await self.async_state_manager.get(
                    "intelligence_hub.decision_history", []
                )
                
                # Add new decision (keep last 100)
                existing_decisions.append(decision_data)
                if len(existing_decisions) > 100:
                    existing_decisions = existing_decisions[-100:]
                
                await self.async_state_manager.update(
                    "intelligence_hub.decision_history",
                    existing_decisions
                )
            
            self.logger.debug(f"Stored decision event for learning: {event.id}")
            
        except Exception as e:
            self.logger.error(f"Error handling decision event: {e}")
    
    async def _handle_task_completion_event(self, event):
        """Handle task completion events."""
        try:
            # Track successful task patterns
            completion_data = {
                "timestamp": event.timestamp.isoformat(),
                "source": event.source_component,
                "task_data": event.data,
                "outcome": "success"
            }
            
            if self.async_state_manager:
                await self._store_task_outcome(completion_data)
            
            self.logger.debug(f"Stored task completion for learning: {event.id}")
            
        except Exception as e:
            self.logger.error(f"Error handling task completion event: {e}")
    
    async def _handle_failure_event(self, event):
        """Handle task failure events."""
        try:
            # Track failure patterns for learning
            failure_data = {
                "timestamp": event.timestamp.isoformat(),
                "source": event.source_component,
                "task_data": event.data,
                "outcome": "failure"
            }
            
            if self.async_state_manager:
                await self._store_task_outcome(failure_data)
            
            self.logger.warning(f"Stored task failure for analysis: {event.id}")
            
        except Exception as e:
            self.logger.error(f"Error handling failure event: {e}")
    
    async def _store_task_outcome(self, outcome_data):
        """Store task outcome for pattern analysis."""
        try:
            # Get existing outcomes
            existing_outcomes = await self.async_state_manager.get(
                "intelligence_hub.task_outcomes", []
            )
            
            # Add new outcome (keep last 200)
            existing_outcomes.append(outcome_data)
            if len(existing_outcomes) > 200:
                existing_outcomes = existing_outcomes[-200:]
            
            await self.async_state_manager.update(
                "intelligence_hub.task_outcomes",
                existing_outcomes
            )
            
        except Exception as e:
            self.logger.error(f"Error storing task outcome: {e}")
    
    async def _handle_pattern_detection(self, pattern):
        """Handle pattern detection events."""
        try:
            # Store detected patterns
            pattern_data = {
                "id": pattern.id,
                "type": pattern.pattern_type,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "frequency": pattern.frequency,
                "detected_at": pattern.last_seen.isoformat()
            }
            
            if self.async_state_manager:
                # Get existing patterns
                existing_patterns = await self.async_state_manager.get(
                    "intelligence_hub.detected_patterns", []
                )
                
                # Add new pattern (keep last 50)
                existing_patterns.append(pattern_data)
                if len(existing_patterns) > 50:
                    existing_patterns = existing_patterns[-50:]
                
                await self.async_state_manager.update(
                    "intelligence_hub.detected_patterns",
                    existing_patterns
                )
            
            self.logger.info(f"Stored detected pattern: {pattern.description}")
            
        except Exception as e:
            self.logger.error(f"Error handling pattern detection: {e}")
    
    async def _handle_insight_generation(self, insight):
        """Handle insight generation events."""
        try:
            # Store generated insights
            insight_data = {
                "id": insight.id,
                "type": insight.insight_type.value,
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence,
                "recommendations": insight.actionable_recommendations,
                "generated_at": insight.timestamp.isoformat()
            }
            
            if self.async_state_manager:
                # Get existing insights
                existing_insights = await self.async_state_manager.get(
                    "intelligence_hub.generated_insights", []
                )
                
                # Add new insight (keep last 30)
                existing_insights.append(insight_data)
                if len(existing_insights) > 30:
                    existing_insights = existing_insights[-30:]
                
                await self.async_state_manager.update(
                    "intelligence_hub.generated_insights",
                    existing_insights
                )
            
            self.logger.info(f"Stored generated insight: {insight.title}")
            
        except Exception as e:
            self.logger.error(f"Error handling insight generation: {e}")
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            "integration_available": INTEGRATION_AVAILABLE,
            "hub_initialized": self.intelligence_hub is not None,
            "state_manager_initialized": self.async_state_manager is not None,
            "connected_components": list(self.connected_components),
            "total_components": len(self.connected_components)
        }
    
    async def shutdown(self):
        """Shutdown intelligence integration."""
        try:
            if self.intelligence_hub:
                await self.intelligence_hub.stop()
            
            if self.async_state_manager:
                await self.async_state_manager.shutdown()
            
            self.logger.info("Intelligence integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during intelligence integration shutdown: {e}")


# Global integrator instance
_global_integrator: Optional[IntelligenceIntegrator] = None


async def get_intelligence_integrator() -> IntelligenceIntegrator:
    """Get or create global intelligence integrator."""
    global _global_integrator
    if _global_integrator is None:
        _global_integrator = IntelligenceIntegrator()
        await _global_integrator.initialize()
    return _global_integrator


async def connect_component_to_intelligence(component, component_name: str) -> bool:
    """Convenience function to connect a component to intelligence hub."""
    integrator = await get_intelligence_integrator()
    
    if component_name == "swarm_intelligence":
        return await integrator.connect_swarm_intelligence(component)
    elif component_name == "ai_brain":
        return await integrator.connect_ai_brain(component)
    elif component_name == "production_orchestrator":
        return await integrator.connect_production_orchestrator(component)
    else:
        integrator.logger.warning(f"Unknown component type: {component_name}")
        return False